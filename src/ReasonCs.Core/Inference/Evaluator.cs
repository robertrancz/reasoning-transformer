using ReasonCs.Core.Data;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Inference;

/// <summary>
/// Evaluates a trained model on a held-out test set using greedy decoding.
/// Records accuracy as the baseline to compare against MCTS later.
///
/// PyTorch equivalent:
///   model.eval()
///   correct = 0
///   with torch.no_grad():
///       for prompt, expected_answer in test_loader:
///           logits   = model(prompt)
///           pred_ids = logits.argmax(dim=-1)   # greedy: pick highest-prob token
///           if tokenizer.decode(pred_ids) == expected_answer:
///               correct += 1
///   accuracy = correct / len(test_loader)
///
/// Key differences:
///   model.eval()       — disables dropout/BatchNorm (not applicable here, but good practice)
///   torch.no_grad()    — disables gradient tracking during inference to save memory.
///                        We don't track gradients at all during eval, so this is implicit.
///   logits.argmax(-1)  — greedy decoding: always pick the most probable next token.
///
/// Accuracy definition (used here):
///   The tokenizer is character-level (one token = one character).
///   A prediction is CORRECT if the first answer.Length generated characters
///   exactly equal the expected answer string.  Tokens beyond answer.Length are
///   forced-generation noise (the model keeps generating after the answer ends)
///   and are ignored.
///
///   Examples (answer="0", 1 char; model decodes 4 tokens):
///     decoded="0   " → prediction="0" == "0" → CORRECT  (noise = spaces)
///     decoded="02XY" → prediction="0" == "0" → CORRECT  (noise = "2XY"; first char is right)
///     decoded="7XYZ" → prediction="7" ≠ "0" → WRONG
///
///   Examples (answer="30", 2 chars):
///     decoded="30  " → prediction="30" == "30" → CORRECT
///     decoded="3012" → prediction="30" == "30" → CORRECT  (noise = "12")
///     decoded="31XY" → prediction="31" ≠ "30" → WRONG
///
///   NOTE: "02" in the log for answer "0" looks confusing but IS correct —
///   the model predicted "0" (right), then "2" as noise.  The log shows
///   exactly prediction.Length chars so this confusion no longer arises.
/// </summary>
public sealed class Evaluator
{
    /// <summary>
    /// Result of one evaluation run.
    /// Accuracy = Correct / Total  ∈ [0, 1].
    /// </summary>
    public record EvalResult(int Correct, int Total, float Accuracy)
    {
        public override string ToString() =>
            $"{Correct}/{Total} = {Accuracy:P1}";
    }

    /// <summary>
    /// Run greedy-decoding accuracy on the given test problems.
    ///
    /// For each problem the model is given the prompt (everything up to and
    /// including "?") and asked to generate up to <paramref name="maxNewTokens"/>
    /// tokens.  A prediction is counted correct if its first
    /// <c>problem.Answer.Length</c> decoded characters equal <c>problem.Answer</c>.
    ///
    /// When <paramref name="logger"/> is provided every individual prediction is
    /// written to the log file (file-only — too verbose for the console).
    /// The aggregate result is always logged to both console and file.
    /// </summary>
    /// <summary>
    /// Run greedy-decoding accuracy in <b>chain-of-thought mode</b>.
    ///
    /// The model generates <c>numSteps × StepWidth</c> tokens.  A prediction
    /// is correct if the LAST step's decoded value (trimmed of padding spaces)
    /// equals <c>problem.Answer</c>.
    ///
    /// This evaluator pairs with the CoT training format
    /// (<c>problem.FullTextChain</c>) and PRM-guided MCTS.
    /// </summary>
    public static EvalResult RunGreedyCoT(
        TransformerModel model,
        Tokenizer tokenizer,
        IEnumerable<SyntheticDataset.Problem> testSet,
        RunLogger? logger = null)
    {
        int correct = 0, total = 0;

        foreach (var problem in testSet)
        {
            int maxNew     = problem.NumSteps * SyntheticDataset.Problem.StepWidth;
            var promptToks = tokenizer.Encode(problem.Prompt);
            var allToks    = model.GreedyDecode(promptToks, maxNew);
            var newToks    = allToks[promptToks.Length..];
            var decoded    = tokenizer.Decode(newToks);

            // Last step starts at (NumSteps-1)*StepWidth
            int lastStart  = (problem.NumSteps - 1) * SyntheticDataset.Problem.StepWidth;
            string lastStep = decoded.Length > lastStart
                ? decoded[lastStart..Math.Min(lastStart + SyntheticDataset.Problem.StepWidth,
                                              decoded.Length)].TrimEnd()
                : "";

            bool isCorrect = string.Equals(lastStep, problem.Answer, StringComparison.Ordinal);
            if (isCorrect) correct++;
            total++;

            logger?.Prediction(problem.Prompt, problem.Answer,
                decoded.Length >= decoded.Length ? decoded[..Math.Min(decoded.Length, maxNew)] : decoded,
                isCorrect);
        }

        float accuracy = total == 0 ? 0f : (float)correct / total;
        var   result   = new EvalResult(correct, total, accuracy);
        logger?.Eval(correct, total, accuracy);
        return result;
    }

    public static EvalResult RunGreedy(
        TransformerModel model,
        Tokenizer tokenizer,
        IEnumerable<SyntheticDataset.Problem> testSet,
        int maxNewTokens = 4,
        RunLogger? logger = null)
    {
        int correct = 0, total = 0;

        foreach (var problem in testSet)
        {
            var promptTokens = tokenizer.Encode(problem.Prompt);

            // Greedy-decode up to maxNewTokens beyond the prompt
            var allTokens = model.GreedyDecode(promptTokens, maxNewTokens);
            var newTokens = allTokens[promptTokens.Length..];
            var decoded   = tokenizer.Decode(newTokens);

            // The tokenizer is character-level: one token = one character.
            // Only the first answer.Length characters are the model's actual
            // answer; characters beyond that are forced-generation noise.
            // Take exactly answer.Length chars and compare — this is equivalent
            // to StartsWith but also fixes the log display (no confusing noise).
            string prediction = decoded.Length >= problem.Answer.Length
                ? decoded[..problem.Answer.Length]
                : decoded;
            bool isCorrect = string.Equals(prediction, problem.Answer, StringComparison.Ordinal);
            if (isCorrect) correct++;
            total++;

            // Per-prediction log line (file only — 500 lines is noisy on console).
            // Log shows exactly 'prediction' (answer.Length chars) — no trailing
            // noise so "expect='0', got='0'" is unambiguous.
            logger?.Prediction(problem.Prompt, problem.Answer, prediction, isCorrect);
        }

        float accuracy = total == 0 ? 0f : (float)correct / total;
        var   result   = new EvalResult(correct, total, accuracy);

        logger?.Eval(correct, total, accuracy);

        return result;
    }
}
