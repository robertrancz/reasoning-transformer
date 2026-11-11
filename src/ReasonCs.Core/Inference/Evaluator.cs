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
///   A prediction is CORRECT if the first len(expected_answer) decoded characters
///   exactly match the expected answer string.  Any trailing generated tokens are
///   ignored.  This mirrors how real evaluation pipelines check final answers.
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

            // Correct if decoded output starts with the expected answer
            bool isCorrect = decoded.StartsWith(problem.Answer, StringComparison.Ordinal);
            if (isCorrect) correct++;
            total++;

            // Per-prediction log line (file only — 500 lines is noisy on console)
            logger?.Prediction(problem.Prompt, problem.Answer,
                               decoded[..Math.Min(decoded.Length, problem.Answer.Length + 2)],
                               isCorrect);
        }

        float accuracy = total == 0 ? 0f : (float)correct / total;
        var   result   = new EvalResult(correct, total, accuracy);

        logger?.Eval(correct, total, accuracy);

        return result;
    }
}
