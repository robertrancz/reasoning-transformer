using ReasonCs.Core.Data;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Reasoning;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Inference;

/// <summary>
/// Evaluates <see cref="MctsSearch"/> accuracy on a held-out test set.
///
/// Pattern: for each problem, call <see cref="MctsSearch.Search"/> on the
/// prompt tokens, decode the continuation, and check whether it starts with
/// the expected answer.  Mirrors <see cref="Evaluator.RunGreedy"/> exactly
/// so both baselines can be compared on the same test set.
///
/// PyTorch equivalent:
///   # No PyTorch equivalent — MCTS is search, not a neural network.
///   # The closest concept is vLLM's best-of-N sampling with a verifier:
///   outputs = llm.generate(prompt, n=budget, temperature=T)
///   scores  = [reward(o) for o in outputs]
///   best    = outputs[argmax(scores)]
///   correct = best.startswith(expected_answer)
/// </summary>
public static class MctsEvaluator
{
    /// <summary>
    /// Run MCTS on every problem in <paramref name="testSet"/> and return aggregate
    /// accuracy.  Can be compared directly to <see cref="Evaluator.RunGreedy"/>.
    ///
    /// Each per-prediction PRED line is written to the log file only (file-only —
    /// too verbose for the console).
    /// </summary>
    public static Evaluator.EvalResult RunMcts(
        MctsSearch                            mcts,
        Tokenizer                             tokenizer,
        IEnumerable<SyntheticDataset.Problem> testSet,
        RunLogger?                            logger = null)
    {
        int correct = 0, total = 0;

        foreach (var problem in testSet)
        {
            var promptTokens = tokenizer.Encode(problem.Prompt);

            // MCTS returns the full sequence (prompt + best continuation)
            var bestTokens = mcts.Search(promptTokens);
            var newTokens  = bestTokens[promptTokens.Length..];
            var decoded    = tokenizer.Decode(newTokens);

            bool isCorrect;
            string logPrediction;

            if (mcts.MaxDepth > 1)
            {
                // Chain-of-thought mode: check the last step value
                int stepW     = SyntheticDataset.Problem.StepWidth;
                int lastStart = (problem.NumSteps - 1) * stepW;
                string lastStep = decoded.Length > lastStart
                    ? decoded[lastStart..Math.Min(lastStart + stepW, decoded.Length)].TrimEnd()
                    : "";
                isCorrect    = string.Equals(lastStep, problem.Answer, StringComparison.Ordinal);
                logPrediction = lastStep;
            }
            else
            {
                // Answer-only mode (MaxDepth=1): take first answer.Length chars
                string prediction = decoded.Length >= problem.Answer.Length
                    ? decoded[..problem.Answer.Length]
                    : decoded;
                isCorrect     = string.Equals(prediction, problem.Answer, StringComparison.Ordinal);
                logPrediction = prediction;
            }

            if (isCorrect) correct++;
            total++;

            logger?.Prediction(problem.Prompt, problem.Answer, logPrediction, isCorrect);
        }

        float accuracy = total == 0 ? 0f : (float)correct / total;
        var   result   = new Evaluator.EvalResult(correct, total, accuracy);
        logger?.Eval(correct, total, accuracy);
        return result;
    }
}
