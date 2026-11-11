using ReasonCs.Core.Data;
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
/// </summary>
public sealed class Evaluator
{
    public record EvalResult(int Correct, int Total, float Accuracy);

    public static EvalResult RunGreedy(
        TransformerModel model,
        Tokenizer tokenizer,
        IEnumerable<SyntheticDataset.Problem> testSet,
        int maxNewTokens = 16)
    {
        // TODO: implement greedy evaluation loop
        throw new NotImplementedException();
    }
}
