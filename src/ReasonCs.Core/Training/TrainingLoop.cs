using ReasonCs.Core.Autograd.Ops;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Training;

/// <summary>
/// Basic training loop: forward → loss → backward → optimizer step.
/// Logs loss to console every <see cref="LogEvery"/> steps.
///
/// PyTorch equivalent:
///   for step, (tokens, targets) in enumerate(dataloader):
///       optimizer.zero_grad()
///       logits = model(tokens)                        # forward
///       loss   = F.cross_entropy(logits, targets)     # loss
///       loss.backward()                               # backward (autograd)
///       optimizer.step()                              # update weights
///       if step % log_every == 0:
///           print(f"step {step}: loss {loss.item():.4f}")
///
/// Key difference: PyTorch's loss.backward() traverses the autograd tape
/// automatically. Here, Backward() is a hand-written method on the model
/// that explicitly calls each op's backward pass in reverse order.
///
/// Batch format: (tokens, targets) where both are int arrays of the same
/// length seqLen.  tokens[i] is the input at position i; targets[i] is the
/// expected next token at position i (i.e. tokens shifted by one for
/// standard language-model training).
/// </summary>
public sealed class TrainingLoop
{
    public int LogEvery { get; init; } = 100;

    /// <summary>
    /// Run the training loop for up to <paramref name="maxSteps"/> batches.
    /// Returns the loss at the final logged step (useful for unit tests).
    /// </summary>
    public float Train(
        TransformerModel model,
        AdamW optimizer,
        IEnumerable<(int[] tokens, int[] targets)> batches,
        int maxSteps)
    {
        float lastLoss = float.NaN;
        int step = 0;

        foreach (var (tokens, targets) in batches)
        {
            if (step >= maxSteps) break;

            // ── Forward ───────────────────────────────────────────────────────
            var logits = model.Forward(tokens);              // [S, VocabSize]

            // ── Loss ──────────────────────────────────────────────────────────
            var (loss, probs) = CrossEntropyOp.Forward(logits, targets);
            lastLoss = loss;

            // ── Backward ──────────────────────────────────────────────────────
            model.ZeroGrad();
            var dLogits = CrossEntropyOp.Backward(probs, targets);
            model.Backward(dLogits);

            // ── Optimizer step ────────────────────────────────────────────────
            optimizer.Step(model.GetParamGrads());

            // ── Logging ───────────────────────────────────────────────────────
            if (step % LogEvery == 0)
                Console.WriteLine($"step {step,6}: loss {loss:F4}");

            step++;
        }

        return lastLoss;
    }
}
