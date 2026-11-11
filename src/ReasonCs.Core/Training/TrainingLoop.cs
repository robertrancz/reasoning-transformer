using System.Diagnostics;
using ReasonCs.Core.Autograd.Ops;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Training;

/// <summary>
/// Basic training loop: forward → loss → backward → optimizer step.
/// Logs loss (and optionally grad norm + step timing) every <see cref="LogEvery"/> steps.
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
    ///
    /// When <paramref name="logger"/> is provided every logged step also writes
    /// loss, gradient norm, and ms/step to the log file.
    ///
    /// Returns the loss at the final logged step (useful for unit tests).
    /// </summary>
    public float Train(
        TransformerModel model,
        AdamW optimizer,
        IEnumerable<(int[] tokens, int[] targets)> batches,
        int maxSteps,
        RunLogger? logger = null)
    {
        float lastLoss = float.NaN;
        int   step     = 0;
        var   sw       = new Stopwatch();

        foreach (var (tokens, targets) in batches)
        {
            if (step >= maxSteps) break;

            sw.Restart();

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

            sw.Stop();

            // ── Logging ───────────────────────────────────────────────────────
            if (step % LogEvery == 0)
            {
                float gradNorm   = model.GradNorm();
                float msPerStep  = sw.ElapsedMilliseconds;

                if (logger is not null)
                    logger.Step(step, loss, gradNorm, msPerStep);
                else
                    Console.WriteLine($"step {step,6}: loss {loss:F4}");
            }

            step++;
        }

        return lastLoss;
    }
}
