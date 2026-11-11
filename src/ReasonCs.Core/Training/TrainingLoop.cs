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
/// </summary>
public sealed class TrainingLoop
{
    public int LogEvery { get; init; } = 100;

    public void Train(
        TransformerModel model,
        AdamW optimizer,
        IEnumerable<(int[] tokens, int[] targets)> batches,
        int maxSteps)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
