using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Autograd.Ops;

/// <summary>
/// Cross-entropy loss over logits, fused with Softmax for numerical stability.
///
/// PyTorch equivalent:
///   # Forward (fused, stable):
///   torch.nn.functional.cross_entropy(logits, targets)
///   # which internally does: log_softmax(logits) then nll_loss.
///
///   # Backward — PyTorch handles automatically.
///   # The fused softmax + cross-entropy gradient is elegantly simple:
///   dLogits[i] = p[i] - 1{i == target}
///   # i.e. subtract 1 from the probability of the correct class.
///
/// Forward:  loss = -log(p[target])  where p = softmax(logits)
/// Backward: dLogits[i] = p[i] - 1{i == target}
///
/// Fusing softmax and cross-entropy avoids a numerically unstable log(exp(...))
/// round-trip — we compute log-softmax directly instead.
/// </summary>
public static class CrossEntropyOp
{
    public static (float loss, Tensor probs) Forward(Tensor logits, int[] targets)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    public static Tensor Backward(Tensor probs, int[] targets)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
