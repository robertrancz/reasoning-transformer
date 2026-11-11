using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Autograd.Ops;

/// <summary>
/// Numerically stable Softmax along the last axis, plus its backward pass.
///
/// PyTorch equivalent:
///   # Forward:
///   torch.nn.functional.softmax(x, dim=-1)
///
///   # Backward — PyTorch handles automatically via autograd.
///   # We implement it explicitly (Jacobian-vector product):
///   # dx_i = p_i * (dy_i - dot(p, dy))
///
/// Forward:  p_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
/// Backward: dx_i = p_i * (dy_i - Σ_j p_j * dy_j)
///
/// The max subtraction is a numerical stability trick — it prevents overflow
/// in exp() without changing the result (cancels in numerator and denominator).
/// </summary>
public static class SoftmaxOp
{
    public static Tensor Forward(Tensor x)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    public static Tensor Backward(Tensor dOut, Tensor probs)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
