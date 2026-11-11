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
    /// <summary>
    /// Numerically stable softmax applied along the last dimension.
    /// Works for any rank — all leading dimensions are treated as a batch.
    /// e.g. [N] → row of N probs; [M, N] → M independent rows of N probs.
    /// </summary>
    public static Tensor Forward(Tensor x)
    {
        var result  = new Tensor(x.Shape);
        int lastDim = x.Shape[^1];
        int batch   = x.Size / lastDim;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * lastDim;

            // 1. Find row maximum for numerical stability (prevents exp overflow)
            float maxVal = float.NegativeInfinity;
            for (int i = 0; i < lastDim; i++)
                maxVal = MathF.Max(maxVal, x.Data[offset + i]);

            // 2. Compute shifted exponentials and accumulate sum
            float sum = 0f;
            for (int i = 0; i < lastDim; i++)
            {
                float e = MathF.Exp(x.Data[offset + i] - maxVal);
                result.Data[offset + i] = e;
                sum += e;
            }

            // 3. Normalise → probabilities in (0, 1) that sum to 1
            for (int i = 0; i < lastDim; i++)
                result.Data[offset + i] /= sum;
        }

        return result;
    }

    /// <summary>
    /// Backward pass: Jacobian-vector product of softmax.
    /// Applied independently to each row (same batch layout as Forward).
    ///
    /// Derivation:
    ///   ∂p_i/∂x_j = p_i (δ_ij − p_j)
    ///   dx_i = Σ_j (∂p_i/∂x_j) · dOut_j
    ///        = p_i · dOut_i − p_i · Σ_j p_j · dOut_j
    ///        = p_i · (dOut_i − dot(p, dOut))
    /// </summary>
    public static Tensor Backward(Tensor dOut, Tensor probs)
    {
        var dx      = new Tensor(probs.Shape);
        int lastDim = probs.Shape[^1];
        int batch   = probs.Size / lastDim;

        for (int b = 0; b < batch; b++)
        {
            int offset = b * lastDim;

            // dot(p, dOut) — scalar per row
            float dot = 0f;
            for (int i = 0; i < lastDim; i++)
                dot += probs.Data[offset + i] * dOut.Data[offset + i];

            // dx_i = p_i * (dOut_i - dot)
            for (int i = 0; i < lastDim; i++)
                dx.Data[offset + i] = probs.Data[offset + i] * (dOut.Data[offset + i] - dot);
        }

        return dx;
    }
}
