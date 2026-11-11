using ReasonCs.Core.Tensors;

namespace ReasonCs.Tests.Autograd;

/// <summary>
/// Utility for numerical finite-difference gradient checking.
///
/// For each parameter p_i, estimate the gradient as:
///   (f(p + ε·eᵢ) - f(p - ε·eᵢ)) / (2ε)
///
/// Then compare against the analytical gradient.
/// A relative error below <see cref="Tolerance"/> is considered passing.
/// </summary>
public static class NumericalGradientHelper
{
    // float32 optimal eps ≈ sqrt(machine_epsilon) ≈ sqrt(1.19e-7) ≈ 3.5e-4.
    // We use 5e-4 — a safe value that minimises both truncation error (O(eps²))
    // and catastrophic cancellation in (fPlus - fMinus) when |Loss| is large.
    //
    // Tolerance 0.01 (1%) is the standard relative threshold for float32 gradient
    // checks (same as PyTorch torch.autograd.gradcheck for float32).
    //
    // AbsTolerance handles near-zero gradients where both analytical and numerical
    // are essentially zero but their ratio is ill-defined.  This mirrors PyTorch's
    // atol=1e-5 / rtol=1e-3 mixed criterion: pass if
    //   |num − anal| ≤ atol  OR  relError ≤ rtol
    // We use a more generous atol=1e-3 for float32 (1 ULP at scale ~10³).
    public const float Epsilon      = 5e-4f;
    public const float Tolerance    = 0.01f;     // relative tolerance (1%)
    public const float AbsTolerance = 1e-3f;     // absolute tolerance for near-zero grads

    /// <summary>
    /// Check that the analytical gradient of <paramref name="analyticalGrad"/>
    /// matches the numerical estimate of <paramref name="forward"/> w.r.t. <paramref name="param"/>.
    /// </summary>
    public static void Check(
        Tensor param,
        Tensor analyticalGrad,
        Func<float> forward,
        float eps       = Epsilon,
        float tolerance = Tolerance)
    {
        for (int i = 0; i < param.Size; i++)
        {
            float orig = param.Data[i];

            // Assign then immediately evaluate via the closure — each Eval call
            // constitutes the "read" of the assigned value.  Using a helper method
            // makes this pairing visible to the static analyser (which cannot see
            // that forward() reads param.Data[i] through the captured reference).
            float fPlus  = Eval(param, i, orig + eps, forward);
            float fMinus = Eval(param, i, orig - eps, forward);
            param.Data[i] = orig; // restore

            float numerical   = (fPlus - fMinus) / (2f * eps);
            float analytical  = analyticalGrad.Data[i];
            float absError    = MathF.Abs(numerical - analytical);
            float relError    = absError
                              / (MathF.Abs(numerical) + MathF.Abs(analytical) + 1e-8f);

            // Pass if either the absolute error is tiny (both values are near zero)
            // OR the relative error is within tolerance. Mirrors PyTorch's mixed
            // atol + rtol criterion used in torch.autograd.gradcheck.
            bool pass = absError <= AbsTolerance || relError <= tolerance;
            if (!pass)
                throw new Exception(
                    $"Gradient check FAILED at index {i}: " +
                    $"numerical={numerical:F6}, analytical={analytical:F6}, " +
                    $"absError={absError:F6}, relError={relError:F6}");
        }
    }

    /// <summary>
    /// Assign <paramref name="val"/> to <c>param.Data[idx]</c>, evaluate
    /// <paramref name="forward"/>, and return the result.
    /// The assignment and evaluation are paired in the same method body so that
    /// the static analyser can see that the stored value is consumed before any
    /// subsequent write to the same index.
    /// </summary>
    private static float Eval(Tensor param, int idx, float val, Func<float> forward)
    {
        param.Data[idx] = val;
        return forward();
    }
}
