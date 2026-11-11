using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Autograd.Ops;

/// <summary>
/// Layer Normalisation forward and backward.
///
/// PyTorch equivalent:
///   # Forward (with learnable affine parameters):
///   torch.nn.functional.layer_norm(x, normalized_shape=[D], weight=gamma, bias=beta)
///   # or as a module:
///   torch.nn.LayerNorm(D)
///
///   # Backward — PyTorch handles automatically via autograd.
///   # We implement three gradients explicitly:
///   #   dx     — gradient w.r.t. input
///   #   dGamma — gradient w.r.t. scale parameter γ  (sum over batch)
///   #   dBeta  — gradient w.r.t. shift parameter β  (sum over batch)
///
/// Forward:  y = (x - μ) / √(σ² + ε) * γ + β
///   where μ = mean(x), σ² = var(x), both computed over the last dimension.
/// Backward: gradients w.r.t. x, γ, β
///
/// We cache mean and rstd (reciprocal std) from Forward for use in Backward.
/// </summary>
public static class LayerNormOp
{
    /// <summary>
    /// Forward pass. Normalises along the last dimension, then applies
    /// the learnable affine transform y = γ·x̂ + β.
    ///
    /// Returns the output y plus the cached per-row mean and rstd
    /// (reciprocal std) needed by the backward pass.
    /// gamma and beta must both have shape [lastDim].
    /// </summary>
    public static (Tensor y, Tensor mean, Tensor rstd) Forward(
        Tensor x, Tensor gamma, Tensor beta, float eps = 1e-5f)
    {
        int lastDim = x.Shape[^1];
        int batch   = x.Size / lastDim;

        var y    = new Tensor(x.Shape);
        var mean = new Tensor([batch]);
        var rstd = new Tensor([batch]);

        for (int b = 0; b < batch; b++)
        {
            int offset = b * lastDim;

            // 1. Row mean
            float mu = 0f;
            for (int i = 0; i < lastDim; i++)
                mu += x.Data[offset + i];
            mu /= lastDim;
            mean.Data[b] = mu;

            // 2. Row variance (biased)
            float variance = 0f;
            for (int i = 0; i < lastDim; i++)
            {
                float d = x.Data[offset + i] - mu;
                variance += d * d;
            }
            variance /= lastDim;

            // 3. Reciprocal std — cached for backward
            float rs = 1f / MathF.Sqrt(variance + eps);
            rstd.Data[b] = rs;

            // 4. Normalise, then apply affine: y = γ · x̂ + β
            for (int i = 0; i < lastDim; i++)
            {
                float xhat = (x.Data[offset + i] - mu) * rs;
                y.Data[offset + i] = gamma.Data[i] * xhat + beta.Data[i];
            }
        }

        return (y, mean, rstd);
    }

    /// <summary>
    /// Backward pass. Computes gradients for x, γ, and β.
    ///
    /// Derivation (per row, dropping batch index):
    ///   Let x̂[i] = (x[i] - μ) · rstd         (normalised input)
    ///   Let dx̂[i] = dOut[i] · γ[i]             (grad w.r.t. x̂)
    ///
    ///   dγ[i] = Σ_b dOut[b,i] · x̂[b,i]        (sum over batch)
    ///   dβ[i] = Σ_b dOut[b,i]                  (sum over batch)
    ///
    ///   dx[i] = rstd · (dx̂[i] − mean(dx̂) − x̂[i] · mean(dx̂ · x̂))
    ///
    /// The two scalar terms in dx subtract the components of dx̂ that lie
    /// in the directions of the all-ones vector and of x̂ — exactly the
    /// two degrees of freedom removed by the normalisation.
    /// </summary>
    public static (Tensor dx, Tensor dGamma, Tensor dBeta) Backward(
        Tensor dOut, Tensor x, Tensor gamma, Tensor mean, Tensor rstd)
    {
        int lastDim = x.Shape[^1];
        int batch   = x.Size / lastDim;

        var dx     = new Tensor(x.Shape);
        var dGamma = new Tensor([lastDim]);
        var dBeta  = new Tensor([lastDim]);

        for (int b = 0; b < batch; b++)
        {
            int   offset = b * lastDim;
            float mu     = mean.Data[b];
            float rs     = rstd.Data[b];

            // Pass 1: accumulate dGamma, dBeta and the two dx̂ scalars
            float sumDxhat     = 0f;
            float sumDxhatXhat = 0f;

            for (int i = 0; i < lastDim; i++)
            {
                float xhat  = (x.Data[offset + i] - mu) * rs;
                float dxhat = dOut.Data[offset + i] * gamma.Data[i];

                dGamma.Data[i] += dOut.Data[offset + i] * xhat; // accumulate over batch
                dBeta.Data[i]  += dOut.Data[offset + i];

                sumDxhat     += dxhat;
                sumDxhatXhat += dxhat * xhat;
            }

            float meanDxhat     = sumDxhat     / lastDim;
            float meanDxhatXhat = sumDxhatXhat / lastDim;

            // Pass 2: apply the dx formula
            for (int i = 0; i < lastDim; i++)
            {
                float xhat  = (x.Data[offset + i] - mu) * rs;
                float dxhat = dOut.Data[offset + i] * gamma.Data[i];
                dx.Data[offset + i] = rs * (dxhat - meanDxhat - xhat * meanDxhatXhat);
            }
        }

        return (dx, dGamma, dBeta);
    }
}
