using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Autograd.Ops;

/// <summary>
/// Rotary Positional Embedding (RoPE) — forward and backward.
///
/// Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
///
/// PyTorch equivalent:
///   # There is no single built-in. The standard reference implementation
///   # (used in LLaMA, Mistral, etc.) looks like:
///   #
///   # def rotate_half(x):
///   #     x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
///   #     return torch.cat([-x2, x1], dim=-1)
///   #
///   # def apply_rotary(q, cos, sin):
///   #     return q * cos + rotate_half(q) * sin
///   #
///   # The HuggingFace Transformers library provides:
///   # transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
///   #
///   # Backward — PyTorch handles via autograd. We implement it explicitly
///   # as an inverse rotation: rotate dOut by the *negative* angle.
///   # This works because rotation matrices are orthogonal: R⁻¹ = Rᵀ.
///
/// Forward: rotate pairs (x_{2i}, x_{2i+1}) by angle φᵢ = position × θᵢ
///          where  θᵢ = 1 / base^(2i / headDim)   (base = 10000 by default)
///
///   y_{2i}   =  x_{2i} · cos(φᵢ) − x_{2i+1} · sin(φᵢ)
///   y_{2i+1} =  x_{2i} · sin(φᵢ) + x_{2i+1} · cos(φᵢ)
///
/// Backward: The Jacobian of a 2D rotation by angle φ is the rotation matrix R(φ).
///           Since R is orthogonal, R⁻¹ = Rᵀ = R(−φ).
///           Therefore the backward pass is simply rotation by −φ:
///
///   dx_{2i}   =  dOut_{2i} · cos(φᵢ) + dOut_{2i+1} · sin(φᵢ)
///   dx_{2i+1} = −dOut_{2i} · sin(φᵢ) + dOut_{2i+1} · cos(φᵢ)
///
/// Shapes:
///   x    : [..., headDim]  — any leading batch dims; headDim must be even
///   y    : [..., headDim]  — same shape as x
///   dOut : [..., headDim]  — upstream gradient (same shape as y)
///   dx   : [..., headDim]  — gradient w.r.t. x
///
/// NOTE: Forward and Backward recompute cos/sin on the fly from `position`.
///       Call BuildFrequencies once and cache for production use.
/// </summary>
public static class RoPEOp
{
    /// <summary>
    /// Default base for the angular frequency computation.
    /// LLaMA/Mistral use 10000; some newer models use higher values.
    /// </summary>
    public const float DefaultBase = 10000f;

    /// <summary>
    /// Apply rotary positional embedding for a single token position.
    ///
    /// x shape: [..., headDim]  (headDim must be even)
    /// Returns y with the same shape.
    /// </summary>
    public static Tensor Forward(Tensor x, int position, float base_ = DefaultBase)
    {
        int headDim = x.Shape[^1];
        if (headDim % 2 != 0)
            throw new ArgumentException($"headDim must be even, got {headDim}.", nameof(x));

        int batch = x.Size / headDim;
        int half  = headDim / 2;
        var y     = new Tensor(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            int offset = b * headDim;
            for (int i = 0; i < half; i++)
            {
                float freq  = 1f / MathF.Pow(base_, 2f * i / headDim);
                float angle = position * freq;
                float cos   = MathF.Cos(angle);
                float sin   = MathF.Sin(angle);

                float xa = x.Data[offset + 2 * i];
                float xb = x.Data[offset + 2 * i + 1];

                y.Data[offset + 2 * i]     = xa * cos - xb * sin;
                y.Data[offset + 2 * i + 1] = xa * sin + xb * cos;
            }
        }

        return y;
    }

    /// <summary>
    /// Backward pass for RoPE at a single token position.
    ///
    /// Because R(φ) is orthogonal, the gradient flows back through R(−φ) — i.e.
    /// the same rotation but with the angle negated, or equivalently sin negated:
    ///
    ///   dx_{2i}   =  dOut_{2i} · cos(φ) + dOut_{2i+1} · sin(φ)
    ///   dx_{2i+1} = −dOut_{2i} · sin(φ) + dOut_{2i+1} · cos(φ)
    ///
    /// dOut shape: [..., headDim]
    /// Returns dx with the same shape.
    /// </summary>
    public static Tensor Backward(Tensor dOut, int position, float base_ = DefaultBase)
    {
        int headDim = dOut.Shape[^1];
        int batch   = dOut.Size / headDim;
        int half    = headDim / 2;
        var dx      = new Tensor(dOut.Shape);

        for (int b = 0; b < batch; b++)
        {
            int offset = b * headDim;
            for (int i = 0; i < half; i++)
            {
                float freq  = 1f / MathF.Pow(base_, 2f * i / headDim);
                float angle = position * freq;
                float cos   = MathF.Cos(angle);
                float sin   = MathF.Sin(angle);

                float da = dOut.Data[offset + 2 * i];
                float db = dOut.Data[offset + 2 * i + 1];

                // Rotate dOut by −φ: cos stays, sin flips sign
                dx.Data[offset + 2 * i]     =  da * cos + db * sin;
                dx.Data[offset + 2 * i + 1] = -da * sin + db * cos;
            }
        }

        return dx;
    }

    /// <summary>
    /// Precompute cos/sin tables for all positions [0, maxSeqLen).
    ///
    /// Returns two flat arrays of length maxSeqLen × (headDim/2).
    /// Index: pos * (headDim/2) + i  →  cos(pos × freq_i) or sin(pos × freq_i)
    ///
    /// Cache these once at model init and pass slices to Forward/Backward to
    /// avoid repeated MathF.Pow calls during training.
    /// </summary>
    public static (float[] Cos, float[] Sin) BuildFrequencies(
        int headDim, int maxSeqLen, float base_ = DefaultBase)
    {
        if (headDim % 2 != 0)
            throw new ArgumentException($"headDim must be even, got {headDim}.", nameof(headDim));

        int half = headDim / 2;
        var cos  = new float[maxSeqLen * half];
        var sin  = new float[maxSeqLen * half];

        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            for (int i = 0; i < half; i++)
            {
                float freq  = 1f / MathF.Pow(base_, 2f * i / headDim);
                float angle = pos * freq;
                cos[pos * half + i] = MathF.Cos(angle);
                sin[pos * half + i] = MathF.Sin(angle);
            }
        }

        return (cos, sin);
    }
}
