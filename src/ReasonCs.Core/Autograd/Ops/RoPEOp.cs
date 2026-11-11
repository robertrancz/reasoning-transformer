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
///   # def apply_rotary(x, cos, sin, position):
///   #     return x * cos[position] + rotate_half(x) * sin[position]
///   #
///   # The HuggingFace Transformers library provides:
///   # transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
///   #
///   # Backward — PyTorch handles via autograd. We implement it explicitly
///   # as an inverse rotation: rotate dOut by the *negative* angle.
///   # This works because rotation matrices are orthogonal: R⁻¹ = Rᵀ.
///
/// WARNING: The backward pass is non-trivial and easy to get subtly wrong.
/// A numerical finite-difference gradient test is mandatory before this op
/// is used in the full transformer (see tests/ReasonCs.Tests/Autograd/).
///
/// Forward:  rotate pairs (x_{2i}, x_{2i+1}) by angle θ_i * position
/// Backward: rotate the incoming gradient by the *negative* angle (inverse rotation)
/// </summary>
public static class RoPEOp
{
    public static Tensor Forward(Tensor x, int position)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    public static Tensor Backward(Tensor dOut, int position)
    {
        // TODO: implement — inverse rotation (negate the angle)
        throw new NotImplementedException();
    }

    /// <summary>Precompute cos/sin tables for all positions up to maxSeqLen.</summary>
    public static (float[] Cos, float[] Sin) BuildFrequencies(int headDim, int maxSeqLen, float theta = 10000f)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
