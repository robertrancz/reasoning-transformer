using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Transformer;

/// <summary>
/// Multi-Head Self-Attention with RoPE positional encoding.
///
/// PyTorch equivalent:
///   # Closest built-in (without RoPE, uses learned positional bias):
///   torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
///
///   # The modern scaled dot-product kernel (FlashAttention-style):
///   torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
///
///   # Our implementation is equivalent to:
///   Q = x @ Wq;  K = x @ Wk;  V = x @ Wv   # linear projections
///   Q, K = apply_rope(Q, position), apply_rope(K, position)
///   scores = (Q @ K.T) / math.sqrt(head_dim)
///   scores = scores + causal_mask             # upper triangle = -inf
///   probs  = F.softmax(scores, dim=-1)
///   out    = probs @ V
///   out    = out @ Wo                         # output projection
///
/// Key difference: PyTorch's nn.MultiheadAttention uses learned positional
/// bias or absolute sinusoidal encoding. We use RoPE, which encodes position
/// via rotation of Q and K — this is what LLaMA/Mistral/Qwen use.
///
/// Forward:
///   Q = x·Wq,  K = x·Wk,  V = x·Wv
///   Apply RoPE to Q and K
///   scores = (Q·Kᵀ) / √headDim
///   probs  = softmax(scores + causal_mask)
///   out    = probs·V
///   out    = out·Wo
/// </summary>
public sealed class Attention
{
    public readonly int NumHeads;
    public readonly int HeadDim;
    public readonly int EmbedDim;

    // Weight matrices  [EmbedDim, EmbedDim]
    public Tensor Wq, Wk, Wv, Wo;

    public Attention(int embedDim, int numHeads)
    {
        EmbedDim = embedDim;
        NumHeads = numHeads;
        HeadDim  = embedDim / numHeads;

        Wq = new Tensor([embedDim, embedDim]);
        Wk = new Tensor([embedDim, embedDim]);
        Wv = new Tensor([embedDim, embedDim]);
        Wo = new Tensor([embedDim, embedDim]);
    }

    public Tensor Forward(Tensor x, int seqLen)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    public Tensor Backward(Tensor dOut)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
