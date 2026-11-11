using ReasonCs.Core.Autograd.Ops;
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
///   Apply RoPE to Q and K (each token position independently)
///   For each head h:
///     scores_h = (Q_h · K_h^T) / √headDim  [seqLen, seqLen]
///     Apply causal mask (upper triangle = −∞)
///     probs_h  = softmax(scores_h)
///     out_h    = probs_h · V_h              [seqLen, headDim]
///   Concatenate heads → [seqLen, embedDim]
///   output = concat · Wo
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

    /// <summary>
    /// Forward pass.
    ///
    /// x       : [seqLen, embedDim]
    /// seqLen  : number of tokens (= x.Shape[0])
    /// returns : [seqLen, embedDim]
    /// </summary>
    public Tensor Forward(Tensor x, int seqLen)
    {
        // ── 1. Linear projections ──────────────────────────────────────────────
        var Q = TensorOps.MatMul(x, Wq);   // [S, D]
        var K = TensorOps.MatMul(x, Wk);   // [S, D]
        var V = TensorOps.MatMul(x, Wv);   // [S, D]

        // ── 2. Apply RoPE per head, per token position ─────────────────────────
        // Each token at position i within each head's slice gets rotated by i.
        for (int h = 0; h < NumHeads; h++)
        {
            int hOff = h * HeadDim;
            for (int pos = 0; pos < seqLen; pos++)
            {
                var qSlice = SliceRow(Q, pos, hOff, HeadDim);
                var kSlice = SliceRow(K, pos, hOff, HeadDim);
                WriteRowSlice(Q, RoPEOp.Forward(qSlice, pos), pos, hOff, HeadDim);
                WriteRowSlice(K, RoPEOp.Forward(kSlice, pos), pos, hOff, HeadDim);
            }
        }

        // ── 3. Per-head scaled dot-product attention ──────────────────────────
        float scale   = 1f / MathF.Sqrt(HeadDim);
        var attnOut   = new Tensor([seqLen, EmbedDim]);

        for (int h = 0; h < NumHeads; h++)
        {
            int hOff = h * HeadDim;

            var Qh = ExtractHead(Q, seqLen, hOff, HeadDim, EmbedDim);  // [S, d]
            var Kh = ExtractHead(K, seqLen, hOff, HeadDim, EmbedDim);  // [S, d]
            var Vh = ExtractHead(V, seqLen, hOff, HeadDim, EmbedDim);  // [S, d]

            // scores [S, S] = Qh @ Kh^T / sqrt(d)
            var scores = TensorOps.Scale(
                TensorOps.MatMul(Qh, TensorOps.Transpose(Kh)), scale);

            // Causal mask: positions i cannot attend to future positions j > i
            for (int i = 0; i < seqLen; i++)
                for (int j = i + 1; j < seqLen; j++)
                    scores[i, j] = float.NegativeInfinity;

            // probs [S, S] = softmax(scores, dim=-1)
            var probs  = SoftmaxOp.Forward(scores);

            // head output [S, d] = probs @ Vh
            var headOut = TensorOps.MatMul(probs, Vh);
            WriteHead(attnOut, headOut, seqLen, hOff, HeadDim, EmbedDim);
        }

        // ── 4. Output projection ───────────────────────────────────────────────
        return TensorOps.MatMul(attnOut, Wo);  // [S, D]
    }

    public Tensor Backward(Tensor dOut)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    // ── Private slice helpers ─────────────────────────────────────────────────

    /// <summary>
    /// Extract a head's columns from a full [seqLen, embedDim] tensor.
    /// Returns [seqLen, headDim] for head starting at column <c>hOff</c>.
    /// </summary>
    private static Tensor ExtractHead(
        Tensor x, int seqLen, int hOff, int headDim, int embedDim)
    {
        var h = new Tensor([seqLen, headDim]);
        for (int i = 0; i < seqLen; i++)
            Array.Copy(x.Data, i * embedDim + hOff, h.Data, i * headDim, headDim);
        return h;
    }

    /// <summary>
    /// Write a [seqLen, headDim] head tensor back into the correct columns of
    /// a [seqLen, embedDim] destination tensor starting at column <c>hOff</c>.
    /// </summary>
    private static void WriteHead(
        Tensor dest, Tensor src,
        int seqLen, int hOff, int headDim, int embedDim)
    {
        for (int i = 0; i < seqLen; i++)
            Array.Copy(src.Data, i * headDim, dest.Data, i * embedDim + hOff, headDim);
    }

    /// <summary>
    /// Extract a contiguous slice of length <c>len</c> starting at column
    /// <c>colOff</c> from row <c>row</c> of a rank-2 tensor.
    /// Returns a rank-1 tensor of shape [len].
    /// </summary>
    private static Tensor SliceRow(Tensor x, int row, int colOff, int len)
    {
        var s = new Tensor([len]);
        Array.Copy(x.Data, row * x.Shape[1] + colOff, s.Data, 0, len);
        return s;
    }

    /// <summary>
    /// Write the contents of a rank-1 tensor <c>src</c> into row <c>row</c>
    /// starting at column <c>colOff</c> of rank-2 tensor <c>dest</c>.
    /// </summary>
    private static void WriteRowSlice(Tensor dest, Tensor src, int row, int colOff, int len)
    {
        Array.Copy(src.Data, 0, dest.Data, row * dest.Shape[1] + colOff, len);
    }
}
