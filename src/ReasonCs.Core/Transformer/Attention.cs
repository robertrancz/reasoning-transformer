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
///
/// Backward (reverse chain):
///   dAttnOut = dOut @ Wo.T;   DWo += attnOut.T @ dOut
///   For each head h:
///     dAttnH_h = dAttnOut[:, h*d:(h+1)*d]
///     dProbs_h = dAttnH_h @ Vh.T;       dVh = probs_h.T @ dAttnH_h
///     dScores_h = SoftmaxOp.Backward(dProbs_h, probs_h)
///     dQh = dScores_h * scale @ Kh_rot;  dKh = (dScores_h * scale).T @ Qh_rot
///     For each pos: dQ[pos, h*d:] += RoPE.Backward(dQh[pos])
///                   dK[pos, h*d:] += RoPE.Backward(dKh[pos])
///     dV[:, h*d:] += dVh
///   DWq += x.T @ dQ;   DWk += x.T @ dK;   DWv += x.T @ dV
///   dx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
/// </summary>
public sealed class Attention
{
    public readonly int NumHeads;
    public readonly int HeadDim;
    public readonly int EmbedDim;

    // Weight matrices  [EmbedDim, EmbedDim]
    public Tensor Wq, Wk, Wv, Wo;

    // Gradient accumulators (same shape as weights; zeroed by ZeroGrad)
    public Tensor DWq, DWk, DWv, DWo;

    // Forward-pass cache (populated by Forward, consumed by Backward)
    private int     _seqLen;
    private Tensor? _cacheX;           // input               [S, D]
    private Tensor? _cacheV;           // V = x @ Wv          [S, D]
    private Tensor[]? _cacheQh;        // Qh post-RoPE per head  [H][S, d]
    private Tensor[]? _cacheKh;        // Kh post-RoPE per head  [H][S, d]
    private Tensor[]? _cacheVh;        // Vh per head (no RoPE)  [H][S, d]
    private Tensor[]? _cacheProbs;     // softmax probs per head [H][S, S]
    private Tensor? _cacheAttnOut;     // concat head outputs     [S, D]

    public Attention(int embedDim, int numHeads)
    {
        EmbedDim = embedDim;
        NumHeads = numHeads;
        HeadDim  = embedDim / numHeads;

        Wq = new Tensor([embedDim, embedDim]);
        Wk = new Tensor([embedDim, embedDim]);
        Wv = new Tensor([embedDim, embedDim]);
        Wo = new Tensor([embedDim, embedDim]);

        DWq = new Tensor([embedDim, embedDim]);
        DWk = new Tensor([embedDim, embedDim]);
        DWv = new Tensor([embedDim, embedDim]);
        DWo = new Tensor([embedDim, embedDim]);
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

        var qhArr    = new Tensor[NumHeads];
        var khArr    = new Tensor[NumHeads];
        var vhArr    = new Tensor[NumHeads];
        var probsArr = new Tensor[NumHeads];

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

            qhArr[h]    = Qh;
            khArr[h]    = Kh;
            vhArr[h]    = Vh;
            probsArr[h] = probs;
        }

        // ── Cache everything needed for the backward pass ────────────────────
        _seqLen       = seqLen;
        _cacheX       = x;
        _cacheV       = V;
        _cacheQh      = qhArr;
        _cacheKh      = khArr;
        _cacheVh      = vhArr;
        _cacheProbs   = probsArr;
        _cacheAttnOut = attnOut;

        // ── 4. Output projection ───────────────────────────────────────────────
        return TensorOps.MatMul(attnOut, Wo);  // [S, D]
    }

    /// <summary>
    /// Backward pass.
    ///
    /// dOut    : upstream gradient [seqLen, embedDim]
    /// returns : gradient w.r.t. input x  [seqLen, embedDim]
    ///
    /// Also accumulates DWq, DWk, DWv, DWo.
    /// </summary>
    public Tensor Backward(Tensor dOut)
    {
        if (_cacheX is null || _cacheQh is null || _cacheKh is null ||
            _cacheVh is null || _cacheProbs is null || _cacheAttnOut is null)
            throw new InvalidOperationException("Backward called before Forward.");

        int seqLen = _seqLen;
        float scale = 1f / MathF.Sqrt(HeadDim);

        // ── Output projection: out = attnOut @ Wo ────────────────────────────
        // dAttnOut = dOut @ Wo.T   [S, D]
        // DWo     += attnOut.T @ dOut
        var dAttnOut = TensorOps.MatMul(dOut, TensorOps.Transpose(Wo));
        TensorOps.AddTo(DWo, TensorOps.MatMul(TensorOps.Transpose(_cacheAttnOut), dOut));

        // Gradient accumulators for pre-RoPE Q, K and V (full [S, D] layout)
        var dQ = new Tensor([seqLen, EmbedDim]);
        var dK = new Tensor([seqLen, EmbedDim]);
        var dV = new Tensor([seqLen, EmbedDim]);

        // ── Per-head backward ─────────────────────────────────────────────────
        for (int h = 0; h < NumHeads; h++)
        {
            int hOff = h * HeadDim;

            var Qh    = _cacheQh[h];    // post-RoPE Qh  [S, d]
            var Kh    = _cacheKh[h];    // post-RoPE Kh  [S, d]
            var Vh    = _cacheVh[h];    // Vh            [S, d]
            var probs = _cacheProbs[h]; // [S, S]

            // Gradient of head's contribution to attnOut
            // attnOut[:, h*d:(h+1)*d] = probs @ Vh
            var dAttnH = ExtractHead(dAttnOut, seqLen, hOff, HeadDim, EmbedDim);  // [S, d]

            // dProbs = dAttnH @ Vh.T    [S, S]
            // dVh    = probs.T @ dAttnH [S, d]
            var dProbs = TensorOps.MatMul(dAttnH, TensorOps.Transpose(Vh));
            var dVh    = TensorOps.MatMul(TensorOps.Transpose(probs), dAttnH);
            AddHead(dV, dVh, seqLen, hOff, HeadDim, EmbedDim);

            // Softmax backward → dScores  [S, S]
            // Then undo scale: dScores = d(score*scale)/d(scores_scaled) = dScores * scale
            var dScores = TensorOps.Scale(SoftmaxOp.Backward(dProbs, probs), scale);

            // scores = Qh @ Kh.T   →  dQh_rot = dScores @ Kh,  dKh_rot = dScores.T @ Qh
            var dQhRot = TensorOps.MatMul(dScores, Kh);                         // [S, d]
            var dKhRot = TensorOps.MatMul(TensorOps.Transpose(dScores), Qh);    // [S, d]

            // Undo RoPE: for each position apply R(−φ) = R(φ)^T
            for (int pos = 0; pos < seqLen; pos++)
            {
                var dqSlice = SliceRow(dQhRot, pos, 0, HeadDim);
                var dkSlice = SliceRow(dKhRot, pos, 0, HeadDim);
                var dqPre   = RoPEOp.Backward(dqSlice, pos);
                var dkPre   = RoPEOp.Backward(dkSlice, pos);
                AddRowSlice(dQ, dqPre, pos, hOff, HeadDim);
                AddRowSlice(dK, dkPre, pos, hOff, HeadDim);
            }
        }

        // ── Q, K, V projections: Q = x @ Wq, K = x @ Wk, V = x @ Wv ─────────
        // DWq += x.T @ dQ;    DWk += x.T @ dK;    DWv += x.T @ dV
        // dx   = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
        var xT = TensorOps.Transpose(_cacheX);
        TensorOps.AddTo(DWq, TensorOps.MatMul(xT, dQ));
        TensorOps.AddTo(DWk, TensorOps.MatMul(xT, dK));
        TensorOps.AddTo(DWv, TensorOps.MatMul(xT, dV));

        var dxFromQ = TensorOps.MatMul(dQ, TensorOps.Transpose(Wq));
        var dxFromK = TensorOps.MatMul(dK, TensorOps.Transpose(Wk));
        var dxFromV = TensorOps.MatMul(dV, TensorOps.Transpose(Wv));

        // dx = dxFromQ + dxFromK + dxFromV
        TensorOps.AddTo(dxFromQ, dxFromK);
        TensorOps.AddTo(dxFromQ, dxFromV);
        return dxFromQ;
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
    /// Accumulate-add: dest[:, hOff:hOff+headDim] += src  (in-place).
    /// Used in backward to scatter head gradients back to the full [S, D] tensor.
    /// </summary>
    private static void AddHead(
        Tensor dest, Tensor src,
        int seqLen, int hOff, int headDim, int embedDim)
    {
        for (int i = 0; i < seqLen; i++)
            for (int d = 0; d < headDim; d++)
                dest.Data[i * embedDim + hOff + d] += src.Data[i * headDim + d];
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

    /// <summary>
    /// Accumulate-add a rank-1 tensor into row <c>row</c> of rank-2 <c>dest</c>
    /// starting at column <c>colOff</c>.
    /// Used in backward to scatter per-position per-head gradients back into
    /// the full [S, D] gradient buffer.
    /// </summary>
    private static void AddRowSlice(Tensor dest, Tensor src, int row, int colOff, int len)
    {
        int baseIdx = row * dest.Shape[1] + colOff;
        for (int d = 0; d < len; d++)
            dest.Data[baseIdx + d] += src.Data[d];
    }
}
