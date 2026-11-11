using ReasonCs.Core.Autograd.Ops;
using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Transformer;

/// <summary>
/// One Transformer decoder block (Pre-Norm variant).
///
/// PyTorch equivalent:
///   # The closest built-in is torch.nn.TransformerEncoderLayer, but it uses
///   # Post-Norm by default. Our Pre-Norm variant matches modern LLMs (GPT-2+,
///   # LLaMA) and is more stable during training.
///   #
///   # Equivalent manual implementation:
///   class TransformerBlock(nn.Module):
///       def forward(self, x):
///           x = x + self.attn(self.norm1(x))   # Pre-Norm: normalize BEFORE sublayer
///           x = x + self.mlp(self.norm2(x))
///           return x
///
///   # Post-Norm (original "Attention is All You Need") would be:
///   #   x = self.norm1(x + self.attn(x))       # normalize AFTER sublayer
///   #   x = self.norm2(x + self.mlp(x))
///
/// Pre-Norm (used here):
///   x = x + Attention(LayerNorm1(x))
///   x = x + MLP(LayerNorm2(x))
///
/// LayerNorm gamma parameters are initialised to 1 so the identity affine
/// transform leaves normalised activations unchanged at model start.
///
/// Backward:
///   Residual 2:  dx1 = dOut + MLP_bwd(LN2_bwd(dOut))
///   Residual 1:  dx  = dx1  + Attn_bwd(LN1_bwd(dx1))
/// </summary>
public sealed class TransformerBlock
{
    public readonly Attention Attn;
    public readonly MLP       Mlp;

    // LayerNorm affine parameters  [EmbedDim]
    // Gamma is initialised to 1 in the constructor (identity transform).
    public Tensor Gamma1, Beta1;
    public Tensor Gamma2, Beta2;

    // Gradient accumulators for LayerNorm parameters
    public Tensor DGamma1, DBeta1;
    public Tensor DGamma2, DBeta2;

    // Forward-pass cache (populated by Forward, consumed by Backward)
    private Tensor? _cacheX;       // block input            [S, D]
    private Tensor? _cacheXNorm1;  // LN1 output             [S, D]
    private Tensor? _cacheMean1;   // LN1 mean               [S]
    private Tensor? _cacheRstd1;   // LN1 reciprocal std     [S]
    private Tensor? _cacheX1;      // after attention residual [S, D]
    private Tensor? _cacheXNorm2;  // LN2 output             [S, D]
    private Tensor? _cacheMean2;   // LN2 mean               [S]
    private Tensor? _cacheRstd2;   // LN2 reciprocal std     [S]

    public TransformerBlock(int embedDim, int numHeads, int mlpHiddenDim)
    {
        Attn   = new Attention(embedDim, numHeads);
        Mlp    = new MLP(embedDim, mlpHiddenDim);

        Gamma1 = new Tensor([embedDim]);
        Beta1  = new Tensor([embedDim]);
        Gamma2 = new Tensor([embedDim]);
        Beta2  = new Tensor([embedDim]);

        DGamma1 = new Tensor([embedDim]);
        DBeta1  = new Tensor([embedDim]);
        DGamma2 = new Tensor([embedDim]);
        DBeta2  = new Tensor([embedDim]);

        // Gamma = 1 → identity affine; Beta = 0 → no bias (default from Tensor ctor)
        TensorOps.Fill(Gamma1, 1f);
        TensorOps.Fill(Gamma2, 1f);
    }

    /// <summary>
    /// Pre-Norm forward pass.
    ///
    /// x      : [seqLen, embedDim]
    /// seqLen : number of tokens (= x.Shape[0])
    /// returns : [seqLen, embedDim]
    /// </summary>
    public Tensor Forward(Tensor x, int seqLen)
    {
        // ── Sublayer 1: Pre-Norm Attention ────────────────────────────────────
        var (xNorm1, mean1, rstd1) = LayerNormOp.Forward(x, Gamma1, Beta1);
        var attnOut = Attn.Forward(xNorm1, seqLen);
        var x1 = TensorOps.Add(x, attnOut);

        // ── Sublayer 2: Pre-Norm MLP ──────────────────────────────────────────
        var (xNorm2, mean2, rstd2) = LayerNormOp.Forward(x1, Gamma2, Beta2);
        var mlpOut = Mlp.Forward(xNorm2);
        var x2 = TensorOps.Add(x1, mlpOut);

        // Cache all intermediates needed for backward
        _cacheX      = x;
        _cacheXNorm1 = xNorm1;
        _cacheMean1  = mean1;
        _cacheRstd1  = rstd1;
        _cacheX1     = x1;
        _cacheXNorm2 = xNorm2;
        _cacheMean2  = mean2;
        _cacheRstd2  = rstd2;

        return x2;
    }

    /// <summary>
    /// Backward pass.
    ///
    /// dOut    : upstream gradient [seqLen, embedDim]
    /// returns : gradient w.r.t. block input x  [seqLen, embedDim]
    ///
    /// Also accumulates DGamma1, DBeta1, DGamma2, DBeta2,
    /// and propagates backward through Attn and Mlp (which accumulate
    /// their own weight gradients).
    ///
    /// Chain (reversed):
    ///   x2 = x1 + mlp(ln2(x1))   →  dx1 = dOut + d(mlp∘ln2)/d(x1) · dOut
    ///   x1 = x  + attn(ln1(x))   →  dx  = dx1  + d(attn∘ln1)/d(x) · dx1
    /// </summary>
    public Tensor Backward(Tensor dOut)
    {
        if (_cacheX is null || _cacheXNorm1 is null || _cacheMean1 is null ||
            _cacheRstd1 is null || _cacheX1 is null || _cacheXNorm2 is null ||
            _cacheMean2 is null || _cacheRstd2 is null)
            throw new InvalidOperationException("Backward called before Forward.");

        // ── Residual 2: x2 = x1 + mlp(ln2(x1)) ──────────────────────────────
        // Upstream gradient splits: direct residual path + MLP sublayer path.
        // dOut flows through both branches; we accumulate them into dx1.

        // MLP sublayer: d(mlp∘ln2)/d(x1) · dOut
        var dXNorm2 = Mlp.Backward(dOut);                               // [S, D]
        var (dx1FromLn2, dGamma2, dBeta2) = LayerNormOp.Backward(
            dXNorm2, _cacheX1, Gamma2, _cacheMean2, _cacheRstd2);       // [S, D]
        TensorOps.AddTo(DGamma2, dGamma2);
        TensorOps.AddTo(DBeta2,  dBeta2);

        // dx1 = dOut (residual passthrough) + dx1_from_mlp_sublayer
        var dx1 = TensorOps.Add(dOut, dx1FromLn2);

        // ── Residual 1: x1 = x + attn(ln1(x)) ───────────────────────────────
        // Attention sublayer: d(attn∘ln1)/d(x) · dx1
        var dXNorm1 = Attn.Backward(dx1);                               // [S, D]
        var (dxFromLn1, dGamma1, dBeta1) = LayerNormOp.Backward(
            dXNorm1, _cacheX, Gamma1, _cacheMean1, _cacheRstd1);        // [S, D]
        TensorOps.AddTo(DGamma1, dGamma1);
        TensorOps.AddTo(DBeta1,  dBeta1);

        // dx = dx1 (residual passthrough) + dx_from_attn_sublayer
        TensorOps.AddTo(dx1, dxFromLn1);
        return dx1;
    }
}
