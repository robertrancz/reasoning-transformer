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
/// </summary>
public sealed class TransformerBlock
{
    public readonly Attention Attn;
    public readonly MLP       Mlp;

    // LayerNorm affine parameters  [EmbedDim]
    // Gamma is initialised to 1 in the constructor (identity transform).
    public Tensor Gamma1, Beta1;
    public Tensor Gamma2, Beta2;

    public TransformerBlock(int embedDim, int numHeads, int mlpHiddenDim)
    {
        Attn   = new Attention(embedDim, numHeads);
        Mlp    = new MLP(embedDim, mlpHiddenDim);

        Gamma1 = new Tensor([embedDim]);
        Beta1  = new Tensor([embedDim]);
        Gamma2 = new Tensor([embedDim]);
        Beta2  = new Tensor([embedDim]);

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
        // mean and rstd are cached here for the backward pass (not yet needed).
        var (xNorm1, _, _) = LayerNormOp.Forward(x, Gamma1, Beta1);
        x = TensorOps.Add(x, Attn.Forward(xNorm1, seqLen));

        // ── Sublayer 2: Pre-Norm MLP ──────────────────────────────────────────
        var (xNorm2, _, _) = LayerNormOp.Forward(x, Gamma2, Beta2);
        x = TensorOps.Add(x, Mlp.Forward(xNorm2));

        return x;
    }

    public Tensor Backward(Tensor dOut)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
