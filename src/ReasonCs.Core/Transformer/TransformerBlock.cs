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
/// </summary>
public sealed class TransformerBlock
{
    public readonly Attention Attn;
    public readonly MLP       Mlp;

    // LayerNorm parameters  [EmbedDim]
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
    }

    public Tensor Forward(Tensor x, int seqLen)
    {
        // TODO: implement pre-norm residual block
        throw new NotImplementedException();
    }

    public Tensor Backward(Tensor dOut)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
