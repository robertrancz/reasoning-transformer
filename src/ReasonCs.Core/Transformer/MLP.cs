using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Transformer;

/// <summary>
/// Feed-Forward block with SwiGLU activation.
///
/// PyTorch equivalent:
///   # There is no single nn.SwiGLU module. The equivalent code is:
///   class SwiGLU(nn.Module):
///       def __init__(self, d_model, d_hidden):
///           self.w1 = nn.Linear(d_model, d_hidden, bias=False)  # gate
///           self.w2 = nn.Linear(d_model, d_hidden, bias=False)  # up
///           self.w3 = nn.Linear(d_hidden, d_model, bias=False)  # down
///       def forward(self, x):
///           return self.w3(F.silu(self.w1(x)) * self.w2(x))
///
///   # For simpler GELU alternative:
///   # torch.nn.GELU() — used in GPT-2/BERT style models
///   # torch.nn.SiLU() — the activation inside SwiGLU (also called Swish)
///
/// Note: SwiGLU uses 3 weight matrices (gate/up/down) vs. 2 for standard FFN.
/// The hidden dim is typically set to 2/3 * 4 * embed_dim to keep param count
/// equivalent.
///
/// SwiGLU:  FFN(x) = (x·W1 ⊙ SiLU(x·W2)) · W3
///
/// Alternative: swap SwiGLU for GELU if simpler is preferred initially.
/// </summary>
public sealed class MLP
{
    public readonly int EmbedDim;
    public readonly int HiddenDim;

    public Tensor W1, W2, W3; // gate, up, down projections

    public MLP(int embedDim, int hiddenDim)
    {
        EmbedDim  = embedDim;
        HiddenDim = hiddenDim;

        W1 = new Tensor([embedDim, hiddenDim]); // gate
        W2 = new Tensor([embedDim, hiddenDim]); // up
        W3 = new Tensor([hiddenDim, embedDim]); // down
    }

    public Tensor Forward(Tensor x)
    {
        // TODO: implement SwiGLU
        throw new NotImplementedException();
    }

    public Tensor Backward(Tensor dOut)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    private static float SiLU(float x) => x / (1f + MathF.Exp(-x));
}
