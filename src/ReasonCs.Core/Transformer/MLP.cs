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
/// SwiGLU:  FFN(x) = (SiLU(x·W1) ⊙ (x·W2)) · W3
///   W1 : gate projection  [embedDim, hiddenDim]
///   W2 : up projection    [embedDim, hiddenDim]
///   W3 : down projection  [hiddenDim, embedDim]
///
/// SiLU(z) = z · σ(z) = z / (1 + exp(−z))   (also called Swish)
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

    /// <summary>
    /// Forward pass.
    ///
    /// x       : [seqLen, embedDim]
    /// returns : [seqLen, embedDim]
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        var gate  = TensorOps.MatMul(x, W1);  // [S, hiddenDim]
        var up    = TensorOps.MatMul(x, W2);  // [S, hiddenDim]

        // SwiGLU gate: SiLU(gate) ⊙ up — element-wise
        var gated = new Tensor(gate.Shape);
        for (int i = 0; i < gate.Size; i++)
            gated.Data[i] = SiLU(gate.Data[i]) * up.Data[i];

        return TensorOps.MatMul(gated, W3);   // [S, embedDim]
    }

    public Tensor Backward(Tensor dOut)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    /// <summary>
    /// SiLU (Sigmoid Linear Unit) — also called Swish.
    /// SiLU(x) = x · σ(x)   where σ(x) = 1 / (1 + exp(−x))
    /// PyTorch equivalent:  torch.nn.functional.silu(x)
    /// </summary>
    private static float SiLU(float x) => x / (1f + MathF.Exp(-x));
}
