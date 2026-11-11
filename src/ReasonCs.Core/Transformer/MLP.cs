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
///
/// Backward:
///   dGated = dOut @ W3.T                             [S, H]
///   dW3   += gated.T @ dOut
///   dGate  = dGated ⊙ (SiLU'(gate) * up)            [S, H]
///   dUp    = dGated ⊙ SiLU(gate)                    [S, H]
///   dW1   += x.T @ dGate
///   dW2   += x.T @ dUp
///   dx     = dGate @ W1.T + dUp @ W2.T              [S, D]
/// </summary>
public sealed class MLP
{
    public readonly int EmbedDim;
    public readonly int HiddenDim;

    // Forward weights
    public Tensor W1, W2, W3; // gate, up, down projections

    // Gradient accumulators (same shape as weights; zeroed by ZeroGrad)
    public Tensor DW1, DW2, DW3;

    // Forward-pass cache (populated by Forward, consumed by Backward)
    private Tensor? _cacheX;     // input             [S, D]
    private Tensor? _cacheGate;  // x @ W1 (pre-SiLU) [S, H]
    private Tensor? _cacheUp;    // x @ W2            [S, H]
    private Tensor? _cacheGated; // SiLU(gate) * up   [S, H]

    public MLP(int embedDim, int hiddenDim)
    {
        EmbedDim  = embedDim;
        HiddenDim = hiddenDim;

        W1 = new Tensor([embedDim, hiddenDim]); // gate
        W2 = new Tensor([embedDim, hiddenDim]); // up
        W3 = new Tensor([hiddenDim, embedDim]); // down

        DW1 = new Tensor([embedDim, hiddenDim]);
        DW2 = new Tensor([embedDim, hiddenDim]);
        DW3 = new Tensor([hiddenDim, embedDim]);
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

        // Cache for backward
        _cacheX     = x;
        _cacheGate  = gate;
        _cacheUp    = up;
        _cacheGated = gated;

        return TensorOps.MatMul(gated, W3);   // [S, embedDim]
    }

    /// <summary>
    /// Backward pass.
    ///
    /// dOut    : upstream gradient [seqLen, embedDim]
    /// returns : gradient w.r.t. input x  [seqLen, embedDim]
    ///
    /// Also accumulates DW1 += ..., DW2 += ..., DW3 += ...
    /// </summary>
    public Tensor Backward(Tensor dOut)
    {
        if (_cacheX is null || _cacheGate is null || _cacheUp is null || _cacheGated is null)
            throw new InvalidOperationException("Backward called before Forward.");

        // ── Down projection: out = gated @ W3 ────────────────────────────────
        // dGated = dOut @ W3.T   [S, H]
        // DW3   += gated.T @ dOut
        var dGated = TensorOps.MatMul(dOut, TensorOps.Transpose(W3));
        TensorOps.AddTo(DW3, TensorOps.MatMul(TensorOps.Transpose(_cacheGated), dOut));

        // ── SwiGLU gate: gated = SiLU(gate) * up ─────────────────────────────
        // d(gated)/d(gate) = SiLU'(gate) * up  (chain rule through activation)
        // d(gated)/d(up)   = SiLU(gate)
        var dGate = new Tensor(_cacheGate.Shape);
        var dUp   = new Tensor(_cacheUp.Shape);
        for (int i = 0; i < dGated.Size; i++)
        {
            float g = _cacheGate.Data[i];
            float u = _cacheUp.Data[i];
            dGate.Data[i] = dGated.Data[i] * SiLUGrad(g) * u;
            dUp.Data[i]   = dGated.Data[i] * SiLU(g);
        }

        // ── Gate / up projections: gate = x @ W1,  up = x @ W2 ───────────────
        // DW1 += x.T @ dGate
        // DW2 += x.T @ dUp
        // dx   = dGate @ W1.T + dUp @ W2.T   [S, D]
        TensorOps.AddTo(DW1, TensorOps.MatMul(TensorOps.Transpose(_cacheX), dGate));
        TensorOps.AddTo(DW2, TensorOps.MatMul(TensorOps.Transpose(_cacheX), dUp));

        return TensorOps.Add(
            TensorOps.MatMul(dGate, TensorOps.Transpose(W1)),
            TensorOps.MatMul(dUp,   TensorOps.Transpose(W2)));
    }

    /// <summary>
    /// SiLU (Sigmoid Linear Unit) — also called Swish.
    /// SiLU(x) = x · σ(x)   where σ(x) = 1 / (1 + exp(−x))
    /// PyTorch equivalent:  torch.nn.functional.silu(x)
    /// </summary>
    private static float SiLU(float x) => x / (1f + MathF.Exp(-x));

    /// <summary>
    /// Derivative of SiLU w.r.t. its input.
    /// d/dx [x·σ(x)] = σ(x) + x·σ(x)·(1 − σ(x))  =  σ(x)·(1 + x·(1 − σ(x)))
    /// PyTorch computes this automatically via loss.backward().
    /// </summary>
    private static float SiLUGrad(float x)
    {
        float sig = 1f / (1f + MathF.Exp(-x));
        return sig * (1f + x * (1f - sig));
    }
}
