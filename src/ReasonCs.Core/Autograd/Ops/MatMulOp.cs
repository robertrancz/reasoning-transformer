using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Autograd.Ops;

/// <summary>
/// Forward and backward pass for Matrix Multiplication.
///
/// PyTorch equivalent:
///   # Forward — PyTorch does this implicitly via matmul:
///   C = A @ B                        →  MatMulOp.Forward(a, b)
///
///   # Backward — PyTorch handles automatically via autograd.
///   # We implement it explicitly:
///   dA = dC @ B.T                    →  MatMulOp.Backward(dC, a, b).dA
///   dB = A.T @ dC                    →  MatMulOp.Backward(dC, a, b).dB
///
/// Forward:  C = A × B
/// Backward: dA = dC × Bᵀ
///           dB = Aᵀ × dC
/// </summary>
public static class MatMulOp
{
    public static Tensor Forward(Tensor a, Tensor b) =>
        TensorOps.MatMul(a, b);

    public static (Tensor dA, Tensor dB) Backward(Tensor dC, Tensor a, Tensor b)
    {
        // dA = dC × Bᵀ   — how much each element of A affected the loss
        //      shapes: [M,N] × [N,K] = [M,K]  ✓ (matches A's shape)
        var dA = TensorOps.MatMul(dC, TensorOps.Transpose(b));

        // dB = Aᵀ × dC   — how much each element of B affected the loss
        //      shapes: [K,M] × [M,N] = [K,N]  ✓ (matches B's shape)
        var dB = TensorOps.MatMul(TensorOps.Transpose(a), dC);

        return (dA, dB);
    }
}
