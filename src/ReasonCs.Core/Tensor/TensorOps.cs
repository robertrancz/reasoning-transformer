namespace ReasonCs.Core.Tensors;

/// <summary>
/// Core tensor math operations.
///
/// PHASE 1 — Naive (loop-based) implementations only. Correctness is the
/// sole priority here. SIMD/AVX2 fast-paths are added later, behind the
/// same interface, once every operation is covered by passing unit tests.
///
/// PyTorch equivalents (summary):
///   TensorOps.MatMul(a, b)          →  torch.matmul(a, b)  /  a @ b
///   TensorOps.Transpose(a)          →  a.T  /  a.transpose(0, 1)
///   TensorOps.Add(a, b)             →  a + b
///   TensorOps.Multiply(a, b)        →  a * b
///   TensorOps.Scale(a, s)           →  a * s
///   TensorOps.Sum(a)                →  a.sum()
///   TensorOps.RandomNormal(t, μ, σ) →  torch.randn(*shape) * σ + μ
///                                       or torch.nn.init.normal_(t, μ, σ)
/// </summary>
public static class TensorOps
{
    // -------------------------------------------------------------------------
    // Matrix Multiplication  (A: [M,K] × B: [K,N] → C: [M,N])
    // -------------------------------------------------------------------------

    /// <summary>
    /// Naive O(M·K·N) MatMul — correctness reference implementation.
    /// PyTorch equivalent:  torch.matmul(a, b)  or  a @ b
    /// </summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMul requires rank-2 tensors.");
        int m = a.Shape[0], k = a.Shape[1], n = b.Shape[1];
        if (k != b.Shape[0])
            throw new ArgumentException($"Shape mismatch: [{m},{k}] × [{b.Shape[0]},{n}]");

        var c = new Tensor([m, n]);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int p = 0; p < k; p++)
                    sum += a[i, p] * b[p, j];
                c[i, j] = sum;
            }
        return c;
    }

    // TODO: MatMulSimd — SIMD-accelerated fast path (Phase 2)

    /// <summary>
    /// Transpose a rank-2 tensor: [M, N] → [N, M].
    /// PyTorch equivalent:  a.T  /  a.transpose(0, 1)
    /// </summary>
    public static Tensor Transpose(Tensor a)
    {
        if (a.Rank != 2)
            throw new ArgumentException("Transpose requires a rank-2 tensor.");
        int m = a.Shape[0], n = a.Shape[1];
        var result = new Tensor([n, m]);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[j, i] = a[i, j];
        return result;
    }

    /// <summary>
    /// Sum all elements to a scalar.
    /// PyTorch equivalent:  a.sum()
    /// </summary>
    public static float Sum(Tensor a)
    {
        float s = 0f;
        for (int i = 0; i < a.Size; i++)
            s += a.Data[i];
        return s;
    }

    // -------------------------------------------------------------------------
    // Element-wise operations
    // -------------------------------------------------------------------------

    public static Tensor Add(Tensor a, Tensor b) =>
        ElementWise(a, b, (x, y) => x + y);

    public static Tensor Multiply(Tensor a, Tensor b) =>
        ElementWise(a, b, (x, y) => x * y);

    public static Tensor Scale(Tensor a, float scalar)
    {
        var result = new Tensor(a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] * scalar;
        return result;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static Tensor ElementWise(Tensor a, Tensor b, Func<float, float, float> op)
    {
        if (a.Size != b.Size)
            throw new ArgumentException("Tensor sizes must match for element-wise ops.");
        var result = new Tensor(a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = op(a.Data[i], b.Data[i]);
        return result;
    }

    /// <summary>
    /// Fill a tensor with random normal values using Box-Muller transform.
    /// PyTorch equivalent:  torch.nn.init.normal_(tensor, mean, std)
    /// </summary>
    public static void RandomNormal(Tensor t, float mean = 0f, float std = 1f, int seed = 42)
    {
        var rng = new Random(seed);
        for (int i = 0; i < t.Size; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            t.Data[i]     = (float)(mean + std * r * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < t.Size)
                t.Data[i + 1] = (float)(mean + std * r * Math.Sin(2 * Math.PI * u2));
        }
    }
}
