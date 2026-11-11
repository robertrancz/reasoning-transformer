using System.Numerics;

namespace ReasonCs.Core.Tensors;

/// <summary>
/// Core tensor math operations.
///
/// PHASE 2 — SIMD fast-path for MatMul using System.Numerics.Vector&lt;float&gt;.
/// On AVX2 hardware this is 8-wide (256-bit); on SSE2 it is 4-wide (128-bit).
/// The naive scalar path is retained as MatMulNaive for correctness tests.
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
    /// Hardware-dispatched MatMul: uses SIMD when Vector.IsHardwareAccelerated,
    /// falls back to the scalar reference otherwise.
    /// PyTorch equivalent:  torch.matmul(a, b)  or  a @ b
    /// </summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMul requires rank-2 tensors.");
        int M = a.Shape[0], K = a.Shape[1], N = b.Shape[1];
        if (K != b.Shape[0])
            throw new ArgumentException($"Shape mismatch: [{M},{K}] × [{b.Shape[0]},{N}]");

        return Vector.IsHardwareAccelerated
            ? MatMulSimdCore(a.Data, b.Data, M, K, N)
            : MatMulNaiveCore(a.Data, b.Data, M, K, N);
    }

    /// <summary>
    /// Scalar reference MatMul — always uses the naive O(M·K·N) loop.
    /// Used by unit tests to validate the SIMD path against this ground truth.
    /// </summary>
    public static Tensor MatMulNaive(Tensor a, Tensor b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMul requires rank-2 tensors.");
        int M = a.Shape[0], K = a.Shape[1], N = b.Shape[1];
        if (K != b.Shape[0])
            throw new ArgumentException($"Shape mismatch: [{M},{K}] × [{b.Shape[0]},{N}]");
        return MatMulNaiveCore(a.Data, b.Data, M, K, N);
    }

    // ---- private cores -------------------------------------------------------

    private static Tensor MatMulNaiveCore(float[] aData, float[] bData, int M, int K, int N)
    {
        var c     = new Tensor([M, N]);
        var cData = c.Data;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum  = 0f;
                int   aOff = i * K;
                int   bOff = j;          // step by N each iteration
                for (int p = 0; p < K; p++, bOff += N)
                    sum += aData[aOff + p] * bData[bOff];
                cData[i * N + j] = sum;
            }
        return c;
    }

    /// <summary>
    /// SIMD MatMul using System.Numerics.Vector&lt;float&gt;.
    ///
    /// Strategy:
    ///   1. Physically transpose B → BT so BT[j, k] = B[k, j].
    ///      Both A rows and BT rows are contiguous in memory, enabling
    ///      SIMD loads of vecSize floats at a time from each.
    ///   2. Accumulate a (vecSize-wide) vector: acc += va * vb,
    ///      then do a single horizontal sum per output element.
    ///   3. Scalar tail loop handles the K % vecSize remaining elements.
    ///
    /// Complexity: O(K·N) one-time transpose + O(M·K·N / vecSize) SIMD work
    /// → ~8× throughput on AVX2 (Vector.Count = 8).
    /// </summary>
    private static Tensor MatMulSimdCore(float[] aData, float[] bData, int M, int K, int N)
    {
        // Step 1: Transpose B into a flat row-major array.
        //         bT[j * K + k] = B[k * N + j]
        var bT = new float[N * K];
        for (int k = 0; k < K; k++)
        {
            int bRow = k * N;
            for (int n = 0; n < N; n++)
                bT[n * K + k] = bData[bRow + n];
        }

        // Step 2: SIMD dot products.
        int vecSize = Vector<float>.Count;   // 8 on AVX2, 4 on SSE2
        var c       = new Tensor([M, N]);
        var cData   = c.Data;

        for (int i = 0; i < M; i++)
        {
            int aOff = i * K;
            for (int j = 0; j < N; j++)
            {
                int  bOff = j * K;
                int  p    = 0;

                // Vector accumulator — one horizontal reduction per output cell.
                var acc = Vector<float>.Zero;
                for (; p <= K - vecSize; p += vecSize)
                    acc += new Vector<float>(aData, aOff + p)
                         * new Vector<float>(bT,    bOff + p);

                float sum = Vector.Sum(acc);

                // Scalar tail for K % vecSize remaining elements.
                for (; p < K; p++)
                    sum += aData[aOff + p] * bT[bOff + p];

                cData[i * N + j] = sum;
            }
        }
        return c;
    }

    // -------------------------------------------------------------------------
    // Transpose  [M, N] → [N, M]
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Reduction
    // -------------------------------------------------------------------------

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
    /// Fill every element with a constant value.
    /// PyTorch equivalent:  torch.full(shape, fill_value)  /  tensor.fill_(value)
    /// </summary>
    public static void Fill(Tensor t, float value) => Array.Fill(t.Data, value);

    /// <summary>
    /// In-place element-wise addition: dest += src.
    /// Tensors must have equal size; dest's shape is preserved.
    /// PyTorch equivalent:  dest.add_(src)
    /// </summary>
    public static void AddTo(Tensor dest, Tensor src)
    {
        if (dest.Size != src.Size)
            throw new ArgumentException(
                $"Size mismatch in AddTo: {dest.Size} vs {src.Size}.");
        for (int i = 0; i < dest.Size; i++)
            dest.Data[i] += src.Data[i];
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
            double r  = Math.Sqrt(-2.0 * Math.Log(u1));
            t.Data[i]     = (float)(mean + std * r * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < t.Size)
                t.Data[i + 1] = (float)(mean + std * r * Math.Sin(2 * Math.PI * u2));
        }
    }
}
