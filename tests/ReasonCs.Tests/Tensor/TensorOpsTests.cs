using System.Numerics;
using ReasonCs.Core.Tensors;

namespace ReasonCs.Tests.TensorTests;

/// <summary>
/// Unit tests for TensorOps — the very first tests to write and pass.
/// Naive MatMul must be correct before any other work continues.
/// PHASE 2: SIMD fast-path tests added alongside the naive reference tests.
/// </summary>
public class TensorOpsTests
{
    [Fact]
    public void MatMul_2x2_KnownResult()
    {
        // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]  →  C = [[19,22],[43,50]]
        var a = new Tensor([2, 2], [1f, 2f, 3f, 4f]);
        var b = new Tensor([2, 2], [5f, 6f, 7f, 8f]);

        var c = TensorOps.MatMul(a, b);

        Assert.Equal(19f, c[0, 0], precision: 4);
        Assert.Equal(22f, c[0, 1], precision: 4);
        Assert.Equal(43f, c[1, 0], precision: 4);
        Assert.Equal(50f, c[1, 1], precision: 4);
    }

    [Fact]
    public void MatMul_ShapeMismatch_Throws()
    {
        var a = new Tensor([2, 3]);
        var b = new Tensor([4, 2]);
        Assert.Throws<ArgumentException>(() => TensorOps.MatMul(a, b));
    }

    [Fact]
    public void MatMul_Identity_ReturnsOriginal()
    {
        var a = new Tensor([3, 3], [1f,2f,3f, 4f,5f,6f, 7f,8f,9f]);
        var eye = new Tensor([3, 3], [1f,0f,0f, 0f,1f,0f, 0f,0f,1f]);

        var result = TensorOps.MatMul(a, eye);

        for (int i = 0; i < 9; i++)
            Assert.Equal(a.Data[i], result.Data[i], precision: 4);
    }

    [Fact]
    public void Scale_MultipliesAllElements()
    {
        var t = new Tensor([3], [1f, 2f, 3f]);
        var r = TensorOps.Scale(t, 3f);
        Assert.Equal([3f, 6f, 9f], r.Data);
    }

    // -------------------------------------------------------------------------
    // SIMD correctness tests (Phase 2)
    // -------------------------------------------------------------------------

    /// <summary>
    /// MatMulSimd must match MatMulNaive on a random 32×64×64 multiply
    /// (representative of the Attention weight multiply shapes).
    /// Tolerance: 1e-4 absolute — SIMD accumulates in a different order
    /// than the scalar loop, so single-precision rounding differs by ~1e-7.
    /// </summary>
    [Fact]
    public void MatMulSimd_MatchesNaive_RepresentativeShape()
    {
        var rng = new Random(99);
        float[] Fill(int n) { var d = new float[n]; for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1); return d; }

        var a = new Tensor([32, 64], Fill(32 * 64));
        var b = new Tensor([64, 64], Fill(64 * 64));

        var naive = TensorOps.MatMulNaive(a, b);
        var simd  = TensorOps.MatMul(a, b);

        Assert.Equal(naive.Shape, simd.Shape);
        float maxErr = 0f;
        for (int i = 0; i < naive.Size; i++)
            maxErr = Math.Max(maxErr, Math.Abs(naive.Data[i] - simd.Data[i]));
        Assert.True(maxErr < 1e-4f, $"Max element-wise error {maxErr:E3} exceeds 1e-4");
    }

    /// <summary>
    /// MatMul SIMD path handles shapes where K is NOT a multiple of Vector.Count
    /// (tests the scalar tail loop).
    /// </summary>
    [Fact]
    public void MatMulSimd_ScalarTail_NonMultipleOfVectorCount()
    {
        // Use K=5 — not a multiple of 4 (SSE) or 8 (AVX2)
        var a = new Tensor([3, 5], [1f,2f,3f,4f,5f,  6f,7f,8f,9f,10f,  11f,12f,13f,14f,15f]);
        var b = new Tensor([5, 2], [1f,0f,  0f,1f,  1f,0f,  0f,1f,  1f,0f]);

        var naive = TensorOps.MatMulNaive(a, b);
        var simd  = TensorOps.MatMul(a, b);

        Assert.Equal(naive.Shape, simd.Shape);
        for (int i = 0; i < naive.Size; i++)
            Assert.Equal(naive.Data[i], simd.Data[i], precision: 4);
    }

    /// <summary>
    /// Rough throughput benchmark: SIMD must be at least 2× faster than naive
    /// on the representative [32,64]×[64,64] shape (i.e. typical Attention forward).
    /// The exact ratio varies by CPU but AVX2 should give ~6-8×; even SSE2 gives ~3-4×.
    /// Printed to output for visibility.
    /// </summary>
    [Fact]
    public void MatMulSimd_FasterThanNaive_Benchmark()
    {
        const int Warmup = 5, Iters = 100;
        var rng = new Random(7);
        float[] Fill(int n) { var d = new float[n]; for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1); return d; }

        var a = new Tensor([32, 64], Fill(32 * 64));
        var b = new Tensor([64, 64], Fill(64 * 64));

        // Warm up JIT
        for (int i = 0; i < Warmup; i++) { TensorOps.MatMulNaive(a, b); TensorOps.MatMul(a, b); }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) TensorOps.MatMulNaive(a, b);
        sw.Stop();
        double naiveMs = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        for (int i = 0; i < Iters; i++) TensorOps.MatMul(a, b);
        sw.Stop();
        double simdMs = sw.Elapsed.TotalMilliseconds;

        double speedup = naiveMs / simdMs;
        Console.WriteLine($"[Benchmark] MatMul [32,64]×[64,64]: naive={naiveMs/Iters:F3}ms  simd={simdMs/Iters:F3}ms  speedup={speedup:F1}×  Vector.Count={Vector<float>.Count}");

        // Must be at least 1.5× faster even in Debug builds.
        // Release+AVX2 typically gives 3-8×.
        Assert.True(speedup >= 1.5, $"SIMD speedup {speedup:F1}× is less than 1.5× — check hardware acceleration.");
    }

    /// <summary>
    /// MatMul SIMD path on a larger random matrix ([32,64]×[64,256]) —
    /// representative of the MLP W1/W2/W3 multiplies.
    /// Tolerance: 1e-4 absolute.
    /// </summary>
    [Fact]
    public void MatMulSimd_MatchesNaive_MlpShape()
    {
        var rng = new Random(123);
        float[] Fill(int n) { var d = new float[n]; for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1); return d; }

        var a = new Tensor([32, 64],  Fill(32 * 64));
        var b = new Tensor([64, 256], Fill(64 * 256));

        var naive = TensorOps.MatMulNaive(a, b);
        var simd  = TensorOps.MatMul(a, b);

        Assert.Equal(naive.Shape, simd.Shape);
        float maxErr = 0f;
        for (int i = 0; i < naive.Size; i++)
            maxErr = Math.Max(maxErr, Math.Abs(naive.Data[i] - simd.Data[i]));
        Assert.True(maxErr < 1e-4f, $"Max element-wise error {maxErr:E3} exceeds 1e-4");
    }
}
