using ReasonCs.Core.Autograd.Ops;
using ReasonCs.Core.Tensors;

namespace ReasonCs.Tests.Autograd;

/// <summary>
/// Numerical gradient checks for every autograd op.
/// ALL tests here must pass before the full Transformer is assembled.
///
/// Pattern: run forward, compute analytical backward, then call
/// NumericalGradientHelper.Check to compare against finite differences.
/// </summary>
public class GradientSanityTests
{
    [Fact]
    public void MatMul_GradientCheck_dA()
    {
        // A [3,4] × B [4,2] → C [3,2]
        // Loss = sum(C)  →  dC = ones [3,2]
        var a = new Tensor([3, 4]);
        var b = new Tensor([4, 2]);
        TensorOps.RandomNormal(a, seed: 10);
        TensorOps.RandomNormal(b, seed: 20);

        var dC = new Tensor([3, 2], Enumerable.Repeat(1f, 6).ToArray());
        var (dA, _) = MatMulOp.Backward(dC, a, b);

        // Numerical check: perturb A, measure change in sum(A @ B)
        NumericalGradientHelper.Check(a, dA,
            forward: () => TensorOps.Sum(MatMulOp.Forward(a, b)));
    }

    [Fact]
    public void MatMul_GradientCheck_dB()
    {
        // A [3,4] × B [4,2] → C [3,2]
        // Loss = sum(C)  →  dC = ones [3,2]
        var a = new Tensor([3, 4]);
        var b = new Tensor([4, 2]);
        TensorOps.RandomNormal(a, seed: 10);
        TensorOps.RandomNormal(b, seed: 20);

        var dC = new Tensor([3, 2], Enumerable.Repeat(1f, 6).ToArray());
        var (_, dB) = MatMulOp.Backward(dC, a, b);

        // Numerical check: perturb B, measure change in sum(A @ B)
        NumericalGradientHelper.Check(b, dB,
            forward: () => TensorOps.Sum(MatMulOp.Forward(a, b)));
    }

    [Fact]
    public void MatMul_GradientCheck_NonUniformLoss()
    {
        // Use a non-uniform dC (explicit upstream gradient) to stress-test
        // the backward beyond the trivial all-ones case.
        //
        // ⚠ We use small explicit values rather than large RandomNormal values.
        // With float32, if |Loss| ≈ 20 and |grad| ≈ 0.004, the finite-diff step
        // (fPlus - fMinus) ≈ 8e-7 is smaller than a ULP of Loss, causing
        // catastrophic cancellation. Small values keep |Loss| ≈ 0.1, which gives
        // ~4 orders of magnitude of headroom and clean numerical gradients.
        var a  = new Tensor([2, 3], [ 0.5f, -0.3f,  0.1f,
                                       0.2f, -0.1f,  0.4f]);
        var b  = new Tensor([3, 2], [ 0.3f, -0.2f,
                                       0.1f,  0.4f,
                                      -0.3f,  0.2f]);
        var dC = new Tensor([2, 2], [ 0.1f, -0.2f,
                                       0.3f, -0.1f]);

        var (dA, dB) = MatMulOp.Backward(dC, a, b);

        // Loss = sum(dC ⊙ (A @ B))  — weighted sum, equivalent to a dot product
        // with a non-trivial upstream gradient
        float Loss() => TensorOps.Sum(TensorOps.Multiply(MatMulOp.Forward(a, b), dC));

        NumericalGradientHelper.Check(a, dA, Loss);
        NumericalGradientHelper.Check(b, dB, Loss);
    }

    // =========================================================================
    // SoftmaxOp
    // =========================================================================

    [Fact]
    public void Softmax_UniformInput_ProducesUniformProbs()
    {
        // softmax([0, 0, 0, 0]) → [0.25, 0.25, 0.25, 0.25]
        var x     = new Tensor([4], [0f, 0f, 0f, 0f]);
        var probs = SoftmaxOp.Forward(x);

        foreach (var p in probs.Data)
            Assert.Equal(0.25f, p, precision: 5);
    }

    [Fact]
    public void Softmax_OutputSumsToOne()
    {
        // For any input the probabilities must sum to 1 (per row).
        var x = new Tensor([3, 5]);
        TensorOps.RandomNormal(x, seed: 7);
        var probs = SoftmaxOp.Forward(x);

        for (int row = 0; row < 3; row++)
        {
            float rowSum = 0f;
            for (int col = 0; col < 5; col++) rowSum += probs[row, col];
            Assert.Equal(1f, rowSum, precision: 5);
        }
    }

    [Fact]
    public void Softmax_GradientCheck_1D()
    {
        // Loss = sum(softmax(x) * w)  →  dOut = w
        // Small values keep |Loss| small → avoids float32 cancellation.
        var x = new Tensor([6], [ 0.5f, -0.3f,  0.1f,  0.8f, -0.2f,  0.4f]);
        var w = new Tensor([6], [ 0.1f,  0.3f, -0.2f,  0.1f,  0.2f, -0.1f]);

        var probs = SoftmaxOp.Forward(x);
        var dx    = SoftmaxOp.Backward(w, probs);  // dOut = w

        NumericalGradientHelper.Check(x, dx,
            forward: () => TensorOps.Sum(TensorOps.Multiply(SoftmaxOp.Forward(x), w)));
    }

    [Fact]
    public void Softmax_GradientCheck_2D()
    {
        // [3, 5] batch — softmax applied independently to each row.
        // Loss = sum(softmax(x) * w) over all elements.
        var x = new Tensor([3, 5], [
             0.4f, -0.1f,  0.6f, -0.3f,  0.2f,
            -0.2f,  0.5f,  0.1f,  0.3f, -0.4f,
             0.1f, -0.5f,  0.3f,  0.2f, -0.1f]);
        var w = new Tensor([3, 5], [
             0.1f, -0.2f,  0.1f,  0.0f,  0.1f,
            -0.1f,  0.2f, -0.1f,  0.1f,  0.0f,
             0.0f,  0.1f, -0.1f,  0.2f, -0.1f]);

        var probs = SoftmaxOp.Forward(x);
        var dx    = SoftmaxOp.Backward(w, probs);

        NumericalGradientHelper.Check(x, dx,
            forward: () => TensorOps.Sum(TensorOps.Multiply(SoftmaxOp.Forward(x), w)));
    }

    // =========================================================================
    // EmbeddingOp
    // =========================================================================

    [Fact]
    public void Embedding_Forward_CorrectRowGather()
    {
        // weight [4, 3], tokens = [2, 0, 2]
        // out should be rows 2, 0, 2 of weight.
        var weight = new Tensor([4, 3], [
            1f, 2f, 3f,   // row 0
            4f, 5f, 6f,   // row 1
            7f, 8f, 9f,   // row 2
           10f,11f,12f]); // row 3
        int[] tokens = [2, 0, 2];

        var output = EmbeddingOp.Forward(weight, tokens);

        Assert.Equal([3, 3], output.Shape);
        // row 0 of output = row 2 of weight
        Assert.Equal(7f, output.Data[0]); Assert.Equal(8f, output.Data[1]); Assert.Equal(9f, output.Data[2]);
        // row 1 of output = row 0 of weight
        Assert.Equal(1f, output.Data[3]); Assert.Equal(2f, output.Data[4]); Assert.Equal(3f, output.Data[5]);
        // row 2 of output = row 2 of weight (same token repeated)
        Assert.Equal(7f, output.Data[6]); Assert.Equal(8f, output.Data[7]); Assert.Equal(9f, output.Data[8]);
    }

    [Fact]
    public void Embedding_Backward_ScatterAdd()
    {
        // dOut [3, 3], tokens = [2, 0, 2]
        // dWeight[2] should be dOut[0] + dOut[2] (token 2 appears twice)
        // dWeight[0] should be dOut[1]
        // dWeight[1] and dWeight[3] should be zero
        var dOut = new Tensor([3, 3], [
            0.1f, 0.2f, 0.3f,   // position 0 → token 2
            0.4f, 0.5f, 0.6f,   // position 1 → token 0
            0.7f, 0.8f, 0.9f]); // position 2 → token 2
        int[] tokens = [2, 0, 2];

        var dWeight = EmbeddingOp.Backward(dOut, tokens, vocabSize: 4);

        Assert.Equal([4, 3], dWeight.Shape);
        // dWeight[0] = dOut[1]
        Assert.Equal(0.4f, dWeight.Data[0], precision: 5);
        Assert.Equal(0.5f, dWeight.Data[1], precision: 5);
        Assert.Equal(0.6f, dWeight.Data[2], precision: 5);
        // dWeight[1] = 0 (token 1 never used)
        Assert.Equal(0f, dWeight.Data[3]); Assert.Equal(0f, dWeight.Data[4]); Assert.Equal(0f, dWeight.Data[5]);
        // dWeight[2] = dOut[0] + dOut[2]  (token 2 used twice)
        Assert.Equal(0.1f + 0.7f, dWeight.Data[6],  precision: 5);
        Assert.Equal(0.2f + 0.8f, dWeight.Data[7],  precision: 5);
        Assert.Equal(0.3f + 0.9f, dWeight.Data[8],  precision: 5);
        // dWeight[3] = 0 (token 3 never used)
        Assert.Equal(0f, dWeight.Data[9]); Assert.Equal(0f, dWeight.Data[10]); Assert.Equal(0f, dWeight.Data[11]);
    }

    [Fact]
    public void Embedding_GradientCheck()
    {
        // weight [8, 4], tokens = [3, 1, 3, 0]  (token 3 appears twice)
        // Loss = sum(w * Forward(weight, tokens))
        // dWeight = Backward(w, tokens, vocabSize=8)
        var weight = new Tensor([8, 4]);
        TensorOps.RandomNormal(weight, seed: 11);
        int[] tokens = [3, 1, 3, 0];
        int seqLen   = tokens.Length;
        int embedDim = 4;

        // Fixed upstream gradient (acts as the dOut / "loss weights")
        var w = new Tensor([seqLen, embedDim], [
             0.1f, -0.2f,  0.3f, -0.1f,
            -0.2f,  0.1f, -0.1f,  0.2f,
             0.3f, -0.1f,  0.2f, -0.3f,
            -0.1f,  0.2f, -0.2f,  0.1f]);

        var dWeight = EmbeddingOp.Backward(w, tokens, vocabSize: 8);

        // NumericalGradientHelper perturbs weight.Data[i] one at a time.
        // Forward uses those rows → the gradient flows through scatter-add.
        NumericalGradientHelper.Check(weight, dWeight,
            forward: () => TensorOps.Sum(
                TensorOps.Multiply(EmbeddingOp.Forward(weight, tokens), w)));
    }

    // =========================================================================
    // LayerNormOp
    // =========================================================================

    [Fact]
    public void LayerNorm_OutputHasMeanZeroStdOne_WithIdentityAffine()
    {
        // With γ=1, β=0 the output should be the normalised input: mean≈0, std≈1.
        var x     = new Tensor([3, 8]);
        var gamma = new Tensor([8], Enumerable.Repeat(1f, 8).ToArray());
        var beta  = new Tensor([8], Enumerable.Repeat(0f, 8).ToArray());
        TensorOps.RandomNormal(x, seed: 5);

        var (y, _, _) = LayerNormOp.Forward(x, gamma, beta);

        for (int row = 0; row < 3; row++)
        {
            float rowMean = 0f, rowVar = 0f;
            for (int col = 0; col < 8; col++) rowMean += y[row, col];
            rowMean /= 8;
            for (int col = 0; col < 8; col++) { float d = y[row, col] - rowMean; rowVar += d * d; }
            rowVar /= 8;

            Assert.Equal(0f,  rowMean,            precision: 4);
            Assert.Equal(1f,  MathF.Sqrt(rowVar), precision: 4);
        }
    }

    [Fact]
    public void LayerNorm_GradientCheck_dx()
    {
        // Loss = sum(LayerNorm(x, γ, β) * w)  →  dOut = w
        // Avoid C# `_` scoping edge-cases by using explicit variable names.
        var x     = new Tensor([2, 4], [ 0.5f, -0.3f,  0.1f,  0.8f,
                                         -0.2f,  0.4f, -0.1f,  0.3f]);
        var gamma = new Tensor([4],    [ 1.0f,  0.8f,  1.2f,  0.9f]);
        var beta  = new Tensor([4],    [ 0.0f, -0.1f,  0.1f,  0.0f]);
        var w     = new Tensor([2, 4], [ 0.1f, -0.2f,  0.1f,  0.0f,
                                        -0.1f,  0.2f, -0.1f,  0.1f]);

        var fwdResult = LayerNormOp.Forward(x, gamma, beta);
        var bwdResult = LayerNormOp.Backward(w, x, gamma, fwdResult.mean, fwdResult.rstd);

        NumericalGradientHelper.Check(x, bwdResult.dx, forward: () =>
        {
            var fwd = LayerNormOp.Forward(x, gamma, beta);
            return TensorOps.Sum(TensorOps.Multiply(fwd.y, w));
        });
    }

    [Fact]
    public void LayerNorm_GradientCheck_dGamma_dBeta()
    {
        // Same setup — check gradients w.r.t. the affine parameters γ and β.
        var x     = new Tensor([2, 4], [ 0.5f, -0.3f,  0.1f,  0.8f,
                                         -0.2f,  0.4f, -0.1f,  0.3f]);
        var gamma = new Tensor([4],    [ 1.0f,  0.8f,  1.2f,  0.9f]);
        var beta  = new Tensor([4],    [ 0.0f, -0.1f,  0.1f,  0.0f]);
        var w     = new Tensor([2, 4], [ 0.1f, -0.2f,  0.1f,  0.0f,
                                        -0.1f,  0.2f, -0.1f,  0.1f]);

        var (yFwd, mean, rstd)     = LayerNormOp.Forward(x, gamma, beta);
        var (dxBwd, dGamma, dBeta) = LayerNormOp.Backward(w, x, gamma, mean, rstd);

        float Loss()
        {
            var (y, m, r) = LayerNormOp.Forward(x, gamma, beta);
            return TensorOps.Sum(TensorOps.Multiply(y, w));
        }

        NumericalGradientHelper.Check(gamma, dGamma, Loss);
        NumericalGradientHelper.Check(beta,  dBeta,  Loss);
    }

    /// <summary>
    /// RoPE backward is non-trivial — this test is mandatory.
    /// The backward pass must be verified against finite differences
    /// before RoPE is integrated into the Attention module.
    /// </summary>
    [Fact]
    public void RoPE_ForwardAtPositionZero_IsIdentity()
    {
        // At position = 0 every angle is 0 → cos = 1, sin = 0 → output equals input.
        var x = new Tensor([2, 8], [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,
                                    -1f, 0f, 0.5f, -0.5f, 1.5f, -1.5f, 2f, -2f]);
        var y = RoPEOp.Forward(x, position: 0);
        for (int i = 0; i < x.Size; i++)
            Assert.Equal(x.Data[i], y.Data[i], precision: 5);
    }

    [Fact]
    public void RoPE_BackwardAtPositionZero_IsIdentity()
    {
        // At position = 0 rotation is identity, so backward is also identity.
        var dOut = new Tensor([1, 8], [0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f]);
        var dx   = RoPEOp.Backward(dOut, position: 0);
        for (int i = 0; i < dOut.Size; i++)
            Assert.Equal(dOut.Data[i], dx.Data[i], precision: 5);
    }

    [Fact]
    public void RoPE_GradientCheck()
    {
        // Apply RoPE to a [1, 8] query vector at a non-trivial position (position=3).
        // Loss = sum(w * RoPE(x, position))  →  dOut = w,  dx = RoPE_Backward(w, position)
        //
        // Using small values to keep |Loss| small and avoid float32 cancellation.
        const int position = 3;
        var x = new Tensor([1, 8], [ 0.5f, -0.3f,  0.2f, -0.1f,
                                      0.4f, -0.2f,  0.1f,  0.3f]);
        var w = new Tensor([1, 8], [ 0.1f, -0.2f,  0.2f, -0.1f,
                                    -0.1f,  0.3f, -0.2f,  0.1f]);

        var y  = RoPEOp.Forward(x, position);
        var dx = RoPEOp.Backward(w, position);   // dOut = w

        NumericalGradientHelper.Check(x, dx,
            forward: () => TensorOps.Sum(TensorOps.Multiply(RoPEOp.Forward(x, position), w)));
    }

    [Fact]
    public void RoPE_GradientCheck_MultipleBatch()
    {
        // Check that RoPE works correctly over a batch dimension [3, 8].
        const int position = 7;
        var x = new Tensor([3, 8], [
             0.5f, -0.3f,  0.2f, -0.1f,  0.4f, -0.2f,  0.1f,  0.3f,
            -0.4f,  0.2f, -0.1f,  0.5f, -0.3f,  0.1f, -0.2f,  0.4f,
             0.1f,  0.3f, -0.5f,  0.2f,  0.3f, -0.4f,  0.5f, -0.1f]);
        var w = new Tensor([3, 8], [
             0.1f, -0.2f,  0.2f, -0.1f, -0.1f,  0.3f, -0.2f,  0.1f,
            -0.1f,  0.2f,  0.1f, -0.3f,  0.2f, -0.1f,  0.1f, -0.2f,
             0.2f, -0.1f,  0.1f,  0.2f, -0.3f,  0.1f,  0.0f, -0.1f]);

        var dx = RoPEOp.Backward(w, position);

        NumericalGradientHelper.Check(x, dx,
            forward: () => TensorOps.Sum(TensorOps.Multiply(RoPEOp.Forward(x, position), w)));
    }

    [Fact]
    public void CrossEntropy_OutputLoss_KnownResult()
    {
        // With a single example and uniform logits, the loss should be -log(1/C) = log(C).
        // logits = [0, 0, 0, 0], target = 2  →  p = [0.25, 0.25, 0.25, 0.25]
        // loss = -log(0.25) = log(4) ≈ 1.38629
        var logits  = new Tensor([1, 4], [0f, 0f, 0f, 0f]);
        int[] targets = [2];
        var (loss, probs) = CrossEntropyOp.Forward(logits, targets);
        Assert.Equal(MathF.Log(4f), loss, precision: 4);
    }

    [Fact]
    public void CrossEntropy_Probs_SumToOne()
    {
        var logits  = new Tensor([3, 8]);
        int[] targets = [0, 3, 7];
        TensorOps.RandomNormal(logits, seed: 42);
        var (_, probs) = CrossEntropyOp.Forward(logits, targets);

        for (int b = 0; b < 3; b++)
        {
            float rowSum = 0f;
            for (int i = 0; i < 8; i++) rowSum += probs[b, i];
            Assert.Equal(1f, rowSum, precision: 5);
        }
    }

    [Fact]
    public void CrossEntropy_GradientCheck()
    {
        // [4 examples, VocabSize=16] logits; check dLogits against finite differences.
        // Small explicit values keep |Loss| small — avoids float32 cancellation.
        var logits  = new Tensor([4, 16], Enumerable.Range(0, 64)
                          .Select(i => (i % 16 == 0 ? 0.5f : -0.1f) + 0.02f * i)
                          .ToArray());
        int[] targets = [0, 3, 7, 15];

        var (_, probs)   = CrossEntropyOp.Forward(logits, targets);
        var dLogits      = CrossEntropyOp.Backward(probs, targets);

        float Loss()
        {
            var (l, _) = CrossEntropyOp.Forward(logits, targets);
            return l;
        }

        NumericalGradientHelper.Check(logits, dLogits, Loss);
    }
}
