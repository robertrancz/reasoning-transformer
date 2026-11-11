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
    [Fact(Skip = "Implement RoPEOp first")]
    public void RoPE_GradientCheck()
    {
        // TODO: apply RoPE to a [1, 8] query vector at position 3, check dx
        throw new NotImplementedException();
    }

    [Fact(Skip = "Implement CrossEntropyOp first")]
    public void CrossEntropy_GradientCheck()
    {
        // TODO: [4, VocabSize=16] logits, 4 targets, check dLogits
        throw new NotImplementedException();
    }
}
