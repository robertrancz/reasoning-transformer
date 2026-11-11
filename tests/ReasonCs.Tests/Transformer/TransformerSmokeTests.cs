using ReasonCs.Core.Autograd.Ops;
using ReasonCs.Core.Training;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Tests.Transformer;

/// <summary>
/// Smoke tests for the assembled Transformer.
/// These tests are skipped until the forward pass is implemented.
/// Their purpose is to verify shapes and that no NaN/Inf values are produced.
/// </summary>
public class TransformerSmokeTests
{
    private static TransformerConfig SmallConfig => new(
        VocabSize: 64, EmbedDim: 32, NumHeads: 2,
        NumLayers: 2, MlpHiddenDim: 64, MaxSeqLen: 16);

    [Fact]
    public void Forward_OutputShape_IsSeqLen_x_VocabSize()
    {
        var model = new TransformerModel(SmallConfig);
        var tokens = new[] { 1, 5, 3, 7, 2 };

        var logits = model.Forward(tokens);

        Assert.Equal(2, logits.Rank);
        Assert.Equal(tokens.Length,       logits.Shape[0]);
        Assert.Equal(SmallConfig.VocabSize, logits.Shape[1]);
    }

    [Fact]
    public void Forward_NoNaNOrInf()
    {
        var model  = new TransformerModel(SmallConfig);
        var tokens = new[] { 1, 5, 3 };
        var logits = model.Forward(tokens);

        foreach (var v in logits.Data)
        {
            Assert.False(float.IsNaN(v),  "NaN detected in logits");
            Assert.False(float.IsInfinity(v), "Inf detected in logits");
        }
    }

    [Fact]
    public void Backward_NoNaNOrInf_InGradients()
    {
        // Verifies that a full forward→CrossEntropy→backward pass produces
        // finite gradients in all weight tensors.
        var model   = new TransformerModel(SmallConfig);
        var tokens  = new[] { 1, 5, 3, 7, 2 };
        var targets = new[] { 5, 3, 7, 2, 0 };   // next-token targets

        var logits      = model.Forward(tokens);
        var (_, probs)  = CrossEntropyOp.Forward(logits, targets);

        model.ZeroGrad();
        var dLogits = CrossEntropyOp.Backward(probs, targets);
        model.Backward(dLogits);

        foreach (var (_, grad) in model.GetParamGrads())
        {
            foreach (var v in grad.Data)
            {
                Assert.False(float.IsNaN(v),      "NaN detected in gradient");
                Assert.False(float.IsInfinity(v), "Inf detected in gradient");
            }
        }
    }

    [Fact]
    public void Backward_GradientNotAllZero()
    {
        // At least some gradients must be non-zero (sanity check that backward
        // is actually computing something, not just returning zeros).
        var model   = new TransformerModel(SmallConfig);
        var tokens  = new[] { 1, 5, 3 };
        var targets = new[] { 5, 3, 7 };

        var logits       = model.Forward(tokens);
        var (_, probs)   = CrossEntropyOp.Forward(logits, targets);
        model.ZeroGrad();
        model.Backward(CrossEntropyOp.Backward(probs, targets));

        bool anyNonZero = false;
        foreach (var (_, grad) in model.GetParamGrads())
            foreach (var v in grad.Data)
                if (v != 0f) { anyNonZero = true; break; }

        Assert.True(anyNonZero, "All gradients are zero — backward may be broken.");
    }

    [Fact]
    public void Training_LossDecreasesAfterGradientSteps()
    {
        // A single gradient-step should move the parameters in the direction
        // that reduces loss; after 20 steps on the same batch the loss must
        // be strictly lower than the initial loss.
        var model     = new TransformerModel(SmallConfig);
        var optimizer = new AdamW(lr: 1e-3f, weightDecay: 0f);
        var tokens    = new[] { 1, 5, 3, 7, 2 };
        var targets   = new[] { 5, 3, 7, 2, 0 };

        // Compute initial loss
        var logits0       = model.Forward(tokens);
        var (loss0, _)    = CrossEntropyOp.Forward(logits0, targets);

        // Take 20 gradient steps on the same batch (over-fit)
        for (int s = 0; s < 20; s++)
        {
            var l         = model.Forward(tokens);
            var (_, p)    = CrossEntropyOp.Forward(l, targets);
            model.ZeroGrad();
            model.Backward(CrossEntropyOp.Backward(p, targets));
            optimizer.Step(model.GetParamGrads());
        }

        var logits1    = model.Forward(tokens);
        var (loss1, _) = CrossEntropyOp.Forward(logits1, targets);

        Assert.True(loss1 < loss0,
            $"Loss did not decrease after 20 gradient steps: {loss0:F4} → {loss1:F4}");
    }

    [Fact]
    public void GreedyDecode_ReturnsPaddedTokenList()
    {
        var model       = new TransformerModel(SmallConfig);
        var prompt      = new[] { 1, 5, 3 };
        int maxNewTokens = 4;

        var result = model.GreedyDecode(prompt, maxNewTokens);

        // Output must be prompt + maxNewTokens tokens
        Assert.Equal(prompt.Length + maxNewTokens, result.Length);

        // Prompt prefix must be preserved
        for (int i = 0; i < prompt.Length; i++)
            Assert.Equal(prompt[i], result[i]);

        // All generated tokens must be valid vocabulary indices
        foreach (var tok in result.Skip(prompt.Length))
            Assert.InRange(tok, 0, SmallConfig.VocabSize - 1);
    }
}
