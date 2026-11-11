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
}
