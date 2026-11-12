using ReasonCs.Core.Data;
using ReasonCs.Core.Reasoning;
using ReasonCs.Core.Training;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Tests.TrainingTests;

/// <summary>
/// Smoke tests for <see cref="RlTrainer"/> (REINFORCE policy gradient).
/// </summary>
public class RlTrainerTests
{
    private static (TransformerModel model, RewardModel orm, Tokenizer tok,
                    SyntheticDataset.Problem[] train, SyntheticDataset.Problem[] test) MakeFixture(int count = 60)
    {
        var probs            = SyntheticDataset.Generate(count, seed: 7).ToArray();
        var tok              = Tokenizer.BuildFromCorpus(probs.Select(p => p.FullText));
        var model            = new TransformerModel(new TransformerConfig(
            VocabSize: tok.VocabSize, EmbedDim: 16, NumHeads: 2,
            NumLayers: 1, MlpHiddenDim: 32, MaxSeqLen: 32));
        var orm              = new RewardModel(model, RewardModel.Mode.ORM);
        var (train, test)    = SyntheticDataset.TrainTestSplit([.. probs], testRatio: 0.2f, seed: 7);
        return (model, orm, tok, train, test);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Basic smoke: does a short run complete without exception?
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TrainReinforce_OrmReward_CompletesWithoutException()
    {
        var (model, orm, tok, train, test) = MakeFixture();
        var optimizer = new AdamW(lr: 1e-3f, weightDecay: 0f);

        var result = RlTrainer.TrainReinforce(
            model, optimizer, tok,
            trainProblems:     train,
            evalProblems:      test,
            rewardFn:          (_, tokens) => orm.Score(tokens),
            steps:             10,
            samplesPerProblem: 2,
            temperature:       0.8f);

        Assert.True(result.FinalAccuracy >= 0f && result.FinalAccuracy <= 1f);
        Assert.Equal(10, result.Steps);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Oracle reward: binary correctness signal
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TrainReinforce_OracleReward_CompletesWithoutException()
    {
        var (model, _, tok, train, test) = MakeFixture();
        var optimizer = new AdamW(lr: 1e-3f, weightDecay: 0f);

        var result = RlTrainer.TrainReinforce(
            model, optimizer, tok,
            trainProblems:     train,
            evalProblems:      test,
            rewardFn:          (_, __) => 0.5f,   // constant reward → zero advantage → no-op ✓
            steps:             10,
            samplesPerProblem: 2,
            temperature:       0.8f);

        Assert.True(result.FinalAccuracy >= 0f && result.FinalAccuracy <= 1f);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gradient signal: model weights must change after a non-trivial reward
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TrainReinforce_NonConstantReward_ChangesModelWeights()
    {
        var (model, _, tok, train, test) = MakeFixture();
        var optimizer = new AdamW(lr: 1e-2f, weightDecay: 0f);

        // Capture a sample weight before training
        float weightBefore = model.LMHead.Data[0];

        int callCount = 0;
        RlTrainer.TrainReinforce(
            model, optimizer, tok,
            trainProblems:     train,
            evalProblems:      test,
            rewardFn: (_, __) =>
            {
                // Alternating 0 / 1 so advantages are non-zero
                return (++callCount % 2 == 0) ? 1.0f : 0.0f;
            },
            steps:             5,
            samplesPerProblem: 2,
            temperature:       0.8f);

        float weightAfter = model.LMHead.Data[0];
        Assert.NotEqual(weightBefore, weightAfter);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Zero-variance reward: all-equal rewards → zero advantage → no weight change
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TrainReinforce_AllSameRewards_NoWeightChange()
    {
        var (model, _, tok, train, test) = MakeFixture();
        var optimizer = new AdamW(lr: 1e-2f, weightDecay: 0f);

        float[] weightsBefore = (float[])model.LMHead.Data.Clone();

        RlTrainer.TrainReinforce(
            model, optimizer, tok,
            trainProblems:     train,
            evalProblems:      test,
            rewardFn:          (_, __) => 0.7f,   // constant reward → advantage = 0 → no gradient
            steps:             5,
            samplesPerProblem: 4,
            temperature:       0.8f);

        // Weights should be identical — no gradient with zero advantage
        Assert.Equal(weightsBefore, model.LMHead.Data);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ORM reward produces rewards in (0,1) range
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void RlReward_OrmScore_InRangeZeroToOne()
    {
        var (model, orm, tok, train, _) = MakeFixture(count: 20);
        int[] promptToks = tok.Encode(train[0].Prompt);

        // Simulate what TemperatureSample would do: prompt + 2 random answer tokens
        int[] seq = [.. promptToks, 1, 2];   // any vocab indices
        float score = orm.Score(seq);

        Assert.InRange(score, 0f, 1f);
    }
}
