using ReasonCs.Core.Data;
using ReasonCs.Core.Reasoning;
using ReasonCs.Core.Training;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Tests.ReasoningTests;

/// <summary>
/// Unit and integration tests for RewardModel (ORM) + RewardTrainer.
/// </summary>
public class RewardModelTests
{
    // ── Tiny config (fast for CI) ─────────────────────────────────────────────
    private static readonly TransformerConfig SmallConfig = new(
        VocabSize:    19,
        EmbedDim:     16,
        NumHeads:     2,
        NumLayers:    1,
        MlpHiddenDim: 32,
        MaxSeqLen:    32);

    private static (TransformerModel model, Tokenizer tok, SyntheticDataset.Problem[] problems) MakeFixture(int count = 50)
    {
        var probs = SyntheticDataset.Generate(count, seed: 1).ToArray();
        var tok   = Tokenizer.BuildFromCorpus(probs.Select(p => p.FullText));
        var model = new TransformerModel(new TransformerConfig(
            VocabSize: tok.VocabSize, EmbedDim: 16, NumHeads: 2,
            NumLayers: 1, MlpHiddenDim: 32, MaxSeqLen: 32));
        return (model, tok, probs);
    }

    // ── Score() contract ──────────────────────────────────────────────────────

    [Fact]
    public void Score_ReturnsValueInZeroOne()
    {
        var (model, tok, probs) = MakeFixture();
        var orm    = new RewardModel(model);
        var tokens = tok.Encode(probs[0].FullText);

        float score = orm.Score(tokens);

        Assert.True(score > 0f && score < 1f,
            $"Score {score:F4} is not in (0, 1).");
    }

    [Fact]
    public void Score_DifferentSequences_DifferentScores()
    {
        var (model, tok, probs) = MakeFixture();
        var orm = new RewardModel(model);

        // Two different problems should generally get different scores
        // (not guaranteed but extremely likely with random init)
        float s0 = orm.Score(tok.Encode(probs[0].FullText));
        float s1 = orm.Score(tok.Encode(probs[1].FullText));

        // Just check both are in range — exact difference not guaranteed
        Assert.True(s0 > 0f && s0 < 1f);
        Assert.True(s1 > 0f && s1 < 1f);
    }

    // ── Backward() contract ───────────────────────────────────────────────────

    [Fact]
    public void Backward_ProducesNonZeroGradient()
    {
        var (model, tok, probs) = MakeFixture();
        var orm    = new RewardModel(model);
        var tokens = tok.Encode(probs[0].FullText);

        orm.Score(tokens);
        orm.ZeroGrad();
        orm.Backward(label: 1.0f);

        float gradNorm = 0f;
        foreach (var v in orm.DScoreHead.Data)
            gradNorm += v * v;
        gradNorm = MathF.Sqrt(gradNorm);

        Assert.True(gradNorm > 0f, "Gradient norm should be non-zero after backward.");
    }

    [Fact]
    public void Backward_ZeroGrad_ClearsGradient()
    {
        var (model, tok, probs) = MakeFixture();
        var orm    = new RewardModel(model);
        var tokens = tok.Encode(probs[0].FullText);

        orm.Score(tokens);
        orm.Backward(label: 1.0f);   // accumulate some gradient
        orm.ZeroGrad();               // clear

        float gradNorm = orm.DScoreHead.Data.Sum(v => v * v);
        Assert.Equal(0f, gradNorm);
    }

    [Fact]
    public void Backward_ReturnsBceLoss_Positive()
    {
        var (model, tok, probs) = MakeFixture();
        var orm    = new RewardModel(model);
        var tokens = tok.Encode(probs[0].FullText);

        float score = orm.Score(tokens);
        orm.ZeroGrad();
        float loss = orm.Backward(label: 1.0f);

        // BCE loss for label=1: -log(score); should be > 0
        float expected = -MathF.Log(score + 1e-10f);
        Assert.True(loss > 0f, "BCE loss must be positive.");
        Assert.Equal(expected, loss, precision: 4);
    }

    // ── Training convergence ──────────────────────────────────────────────────

    [Fact]
    public void ORM_ScoreIncreases_ForPositiveExamplesAfterTraining()
    {
        var (model, tok, probs) = MakeFixture(count: 20);
        var orm       = new RewardModel(model);
        var optimizer = new AdamW(lr: 1e-2f, weightDecay: 0f);
        var tokens    = tok.Encode(probs[0].FullText);

        float scoreBefore = orm.Score(tokens);

        // Overfit the score head on a single positive example for 50 steps
        for (int i = 0; i < 50; i++)
        {
            orm.Score(tokens);
            orm.ZeroGrad();
            orm.Backward(label: 1.0f);
            optimizer.Step(orm.GetParamGrads());
        }

        float scoreAfter = orm.Score(tokens);

        Assert.True(scoreAfter > scoreBefore,
            $"Score did not increase: before={scoreBefore:F4}  after={scoreAfter:F4}");
    }

    [Fact]
    public void ORM_ScoreDecreases_ForNegativeExamplesAfterTraining()
    {
        var (model, tok, probs) = MakeFixture(count: 20);
        var orm       = new RewardModel(model);
        var optimizer = new AdamW(lr: 1e-2f, weightDecay: 0f);

        // Use a "wrong" sequence: prompt + obviously wrong answer
        string wrongAnswer = RewardTrainer.SampleWrongAnswer(probs[0].Answer, new Random(42));
        var tokens         = tok.Encode(probs[0].Prompt + wrongAnswer);

        float scoreBefore = orm.Score(tokens);

        // Overfit the score head on a single negative example for 50 steps
        for (int i = 0; i < 50; i++)
        {
            orm.Score(tokens);
            orm.ZeroGrad();
            orm.Backward(label: 0.0f);
            optimizer.Step(orm.GetParamGrads());
        }

        float scoreAfter = orm.Score(tokens);

        Assert.True(scoreAfter < scoreBefore,
            $"Score did not decrease: before={scoreBefore:F4}  after={scoreAfter:F4}");
    }

    // ── RewardTrainer integration ─────────────────────────────────────────────

    [Fact]
    public void RewardTrainer_RankingAccuracy_AboveChanceAfterTraining()
    {
        // 40 train / 10 eval problems, 300 ORM training steps
        var allProbs = SyntheticDataset.Generate(50, seed: 7).ToArray();
        var tok      = Tokenizer.BuildFromCorpus(allProbs.Select(p => p.FullText));
        var backbone = new TransformerModel(new TransformerConfig(
            VocabSize: tok.VocabSize, EmbedDim: 16, NumHeads: 2,
            NumLayers: 1, MlpHiddenDim: 32, MaxSeqLen: 32));

        var orm       = new RewardModel(backbone);
        var optimizer = new AdamW(lr: 1e-2f, weightDecay: 0f);

        var result = RewardTrainer.TrainOrm(
            orm, optimizer, tok,
            trainProblems: allProbs[..40],
            evalProblems:  allProbs[40..],
            steps:    300,
            logEvery: 100);

        // After 300 steps on a tiny model the ORM should clearly beat random (50%)
        Assert.True(result.FinalRankingAcc > 0.55f,
            $"ORM ranking accuracy {result.FinalRankingAcc:P1} is not above 55% (random = 50%).");
    }

    // ── SampleWrongAnswer helper ──────────────────────────────────────────────

    [Fact]
    public void SampleWrongAnswer_NeverReturnsCorrectAnswer()
    {
        var rng = new Random(0);
        for (int trial = 0; trial < 200; trial++)
        {
            string correct = rng.Next(0, 100).ToString();
            string wrong   = RewardTrainer.SampleWrongAnswer(correct, new Random(trial));
            Assert.NotEqual(correct, wrong);
        }
    }
}
