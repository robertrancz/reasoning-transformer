using ReasonCs.Core.Autograd.Ops;
using ReasonCs.Core.Data;
using ReasonCs.Core.Inference;
using ReasonCs.Core.Training;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Tests.Data;

/// <summary>
/// Tests for SyntheticDataset, Tokenizer, and Evaluator.
/// These verify the data pipeline end-to-end, independent of gradient math.
/// </summary>
public class SyntheticDatasetTests
{
    // ── SyntheticDataset ──────────────────────────────────────────────────────

    [Fact]
    public void Generate_ProducesExactCount()
    {
        var problems = SyntheticDataset.Generate(count: 100, seed: 0).ToList();
        Assert.Equal(100, problems.Count);
    }

    [Fact]
    public void Generate_ProblemsHaveCorrectStepCounts()
    {
        var problems = SyntheticDataset.Generate(count: 200, minSteps: 2, maxSteps: 3, seed: 1).ToList();

        foreach (var p in problems)
        {
            Assert.InRange(p.Steps.Length, 2, 3);
            Assert.Equal(p.Steps.Length, p.StepCorrect.Length);
            Assert.True(p.StepCorrect.All(c => c), "All generated steps should be correct.");
        }
    }

    [Fact]
    public void Generate_AnswerMatchesLastStep()
    {
        // The answer must equal the value in the last evaluated step.
        // E.g. for "A=3 B=A+2 B=?", Steps=["A=3","B=5"], Answer="5"
        foreach (var p in SyntheticDataset.Generate(count: 200, seed: 2))
        {
            string lastStep = p.Steps[^1];                // e.g. "B=5"
            string stepValue = lastStep.Split('=')[1];    // e.g. "5"
            Assert.Equal(stepValue, p.Answer);
        }
    }

    [Fact]
    public void Generate_PromptEndsWithQuestionMark()
    {
        foreach (var p in SyntheticDataset.Generate(count: 50, seed: 3))
            Assert.True(p.Prompt.EndsWith('?'), $"Prompt should end with '?': '{p.Prompt}'");
    }

    [Fact]
    public void Generate_FullTextIsPrefixedByPrompt()
    {
        foreach (var p in SyntheticDataset.Generate(count: 50, seed: 4))
        {
            Assert.StartsWith(p.Prompt, p.FullText);
            Assert.EndsWith(p.Answer, p.FullText);
        }
    }

    [Fact]
    public void Generate_AnswerValuesInRange()
    {
        // No intermediate or final value should fall outside [0, 99]
        foreach (var p in SyntheticDataset.Generate(count: 500, seed: 5))
        {
            int answer = int.Parse(p.Answer);
            Assert.InRange(answer, 0, 99);
        }
    }

    [Fact]
    public void Generate_IsDeterministicWithSameSeed()
    {
        var a = SyntheticDataset.Generate(count: 20, seed: 99).ToList();
        var b = SyntheticDataset.Generate(count: 20, seed: 99).ToList();

        for (int i = 0; i < a.Count; i++)
            Assert.Equal(a[i].FullText, b[i].FullText);
    }

    [Fact]
    public void TrainTestSplit_SumsToOriginalCount()
    {
        var all = SyntheticDataset.Generate(count: 100, seed: 0).ToList();
        var (train, test) = SyntheticDataset.TrainTestSplit(all, testRatio: 0.1f, seed: 0);

        Assert.Equal(100, train.Length + test.Length);
        Assert.Equal(10, test.Length);
        Assert.Equal(90, train.Length);
    }

    [Fact]
    public void TrainTestSplit_NoIndexOverlap()
    {
        // TrainTestSplit partitions the input list by *index*, not by content.
        // There can be duplicate problem texts (small search space), but each
        // array element must appear in exactly one split.
        var all = SyntheticDataset.Generate(count: 100, seed: 0).ToList();
        var (train, test) = SyntheticDataset.TrainTestSplit(all, testRatio: 0.2f, seed: 0);

        // Every original object reference must land in exactly one partition
        var trainSet = new HashSet<SyntheticDataset.Problem>(ReferenceEqualityComparer.Instance);
        foreach (var p in train) trainSet.Add(p);

        foreach (var p in test)
            Assert.DoesNotContain(p, trainSet, ReferenceEqualityComparer.Instance);
    }

    // ── Tokenizer ─────────────────────────────────────────────────────────────

    [Fact]
    public void Tokenizer_EncodeDecodeRoundTrip()
    {
        var problems  = SyntheticDataset.Generate(count: 50, seed: 0).ToList();
        var tokenizer = Tokenizer.BuildFromCorpus(problems.Select(p => p.FullText));

        foreach (var p in problems)
        {
            var encoded = tokenizer.Encode(p.FullText);
            var decoded = tokenizer.Decode(encoded);
            Assert.Equal(p.FullText, decoded);
        }
    }

    [Fact]
    public void Tokenizer_VocabCoversAllDatasetChars()
    {
        var problems  = SyntheticDataset.Generate(count: 200, seed: 0).ToList();
        var tokenizer = Tokenizer.BuildFromCorpus(problems.Select(p => p.FullText));
        var allChars  = problems.SelectMany(p => p.FullText).ToHashSet();

        // Every character in the dataset must be encodable
        foreach (var ch in allChars)
        {
            var encoded = tokenizer.Encode(ch.ToString());
            Assert.Single(encoded);
        }
    }

    // ── Evaluator ─────────────────────────────────────────────────────────────

    [Fact]
    public void Evaluator_PipelineSmoke_LossDropsAfterTraining()
    {
        // Verifies the full generate→tokenize→train→evaluate pipeline is wired
        // correctly.  We do NOT test high accuracy here (that would require
        // thousands of steps on our unoptimized MatMul) — we only verify:
        //   1. Evaluator returns a valid fraction in [0, 1]
        //   2. Loss decreases after a short burst of over-fitting
        //
        // Kept tiny intentionally: naive O(S²·D) attention means full-length
        // sequences (~20 tokens) are ~16× slower per step than the 5-token
        // smoke tests.  100 steps × ~40 ms ≈ 4 seconds total.
        const int N          = 5;
        const int TrainSteps = 100;

        var problems  = SyntheticDataset.Generate(count: N, seed: 42).ToList();
        var tokenizer = Tokenizer.BuildFromCorpus(problems.Select(p => p.FullText));

        var config = new TransformerConfig(
            VocabSize:    tokenizer.VocabSize,
            EmbedDim:     32,
            NumHeads:     2,
            NumLayers:    1,
            MlpHiddenDim: 64,
            MaxSeqLen:    32);
        var model     = new TransformerModel(config);
        var optimizer = new AdamW(lr: 5e-3f, weightDecay: 0f);

        // Encode initial loss before any training
        float lossBeforeSum = 0f;
        foreach (var p in problems)
        {
            var full = tokenizer.Encode(p.FullText);
            if (full.Length < 2) continue;
            var logits = model.Forward(full[..^1]);
            var (loss, _) = CrossEntropyOp.Forward(logits, full[1..]);
            lossBeforeSum += loss;
        }
        float lossBefore = lossBeforeSum / N;

        // Training batches: tokens[..^1] → targets = tokens[1..]
        IEnumerable<(int[], int[])> Batches()
        {
            while (true)
                foreach (var p in problems)
                {
                    var full = tokenizer.Encode(p.FullText);
                    if (full.Length < 2) continue;
                    yield return (full[..^1], full[1..]);
                }
        }

        new TrainingLoop { LogEvery = int.MaxValue }
            .Train(model, optimizer, Batches(), maxSteps: TrainSteps);

        // Loss after training must be lower (over-fitting on 5 examples)
        float lossAfterSum = 0f;
        foreach (var p in problems)
        {
            var full = tokenizer.Encode(p.FullText);
            if (full.Length < 2) continue;
            var logits = model.Forward(full[..^1]);
            var (loss, _) = CrossEntropyOp.Forward(logits, full[1..]);
            lossAfterSum += loss;
        }
        float lossAfter = lossAfterSum / N;

        Assert.True(lossAfter < lossBefore,
            $"Loss should drop after {TrainSteps} steps: {lossBefore:F4} → {lossAfter:F4}");

        // Evaluator must return a valid fraction — pipeline is wired
        var result = Evaluator.RunGreedy(model, tokenizer, problems, maxNewTokens: 4);
        Assert.Equal(N, result.Total);
        Assert.InRange(result.Accuracy, 0f, 1f);
    }
}
