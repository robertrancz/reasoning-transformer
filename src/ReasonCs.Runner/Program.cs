using ReasonCs.Core.Data;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Inference;
using ReasonCs.Core.Transformer;
using ReasonCs.Core.Training;

// ─────────────────────────────────────────────────────────────────────────────
//  reason.cs — entry point
// ─────────────────────────────────────────────────────────────────────────────
//
//  Stages (uncomment as each is implemented):
//
//  1. Tensor + Autograd tests      → run:  dotnet test
//  2. Train base model             → done below
//  3. Evaluate greedy baseline     → done below
//  4. MCTS reasoning               → uncomment RunMcts()
//

Console.WriteLine("reason.cs — From-Scratch Reasoning Transformer");
Console.WriteLine("================================================");

TrainBaseModel();

// RunMcts();   // TODO: uncomment once MCTS + RewardModel are implemented

// ─────────────────────────────────────────────────────────────────────────────

static void TrainBaseModel()
{
    using var logger = RunLogger.Create(logDir: "logs");

    // ── 1. Generate dataset ───────────────────────────────────────────────────
    logger.Info("\n=== Dataset ===");
    var allProblems = SyntheticDataset.Generate(count: 5_000, seed: 42).ToList();
    var (trainSet, testSet) = SyntheticDataset.TrainTestSplit(allProblems, testRatio: 0.1f, seed: 42);

    logger.Kv("Problems",  $"{allProblems.Count}  (train {trainSet.Length} / test {testSet.Length})");
    logger.Info("Samples:");
    foreach (var p in trainSet.Take(3))
        logger.Info($"  prompt='{p.Prompt}'  answer='{p.Answer}'  steps=[{string.Join(", ", p.Steps)}]");

    // ── 2. Build character-level tokenizer ────────────────────────────────────
    logger.Info("\n=== Tokenizer ===");
    var tokenizer = Tokenizer.BuildFromCorpus(allProblems.Select(p => p.FullText));
    logger.Kv("Vocab size", tokenizer.VocabSize);
    logger.Kv("Characters", $"[{string.Join("", Enumerable.Range(0, tokenizer.VocabSize)
                                                           .Select(i => tokenizer.Decode([i])))}]");

    // ── 3. Create model ───────────────────────────────────────────────────────
    // NOTE: Compact config so the demo completes in ~3-5 min on naive MatMul.
    // Phase 2 will add AVX2 SIMD for 10-20× speed-up, enabling full config.
    logger.Info("\n=== Model ===");
    var config = new TransformerConfig(
        VocabSize:    tokenizer.VocabSize,
        EmbedDim:     64,
        NumHeads:     4,
        NumLayers:    2,
        MlpHiddenDim: 256,
        MaxSeqLen:    32);
    var model     = new TransformerModel(config);
    var optimizer = new AdamW(lr: 3e-4f, weightDecay: 0.01f);
    long paramCount = model.GetParamGrads().Sum(pg => (long)pg.param.Size);

    logger.Kv("VocabSize",    config.VocabSize);
    logger.Kv("EmbedDim",     config.EmbedDim);
    logger.Kv("NumHeads",     config.NumHeads);
    logger.Kv("NumLayers",    config.NumLayers);
    logger.Kv("MlpHiddenDim", config.MlpHiddenDim);
    logger.Kv("Parameters",   $"{paramCount:N0}");
    logger.Kv("LearningRate", optimizer.LearningRate);
    logger.Kv("WeightDecay",  optimizer.WeightDecay);
    logger.Info("(Phase 2: swap in AVX2 MatMul for 10-20× speed-up)");

    // ── 4. Train ──────────────────────────────────────────────────────────────
    logger.Info("\n=== Training (2 000 steps) ===");
    var batches = MakeBatches(trainSet, tokenizer);
    var loop    = new TrainingLoop { LogEvery = 200 };
    loop.Train(model, optimizer, RepeatForever(batches), maxSteps: 2_000, logger: logger);

    // ── 5. Evaluate greedy baseline ───────────────────────────────────────────
    logger.Info("\n=== Greedy baseline ===");
    var result = Evaluator.RunGreedy(model, tokenizer, testSet,
                                     maxNewTokens: 4, logger: logger);
    logger.Info($"Accuracy: {result}");
    logger.Info("\n(Run MCTS on top of this model to beat the greedy baseline.)");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Convert training problems into (tokens, targets) pairs for the training loop.
///
/// Each problem becomes one sequence:
///   full    = encode(Prompt + Answer)           e.g. [A,=,3, ,B,=,A,+,2, ,B,=,?,5]
///   tokens  = full[..^1]   (drop last)
///   targets = full[1..]    (drop first)
///
/// The model predicts the next character at every position.
/// </summary>
static IEnumerable<(int[] tokens, int[] targets)> MakeBatches(
    IEnumerable<SyntheticDataset.Problem> problems, Tokenizer tokenizer)
{
    foreach (var p in problems)
    {
        var full = tokenizer.Encode(p.FullText);
        if (full.Length < 2) continue;
        yield return (full[..^1], full[1..]);
    }
}

/// <summary>Repeat a sequence forever (training loop uses maxSteps to stop).</summary>
static IEnumerable<T> RepeatForever<T>(IEnumerable<T> source)
{
    while (true)
        foreach (var item in source)
            yield return item;
}
