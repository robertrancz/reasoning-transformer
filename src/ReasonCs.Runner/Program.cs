using ReasonCs.Core.Data;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Inference;
using ReasonCs.Core.Reasoning;
using ReasonCs.Core.Training;
using ReasonCs.Core.Transformer;

// ─────────────────────────────────────────────────────────────────────────────
//  reason.cs — entry point
// ─────────────────────────────────────────────────────────────────────────────
//
//  Pipeline stages:
//    1. Train base LM                → TrainBaseModel()
//    2. Greedy baseline eval         → Evaluator.RunGreedy()
//    3. Train ORM reward model       → RewardTrainer.TrainOrm()
//    4. ORM ranking eval             → RewardTrainer.EvalOrmRanking()
//    5. MCTS reasoning               → RunMcts()  [TODO]
//

Console.WriteLine("reason.cs — From-Scratch Reasoning Transformer");
Console.WriteLine("================================================");

RunPipeline();

// RunMcts();   // TODO: uncomment once MctsSearch is implemented

// ─────────────────────────────────────────────────────────────────────────────

static void RunPipeline()
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

    // ── 3. Create and train base language model ───────────────────────────────
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

    logger.Info("\n=== Training base LM (2 000 steps) ===");
    var batches = MakeBatches(trainSet, tokenizer);
    var loop    = new TrainingLoop { LogEvery = 200 };
    loop.Train(model, optimizer, RepeatForever(batches), maxSteps: 2_000, logger: logger);

    // ── 4. Greedy baseline ────────────────────────────────────────────────────
    logger.Info("\n=== Greedy baseline ===");
    var baselineResult = Evaluator.RunGreedy(model, tokenizer, testSet,
                                             maxNewTokens: 4, logger: logger);
    logger.Info($"Greedy accuracy: {baselineResult}");

    // ── 5. Train ORM (Outcome Reward Model) ───────────────────────────────────
    logger.Info("\n=== Reward Model (ORM) — training ===");
    var orm          = new RewardModel(model, RewardModel.Mode.ORM);
    var ormOptimizer = new AdamW(lr: 1e-3f, weightDecay: 0f);
    long ormParams   = orm.GetParamGrads().Sum(pg => (long)pg.param.Size);

    logger.Kv("ORM params",     $"{ormParams:N0}  (score head only — backbone frozen)");
    logger.Kv("ORM lr",         ormOptimizer.LearningRate);
    logger.Kv("ORM steps",      "1 000");

    var ormResult = RewardTrainer.TrainOrm(
        orm, ormOptimizer, tokenizer,
        trainProblems: trainSet,
        evalProblems:  testSet,
        steps:    1_000,
        logEvery: 100,
        logger:   logger);

    // ── 6. ORM ranking evaluation ─────────────────────────────────────────────
    logger.Info("\n=== ORM ranking accuracy ===");
    logger.Info($"ORM ranking accuracy (correct > wrong): {ormResult.FinalRankingAcc:P1}");
    logger.Info("(Random baseline = 50%; perfect ORM = 100%)");
    logger.Info("\n(Next: MctsSearch — use ORM scores to guide tree search and beat greedy baseline.)");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Convert training problems into (tokens, targets) pairs for the training loop.
/// Each full sequence (prompt + answer) is shifted by one for next-token prediction.
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
