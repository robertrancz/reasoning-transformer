using ReasonCs.Core.Data;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Inference;
using ReasonCs.Core.Reasoning;
using ReasonCs.Core.Training;
using ReasonCs.Core.Transformer;
using System.Diagnostics;

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

    // 10 000 steps: at ~7 ms/step after SIMD = ~70 seconds.
    // More training significantly improves greedy accuracy, giving MCTS
    // more non-trivial candidates to rank.
    logger.Info("\n=== Training base LM (10 000 steps) ===");
    var batches = MakeBatches(trainSet, tokenizer);
    var loop    = new TrainingLoop { LogEvery = 1_000 };
    loop.Train(model, optimizer, RepeatForever(batches), maxSteps: 10_000, logger: logger);

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
        steps:         1_000,
        logEvery:      100,
        logger:        logger,
        baseModel:     model,       // hard negatives — model-sampled wrong answers
        negativeTemp:  0.8f);       //   matching MCTS's inference temperature

    // ── 6. ORM ranking evaluation ─────────────────────────────────────────────
    logger.Info("\n=== ORM ranking accuracy ===");
    logger.Info($"ORM ranking acc (vs random wrong)  : {ormResult.FinalRankingAcc:P1}");
    // Also evaluate on model-sampled negatives — the actual distribution MCTS faces.
    float ormHardAcc = RewardTrainer.EvalOrmRanking(
        orm, tokenizer, testSet,
        baseModel:    model,
        negativeTemp: 0.5f,
        logger:       null);
    logger.Info($"ORM ranking acc (vs model-sampled) : {ormHardAcc:P1}");
    logger.Info("(Random baseline = 50%; perfect ORM = 100%)");

    // ── 7. MCTS reasoning ─────────────────────────────────────────────────────
    logger.Info("\n=== MCTS reasoning (Test-Time Compute) ===");
    // StepTokenLen=2 MUST match RewardTrainer.ContinuationLen so the ORM sees the
    // same sequence structure during MCTS as it was trained on (prompt + 2 tokens).
    // Temperature=0.5: lower than 0.8 so each sample has higher probability of
    // matching the correct answer while still providing diversity beyond greedy.
    var mcts = new MctsSearch(model, orm)
    {
        StepTokenLen          = RewardTrainer.ContinuationLen,  // =2, matches ORM training
        TopK                  = 4,     // branching factor
        SearchBudget          = 4,     // = TopK: budget fully spent on unique candidates
        Temperature           = 0.2f,  // low T → samples close to greedy, higher p(correct)
        ExplorationC          = 1.41f,
        Seed                  = 42,
        IncludeGreedyCandidate = false  // pure temperature: all 4 slots are explorations
    };
    logger.Kv("MCTS TopK",                  mcts.TopK);
    logger.Kv("MCTS SearchBudget",          mcts.SearchBudget);
    logger.Kv("MCTS Temperature",           mcts.Temperature);
    logger.Kv("MCTS StepTokenLen",          mcts.StepTokenLen);
    logger.Kv("MCTS IncludeGreedyCandidate", mcts.IncludeGreedyCandidate);

    // Run on all 500 test problems — with TopK=4 / Budget=4 it takes ≈20 s
    var mctsTestSet = testSet.ToArray();
    logger.Info($"Evaluating on all {mctsTestSet.Length} test problems ...");

    var sw = Stopwatch.StartNew();
    var mctsResult = MctsEvaluator.RunMcts(mcts, tokenizer, mctsTestSet, logger: logger);
    sw.Stop();

    // Greedy baseline on same subset for a fair comparison
    var greedySubset = Evaluator.RunGreedy(model, tokenizer, mctsTestSet, maxNewTokens: 4);

    logger.Info($"\n--- Results on first {mctsTestSet.Length} problems ---");
    logger.Info($"Greedy accuracy : {greedySubset}");
    logger.Info($"MCTS   accuracy : {mctsResult}");
    logger.Info($"Delta           : {(mctsResult.Accuracy - greedySubset.Accuracy):+0.0%;-0.0%;0.0%}");
    logger.Info($"MCTS elapsed    : {sw.Elapsed.TotalSeconds:F1} s");

    if (mctsResult.Accuracy > greedySubset.Accuracy)
        logger.Info("✓ MCTS beats greedy — Test-Time Compute helps!");
    else
        logger.Info("✗ MCTS did not beat greedy on this run.");

    // ── 8. Chain-of-Thought retraining + PRM ─────────────────────────────────
    // The PRM scores intermediate steps.  This requires the base LM to generate
    // a chain-of-thought ("2  3  9  " for A=2, B=3, C=9) rather than just the
    // final answer.  We retrain a fresh model on the CoT format, then train the
    // PRM on partial chains, and run multi-step MCTS.
    logger.Info("\n=== Chain-of-Thought retraining (PRM phase) ===");

    // Use only 3-step problems so all chains have the same length (9 chars).
    var allProblems3 = SyntheticDataset.Generate(count: 5_000, minSteps: 3, maxSteps: 3, seed: 99).ToList();
    var (train3, test3) = SyntheticDataset.TrainTestSplit(allProblems3, testRatio: 0.1f, seed: 99);

    // Build tokenizer from CoT corpus (FullTextChain includes step padding spaces)
    var tokenizerCot = Tokenizer.BuildFromCorpus(allProblems3.Select(p => p.FullTextChain));
    logger.Kv("CoT vocab size", tokenizerCot.VocabSize);

    // Fresh model for CoT training
    var configCot  = new TransformerConfig(
        VocabSize:    tokenizerCot.VocabSize,
        EmbedDim:     64,
        NumHeads:     4,
        NumLayers:    2,
        MlpHiddenDim: 256,
        MaxSeqLen:    48);   // longer: prompt (~20 chars) + chain (9 chars) = ~29 tokens
    var modelCot   = new TransformerModel(configCot);
    var optCot     = new AdamW(lr: 3e-4f, weightDecay: 0.01f);

    // CoT batches: train on Prompt + StepsChain
    var batchesCot = MakeBatchesCoT(train3, tokenizerCot);
    var loopCot    = new TrainingLoop { LogEvery = 1_000 };
    logger.Info("Training CoT base LM (10 000 steps) ...");
    loopCot.Train(modelCot, optCot, RepeatForever(batchesCot), maxSteps: 10_000, logger: logger);

    // Greedy CoT baseline
    logger.Info("\n=== CoT greedy baseline ===");
    var cotGreedy = Evaluator.RunGreedyCoT(modelCot, tokenizerCot, test3, logger: logger);
    logger.Info($"CoT greedy accuracy: {cotGreedy}");

    // Train PRM (backbone unfrozen — fine-tune transformer to learn step-quality signal)
    // IMPORTANT: Use a *copy* of modelCot as the PRM backbone so that PRM fine-tuning
    // does not modify the generation model's weights.  After PRM training the generation
    // model (modelCot) is unchanged; only the prmBackbone has been adapted for scoring.
    logger.Info("\n=== PRM (Process Reward Model) — training ===");
    // Frozen backbone: keeps PRM and generation model in the same representation space,
    // which improves MCTS candidate scoring consistency over unfreezing.
    var prm           = new RewardModel(modelCot, RewardModel.Mode.PRM);   // FreezeBackbone=true
    var prmOptimizer  = new AdamW(lr: 1e-3f, weightDecay: 0f);
    AdamW? prmBackboneOpt = null;   // backbone stays frozen — no backbone optimizer needed
    long headParams = prm.GetParamGrads().Sum(pg => (long)pg.param.Size);
    logger.Kv("PRM params", $"{headParams:N0}  (score head only — backbone frozen)");
    logger.Kv("PRM steps",  "8 000");

    var prmResult = RewardTrainer.TrainPrm(
        prm, prmOptimizer, tokenizerCot,
        trainProblems: train3,
        evalProblems:  test3,
        steps:         8_000,
        logEvery:      400,
        logger:        logger,
        backboneOptimizer: prmBackboneOpt);  // null → backbone stays frozen

    // Dual PRM eval: random-wrong (upper bound) and model-sampled-wrong (realistic)
    float prmRandAcc  = prmResult.FinalRankingAcc;
    float prmModelAcc = RewardTrainer.EvalPrmRanking(
        prm, tokenizerCot, test3,
        baseModel: modelCot, negTemp: 0.8f);
    logger.Info($"PRM ranking acc (vs random wrong)       : {prmRandAcc:P1}");
    logger.Info($"PRM ranking acc (vs model-sampled wrong): {prmModelAcc:P1}");
    logger.Info("(Random baseline = 50%; perfect PRM = 100%)");

    // ORM for the CoT model (scores full chain at leaf depth)
    var ormCot         = new RewardModel(modelCot, RewardModel.Mode.ORM);
    var ormCotOptimizer = new AdamW(lr: 1e-3f, weightDecay: 0f);
    var ormCotResult   = RewardTrainer.TrainOrm(
        ormCot, ormCotOptimizer, tokenizerCot,
        trainProblems: train3,
        evalProblems:  test3,
        steps:         1_000,
        logEvery:      200,
        logger:        logger);
    logger.Info($"CoT ORM ranking accuracy: {ormCotResult.FinalRankingAcc:P1}");

    // Multi-step MCTS with PRM + ORM
    logger.Info("\n=== Multi-step MCTS (PRM + ORM, MaxDepth=3) ===");
    int stepW  = SyntheticDataset.Problem.StepWidth;   // = 3
    var mctsPrm = new MctsSearch(modelCot, ormCot)
    {
        StepTokenLen           = stepW,          // 3 tokens per step
        TopK                   = 4,
        // Budget: 4 depth-1 nodes + best 4 depth-2 + best 4 depth-3 = 12 expansions.
        // Add extra sims for UCB1 to refine estimates → 32 total.
        SearchBudget           = 32,
        Temperature            = 0.3f,
        ExplorationC           = 1.41f,
        Seed                   = 42,
        MaxDepth               = 3,              // 3-step chain
        IncludeGreedyCandidate  = false,
        PrmModel               = prm             // PRM at all depths (trained on all chain lengths)
    };
    logger.Kv("MCTS MaxDepth",    mctsPrm.MaxDepth);
    logger.Kv("MCTS StepTokenLen", mctsPrm.StepTokenLen);
    logger.Kv("MCTS TopK",        mctsPrm.TopK);
    logger.Kv("MCTS SearchBudget", mctsPrm.SearchBudget);
    logger.Kv("PRM scoring",      "all depths 1-3 (PRM trained at all chain lengths)");

    logger.Info($"Evaluating on all {test3.Length} CoT test problems ...");
    var swPrm = Stopwatch.StartNew();
    var prmMctsResult = MctsEvaluator.RunMcts(mctsPrm, tokenizerCot, test3, logger: logger);
    swPrm.Stop();

    // CoT greedy on same test set for comparison
    var cotGreedySubset = Evaluator.RunGreedyCoT(modelCot, tokenizerCot, test3);

    logger.Info($"\n--- PRM MCTS Results (CoT, 3-step problems) ---");
    logger.Info($"CoT Greedy accuracy : {cotGreedySubset}");
    logger.Info($"PRM MCTS accuracy   : {prmMctsResult}");
    logger.Info($"Delta               : {(prmMctsResult.Accuracy - cotGreedySubset.Accuracy):+0.0%;-0.0%;0.0%}");
    logger.Info($"PRM MCTS elapsed    : {swPrm.Elapsed.TotalSeconds:F1} s");

    if (prmMctsResult.Accuracy > cotGreedySubset.Accuracy)
        logger.Info("✓ PRM MCTS beats CoT greedy — Process Reward Model helps!");
    else
        logger.Info("✗ PRM MCTS did not beat CoT greedy on this run.");
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

/// <summary>
/// CoT training: use <c>FullTextChain</c> (prompt + step values) as the target.
/// The model learns to generate all intermediate values, not just the final answer.
/// </summary>
static IEnumerable<(int[] tokens, int[] targets)> MakeBatchesCoT(
    IEnumerable<SyntheticDataset.Problem> problems, Tokenizer tokenizer)
{
    foreach (var p in problems)
    {
        var full = tokenizer.Encode(p.FullTextChain);
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
