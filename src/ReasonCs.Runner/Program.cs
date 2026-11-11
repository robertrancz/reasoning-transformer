using ReasonCs.Core.Data;
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
    // ── 1. Generate dataset ───────────────────────────────────────────────────
    Console.WriteLine("\n=== Dataset ===");
    var allProblems = SyntheticDataset.Generate(count: 5_000, seed: 42).ToList();
    var (trainSet, testSet) = SyntheticDataset.TrainTestSplit(allProblems, testRatio: 0.1f, seed: 42);
    Console.WriteLine($"Problems  : {allProblems.Count}  (train {trainSet.Length} / test {testSet.Length})");

    // Print a few examples so the reader can see the data format
    Console.WriteLine("Samples:");
    foreach (var p in trainSet.Take(3))
        Console.WriteLine($"  prompt='{p.Prompt}'  answer='{p.Answer}'  " +
                          $"steps=[{string.Join(", ", p.Steps)}]");

    // ── 2. Build character-level tokenizer from the full corpus ───────────────
    Console.WriteLine("\n=== Tokenizer ===");
    var tokenizer = Tokenizer.BuildFromCorpus(allProblems.Select(p => p.FullText));
    Console.WriteLine($"Vocab size : {tokenizer.VocabSize}");
    Console.WriteLine($"Chars      : [{string.Join("", Enumerable.Range(0, tokenizer.VocabSize)
                                                              .Select(i => tokenizer.Decode([i])))}]");

    // ── 3. Create model ───────────────────────────────────────────────────────
    // NOTE: We use a compact config here so the demo completes in ~3-5 minutes
    // on the naive O(M·K·N) MatMul.  Phase 2 will add AVX2 SIMD, making the
    // full 4-layer / D=128 model tractable.
    Console.WriteLine("\n=== Model ===");
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
    Console.WriteLine($"Config     : {config}");
    Console.WriteLine($"Parameters : {paramCount:N0}");
    Console.WriteLine("(Phase 2: swap in AVX2 MatMul for 10-20× speed-up)");

    // ── 4. Train ──────────────────────────────────────────────────────────────
    // 2 000 steps × ~200 ms/step ≈ 6-7 minutes on a single CPU core.
    Console.WriteLine("\n=== Training (2 000 steps) ===");
    var batches = MakeBatches(trainSet, tokenizer);
    var loop    = new TrainingLoop { LogEvery = 200 };
    loop.Train(model, optimizer, RepeatForever(batches), maxSteps: 2_000);

    // ── 5. Evaluate greedy baseline ───────────────────────────────────────────
    Console.WriteLine("\n=== Greedy baseline ===");
    var result = Evaluator.RunGreedy(model, tokenizer, testSet, maxNewTokens: 4);
    Console.WriteLine($"Accuracy   : {result}");
    Console.WriteLine();
    Console.WriteLine("(Run MCTS on top of this model to beat the greedy baseline.)");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Convert training problems into (tokens, targets) pairs for the training loop.
///
/// Each problem becomes one sequence:
///   full = encode(Prompt + Answer)              e.g. [A,=,3, ,B,=,A,+,2, ,B,=,?,5]
///   tokens  = full[..^1]   (drop last)
///   targets = full[1..]    (drop first)
///
/// The model predicts the next character at every position — the last target
/// token is the final answer character(s), which is where cross-entropy loss
/// actually matters for answer quality.
/// </summary>
static IEnumerable<(int[] tokens, int[] targets)> MakeBatches(
    IEnumerable<SyntheticDataset.Problem> problems, Tokenizer tokenizer)
{
    foreach (var p in problems)
    {
        var full = tokenizer.Encode(p.FullText);
        if (full.Length < 2) continue;        // degenerate, skip
        yield return (full[..^1], full[1..]);
    }
}

/// <summary>
/// Repeat a sequence forever (for the training loop's maxSteps stopping criterion).
/// </summary>
static IEnumerable<T> RepeatForever<T>(IEnumerable<T> source)
{
    while (true)
        foreach (var item in source)
            yield return item;
}
