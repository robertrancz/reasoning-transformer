using ReasonCs.Core.Data;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Reasoning;
using ReasonCs.Core.Tensors;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Training;

/// <summary>
/// Training and evaluation utilities for <see cref="RewardModel"/>.
///
/// Training strategy (ORM)
/// ───────────────────────
/// Each step samples one problem from the training set.  With probability 0.5
/// the step is a <em>positive</em> example (prompt + correct answer, label = 1);
/// otherwise it is a <em>negative</em> example (prompt + wrong answer, label = 0).
/// This balanced curriculum keeps the score head from collapsing to a trivial
/// constant predictor.
///
/// Negative examples
/// ─────────────────
/// By default negatives are model-sampled (temperature-sampled from the base LM),
/// which directly matches the candidate distribution seen by MCTS at inference time.
/// This "hard negative" strategy forces the ORM to distinguish correct answers from
/// plausible-but-wrong model outputs.
///
/// If no base model is provided, negatives fall back to random integers in [0, 99],
/// which is weaker but sufficient when MCTS is not used.
///
/// Evaluation metric — ranking accuracy
/// ──────────────────────────────────────
/// For each test problem we ask: does the ORM rank the correct answer above a
/// randomly sampled wrong answer?
///
///   ranking_accuracy = #{correct > wrong} / #{pairs}
///
/// A random scorer achieves 50%; a perfect ORM achieves 100%.
/// This metric does not require a score-threshold and is directly relevant to
/// how the ORM will be used inside MCTS (to rank candidate steps).
///
/// PyTorch equivalent pattern:
///   # No single framework class; the pattern is:
///   optimizer = torch.optim.AdamW(reward_model.score_head.parameters(), lr=1e-3)
///   loss_fn   = torch.nn.BCELoss()
///   for tokens, label in loader:
///       optimizer.zero_grad()
///       score = reward_model(tokens)
///       loss  = loss_fn(score, label)
///       loss.backward()
///       optimizer.step()
/// </summary>
public static class RewardTrainer
{
    public record TrainResult(int Steps, float LastLoss, float FinalRankingAcc);

    // ── Training ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Train the ORM score head for <paramref name="steps"/> gradient steps.
    ///
    /// The backbone is frozen — only <see cref="RewardModel.ScoreHead"/> is updated.
    ///
    /// Each step:
    ///   1. Sample problem at random.
    ///   2. With p=0.5 use positive (correct answer, label=1) or
    ///      negative (wrong answer, label=0).
    ///   3. Score → ZeroGrad → Backward → optimizer.Step
    ///
    /// Returns the last training loss.
    /// </summary>
    public static TrainResult TrainOrm(
        RewardModel    orm,
        AdamW          optimizer,
        Tokenizer      tokenizer,
        SyntheticDataset.Problem[] trainProblems,
        SyntheticDataset.Problem[] evalProblems,
        int               steps       = 1_000,
        int               logEvery    = 100,
        RunLogger?        logger      = null,
        TransformerModel? baseModel   = null,   // when set: hard negatives from model sampling
        float             negativeTemp = 0.8f)  // temperature for hard-negative sampling
    {
        var rng      = new Random(42);
        float lastLoss = float.NaN;
        float runningLoss = 0f;
        int   logCount    = 0;

        for (int step = 0; step < steps; step++)
        {
            var problem    = trainProblems[rng.Next(trainProblems.Length)];
            bool positive  = rng.Next(2) == 0;

            // All sequences are padded/truncated to promptLen + ContinuationLen tokens
            // so the last-token hidden state is always at the same relative position.
            // This ensures ORM training and MCTS inference share the same distribution.
            int[] tokens;
            float label;
            if (positive)
            {
                string padded = PadContinuation(problem.Answer);
                tokens = tokenizer.Encode(problem.Prompt + padded);
                label  = 1.0f;
            }
            else
            {
                // Hard negatives: sample from base model so ORM training distribution
                // matches MCTS inference distribution (plausible-but-wrong answers).
                // Fall back to random wrong answer when no base model is available.
                string wrong = baseModel is not null
                    ? SampleModelWrongAnswer(problem, tokenizer, baseModel, rng, negativeTemp)
                      ?? SampleWrongAnswer(problem.Answer, rng)
                    : SampleWrongAnswer(problem.Answer, rng);
                string padded = PadContinuation(wrong);
                tokens = tokenizer.Encode(problem.Prompt + padded);
                label  = 0.0f;
            }

            // Forward → backward → update
            orm.Score(tokens);
            orm.ZeroGrad();
            float loss = orm.Backward(label);
            optimizer.Step(orm.GetParamGrads());

            lastLoss    = loss;
            runningLoss += loss;
            logCount++;

            if (step % logEvery == 0)
            {
                float avgLoss = runningLoss / logCount;
                logger?.Info($"ORM  step={step,-6}  avg_bce={avgLoss:F4}  label={label:F0}");
                runningLoss = 0f;
                logCount    = 0;
            }
        }

        float rankAcc = EvalOrmRanking(orm, tokenizer, evalProblems, logger: logger);
        return new TrainResult(steps, lastLoss, rankAcc);
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// <summary>
    /// Ranking accuracy: fraction of (correct, wrong) pairs where the ORM correctly
    /// assigns a higher score to the correct answer.
    ///
    /// Random baseline = 50%.  Perfect ORM = 100%.
    /// </summary>
    public static float EvalOrmRanking(
        RewardModel                           orm,
        Tokenizer                             tokenizer,
        IEnumerable<SyntheticDataset.Problem> testProblems,
        int                                   samplesPerProblem = 1,
        RunLogger?                            logger            = null,
        TransformerModel?                     baseModel         = null,
        float                                 negativeTemp      = 0.8f)
    {
        var rng     = new Random(0);
        int correct = 0, total = 0;

        foreach (var problem in testProblems)
        {
            for (int s = 0; s < samplesPerProblem; s++)
            {
                string posStr  = PadContinuation(problem.Answer);
                float posScore = orm.Score(tokenizer.Encode(problem.Prompt + posStr));

                string wrong = baseModel is not null
                    ? SampleModelWrongAnswer(problem, tokenizer, baseModel, rng, negativeTemp)
                      ?? SampleWrongAnswer(problem.Answer, rng)
                    : SampleWrongAnswer(problem.Answer, rng);

                string negStr  = PadContinuation(wrong);
                float negScore = orm.Score(tokenizer.Encode(problem.Prompt + negStr));

                if (posScore > negScore) correct++;
                total++;
            }
        }

        float acc = total == 0 ? 0f : (float)correct / total;
        logger?.Eval(correct, total, acc);
        return acc;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// The fixed continuation length used for all ORM training and MCTS inference.
    /// All sequences passed to the ORM are exactly <c>promptLen + ContinuationLen</c>
    /// tokens long. This ensures the last-token hidden state is a consistent
    /// representation regardless of the actual answer length.
    ///
    /// Must match <see cref="MctsSearch.StepTokenLen"/> in Program.cs.
    /// </summary>
    public const int ContinuationLen = 2;

    /// <summary>
    /// Pad or truncate <paramref name="continuation"/> to exactly
    /// <see cref="ContinuationLen"/> characters using space as the pad character.
    ///
    /// Space is guaranteed to be in the synthetic dataset's vocabulary (used as
    /// a separator between variable assignments).
    ///
    /// Examples:
    ///   "8"  → "8 "    (1-char answer padded to 2)
    ///   "10" → "10"    (already 2 chars — no change)
    ///   "123"→ "12"    (truncated to 2)
    /// </summary>
    public static string PadContinuation(string continuation)
    {
        if (continuation.Length >= ContinuationLen)
            return continuation[..ContinuationLen];
        return continuation.PadRight(ContinuationLen, ' ');
    }

    /// <summary>
    /// Sample a wrong answer string: a random integer in [0, 99] that differs
    /// from <paramref name="correctAnswer"/>.
    /// All decimal digits are in the synthetic dataset's vocabulary, so any
    /// integer 0–99 is safely encodable.
    /// </summary>
    public static string SampleWrongAnswer(string correctAnswer, Random rng)
    {
        for (int i = 0; i < 100; i++)
        {
            string candidate = rng.Next(0, 100).ToString();
            if (candidate != correctAnswer)
                return candidate;
        }
        // Fallback (only reached if correctAnswer covers every 0-99 value, impossible)
        return correctAnswer == "0" ? "1" : "0";
    }

    // ── Hard-negative sampling ────────────────────────────────────────────────

    /// <summary>
    /// Generate a model-sampled wrong answer by temperature-decoding
    /// <see cref="ContinuationLen"/> tokens from the prompt using the base LM.
    ///
    /// Returns <c>null</c> if the sampled continuation accidentally matches the
    /// correct answer (the caller should fall back to <see cref="SampleWrongAnswer"/>).
    ///
    /// Using model-sampled negatives closes the training/inference distribution
    /// gap for the ORM: MCTS presents plausible-but-wrong model outputs, so the
    /// ORM must learn to distinguish those — not arbitrary random integers.
    /// </summary>
    private static string? SampleModelWrongAnswer(
        SyntheticDataset.Problem problem,
        Tokenizer                tokenizer,
        TransformerModel         model,
        Random                   rng,
        float                    temperature)
    {
        int[] promptTokens = tokenizer.Encode(problem.Prompt);
        var   tokens       = new List<int>(promptTokens.Length + ContinuationLen);
        tokens.AddRange(promptTokens);

        int vocabSize = model.Config.VocabSize;

        for (int step = 0; step < ContinuationLen; step++)
        {
            var logits = model.Forward([.. tokens]);
            int lastOffset = (tokens.Count - 1) * vocabSize;
            float[] lastLogits = new float[vocabSize];
            Array.Copy(logits.Data, lastOffset, lastLogits, 0, vocabSize);
            int token = SampleTokenFromLogits(lastLogits, temperature, rng);
            tokens.Add(token);
        }

        // Decode only the new tokens
        int[] contTokens  = [.. tokens.Skip(promptTokens.Length)];
        string continuation = tokenizer.Decode(contTokens).TrimEnd();
        return continuation == problem.Answer ? null : continuation;
    }

    /// <summary>
    /// Numerically stable temperature sampling over raw logits.
    /// Returns the sampled token index.
    /// </summary>
    private static int SampleTokenFromLogits(float[] logits, float temperature, Random rng)
    {
        float maxVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxVal) maxVal = logits[i];

        float sumExp = 0f;
        float[] probs = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i]  = MathF.Exp((logits[i] - maxVal) / temperature);
            sumExp   += probs[i];
        }
        for (int i = 0; i < logits.Length; i++) probs[i] /= sumExp;

        float u = (float)rng.NextDouble(), cumsum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cumsum += probs[i];
            if (u < cumsum) return i;
        }
        return probs.Length - 1;
    }

    // ── PRM training ──────────────────────────────────────────────────────────

    /// <summary>
    /// Train the PRM score head on step-level supervision.
    ///
    /// Negative examples
    /// ─────────────────
    /// When <paramref name="baseModel"/> is provided, wrong steps at depth k are
    /// temperature-sampled from the base LM given <c>prompt + correct_prefix[..k-1]</c>.
    /// This closes the training/inference distribution gap: MCTS presents model-sampled
    /// candidates, so the PRM must learn to distinguish those from the correct step —
    /// not arbitrary random integers.
    ///
    /// If no base model is provided, negatives fall back to random integers in [0, 99].
    ///
    /// Training distribution
    /// ─────────────────────
    /// Each gradient step samples one problem and one step depth k uniformly.
    /// With 50% probability the example is positive (correct chain up to step k);
    /// otherwise it is negative (correct prefix + wrong step k).
    ///
    /// Sequence format
    /// ───────────────
    /// prompt + chain[0..k] where each chain slot is <see cref="SyntheticDataset.Problem.StepWidth"/>
    /// characters wide (value right-padded with spaces to StepWidth).
    ///
    /// PyTorch equivalent:
    ///   Similar to Lightman et al. "Let's Verify Step by Step" (2023) — but with
    ///   programmatic step labels instead of human annotation.
    /// </summary>
    /// <param name="backboneOptimizer">
    /// Optional separate optimizer for the backbone parameters.  Pass a low-lr
    /// AdamW (e.g. 3e-5) when <c>prm.FreezeBackbone = false</c> to fine-tune the
    /// backbone while keeping the step size from disrupting LM representations.
    /// When null, backbone parameters are not updated even if unfrozen.
    /// </param>
    /// <param name="baseModel">
    /// Optional base language model for hard-negative step sampling.
    /// When set, negative examples use model-generated wrong steps rather than
    /// random integers, matching the MCTS inference distribution.
    /// </param>
    /// <param name="negativeTemp">
    /// Temperature for model-sampled negative steps (default 0.8).
    /// </param>
    public static TrainResult TrainPrm(
        RewardModel                        prm,
        AdamW                              optimizer,
        Tokenizer                          tokenizer,
        SyntheticDataset.Problem[]         trainProblems,
        SyntheticDataset.Problem[]         evalProblems,
        int                                steps             = 1_000,
        int                                logEvery          = 100,
        RunLogger?                         logger            = null,
        AdamW?                             backboneOptimizer = null,
        TransformerModel?                  baseModel         = null,
        float                              negativeTemp      = 0.8f)
    {
        var rng       = new Random(42);
        float lastLoss = float.NaN;
        float runningLoss = 0f;
        int   logCount    = 0;

        for (int step = 0; step < steps; step++)
        {
            var problem  = trainProblems[rng.Next(trainProblems.Length)];
            int numSteps = problem.NumSteps;

            // Sample a random step depth (1-indexed)
            int stepDepth = rng.Next(1, numSteps + 1);
            bool positive = rng.Next(2) == 0;

            int[] tokens;
            float label;

            if (positive)
            {
                // Correct prefix up to stepDepth
                string chain = string.Concat(
                    problem.StepValues[..stepDepth]
                           .Select(v => v.ToString().PadRight(SyntheticDataset.Problem.StepWidth)));
                tokens = tokenizer.Encode(problem.Prompt + chain);
                label  = 1.0f;
            }
            else
            {
                // Hard negatives: model-sampled wrong step, matching MCTS inference distribution.
                // Fall back to random wrong step when no base model is available.
                string correctVal = problem.StepValues[stepDepth - 1].ToString();
                string wrongVal = baseModel is not null
                    ? SampleModelWrongStep(problem, stepDepth, tokenizer, baseModel, rng, negativeTemp)
                      ?? SampleWrongAnswer(correctVal, rng)
                    : SampleWrongAnswer(correctVal, rng);

                string prefix = stepDepth > 1
                    ? string.Concat(problem.StepValues[..(stepDepth - 1)]
                                           .Select(v => v.ToString().PadRight(SyntheticDataset.Problem.StepWidth)))
                    : "";
                string wrongStep = wrongVal.PadRight(SyntheticDataset.Problem.StepWidth);
                tokens = tokenizer.Encode(problem.Prompt + prefix + wrongStep);
                label  = 0.0f;
            }

            prm.ZeroAllGrads();   // zero score head + backbone grads (if unfrozen)
            prm.Score(tokens);
            float loss = prm.Backward(label);
            optimizer.Step(prm.GetParamGrads());
            backboneOptimizer?.Step(prm.GetBackboneParamGrads());

            lastLoss    = loss;
            runningLoss += loss;
            logCount++;

            if (step % logEvery == 0)
            {
                float avgLoss = runningLoss / logCount;
                logger?.Info($"PRM  step={step,-6}  avg_bce={avgLoss:F4}  label={label:F0}  depth={stepDepth}");
                runningLoss = 0f;
                logCount    = 0;
            }
        }

        float rankAcc = EvalPrmRanking(prm, tokenizer, evalProblems, logger: logger);
        return new TrainResult(steps, lastLoss, rankAcc);
    }

    /// <summary>
    /// PRM ranking accuracy: fraction of (correct-step, wrong-step) pairs where
    /// the PRM correctly assigns a higher score to the correct step.
    ///
    /// Evaluated at every step depth, averaged over all depths and problems.
    /// Random baseline = 50%; perfect PRM = 100%.
    ///
    /// When <paramref name="baseModel"/> is provided, wrong steps are model-sampled
    /// rather than random, measuring PRM quality against the MCTS inference distribution.
    /// </summary>
    public static float EvalPrmRanking(
        RewardModel                           prm,
        Tokenizer                             tokenizer,
        IEnumerable<SyntheticDataset.Problem> testProblems,
        RunLogger?                            logger    = null,
        TransformerModel?                     baseModel = null,
        float                                 negTemp   = 0.8f)
    {
        var rng     = new Random(0);
        int correct = 0, total = 0;

        foreach (var problem in testProblems)
        {
            int numSteps = problem.NumSteps;

            for (int k = 1; k <= numSteps; k++)
            {
                // Build correct prefix up to step k
                string posChain = string.Concat(
                    problem.StepValues[..k]
                           .Select(v => v.ToString().PadRight(SyntheticDataset.Problem.StepWidth)));
                float posScore = prm.Score(tokenizer.Encode(problem.Prompt + posChain));

                // Build incorrect prefix: correct up to k-1, wrong at k
                string correctVal = problem.StepValues[k - 1].ToString();
                string wrongVal = baseModel is not null
                    ? SampleModelWrongStep(problem, k, tokenizer, baseModel, rng, negTemp)
                      ?? SampleWrongAnswer(correctVal, rng)
                    : SampleWrongAnswer(correctVal, rng);
                string negPrefix = k > 1
                    ? string.Concat(problem.StepValues[..(k - 1)]
                                           .Select(v => v.ToString().PadRight(SyntheticDataset.Problem.StepWidth)))
                    : "";
                string negChain = negPrefix + wrongVal.PadRight(SyntheticDataset.Problem.StepWidth);
                float negScore  = prm.Score(tokenizer.Encode(problem.Prompt + negChain));

                if (posScore > negScore) correct++;
                total++;
            }
        }

        float acc = total == 0 ? 0f : (float)correct / total;
        logger?.Eval(correct, total, acc);
        return acc;
    }

    // ── Hard-negative step sampling (PRM) ────────────────────────────────────

    /// <summary>
    /// Temperature-sample step <paramref name="stepDepth"/> from the base LM
    /// given <c>prompt + correct_prefix[..stepDepth-1]</c>.
    ///
    /// Returns the decoded step string (trimmed), or <c>null</c> if the sampled
    /// step accidentally matches the correct step (caller should fall back to
    /// <see cref="SampleWrongAnswer"/>).
    ///
    /// Exactly <see cref="SyntheticDataset.Problem.StepWidth"/> tokens are sampled,
    /// matching the fixed step width used in MCTS CoT mode.
    /// </summary>
    private static string? SampleModelWrongStep(
        SyntheticDataset.Problem problem,
        int                      stepDepth,
        Tokenizer                tokenizer,
        TransformerModel         model,
        Random                   rng,
        float                    temperature)
    {
        // Build prompt + correct prefix up to stepDepth-1
        string prefix = stepDepth > 1
            ? string.Concat(problem.StepValues[..(stepDepth - 1)]
                                   .Select(v => v.ToString().PadRight(SyntheticDataset.Problem.StepWidth)))
            : "";
        int[] promptTokens = tokenizer.Encode(problem.Prompt + prefix);
        var   tokens       = new List<int>(promptTokens.Length + SyntheticDataset.Problem.StepWidth);
        tokens.AddRange(promptTokens);

        int vocabSize = model.Config.VocabSize;

        // Sample StepWidth tokens (one char per token in the char-level tokenizer)
        for (int s = 0; s < SyntheticDataset.Problem.StepWidth; s++)
        {
            var logits = model.Forward([.. tokens]);
            int lastOffset = (tokens.Count - 1) * vocabSize;
            float[] lastLogits = new float[vocabSize];
            Array.Copy(logits.Data, lastOffset, lastLogits, 0, vocabSize);
            int token = SampleTokenFromLogits(lastLogits, temperature, rng);
            tokens.Add(token);
        }

        // Decode only the newly sampled step tokens
        int[]  stepTokens  = [.. tokens.Skip(promptTokens.Length)];
        string sampledStep = tokenizer.Decode(stepTokens).TrimEnd();
        string correctStep = problem.StepValues[stepDepth - 1].ToString();
        return sampledStep == correctStep ? null : sampledStep;
    }
}
