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
}
