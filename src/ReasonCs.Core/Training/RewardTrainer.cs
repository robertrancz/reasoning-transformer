using ReasonCs.Core.Data;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Reasoning;
using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Training;

/// <summary>
/// Training and evaluation utilities for <see cref="RewardModel"/>.
///
/// Training strategy (ORM)
/// ───────────────────────
/// Each step samples one problem from the training set.  With probability 0.5
/// the step is a <em>positive</em> example (prompt + correct answer, label = 1);
/// otherwise it is a <em>negative</em> example (prompt + randomly sampled wrong
/// answer, label = 0).  This balanced curriculum keeps the score head from
/// collapsing to a trivial constant predictor.
///
/// Negative examples
/// ─────────────────
/// A wrong answer is a random integer in [0, 99] that differs from the correct
/// answer string.  Because the synthetic dataset's tokenizer vocabulary includes
/// all decimal digits, every such integer is encodable.
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
        int            steps    = 1_000,
        int            logEvery = 100,
        RunLogger?     logger   = null)
    {
        var rng      = new Random(42);
        float lastLoss = float.NaN;
        float runningLoss = 0f;
        int   logCount    = 0;

        for (int step = 0; step < steps; step++)
        {
            var problem    = trainProblems[rng.Next(trainProblems.Length)];
            bool positive  = rng.Next(2) == 0;

            int[] tokens;
            float label;
            if (positive)
            {
                tokens = tokenizer.Encode(problem.FullText);
                label  = 1.0f;
            }
            else
            {
                string wrong = SampleWrongAnswer(problem.Answer, rng);
                tokens = tokenizer.Encode(problem.Prompt + wrong);
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
        RewardModel orm,
        Tokenizer   tokenizer,
        IEnumerable<SyntheticDataset.Problem> testProblems,
        int         samplesPerProblem = 1,
        RunLogger?  logger = null)
    {
        var rng     = new Random(0);
        int correct = 0, total = 0;

        foreach (var problem in testProblems)
        {
            for (int s = 0; s < samplesPerProblem; s++)
            {
                float posScore = orm.Score(tokenizer.Encode(problem.FullText));
                string wrong   = SampleWrongAnswer(problem.Answer, rng);
                float negScore = orm.Score(tokenizer.Encode(problem.Prompt + wrong));

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
}
