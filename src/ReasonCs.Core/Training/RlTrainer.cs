using ReasonCs.Core.Data;
using ReasonCs.Core.Diagnostics;
using ReasonCs.Core.Inference;
using ReasonCs.Core.Tensors;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Training;

/// <summary>
/// REINFORCE policy-gradient fine-tuning of the base language model.
///
/// Algorithm — REINFORCE with baseline (Williams, 1992)
/// ─────────────────────────────────────────────────────────────────────────────
/// For each gradient step:
///
///   1. Sample N completions from the current policy (model) with temperature.
///
///   2. Score each completion with the reward function.
///      Default: ORM score ∈ (0, 1) — links the trained reward model back to
///      policy optimisation, closing the LM → ORM → RL loop.
///      Oracle option: binary correctness check (best signal for synthetic tasks).
///
///   3. Compute per-sample normalised advantage:
///        A_s = (r_s − mean(r)) / (std(r) + ε)
///      Subtracting the baseline (mean) reduces gradient variance; normalising
///      by std keeps the effective learning rate stable across problems.
///
///   4. For each sample s, accumulate the REINFORCE gradient:
///        dL/d_logit[t, v] = A_s / N · (softmax(logit_t)[v] − 1{v == a_t})
///      where a_t is the sampled token at completion step t.
///      This is the standard cross-entropy gradient scaled by advantage / N.
///
///      Only the logit rows corresponding to completion tokens get a non-zero
///      gradient; prompt positions are left at zero (no policy update for input).
///
///   5. After accumulating over all N samples, step the optimizer.
///
/// Connection to CrossEntropyOp
/// ────────────────────────────
/// The gradient is structurally identical to cross-entropy (CE) backward —
/// just without the 1/seqLen averaging and scaled by advantage:
///   CE backward:        (softmax - one_hot) / T
///   REINFORCE backward: (softmax - one_hot) * advantage / N
///
/// PyTorch equivalent
/// ──────────────────
///   optimizer.zero_grad()
///   for sample, advantage in zip(samples, advantages):
///       log_prob = Categorical(logits=logits[completion_pos]).log_prob(tokens)
///       (-advantage * log_prob / N).backward()
///   optimizer.step()
///
/// References
/// ──────────
///   Williams, R. J. (1992). Simple statistical gradient-following algorithms
///   for connectionist reinforcement learning. Machine Learning, 8, 229–256.
///
///   Schulman et al. (2017). Proximal Policy Optimization Algorithms.
///   (PPO builds on REINFORCE; see for KL-penalty extension.)
/// </summary>
public static class RlTrainer
{
    /// <summary>
    /// Result of one REINFORCE training run.
    ///
    /// <see cref="FinalAccuracy"/> is the greedy-decoding accuracy measured AFTER
    /// RL fine-tuning — compare with the pre-RL baseline to gauge improvement.
    ///
    /// <see cref="LastAvgReward"/> is the mean reward across the last batch of
    /// sampled completions (reward model score or oracle binary reward).
    /// </summary>
    public record RlTrainResult(int Steps, float LastAvgReward, float FinalAccuracy);

    // ── Training ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Fine-tune <paramref name="model"/> using REINFORCE with the supplied
    /// <paramref name="rewardFn"/> as the scalar reward signal.
    ///
    /// <paramref name="rewardFn"/> receives the <see cref="SyntheticDataset.Problem"/>
    /// (so it can check correctness) and the full token sequence
    /// (prompt + sampled continuation); it returns a reward in [0, 1].
    ///
    /// Typical choices:
    ///   · Oracle:  <c>(p, toks) => decoded[..p.Answer.Length] == p.Answer ? 1f : 0f</c>
    ///              — binary correctness; no reward hacking possible; best for synthetic tasks
    ///   · ORM:     <c>(_, toks) => orm.Score(toks)</c>
    ///              — proxy reward; demonstrates reward hacking when ORM quality &lt; 100%
    ///
    /// <paramref name="continuationLen"/> must match the ORM's
    /// <see cref="RewardTrainer.ContinuationLen"/> so reward scores are comparable
    /// to those seen during ORM training.
    /// </summary>
    public static RlTrainResult TrainReinforce(
        TransformerModel                                    model,
        AdamW                                               optimizer,
        Tokenizer                                           tokenizer,
        SyntheticDataset.Problem[]                          trainProblems,
        SyntheticDataset.Problem[]                          evalProblems,
        Func<SyntheticDataset.Problem, int[], float>        rewardFn,
        int                                steps             = 2_000,
        int                                samplesPerProblem = 4,
        float                              temperature       = 0.8f,
        int                                continuationLen   = RewardTrainer.ContinuationLen,
        int                                logEvery          = 200,
        RunLogger?                         logger            = null)
    {
        var   rng         = new Random(42);
        float lastReward  = float.NaN;
        float runningRew  = 0f;
        int   logCount    = 0;

        for (int step = 0; step < steps; step++)
        {
            var problem      = trainProblems[rng.Next(trainProblems.Length)];
            int[] promptToks = tokenizer.Encode(problem.Prompt);

            // ── 1. Sample N completions ───────────────────────────────────────
            var completions = new int[samplesPerProblem][];
            var rewards     = new float[samplesPerProblem];

            for (int s = 0; s < samplesPerProblem; s++)
            {
                completions[s] = TemperatureSample(model, promptToks, continuationLen, temperature, rng);
                rewards[s]     = rewardFn(problem, completions[s]);
            }

            // ── 2. Compute normalised advantages ─────────────────────────────
            //   baseline = mean reward → reduces variance without biasing gradient
            //   std normalisation → keeps effective step size stable
            float mean = 0f;
            for (int s = 0; s < samplesPerProblem; s++) mean += rewards[s];
            mean /= samplesPerProblem;

            float varSum = 0f;
            for (int s = 0; s < samplesPerProblem; s++)
                varSum += (rewards[s] - mean) * (rewards[s] - mean);
            float std = MathF.Sqrt(varSum / samplesPerProblem + 1e-8f);

            // Divide by N now so accumulated gradients are automatically averaged
            float[] advantages = new float[samplesPerProblem];
            for (int s = 0; s < samplesPerProblem; s++)
                advantages[s] = (rewards[s] - mean) / (std * samplesPerProblem);

            // ── 3. Policy gradient update (REINFORCE) ─────────────────────────
            model.ZeroGrad();
            for (int s = 0; s < samplesPerProblem; s++)
            {
                if (MathF.Abs(advantages[s]) < 1e-8f) continue; // no signal when all rewards equal

                var logits  = model.Forward(completions[s]);
                var dLogits = PolicyGradient(logits, completions[s], promptToks.Length,
                                             continuationLen, advantages[s]);
                model.Backward(dLogits);
            }
            optimizer.Step(model.GetParamGrads());

            lastReward    = mean;
            runningRew   += mean;
            logCount++;

            if (step % logEvery == 0)
            {
                float avgRew = runningRew / logCount;
                logger?.Info($"RL   step={step,-6}  avg_reward={avgRew:F4}  last_baseline={mean:F4}");
                runningRew = 0f;
                logCount   = 0;
            }
        }

        // ── 4. Evaluate greedy accuracy after RL ─────────────────────────────
        var eval = Evaluator.RunGreedy(model, tokenizer, evalProblems, maxNewTokens: 4, logger: null);
        logger?.Eval(eval.Correct, eval.Total, eval.Accuracy);
        return new RlTrainResult(steps, lastReward, eval.Accuracy);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>
    /// Temperature-sample <paramref name="continuationLen"/> tokens after the
    /// prompt, returning the full token sequence (prompt + continuation).
    ///
    /// The sampled sequence is in the same format as the ORM training sequences
    /// so <c>orm.Score(result)</c> can be called directly.
    ///
    /// PyTorch equivalent:
    ///   with torch.no_grad():
    ///       tokens = prompt_tokens.tolist()
    ///       for _ in range(continuation_len):
    ///           logits = model(torch.tensor(tokens))
    ///           probs  = F.softmax(logits[-1] / temperature, dim=-1)
    ///           tokens.append(torch.multinomial(probs, 1).item())
    /// </summary>
    private static int[] TemperatureSample(
        TransformerModel model,
        int[]            promptTokens,
        int              continuationLen,
        float            temperature,
        Random           rng)
    {
        int   vocabSize = model.Config.VocabSize;
        var   tokens    = new List<int>(promptTokens.Length + continuationLen);
        tokens.AddRange(promptTokens);

        for (int t = 0; t < continuationLen; t++)
        {
            var   logits  = model.Forward([.. tokens]);
            int   lastOff = (tokens.Count - 1) * vocabSize;
            float[] row   = new float[vocabSize];
            Array.Copy(logits.Data, lastOff, row, 0, vocabSize);
            tokens.Add(SampleFromLogits(row, temperature, rng));
        }

        return [.. tokens];
    }

    /// <summary>
    /// Compute the REINFORCE policy-gradient tensor.
    ///
    /// In next-token-prediction convention:
    ///   • completion token k is at index <c>promptLen + k</c> in <c>tokens</c>.
    ///   • it is predicted by logit row <c>promptLen + k − 1</c>.
    ///
    /// The gradient at that row is:
    ///   <c>advantage · (softmax(logit_row) − one_hot(target_token))</c>
    ///
    /// All other logit rows are left at zero — no gradient flows from prompt
    /// positions, keeping the update focused on the policy's completion choices.
    ///
    /// When accumulated over N samples (each pre-divided by N), the result
    /// is an unbiased estimate of the policy gradient.
    ///
    /// PyTorch equivalent:
    ///   log_prob = sum(Categorical(logits=logits[pos]).log_prob(tok)
    ///                  for pos, tok in zip(completion_pos, completion_tokens))
    ///   loss = -(advantage * log_prob)      # gradient ≡ advantage*(softmax−one_hot)
    ///   loss.backward()
    /// </summary>
    private static Tensor PolicyGradient(
        Tensor logits,
        int[]  tokens,
        int    promptLen,
        int    continuationLen,
        float  advantage)
    {
        int seqLen    = logits.Shape[0];
        int vocabSize = logits.Shape[1];
        var dLogits   = new Tensor([seqLen, vocabSize]);   // zero-initialised

        for (int k = 0; k < continuationLen; k++)
        {
            int logitRow  = promptLen + k - 1;       // logit position predicting completion token k
            int targetTok = tokens[promptLen + k];   // the sampled token we are reinforcing

            if (logitRow < 0 || logitRow >= seqLen) continue;

            int rowOff = logitRow * vocabSize;

            // Numerically stable softmax over this logit row
            float maxVal = float.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
                if (logits.Data[rowOff + v] > maxVal) maxVal = logits.Data[rowOff + v];

            float sumExp = 0f;
            for (int v = 0; v < vocabSize; v++)
                sumExp += MathF.Exp(logits.Data[rowOff + v] - maxVal);

            // dLogits[logitRow, v] = advantage · (p_v − 1{v == target})
            for (int v = 0; v < vocabSize; v++)
            {
                float p       = MathF.Exp(logits.Data[rowOff + v] - maxVal) / sumExp;
                float oneHot  = v == targetTok ? 1f : 0f;
                dLogits.Data[rowOff + v] = advantage * (p - oneHot);
            }
        }

        return dLogits;
    }

    /// <summary>
    /// Numerically stable temperature sampling over raw logits.
    /// Identical to the private helper in <see cref="RewardTrainer"/> — duplicated
    /// here to keep <c>RlTrainer</c> self-contained.
    /// </summary>
    private static int SampleFromLogits(float[] logits, float temperature, Random rng)
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
        for (int i = 0; i < probs.Length; i++) probs[i] /= sumExp;

        float u = (float)rng.NextDouble(), cum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (u < cum) return i;
        }
        return probs.Length - 1;
    }
}
