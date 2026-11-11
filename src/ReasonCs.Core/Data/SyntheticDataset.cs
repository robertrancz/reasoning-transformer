using System.Text;

namespace ReasonCs.Core.Data;

/// <summary>
/// Generates chained multi-step arithmetic reasoning problems.
///
/// Each problem requires ≥2 reasoning steps so that MCTS + PRM can
/// produce a measurable accuracy gain over greedy decoding.
///
/// Problem format
/// ──────────────
///   2-step:  "A=3 B=A+2 B=?"   →  answer "5"
///   3-step:  "A=2 B=A+1 C=B*3 C=?"  →  answer "9"
///
/// The prompt ends with "?"; the training sequence is prompt + answer,
/// so the model learns to complete "A=3 B=A+2 B=?" → "5".
///
/// Variables use the letters A, B, C, D (one per step).
/// Operators:  +  (add),  -  (subtract, result ≥ 0),  *  (multiply).
/// Values are capped at 99 so answers fit in ≤2 characters.
/// Problems where any intermediate value falls outside [0, 99] are
/// silently retried (rejection sampling).
///
/// Step-level correctness labels
/// ─────────────────────────────
/// Every problem generated here is correct by construction, so all
/// StepCorrect values are true.  These labels become non-trivial when
/// the model makes mistakes during MCTS rollouts — the PRM will score
/// each decoded step against the expected value.
///
/// PyTorch equivalent:
///   # PyTorch itself has no dataset generation utilities.
///   # The pattern maps to:
///   from torch.utils.data import Dataset, DataLoader
///
///   class ReasoningDataset(Dataset):
///       def __init__(self, problems):
///           self.problems = problems
///       def __len__(self):
///           return len(self.problems)
///       def __getitem__(self, idx):
///           p = self.problems[idx]
///           return tokenizer.encode(p.prompt), tokenizer.encode(p.answer)
///
///   loader = DataLoader(dataset, batch_size=32, shuffle=True)
///
/// Key difference: we generate step-level correctness labels alongside the
/// problems. Real PRM datasets (e.g. PRM800K from Lightman et al.) are
/// human-annotated; ours are derived programmatically since every arithmetic
/// sub-step is machine-verifiable.
/// </summary>
public sealed class SyntheticDataset
{
    /// <summary>
    /// One reasoning problem.
    ///
    /// Prompt      — the question string, ending with "?", e.g. "A=2 B=A+1 C=B*3 C=?"
    /// Answer      — the expected answer string,           e.g. "9"
    /// Steps       — evaluated intermediate assignments,   e.g. ["A=2", "B=3", "C=9"]
    /// StepCorrect — correctness of each step (all true for generated data)
    /// FullText    — Prompt + Answer (the sequence the model is trained on)
    /// </summary>
    public record Problem(string Prompt, string Answer, string[] Steps, bool[] StepCorrect)
    {
        public string FullText => Prompt + Answer;
    }

    // Letters used as variable names, one per step
    private static readonly char[] VarNames = ['A', 'B', 'C', 'D'];
    private static readonly char[] Operators = ['+', '-', '*'];

    /// <summary>
    /// Generate <paramref name="count"/> problems with between
    /// <paramref name="minSteps"/> and <paramref name="maxSteps"/> reasoning steps.
    ///
    /// Problems where any intermediate result falls outside [0, 99] are
    /// discarded and retried (rejection sampling), so exactly <paramref name="count"/>
    /// valid problems are returned.
    /// </summary>
    public static IEnumerable<Problem> Generate(
        int count, int minSteps = 2, int maxSteps = 3, int seed = 0)
    {
        if (minSteps < 2) throw new ArgumentOutOfRangeException(nameof(minSteps), "Need at least 2 steps.");
        if (maxSteps > VarNames.Length) throw new ArgumentOutOfRangeException(nameof(maxSteps));

        var rng = new Random(seed);
        int yielded = 0;

        while (yielded < count)
        {
            int numSteps = rng.Next(minSteps, maxSteps + 1);
            var p = TryBuildProblem(rng, numSteps);
            if (p is null) continue;
            yield return p;
            yielded++;
        }
    }

    /// <summary>
    /// Split problems into train / test sets.
    ///
    /// The collection is shuffled deterministically with <paramref name="seed"/>
    /// before splitting, so results are reproducible across runs.
    /// </summary>
    public static (Problem[] Train, Problem[] Test) TrainTestSplit(
        IEnumerable<Problem> problems, float testRatio = 0.1f, int seed = 0)
    {
        var all = problems.ToArray();

        // Fisher-Yates shuffle
        var rng = new Random(seed);
        for (int i = all.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (all[i], all[j]) = (all[j], all[i]);
        }

        int testCount = Math.Max(1, (int)(all.Length * testRatio));
        return (Train: all[testCount..], Test: all[..testCount]);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// <summary>
    /// Attempt to build one problem with exactly <paramref name="numSteps"/> steps.
    /// Returns null if any intermediate value falls outside [0, 99].
    /// </summary>
    private static Problem? TryBuildProblem(Random rng, int numSteps)
    {
        // Step 0: pick initial value in [1, 9]
        int[] values = new int[numSteps];
        values[0] = rng.Next(1, 10);

        char[] ops     = new char[numSteps - 1];
        int[]  operands = new int[numSteps - 1];

        for (int s = 1; s < numSteps; s++)
        {
            char op  = Operators[rng.Next(Operators.Length)];
            // For *, cap operand at 5 so even 9*5=45 stays within range
            int  rhs = op == '*' ? rng.Next(1, 6) : rng.Next(1, 10);
            int  result = op switch {
                '+' => values[s - 1] + rhs,
                '-' => values[s - 1] - rhs,
                '*' => values[s - 1] * rhs,
                _   => throw new InvalidOperationException()
            };

            if (result < 0 || result > 99) return null;   // reject; caller retries

            ops[s - 1]      = op;
            operands[s - 1] = rhs;
            values[s]       = result;
        }

        return BuildProblem(numSteps, values, ops, operands);
    }

    /// <summary>Assemble a validated Problem record from its components.</summary>
    private static Problem BuildProblem(
        int numSteps, int[] values, char[] ops, int[] operands)
    {
        // ── Prompt: "A=2 B=A+1 C=B*3 C=?" ───────────────────────────────────
        var sb = new StringBuilder();
        sb.Append($"{VarNames[0]}={values[0]}");
        for (int s = 1; s < numSteps; s++)
            sb.Append($" {VarNames[s]}={VarNames[s - 1]}{ops[s - 1]}{operands[s - 1]}");
        sb.Append($" {VarNames[numSteps - 1]}=?");
        string prompt = sb.ToString();

        // ── Answer: e.g. "9" ─────────────────────────────────────────────────
        string answer = values[numSteps - 1].ToString();

        // ── Steps: evaluated form, e.g. ["A=2", "B=3", "C=9"] ───────────────
        var steps = new string[numSteps];
        steps[0] = $"{VarNames[0]}={values[0]}";
        for (int s = 1; s < numSteps; s++)
            steps[s] = $"{VarNames[s]}={values[s]}";

        // All generated steps are correct by construction
        var stepCorrect = Enumerable.Repeat(true, numSteps).ToArray();

        return new Problem(prompt, answer, steps, stepCorrect);
    }
}
