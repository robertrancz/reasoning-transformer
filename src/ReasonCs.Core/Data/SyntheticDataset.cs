namespace ReasonCs.Core.Data;

/// <summary>
/// Generates chained multi-step arithmetic reasoning problems.
///
/// Each problem requires ≥2 reasoning steps so that MCTS + PRM can
/// produce a measurable accuracy gain over greedy decoding.
///
/// Example (2-step):
///   "A=3 B=A+2 B=?"   →  answer "5"
///
/// Example (3-step):
///   "A=2 B=A+1 C=B*3 C=?"  →  answer "9"
///
/// The dataset also generates programmatic step-level correctness labels
/// (required for PRM training) by evaluating each sub-expression.
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
    public record Problem(string Prompt, string Answer, string[] Steps, bool[] StepCorrect);

    public static IEnumerable<Problem> Generate(int count, int minSteps = 2, int maxSteps = 3, int seed = 0)
    {
        // TODO: implement problem generator
        throw new NotImplementedException();
    }

    /// <summary>Split into train / test by ratio (default 90/10).</summary>
    public static (Problem[] Train, Problem[] Test) TrainTestSplit(
        IEnumerable<Problem> problems, float testRatio = 0.1f, int seed = 0)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
