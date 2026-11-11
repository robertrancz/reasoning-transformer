namespace ReasonCs.Core.Reasoning;

/// <summary>
/// A single node in the MCTS search tree.
///
/// Each node represents a partial reasoning sequence (a prefix of decoded steps).
/// Children correspond to top-k sampled step-candidates from this state.
///
/// PyTorch equivalent:
///   # There is no PyTorch equivalent — MCTS is a pure search algorithm,
///   # not a neural network operation.
///   #
///   # In Python, this would typically be a plain dataclass or namedtuple:
///   @dataclass
///   class MctsNode:
///       tokens:      list[int]
///       parent:      MctsNode | None = None
///       children:    list[MctsNode]  = field(default_factory=list)
///       visits:      int   = 0
///       total_value: float = 0.0
///       prior_score: float = 0.0
///
///       @property
///       def q(self): return self.total_value / self.visits if self.visits else 0.0
///
///       def ucb(self, c=1.41):
///           parent_visits = self.parent.visits if self.parent else 1
///           return self.q + c * math.sqrt(math.log(parent_visits) / (self.visits + 1))
/// </summary>
public sealed class MctsNode
{
    public MctsNode? Parent { get; }
    public int[] Tokens { get; }          // full token sequence to this node
    public List<MctsNode> Children { get; } = new();

    public int  Visits       { get; set; }
    public float TotalValue  { get; set; }
    public float PriorScore  { get; set; } // PRM/ORM score assigned at expansion

    public float Q => Visits == 0 ? 0f : TotalValue / Visits;

    /// <summary>UCB1 score used during Selection.</summary>
    public float UCB(float explorationC = 1.41f) =>
        Q + explorationC * MathF.Sqrt(MathF.Log(Parent?.Visits ?? 1) / (Visits + 1));

    public MctsNode(int[] tokens, MctsNode? parent = null, float priorScore = 0f)
    {
        Tokens     = tokens;
        Parent     = parent;
        PriorScore = priorScore;
    }

    public bool IsLeaf => Children.Count == 0;
}
