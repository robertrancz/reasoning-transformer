using ReasonCs.Core.Tensors;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Reasoning;

/// <summary>
/// Reward model hierarchy — build ORM first, then extend to PRM.
///
/// ORM (Outcome Reward Model):
///   A small linear head on top of the frozen base model that scores
///   whether the *final answer* is correct. No step-level labels needed.
///
/// PRM (Process Reward Model):
///   Same architecture but scores each *intermediate step*.
///   Step-level correctness labels are derived programmatically from
///   the synthetic dataset (each sub-expression is verifiable).
///
/// PyTorch equivalent:
///   # ORM / PRM share the same architecture — a linear "value head"
///   # on top of the last hidden state of the base model:
///   class RewardModel(nn.Module):
///       def __init__(self, base_model, embed_dim):
///           self.backbone  = base_model      # frozen TransformerModel
///           self.score_head = nn.Linear(embed_dim, 1)   # scalar score
///
///       def forward(self, tokens):
///           with torch.no_grad():
///               hidden = self.backbone(tokens)   # [seq_len, embed_dim]
///           last  = hidden[-1]                   # take last token's repr
///           score = torch.sigmoid(self.score_head(last))  # → [0, 1]
///           return score.item()
///
///   # ORM: call once on the full sequence (final answer).
///   # PRM: call after each decoded reasoning step.
///
/// Reference: Lightman et al., "Let's Verify Step by Step" (2023) for PRM.
///            Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021) for ORM.
/// </summary>
public sealed class RewardModel
{
    public enum Mode { ORM, PRM }

    public readonly Mode RewardMode;
    private readonly TransformerModel _backbone;

    // Linear head  [EmbedDim, 1]
    public Tensor ScoreHead;

    public RewardModel(TransformerModel backbone, Mode mode = Mode.ORM)
    {
        _backbone  = backbone;
        RewardMode = mode;
        ScoreHead  = new Tensor([backbone.Config.EmbedDim, 1]);
    }

    /// <summary>
    /// Score a (partial) sequence.
    /// ORM: call once on the full sequence.
    /// PRM: call after each reasoning step.
    /// Returns a scalar score in [0, 1].
    /// </summary>
    public float Score(int[] tokens)
    {
        // TODO: implement
        throw new NotImplementedException();
    }
}
