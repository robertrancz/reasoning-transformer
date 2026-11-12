using ReasonCs.Core.Tensors;
using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Reasoning;

/// <summary>
/// Outcome Reward Model (ORM) — a linear scoring head on top of a frozen
/// TransformerModel backbone.
///
/// Architecture
/// ────────────
///   hidden  = backbone.ForwardToHidden(tokens)   // [seqLen, EmbedDim]  (frozen)
///   last    = hidden[seqLen-1, :]                // [EmbedDim]
///   logit   = dot(last, ScoreHead)               // scalar
///   score   = σ(logit)                           // ∈ (0, 1)
///
/// Training (only ScoreHead is updated — backbone stays frozen)
/// ──────────────────────────────────────────────────────────────
///   label   ∈ {0, 1}  (1 = correct answer, 0 = wrong answer)
///   loss    = BCE(score, label)
///           = −label·log(score) − (1−label)·log(1−score)
///
///   Combined sigmoid + BCE gradient (same derivation as softmax + cross-entropy):
///   dLogit  = score − label
///   dHead[d] += last[d] · dLogit   for all d in [0, EmbedDim)
///
/// Inference
/// ─────────
///   score = Score(encode(prompt + candidate_answer))
///   Higher score → more likely the answer is correct.
///   Use for ranking / filtering in MCTS.
///
/// PRM extension (future)
/// ─────────────────────
///   Same architecture, but Score() is called after each decoded reasoning step
///   rather than once on the full sequence.  Step-level labels are derived
///   programmatically from SyntheticDataset (all generated steps are correct).
///
/// PyTorch equivalent:
///   class RewardModel(nn.Module):
///       def __init__(self, backbone, embed_dim):
///           self.backbone   = backbone                 # frozen
///           self.score_head = nn.Linear(embed_dim, 1, bias=False)
///
///       def forward(self, tokens):
///           with torch.no_grad():
///               hidden = self.backbone.forward_to_hidden(tokens)  # [S, D]
///           last  = hidden[-1]                                     # [D]
///           return torch.sigmoid(self.score_head(last))            # scalar
///
///   loss_fn = nn.BCELoss()
///   # Only score_head parameters are passed to the optimizer:
///   optimizer = torch.optim.AdamW(reward_model.score_head.parameters(), lr=1e-3)
///
/// Reference:
///   ORM — Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021)
///   PRM — Lightman et al., "Let's Verify Step by Step" (2023)
/// </summary>
public sealed class RewardModel
{
    public enum Mode { ORM, PRM }

    public readonly Mode RewardMode;
    private readonly TransformerModel _backbone;

    // Score head parameters and gradient accumulator  [EmbedDim, 1]
    public Tensor ScoreHead;
    public Tensor DScoreHead;

    // Forward-pass cache (populated by Score(), consumed by Backward())
    private Tensor? _cacheLastHidden;   // last token's representation  [1, EmbedDim]
    private float   _cacheScore;        // sigmoid output of last Score() call

    public RewardModel(TransformerModel backbone, Mode mode = Mode.ORM)
    {
        _backbone  = backbone;
        RewardMode = mode;

        int D      = backbone.Config.EmbedDim;
        ScoreHead  = new Tensor([D, 1]);
        DScoreHead = new Tensor([D, 1]);

        InitWeights(seed: 99);
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    // ── Forward ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Score a token sequence.
    ///
    /// ORM: call once on the full sequence (prompt + padded continuation).
    /// PRM: call after each reasoning step.
    ///
    /// Returns a scalar in (0, 1) — higher → answer more likely correct.
    /// Caches the last-token hidden state and sigmoid output for <see cref="Backward"/>.
    ///
    /// For training / MCTS consistency, always pass a sequence whose length is
    /// <c>promptLen + ContinuationLen</c> tokens (see <see cref="RewardTrainer.PadContinuation"/>
    /// and <see cref="MctsSearch.StepTokenLen"/>).  When all sequences share the same
    /// total length the last-token representation is a stable, well-defined signal.
    /// </summary>
    public float Score(int[] tokens)
    {
        // 1. Run frozen backbone: hidden [seqLen, D]
        var hidden   = _backbone.ForwardToHidden(tokens);
        int seqLen   = hidden.Shape[0];
        int embedDim = hidden.Shape[1];

        // 2. Extract last token's representation → [1, D]
        var last   = new Tensor([1, embedDim]);
        int srcOff = (seqLen - 1) * embedDim;
        Array.Copy(hidden.Data, srcOff, last.Data, 0, embedDim);

        // 3. Linear: logit = last @ ScoreHead  →  [1, 1]
        var logitTensor = TensorOps.MatMul(last, ScoreHead);
        float logit     = logitTensor.Data[0];

        // 4. Sigmoid
        float score = Sigmoid(logit);

        // Cache for backward
        _cacheLastHidden = last;
        _cacheScore      = score;

        return score;
    }

    // ── Backward ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Backward pass through the score head (backbone is frozen — not backpropagated).
    ///
    /// Must be called after <see cref="Score"/>; call <see cref="ZeroGrad"/> before
    /// the first Backward() in each update step to avoid cross-step accumulation.
    ///
    /// Combined sigmoid + BCE gradient:
    ///   dLogit   = score − label
    ///   dHead[d] += last[d] · dLogit  for all d
    ///
    /// Returns the scalar BCE loss for logging.
    /// </summary>
    public float Backward(float label)
    {
        if (_cacheLastHidden is null)
            throw new InvalidOperationException("Backward() called before Score().");

        float loss   = BceLoss(_cacheScore, label);

        // Combined sigmoid + BCE gradient: dLogit = score − label
        float dLogit = _cacheScore - label;

        // dHead[d, 0] += last[0, d] * dLogit  for all d
        int D = _cacheLastHidden.Shape[1];
        for (int d = 0; d < D; d++)
            DScoreHead.Data[d] += _cacheLastHidden.Data[d] * dLogit;

        return loss;
    }

    // ── Optimizer interface ───────────────────────────────────────────────────

    /// <summary>
    /// Returns just (ScoreHead, DScoreHead) for the AdamW optimizer.
    /// The backbone parameters are NOT included — they remain frozen.
    /// </summary>
    public IEnumerable<(Tensor param, Tensor grad)> GetParamGrads()
    {
        yield return (ScoreHead, DScoreHead);
    }

    /// <summary>Zero the score head gradient accumulator before each update.</summary>
    public void ZeroGrad() => Array.Clear(DScoreHead.Data);

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void InitWeights(int seed)
    {
        // Small N(0, 0.02²) init — same scale as the backbone's projection matrices.
        TensorOps.RandomNormal(ScoreHead, std: 0.02f, seed: seed);
    }

    private static float Sigmoid(float x)
    {
        if (x >= 0f)
        {
            float e = MathF.Exp(-x);
            return 1f / (1f + e);
        }
        else
        {
            float e = MathF.Exp(x);
            return e / (1f + e);
        }
    }

    private static float BceLoss(float score, float label) =>
        -(label * MathF.Log(score + 1e-10f) + (1f - label) * MathF.Log(1f - score + 1e-10f));
}
