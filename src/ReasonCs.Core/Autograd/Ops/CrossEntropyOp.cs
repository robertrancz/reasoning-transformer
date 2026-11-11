using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Autograd.Ops;

/// <summary>
/// Cross-entropy loss over logits, fused with Softmax for numerical stability.
///
/// PyTorch equivalent:
///   # Forward (fused, stable):
///   torch.nn.functional.cross_entropy(logits, targets)
///   # which internally does: log_softmax(logits) then nll_loss.
///
///   # Backward — PyTorch handles automatically.
///   # The fused softmax + cross-entropy gradient is elegantly simple:
///   dLogits[i] = p[i] - 1{i == target}
///   # i.e. subtract 1 from the probability of the correct class.
///
/// Forward:  loss = -log(p[target])  where p = softmax(logits)
/// Backward: dLogits[i] = p[i] - 1{i == target}
///
/// Fusing softmax and cross-entropy avoids a numerically unstable log(exp(...))
/// round-trip — we compute log-softmax directly instead.
///
/// Shapes:
///   logits  : [B, C]  — B examples, C classes
///   targets : int[B]  — one target class index per example  (0 ≤ t < C)
///   probs   : [B, C]  — softmax probabilities (cached for backward)
///   loss    : scalar  — mean cross-entropy over the batch
///   dLogits : [B, C]  — gradient w.r.t. logits
///
/// Derivation of the backward gradient:
///   L(z) = -log( exp(z_t) / Σ_j exp(z_j) )
///         = -z_t + log( Σ_j exp(z_j) )                  (log-sum-exp form)
///
///   dL/dz_i = -1{i == t} + exp(z_i) / Σ_j exp(z_j)
///           = p_i - 1{i == t}
///
///   When we average over the batch (loss = mean), the gradient is divided by B.
/// </summary>
public static class CrossEntropyOp
{
    /// <summary>
    /// Forward pass. Computes numerically stable log-softmax internally,
    /// then NLL loss averaged over the batch.
    ///
    /// Returns the scalar mean loss and the full softmax probability matrix
    /// [B, C] needed by the backward pass.
    /// </summary>
    public static (float loss, Tensor probs) Forward(Tensor logits, int[] targets)
    {
        // logits shape: [B, C]
        if (logits.Rank != 2)
            throw new ArgumentException("logits must be rank-2 [B, C].", nameof(logits));

        int B = logits.Shape[0];
        int C = logits.Shape[1];

        if (targets.Length != B)
            throw new ArgumentException(
                $"targets.Length ({targets.Length}) must equal batch size ({B}).", nameof(targets));

        var probs = new Tensor(logits.Shape);
        float totalLoss = 0f;

        for (int b = 0; b < B; b++)
        {
            int offset = b * C;

            // 1. Numerically stable softmax: subtract row max before exp
            float maxVal = float.NegativeInfinity;
            for (int i = 0; i < C; i++)
                if (logits.Data[offset + i] > maxVal)
                    maxVal = logits.Data[offset + i];

            float sumExp = 0f;
            for (int i = 0; i < C; i++)
            {
                float e = MathF.Exp(logits.Data[offset + i] - maxVal);
                probs.Data[offset + i] = e;
                sumExp += e;
            }

            for (int i = 0; i < C; i++)
                probs.Data[offset + i] /= sumExp;

            // 2. NLL loss: -log(p[target])
            int t = targets[b];
            if (t < 0 || t >= C)
                throw new ArgumentOutOfRangeException(nameof(targets),
                    $"targets[{b}] = {t} is out of range [0, {C}).");

            totalLoss += -MathF.Log(probs.Data[offset + t] + 1e-10f); // +eps avoids log(0)
        }

        float meanLoss = totalLoss / B;
        return (meanLoss, probs);
    }

    /// <summary>
    /// Backward pass. Returns dLogits with shape [B, C].
    ///
    ///   dLogits[b, i] = (p[b, i] - 1{i == target[b]}) / B
    ///
    /// The /B comes from the mean reduction in Forward.
    /// </summary>
    public static Tensor Backward(Tensor probs, int[] targets)
    {
        int B = probs.Shape[0];
        int C = probs.Shape[1];

        var dLogits = new Tensor(probs.Shape);

        for (int b = 0; b < B; b++)
        {
            int offset = b * C;
            int t = targets[b];

            for (int i = 0; i < C; i++)
            {
                float indicator = (i == t) ? 1f : 0f;
                dLogits.Data[offset + i] = (probs.Data[offset + i] - indicator) / B;
            }
        }

        return dLogits;
    }
}
