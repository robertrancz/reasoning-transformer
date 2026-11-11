using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Training;

/// <summary>
/// AdamW optimizer — Adam with decoupled weight decay.
///
/// PyTorch equivalent:
///   torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999),
///                     eps=1e-8, weight_decay=0.1)
///
///   # The training loop pattern in PyTorch:
///   optimizer.zero_grad()
///   loss.backward()
///   optimizer.step()
///
///   # Our equivalent (manual):
///   model.ZeroGrad()
///   model.Backward(dLogits)        // fills grad tensors manually
///   optimizer.Step(model.GetParamGrads())
///
/// Key difference from Adam: weight decay is applied directly to the weights
/// (θ -= lr·λ·θ) rather than being folded into the gradient. This is
/// mathematically cleaner and empirically better for Transformers.
///
/// Reference: Loshchilov &amp; Hutter, "Decoupled Weight Decay Regularization" (2017)
///
/// Update rule:
///   m_t = β1·m_{t-1} + (1-β1)·g_t
///   v_t = β2·v_{t-1} + (1-β2)·g_t²
///   m̂ = m_t / (1-β1^t),  v̂ = v_t / (1-β2^t)
///   θ_t = θ_{t-1} - lr·(m̂ / (√v̂ + ε) + λ·θ_{t-1})
/// </summary>
public sealed class AdamW
{
    public float LearningRate;
    public readonly float Beta1;
    public readonly float Beta2;
    public readonly float Epsilon;
    public readonly float WeightDecay;

    private int _step;
    private readonly Dictionary<Tensor, (float[] m, float[] v)> _state = new();

    public AdamW(float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
                 float eps = 1e-8f, float weightDecay = 0.1f)
    {
        LearningRate = lr;
        Beta1        = beta1;
        Beta2        = beta2;
        Epsilon      = eps;
        WeightDecay  = weightDecay;
    }

    /// <summary>
    /// Run one AdamW update step.
    ///
    /// paramGrads : all (parameter, gradient) pairs from model.GetParamGrads()
    ///
    /// Moment vectors are lazily allocated per parameter on first call and
    /// reused across steps (keyed by the Tensor object identity).
    /// </summary>
    public void Step(IEnumerable<(Tensor param, Tensor grad)> paramGrads)
    {
        _step++;
        float bc1 = 1f - MathF.Pow(Beta1, _step);  // bias-correction denominator 1
        float bc2 = 1f - MathF.Pow(Beta2, _step);  // bias-correction denominator 2

        foreach (var (param, grad) in paramGrads)
        {
            // Lazily allocate first-moment (m) and second-moment (v) vectors
            if (!_state.TryGetValue(param, out var state))
            {
                state = (new float[param.Size], new float[param.Size]);
                _state[param] = state;
            }
            var (m, v) = state;

            for (int i = 0; i < param.Size; i++)
            {
                float g = grad.Data[i];

                // Moment updates
                m[i] = Beta1 * m[i] + (1f - Beta1) * g;
                v[i] = Beta2 * v[i] + (1f - Beta2) * g * g;

                // Bias-corrected estimates
                float mHat = m[i] / bc1;
                float vHat = v[i] / bc2;

                // θ -= lr · (m̂ / (√v̂ + ε) + λ · θ)
                param.Data[i] -= LearningRate * (mHat / (MathF.Sqrt(vHat) + Epsilon)
                                               + WeightDecay * param.Data[i]);
            }
        }
    }

    /// <summary>
    /// Zero out a collection of gradient tensors.
    /// Call before Backward() to avoid accumulating across batches.
    /// </summary>
    public void ZeroGrad(IEnumerable<Tensor> grads)
    {
        foreach (var g in grads)
            Array.Clear(g.Data);
    }
}
