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
///   optimizer.ZeroGrad(grads)
///   model.Backward(dLogits)        // fills grad tensors manually
///   optimizer.Step(paramGrads)
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

    public void Step(IEnumerable<(Tensor param, Tensor grad)> paramGrads)
    {
        _step++;
        // TODO: implement AdamW update
        throw new NotImplementedException();
    }

    public void ZeroGrad(IEnumerable<Tensor> grads)
    {
        foreach (var g in grads)
            Array.Clear(g.Data);
    }
}
