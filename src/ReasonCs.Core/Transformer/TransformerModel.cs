using ReasonCs.Core.Autograd.Ops;
using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Transformer;

/// <summary>
/// The full tiny Transformer language model (decoder-only, GPT-style).
///
/// PyTorch equivalent:
///   # The closest reference implementation is Karpathy's nanoGPT:
///   # https://github.com/karpathy/nanoGPT
///   #
///   # Rough PyTorch skeleton:
///   class TransformerModel(nn.Module):
///       def __init__(self, config):
///           self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
///           self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
///           self.final_norm = nn.LayerNorm(config.embed_dim)
///           self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
///
///       def forward(self, tokens):                  # tokens: [seq_len]
///           x = self.token_embedding(tokens)        # [seq_len, embed_dim]
///           for block in self.blocks:
///               x = block(x)
///           x = self.final_norm(x)
///           return self.lm_head(x)                  # [seq_len, vocab_size]
///
///   # Key difference: PyTorch's nn.Module tracks parameters automatically.
///   # Here we manage all weight tensors manually and pass them explicitly
///   # to the optimizer.
///
/// Architecture:
///   TokenEmbedding [VocabSize, EmbedDim]
///   N × TransformerBlock (pre-norm attention + SwiGLU MLP)
///   FinalLayerNorm
///   LM Head (unembedding): [EmbedDim, VocabSize]
///
/// Weight initialisation (called from constructor):
///   Embedding and projection weights ~ N(0, 0.02²)  — standard GPT init
///   LayerNorm gamma = 1, beta = 0
/// </summary>
public sealed class TransformerModel
{
    public readonly TransformerConfig Config;

    public Tensor TokenEmbedding;       // [VocabSize, EmbedDim]
    public TransformerBlock[] Blocks;
    public Tensor FinalGamma, FinalBeta;
    public Tensor LMHead;               // [EmbedDim, VocabSize]

    // Gradient accumulators for model-level parameters
    public Tensor DTokenEmbedding;      // [VocabSize, EmbedDim]
    public Tensor DFinalGamma, DFinalBeta;
    public Tensor DLMHead;              // [EmbedDim, VocabSize]

    // Forward-pass cache (populated by Forward, consumed by Backward)
    private int[]?  _cacheTokens;          // input token indices
    private Tensor? _cacheXAfterBlocks;    // x after all blocks, before FinalLN [S, D]
    private Tensor? _cacheXNorm;           // output of FinalLN                  [S, D]
    private Tensor? _cacheFinalMean;       // FinalLN mean                       [S]
    private Tensor? _cacheFinalRstd;       // FinalLN reciprocal std             [S]

    public TransformerModel(TransformerConfig config)
    {
        Config = config;

        TokenEmbedding = new Tensor([config.VocabSize, config.EmbedDim]);
        Blocks         = new TransformerBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            Blocks[i] = new TransformerBlock(config.EmbedDim, config.NumHeads, config.MlpHiddenDim);
        FinalGamma = new Tensor([config.EmbedDim]);
        FinalBeta  = new Tensor([config.EmbedDim]);
        LMHead     = new Tensor([config.EmbedDim, config.VocabSize]);

        DTokenEmbedding = new Tensor([config.VocabSize, config.EmbedDim]);
        DFinalGamma     = new Tensor([config.EmbedDim]);
        DFinalBeta      = new Tensor([config.EmbedDim]);
        DLMHead         = new Tensor([config.EmbedDim, config.VocabSize]);

        InitWeights(seed: 42);
    }

    /// <summary>
    /// Forward pass.
    ///
    /// tokens  : int[seqLen]  — token indices in [0, VocabSize)
    /// returns : [seqLen, VocabSize]  — raw logits (pre-softmax)
    /// </summary>
    public Tensor Forward(int[] tokens)
    {
        int seqLen = tokens.Length;

        // 1. Token embedding lookup: [seqLen, EmbedDim]
        var x = EmbeddingOp.Forward(TokenEmbedding, tokens);

        // 2. Transformer blocks (pre-norm residual)
        foreach (var block in Blocks)
            x = block.Forward(x, seqLen);

        // 3. Final LayerNorm
        var (xNorm, finalMean, finalRstd) = LayerNormOp.Forward(x, FinalGamma, FinalBeta);

        // Cache for backward
        _cacheTokens       = tokens;
        _cacheXAfterBlocks = x;
        _cacheXNorm        = xNorm;
        _cacheFinalMean    = finalMean;
        _cacheFinalRstd    = finalRstd;

        // 4. LM Head (unembedding): [seqLen, VocabSize]
        return TensorOps.MatMul(xNorm, LMHead);
    }

    /// <summary>
    /// Backward pass through the full model.
    ///
    /// dLogits : upstream gradient from CrossEntropyOp.Backward  [seqLen, VocabSize]
    ///
    /// Accumulates gradients into all D* tensors; call ZeroGrad() first.
    ///
    /// Reverse chain:
    ///   dLogits
    ///     → LM Head    : dXNorm, DLMHead
    ///     → FinalLN    : dx, DFinalGamma, DFinalBeta
    ///     → Blocks[N-1..0] : dx
    ///     → Embedding  : DTokenEmbedding
    /// </summary>
    public void Backward(Tensor dLogits)
    {
        if (_cacheXNorm is null || _cacheXAfterBlocks is null ||
            _cacheFinalMean is null || _cacheFinalRstd is null || _cacheTokens is null)
            throw new InvalidOperationException("Backward called before Forward.");

        // ── LM Head: logits = xNorm @ LMHead ─────────────────────────────────
        // dXNorm  = dLogits @ LMHead.T   [S, D]
        // DLMHead += xNorm.T @ dLogits   [D, V]
        var dXNorm = TensorOps.MatMul(dLogits, TensorOps.Transpose(LMHead));
        TensorOps.AddTo(DLMHead, TensorOps.MatMul(TensorOps.Transpose(_cacheXNorm), dLogits));

        // ── Final LayerNorm ───────────────────────────────────────────────────
        var (dx, dFinalGamma, dFinalBeta) = LayerNormOp.Backward(
            dXNorm, _cacheXAfterBlocks, FinalGamma, _cacheFinalMean, _cacheFinalRstd);
        TensorOps.AddTo(DFinalGamma, dFinalGamma);
        TensorOps.AddTo(DFinalBeta,  dFinalBeta);

        // ── Transformer blocks in reverse order ───────────────────────────────
        for (int i = Blocks.Length - 1; i >= 0; i--)
            dx = Blocks[i].Backward(dx);

        // ── Token embedding: scatter-add gradient back to embedding table ─────
        var dEmb = EmbeddingOp.Backward(dx, _cacheTokens, Config.VocabSize);
        TensorOps.AddTo(DTokenEmbedding, dEmb);
    }

    /// <summary>
    /// Returns all (parameter, gradient) pairs in the model.
    /// Consumed by the optimizer and by ZeroGrad.
    /// </summary>
    public IEnumerable<(Tensor param, Tensor grad)> GetParamGrads()
    {
        yield return (TokenEmbedding, DTokenEmbedding);
        yield return (FinalGamma,     DFinalGamma);
        yield return (FinalBeta,      DFinalBeta);
        yield return (LMHead,         DLMHead);

        foreach (var block in Blocks)
        {
            // Attention weights
            yield return (block.Attn.Wq, block.Attn.DWq);
            yield return (block.Attn.Wk, block.Attn.DWk);
            yield return (block.Attn.Wv, block.Attn.DWv);
            yield return (block.Attn.Wo, block.Attn.DWo);
            // MLP weights
            yield return (block.Mlp.W1,  block.Mlp.DW1);
            yield return (block.Mlp.W2,  block.Mlp.DW2);
            yield return (block.Mlp.W3,  block.Mlp.DW3);
            // LayerNorm parameters
            yield return (block.Gamma1, block.DGamma1);
            yield return (block.Beta1,  block.DBeta1);
            yield return (block.Gamma2, block.DGamma2);
            yield return (block.Beta2,  block.DBeta2);
        }
    }

    /// <summary>
    /// Compute the global L2 norm of all gradient tensors.
    /// Useful for diagnosing exploding / vanishing gradients.
    /// PyTorch equivalent: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    /// (we compute but do not clip here)
    /// </summary>
    public float GradNorm()
    {
        float sumSq = 0f;
        foreach (var (_, grad) in GetParamGrads())
            foreach (var v in grad.Data)
                sumSq += v * v;
        return MathF.Sqrt(sumSq);
    }

    /// <summary>
    /// Zero all gradient accumulators before starting a new backward pass.
    /// PyTorch equivalent: optimizer.zero_grad()
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var (_, grad) in GetParamGrads())
            Array.Clear(grad.Data);
    }

    /// <summary>
    /// Backbone-only forward pass — returns the final LayerNorm hidden states
    /// [seqLen, EmbedDim] <em>without</em> applying the LM head.
    ///
    /// Used by <see cref="ReasonCs.Core.Reasoning.RewardModel"/> to extract
    /// per-token contextual representations from the frozen backbone.
    ///
    /// Note: updates the TransformerBlock internal caches (those from the most
    /// recent Forward() call are overwritten).  This is intentional — the
    /// backbone is frozen during reward-model training so its own Backward()
    /// will not be called again.
    ///
    /// PyTorch equivalent:
    ///   with torch.no_grad():
    ///       hidden = backbone(tokens)   # [S, D] — before lm_head
    /// </summary>
    public Tensor ForwardToHidden(int[] tokens)
    {
        int seqLen = tokens.Length;
        var x = EmbeddingOp.Forward(TokenEmbedding, tokens);
        foreach (var block in Blocks)
            x = block.Forward(x, seqLen);
        var (xNorm, _, _) = LayerNormOp.Forward(x, FinalGamma, FinalBeta);
        return xNorm;   // [seqLen, EmbedDim]
    }

    /// <summary>
    /// Greedy decoding — appends <paramref name="maxNewTokens"/> tokens to the prompt,
    /// choosing the argmax logit at each step.
    /// </summary>
    public int[] GreedyDecode(int[] promptTokens, int maxNewTokens)
    {
        var tokens = new List<int>(promptTokens);
        for (int i = 0; i < maxNewTokens; i++)
        {
            var logits   = Forward([.. tokens]);  // [seqLen, V]
            int seqLen   = tokens.Count;
            // Pick argmax at the LAST position (next-token prediction)
            int vocabSize = Config.VocabSize;
            int bestToken = 0;
            float bestLogit = float.NegativeInfinity;
            int lastRowOffset = (seqLen - 1) * vocabSize;
            for (int v = 0; v < vocabSize; v++)
            {
                float val = logits.Data[lastRowOffset + v];
                if (val > bestLogit) { bestLogit = val; bestToken = v; }
            }
            tokens.Add(bestToken);
        }
        return [.. tokens];
    }

    // ── Weight initialisation ─────────────────────────────────────────────────

    /// <summary>
    /// Initialise all learnable weights.
    ///
    /// Projection matrices ~ N(0, σ²) with σ = 0.02 (standard GPT-2 init).
    /// LayerNorm gamma = 1, beta = 0 (identity affine at start of training).
    ///
    /// Each weight tensor gets an independent seed derived from <paramref name="seed"/>
    /// so the initialisation is deterministic but different per parameter.
    /// </summary>
    public void InitWeights(int seed = 42)
    {
        const float std = 0.02f;
        int s = seed;

        TensorOps.RandomNormal(TokenEmbedding, std: std, seed: s++);

        foreach (var block in Blocks)
        {
            TensorOps.RandomNormal(block.Attn.Wq, std: std, seed: s++);
            TensorOps.RandomNormal(block.Attn.Wk, std: std, seed: s++);
            TensorOps.RandomNormal(block.Attn.Wv, std: std, seed: s++);
            TensorOps.RandomNormal(block.Attn.Wo, std: std, seed: s++);
            TensorOps.RandomNormal(block.Mlp.W1,  std: std, seed: s++);
            TensorOps.RandomNormal(block.Mlp.W2,  std: std, seed: s++);
            TensorOps.RandomNormal(block.Mlp.W3,  std: std, seed: s++);
            // LayerNorm gamma is already 1 from TransformerBlock ctor;
            // beta is already 0 from Tensor ctor.
        }

        TensorOps.Fill(FinalGamma, 1f);
        // FinalBeta stays 0.
        TensorOps.RandomNormal(LMHead, std: std, seed: s);
    }
}

public sealed record TransformerConfig(
    int VocabSize    = 256,
    int EmbedDim     = 128,
    int NumHeads     = 4,
    int NumLayers    = 4,
    int MlpHiddenDim = 512,
    int MaxSeqLen    = 128
);
