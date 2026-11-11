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
        var (xNorm, _, _) = LayerNormOp.Forward(x, FinalGamma, FinalBeta);

        // 4. LM Head (unembedding): [seqLen, VocabSize]
        return TensorOps.MatMul(xNorm, LMHead);
    }

    public void Backward(Tensor dLogits)
    {
        // TODO: implement full backward pass
        throw new NotImplementedException();
    }

    /// <summary>
    /// Greedy decoding — appends <paramref name="maxNewTokens"/> tokens to the prompt,
    /// choosing the argmax logit at each step.
    /// </summary>
    public int[] GreedyDecode(int[] promptTokens, int maxNewTokens)
    {
        // TODO: implement autoregressive decode loop
        throw new NotImplementedException();
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
