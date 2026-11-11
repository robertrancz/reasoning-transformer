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
///   N × TransformerBlock
///   FinalLayerNorm
///   LM Head (unembedding): [EmbedDim, VocabSize]
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
    }

    /// <summary>Forward pass. Returns logits [SeqLen, VocabSize].</summary>
    public Tensor Forward(int[] tokens)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    public void Backward(Tensor dLogits)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    /// <summary>Greedy decoding — returns the argmax token at each position.</summary>
    public int[] GreedyDecode(int[] promptTokens, int maxNewTokens)
    {
        // TODO: implement
        throw new NotImplementedException();
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
