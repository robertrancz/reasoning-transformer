using ReasonCs.Core.Tensors;

namespace ReasonCs.Core.Autograd.Ops;

/// <summary>
/// Token embedding lookup — forward and backward.
///
/// PyTorch equivalent:
///   # Forward:
///   nn.Embedding(vocab_size, embed_dim)
///   embedding(tokens)              →  EmbeddingOp.Forward(weight, tokens)
///
///   # As a raw operation:
///   weight[tokens]                 →  EmbeddingOp.Forward(weight, tokens)
///   # (PyTorch uses F.embedding internally)
///
///   # Backward — PyTorch handles automatically.
///   # The gradient is a sparse scatter-add into the weight matrix:
///   # for each token t at position i: dWeight[t] += dOut[i]
///   # PyTorch accumulates these as a sparse gradient on the embedding weight.
///
/// Forward:  out[i] = weight[token[i]]          (index into embedding table)
/// Backward: dWeight[token[i]] += dOut[i]        (scatter-add)
/// </summary>
public static class EmbeddingOp
{
    public static Tensor Forward(Tensor weight, int[] tokens)
    {
        // TODO: implement
        throw new NotImplementedException();
    }

    public static Tensor Backward(Tensor dOut, int[] tokens, int vocabSize)
    {
        // TODO: implement — scatter-add gradients into dWeight
        throw new NotImplementedException();
    }
}
