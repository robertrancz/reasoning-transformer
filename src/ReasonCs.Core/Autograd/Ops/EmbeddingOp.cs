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
/// Forward:  out[i]       = weight[token[i], :]       (gather rows)
/// Backward: dWeight[t]  += dOut[i]  for each i where token[i] == t  (scatter-add)
///
/// Shapes:
///   weight  : [vocabSize, embedDim]
///   tokens  : int[seqLen]           — each value in [0, vocabSize)
///   out     : [seqLen, embedDim]
///   dOut    : [seqLen, embedDim]    — upstream gradient
///   dWeight : [vocabSize, embedDim] — dense accumulation of the sparse gradient
///
/// Note on the backward:
///   When the same token appears multiple times in the sequence (common with
///   padding tokens or repeated words), its gradient row is accumulated over
///   all positions. This is identical to PyTorch's dense fallback mode.
///   PyTorch also supports a sparse tensor gradient (for large vocabularies)
///   but we use the dense form for simplicity.
/// </summary>
public static class EmbeddingOp
{
    /// <summary>
    /// Gather embedding rows for the given token sequence.
    ///
    /// weight : [vocabSize, embedDim]
    /// tokens : int[seqLen]
    /// returns : [seqLen, embedDim]
    /// </summary>
    public static Tensor Forward(Tensor weight, int[] tokens)
    {
        if (weight.Rank != 2)
            throw new ArgumentException("weight must be rank-2 [vocabSize, embedDim].", nameof(weight));

        int vocabSize = weight.Shape[0];
        int embedDim  = weight.Shape[1];
        int seqLen    = tokens.Length;

        var output = new Tensor([seqLen, embedDim]);

        for (int i = 0; i < seqLen; i++)
        {
            int t = tokens[i];
            if (t < 0 || t >= vocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokens),
                    $"tokens[{i}] = {t} is out of range [0, {vocabSize}).");

            int srcOffset = t * embedDim;
            int dstOffset = i * embedDim;
            Array.Copy(weight.Data, srcOffset, output.Data, dstOffset, embedDim);
        }

        return output;
    }

    /// <summary>
    /// Scatter-add the upstream gradient back into the embedding weight matrix.
    ///
    /// dOut     : [seqLen, embedDim]
    /// tokens   : int[seqLen]  — same tokens used in Forward
    /// vocabSize: int           — number of rows in weight
    /// returns  : [vocabSize, embedDim]  — dWeight (dense)
    ///
    /// For each sequence position i: dWeight[token[i], :] += dOut[i, :]
    /// Multiple positions with the same token accumulate (add) their gradients.
    /// </summary>
    public static Tensor Backward(Tensor dOut, int[] tokens, int vocabSize)
    {
        if (dOut.Rank != 2)
            throw new ArgumentException("dOut must be rank-2 [seqLen, embedDim].", nameof(dOut));

        int seqLen  = dOut.Shape[0];
        int embedDim = dOut.Shape[1];

        if (tokens.Length != seqLen)
            throw new ArgumentException(
                $"tokens.Length ({tokens.Length}) must equal seqLen ({seqLen}).", nameof(tokens));

        var dWeight = new Tensor([vocabSize, embedDim]); // zeroed by default

        for (int i = 0; i < seqLen; i++)
        {
            int t = tokens[i];
            int dstOffset = t   * embedDim;
            int srcOffset = i   * embedDim;

            for (int d = 0; d < embedDim; d++)
                dWeight.Data[dstOffset + d] += dOut.Data[srcOffset + d];
        }

        return dWeight;
    }
}
