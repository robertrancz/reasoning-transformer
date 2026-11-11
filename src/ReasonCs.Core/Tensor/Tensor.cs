namespace ReasonCs.Core.Tensors;

/// <summary>
/// A rank-N tensor backed by a flat 1-D array.
/// Shape and strides are stored separately so reshape/transpose are zero-copy.
///
/// PyTorch equivalent:
///   torch.Tensor
///
///   # Creation equivalents:
///   torch.zeros(*shape)          →  new Tensor(shape)
///   torch.tensor(data)           →  new Tensor(shape, data)
///   t[i, j]                      →  tensor[i, j]
///   t.shape / t.ndim / t.numel() →  Shape / Rank / Size
///
/// Key difference: PyTorch tensors support automatic differentiation and
/// device placement (CPU/CUDA). This class is CPU-only and grad-free by
/// design — gradients are tracked manually in the Autograd/Ops layer.
/// </summary>
public sealed class Tensor
{
    public float[] Data { get; }
    public int[] Shape { get; }
    public int[] Strides { get; }
    public int Rank => Shape.Length;
    public int Size => Data.Length;

    public Tensor(int[] shape)
    {
        Shape = shape;
        Strides = ComputeStrides(shape);
        Data = new float[Strides[0] * shape[0]];
    }

    public Tensor(int[] shape, float[] data)
    {
        Shape = shape;
        Strides = ComputeStrides(shape);
        Data = data;
    }

    public float this[params int[] indices]
    {
        get => Data[FlatIndex(indices)];
        set => Data[FlatIndex(indices)] = value;
    }

    private int FlatIndex(int[] indices)
    {
        int idx = 0;
        for (int i = 0; i < indices.Length; i++)
            idx += indices[i] * Strides[i];
        return idx;
    }

    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        strides[^1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }

    public override string ToString() =>
        $"Tensor(shape=[{string.Join(", ", Shape)}])";
}
