using ReasonCs.Core.Tensors;

namespace ReasonCs.Tests.TensorTests;

/// <summary>
/// Unit tests for TensorOps — the very first tests to write and pass.
/// Naive MatMul must be correct before any other work continues.
/// </summary>
public class TensorOpsTests
{
    [Fact]
    public void MatMul_2x2_KnownResult()
    {
        // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]  →  C = [[19,22],[43,50]]
        var a = new Tensor([2, 2], [1f, 2f, 3f, 4f]);
        var b = new Tensor([2, 2], [5f, 6f, 7f, 8f]);

        var c = TensorOps.MatMul(a, b);

        Assert.Equal(19f, c[0, 0], precision: 4);
        Assert.Equal(22f, c[0, 1], precision: 4);
        Assert.Equal(43f, c[1, 0], precision: 4);
        Assert.Equal(50f, c[1, 1], precision: 4);
    }

    [Fact]
    public void MatMul_ShapeMismatch_Throws()
    {
        var a = new Tensor([2, 3]);
        var b = new Tensor([4, 2]);
        Assert.Throws<ArgumentException>(() => TensorOps.MatMul(a, b));
    }

    [Fact]
    public void MatMul_Identity_ReturnsOriginal()
    {
        var a = new Tensor([3, 3], [1f,2f,3f, 4f,5f,6f, 7f,8f,9f]);
        var eye = new Tensor([3, 3], [1f,0f,0f, 0f,1f,0f, 0f,0f,1f]);

        var result = TensorOps.MatMul(a, eye);

        for (int i = 0; i < 9; i++)
            Assert.Equal(a.Data[i], result.Data[i], precision: 4);
    }

    [Fact]
    public void Scale_MultipliesAllElements()
    {
        var t = new Tensor([3], [1f, 2f, 3f]);
        var r = TensorOps.Scale(t, 3f);
        Assert.Equal([3f, 6f, 9f], r.Data);
    }
}
