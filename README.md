# reason.cs — From-Scratch Reasoning Transformer

A ground-up implementation of a tiny Transformer with a **Test-Time Compute reasoning loop**, written entirely in **C# (.NET 10)**. No PyTorch, no ONNX, no ML frameworks — every tensor operation, every backward pass, and the full MCTS search tree are written by hand.

The project bridges the educational depth of Andrej Karpathy's [`llm.c`](https://github.com/karpathy/llm.c) / [`micrograd`](https://github.com/karpathy/micrograd) with the cutting-edge ideas behind OpenAI's o1 (Test-Time Compute / "System 2 thinking").

---

## Why?

Most ML education stops at "here is PyTorch, call `loss.backward()`". This project goes one level deeper: we derive and implement every gradient by hand, then add a reasoning layer on top that lets the model *think before it answers* — all in a language and runtime most ML practitioners have never used for this purpose.

---

## Architecture Overview

```
Input tokens
     │
     ▼
Token Embedding  [VocabSize × EmbedDim]
     │
     ▼
 ┌─────────────────────────────┐
 │  TransformerBlock × N       │
 │  ┌─────────────────────┐    │
 │  │ LayerNorm            │    │
 │  │ Multi-Head Attention │    │
 │  │   └─ RoPE           │    │
 │  │ Residual Add         │    │
 │  │ LayerNorm            │    │
 │  │ SwiGLU MLP           │    │
 │  │ Residual Add         │    │
 │  └─────────────────────┘    │
 └─────────────────────────────┘
     │
     ▼
Final LayerNorm
     │
     ▼
LM Head  [EmbedDim × VocabSize]
     │
     ▼
  Logits
     │
     ├─── Greedy Decode (baseline)
     │
     └─── MCTS + Reward Model (reasoning)
              │
              ├── ORM  (Outcome Reward Model)
              └── PRM  (Process Reward Model)
```

**Default config** (`TransformerConfig`): VocabSize=256 · EmbedDim=128 · 4 heads · 4 layers · MLP hidden=512 · MaxSeqLen=128

---

## Tech Stack

| Concern | Choice |
|---|---|
| Language | C# 13 / .NET 10 |
| Tensor backend | Hand-rolled (`Span<T>`, flat arrays) |
| SIMD | `System.Runtime.Intrinsics` (Phase 2) |
| Testing | xUnit |
| External ML libs | **None** |

---

## Project Structure

```
reason.cs/
├── src/
│   ├── ReasonCs.Core/              # Core library — all model logic
│   │   ├── Tensor/
│   │   │   ├── Tensor.cs           # Rank-N tensor (shape, strides, flat array)
│   │   │   └── TensorOps.cs        # MatMul, Transpose, Sum, Add, Multiply, Scale, RandomNormal
│   │   ├── Autograd/Ops/
│   │   │   ├── MatMulOp.cs         # ✅ Forward + backward (dA = dC×Bᵀ, dB = Aᵀ×dC)
│   │   │   ├── SoftmaxOp.cs        # ✅ Numerically stable softmax + Jacobian-vector backward
│   │   │   ├── LayerNormOp.cs      # LayerNorm + backward (dx, dγ, dβ)
│   │   │   ├── CrossEntropyOp.cs   # Fused softmax + cross-entropy
│   │   │   ├── RoPEOp.cs           # Rotary positional encoding + backward
│   │   │   └── EmbeddingOp.cs      # Token embedding lookup + scatter-add grad
│   │   ├── Transformer/
│   │   │   ├── Attention.cs        # Multi-head self-attention (Q/K/V/O projections)
│   │   │   ├── MLP.cs              # SwiGLU feed-forward block
│   │   │   ├── TransformerBlock.cs # Pre-norm residual block
│   │   │   └── TransformerModel.cs # Full model + TransformerConfig record
│   │   ├── Training/
│   │   │   ├── AdamW.cs            # AdamW optimizer (decoupled weight decay)
│   │   │   └── TrainingLoop.cs     # Forward → loss → backward → step loop
│   │   ├── Data/
│   │   │   ├── Tokenizer.cs        # Character-level tokenizer
│   │   │   └── SyntheticDataset.cs # Chained multi-step arithmetic problems
│   │   ├── Inference/
│   │   │   └── Evaluator.cs        # Greedy-decoding accuracy (baseline)
│   │   └── Reasoning/
│   │       ├── RewardModel.cs      # ORM / PRM scoring head
│   │       ├── MctsNode.cs         # MCTS tree node (UCB1, visits, value)
│   │       └── MctsSearch.cs       # 4-phase MCTS over reasoning steps
│   └── ReasonCs.Runner/
│       └── Program.cs              # Staged entry point
└── tests/
    └── ReasonCs.Tests/
        ├── Tensor/TensorOpsTests.cs            # Tensor + MatMul correctness
        ├── Autograd/GradientSanityTests.cs     # Finite-difference gradient gates
        ├── Autograd/NumericalGradientHelper.cs # Reusable grad-check utility
        └── Transformer/TransformerSmokeTests.cs # Shape / NaN smoke tests
```

---

## Build & Test

```bash
# Build everything
dotnet build ReasonCs.slnx

# Run all tests
dotnet test ReasonCs.slnx
```

**Current test status:**

| Suite | Tests | Status |
|---|---|---|
| `TensorOpsTests` — MatMul (2×2, identity, shape mismatch), Scale | 4 | ✅ passing |
| `GradientSanityTests` — MatMul dA, dB, non-uniform loss | 3 | ✅ passing |
| `GradientSanityTests` — Softmax uniform, sums-to-one, grad 1D, grad 2D | 4 | ✅ passing |
| `GradientSanityTests` — LayerNorm (×2), RoPE, CrossEntropy | 4 | ⏸ skipped (gates) |
| `TransformerSmokeTests` — output shape, no NaN/Inf | 2 | ⏸ skipped (gates) |

**Total: 11 passing · 6 skipped · 0 failing**

Skipped tests are **intentional gates** — each `[Fact(Skip = "...")]` carries the name of the op that must be implemented first. Remove the `Skip` attribute once the op is done.

---

## Implementation Status

### ✅ Done

#### `Tensor` — `src/ReasonCs.Core/Tensor/`
- **`Tensor`**: rank-N tensor backed by a flat `float[]`. Stores shape and computed strides separately so reshape/transpose are zero-copy. Multi-index accessor (`tensor[i, j, k]`).
- **`TensorOps`**:
  - `MatMul(a, b)` — O(M·K·N) naive triple-loop ✅ verified
  - `Transpose(a)` — rank-2 transpose `[M,N] → [N,M]` ✅
  - `Sum(a)` — reduce all elements to a scalar ✅
  - `Add`, `Multiply` — element-wise ops ✅
  - `Scale(t, scalar)` — scalar multiplication ✅
  - `RandomNormal(t, mean, std, seed)` — Box-Muller initialization ✅
  - *(SIMD fast-path stubbed as `// TODO: MatMulSimd` — Phase 2)*

#### `MatMulOp` — `src/ReasonCs.Core/Autograd/Ops/MatMulOp.cs`
- **`Forward(a, b)`** — delegates to `TensorOps.MatMul` ✅
- **`Backward(dC, a, b)`** — `dA = dC × Bᵀ`, `dB = Aᵀ × dC` ✅ verified by 3 gradient tests

#### `SoftmaxOp` — `src/ReasonCs.Core/Autograd/Ops/SoftmaxOp.cs`
- **`Forward(x)`** — numerically stable (max subtraction), works on any rank, applies along last dim ✅
- **`Backward(dOut, probs)`** — Jacobian-vector product: `dx_i = p_i · (dOut_i − dot(p, dOut))` ✅ verified by 1D + 2D gradient tests

#### `NumericalGradientHelper` — `tests/ReasonCs.Tests/Autograd/`
- Finite-difference gradient checker with configurable `eps` and `tolerance` ✅
- Calibrated for float32: `eps=5e-4`, `tolerance=0.01` (1%) with documented rationale

#### `MctsNode` — `src/ReasonCs.Core/Reasoning/MctsNode.cs`
- Tree node with `Tokens`, `Parent`, `Children`, `Visits`, `TotalValue`, `PriorScore` ✅
- `Q` property (mean value), `UCB(explorationC)` — Upper Confidence Bound ✅

#### Scaffolding (stubs — full signatures, PyTorch annotations, math in XML doc comments)

| Module | File | Status |
|---|---|---|
| `SoftmaxOp` | `Autograd/Ops/SoftmaxOp.cs` | ✅ done |
| `LayerNormOp` | `Autograd/Ops/LayerNormOp.cs` | 🔲 TODO |
| `CrossEntropyOp` | `Autograd/Ops/CrossEntropyOp.cs` | 🔲 TODO |
| `RoPEOp` | `Autograd/Ops/RoPEOp.cs` | 🔲 TODO |
| `EmbeddingOp` | `Autograd/Ops/EmbeddingOp.cs` | 🔲 TODO |
| `Attention` | `Transformer/Attention.cs` | 🔲 TODO |
| `MLP` | `Transformer/MLP.cs` | 🔲 TODO |
| `TransformerBlock` | `Transformer/TransformerBlock.cs` | 🔲 TODO |
| `TransformerModel` | `Transformer/TransformerModel.cs` | 🔲 TODO |
| `AdamW` | `Training/AdamW.cs` | 🔲 TODO |
| `TrainingLoop` | `Training/TrainingLoop.cs` | 🔲 TODO |
| `Tokenizer` | `Data/Tokenizer.cs` | 🔲 TODO |
| `SyntheticDataset` | `Data/SyntheticDataset.cs` | 🔲 TODO |
| `Evaluator` | `Inference/Evaluator.cs` | 🔲 TODO |
| `RewardModel` | `Reasoning/RewardModel.cs` | 🔲 TODO |
| `MctsSearch` | `Reasoning/MctsSearch.cs` | 🔲 TODO |

---

### 🔜 What's Next (ordered by plan)

1. **`LayerNormOp`** — forward + backward (dx, dγ, dβ), remove 2 skips
3. **`CrossEntropyOp`** — fused log-softmax + backward (`p - 1_target`), remove skip
4. **`RoPEOp`** — forward + backward (⚠ verify against finite-diff — this is the hard one), remove skip
5. **`EmbeddingOp`** — lookup + scatter-add gradient
6. **`Attention.Forward` + `Backward`** — wire all ops; smoke tests pass
7. **`MLP.Forward` + `Backward`** — SwiGLU
8. **`TransformerBlock` + `TransformerModel`** — assemble; smoke tests pass
9. **`AdamW.Step`** + **`TrainingLoop`** — end-to-end training
10. **`SyntheticDataset.Generate`** + **`Tokenizer`** — multi-step arithmetic data
11. **Train & eval** — greedy baseline accuracy on held-out test set
12. **`RewardModel`** — ORM head, then PRM
13. **`MctsSearch`** — full MCTS loop; compare against greedy baseline

---

## Key Design Decisions

**Manual backward passes, not autograd tape.** For a fixed Transformer architecture, explicit gradients (à la `llm.c`) are faster, simpler, and more educational than a general tape-based engine. Every op carries its own documented backward formula.

**Naive MatMul before SIMD.** Correctness is gated by unit tests before any `System.Runtime.Intrinsics` optimization. The SIMD path is a drop-in replacement behind the same interface.

**MCTS over reasoning steps, not tokens.** Vocabulary-level MCTS (branching factor ~10k–50k) is intractable. The search operates over *steps* — fixed-length decoded sequences of 20–50 tokens — with a branching factor of k=4–8 and a node budget of ~64 expansions per query.

**ORM before PRM.** The Outcome Reward Model (scores final answers) is simpler to train and is built first as a stepping stone. The Process Reward Model (scores intermediate steps) extends it, using programmatically derived step-correctness labels from the synthetic dataset.

**Greedy baseline first.** The trained model is evaluated with greedy decoding before MCTS is turned on. Without this baseline, it is impossible to demonstrate that the reasoning loop actually helps.

---

## PyTorch Rosetta Stone

Every handcrafted implementation carries a full `/// PyTorch equivalent:` section in its XML doc comment. The table below is a quick reference — open the source file for the complete side-by-side comparison.

| reason.cs class | PyTorch equivalent | Notes |
|---|---|---|
| `Tensor` | `torch.Tensor` | CPU-only, no autograd tape |
| `TensorOps.MatMul` | `a @ b` / `torch.matmul(a, b)` | Naive O(M·K·N); SIMD in Phase 2 |
| `TensorOps.RandomNormal` | `torch.nn.init.normal_(t, μ, σ)` | Box-Muller transform |
| `MatMulOp.Forward/Backward` | `a @ b` + autograd | Backward: `dA = dC @ B.T`, `dB = A.T @ dC` |
| `SoftmaxOp` | `F.softmax(x, dim=-1)` | Numerically stable (max subtraction) |
| `LayerNormOp` | `F.layer_norm` / `nn.LayerNorm` | Returns cached mean & rstd for backward |
| `CrossEntropyOp` | `F.cross_entropy(logits, targets)` | Fused log-softmax + NLL for stability |
| `RoPEOp` | `LlamaRotaryEmbedding` (HuggingFace) | No PyTorch built-in; backward = inverse rotation |
| `EmbeddingOp` | `nn.Embedding(vocab, dim)` | Backward = scatter-add (sparse grad) |
| `Attention` | `nn.MultiheadAttention` + `F.scaled_dot_product_attention` | Uses RoPE instead of learned positional bias |
| `MLP` (SwiGLU) | `F.silu(x @ W1) * (x @ W2)) @ W3` | No `nn.SwiGLU` exists; 3 matrices vs. standard 2 |
| `TransformerBlock` | `nn.TransformerEncoderLayer` (Pre-Norm) | Pre-Norm: normalize *before* sublayer |
| `TransformerModel` | nanoGPT / `nn.Transformer` | Manually managed weights; no `nn.Module` |
| `AdamW` | `torch.optim.AdamW` | Decoupled weight decay (not folded into gradient) |
| `TrainingLoop` | `loss.backward()` + `optimizer.step()` | Backward is a hand-written reverse pass |
| `Tokenizer` | `tiktoken` / `AutoTokenizer` | Character-level; vocabulary from corpus |
| `SyntheticDataset` | `torch.utils.data.Dataset` | Programmatic step labels (no human annotation) |
| `Evaluator` | `model.eval()` + `torch.no_grad()` + `argmax` | Greedy decoding accuracy |
| `RewardModel` | Custom `nn.Module` + `nn.Linear` head | ORM first → extend to PRM |
| `MctsNode` | Plain dataclass (no PyTorch equivalent) | Pure search data structure |
| `MctsSearch` | No PyTorch equivalent | Pure search algorithm; LLM used only in Expand |

---

## References

| Paper | Authors | Year |
|---|---|---|
| Attention Is All You Need | Vaswani et al. | 2017 |
| RoFormer: Rotary Position Embedding | Su et al. | 2021 |
| Decoupled Weight Decay Regularization (AdamW) | Loshchilov & Hutter | 2017 |
| Training Verifiers to Solve Math Word Problems (ORM) | Cobbe et al. | 2021 |
| Let's Verify Step by Step (PRM) | Lightman et al. | 2023 |
| Quiet-STaR: Language Models Can Teach Themselves to Think | Zelikman et al. | 2024 |
