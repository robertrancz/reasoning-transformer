---
overview: A from-scratch implementation of a tiny Transformer with a Test-Time Compute reasoning loop, written entirely in C#. We will build a minimalist tensor core, implement forward/backward passes, and add an MCTS loop to allow the model to "think" during inference.
todos:
  - id: naive-matmul-first
    content: Implement a naive (loop-based) correct MatMul first, fully tested, before any SIMD optimization
    status: done
  - id: init-tensor-core
    content: Implement minimalist tensor operations and core math functions — MatMul, Transpose, Add, Multiply, Scale, Fill, AddTo, RandomNormal all done ✅; SIMD fast-path (AVX2) stubbed for Phase 2
    status: in-progress
  - id: build-autograd
    content: Implement manual backward passes (not a tape-based autograd engine) for all essential ops — MatMul ✅, Softmax ✅, LayerNorm ✅, CrossEntropyLoss ✅, RoPE ✅, Embedding ✅ — ALL OPS DONE
    status: done
  - id: gradient-sanity-tests
    content: Write numerical finite-difference gradient checks for every operation — MatMul ✅, Softmax ✅, LayerNorm ✅, CrossEntropy ✅, RoPE ✅, Embedding ✅ — 26 gradient gates cleared; NumericalGradientHelper uses mixed atol+rtol criterion
    status: done
  - id: build-transformer
    content: Build the Tiny Transformer (RoPE, Multi-Head Attention, SwiGLU MLP, LayerNorm) — Forward ✅ + Backward ✅ for Attention, MLP, TransformerBlock, TransformerModel; backward integration tests pass (no NaN/Inf, loss decreases after gradient steps)
    status: done
  - id: training-loop
    content: Implement the AdamW optimizer (decoupled weight decay, bias-corrected moments) and a basic training loop (forward → CrossEntropy → ZeroGrad → backward → optimizer.Step → log). Done ✅; GreedyDecode also implemented.
    status: done
  - id: design-step-representation
    content: Define what a "reasoning step" is (e.g., a decoded sequence of 20–50 tokens), the branching factor for MCTS (top-k=4–8 candidates per step), and the overall search budget before building MCTS or PRM
    status: pending
  - id: train-base-model
    content: "DONE ✅. 5000 problems, 19-char vocab, D=64/H=4/L=2/MLP=256 (134k params), 2000 steps (~25 min). Loss: 2.92→0.91→0.73→0.66→0.62→0.58. RunLogger writes to logs/run_*.log."
    status: done
  - id: eval-baseline
    content: "DONE ✅. Greedy baseline: 46/500 = 9.2% on held-out test set. Random baseline ≈ 5.3%. Model beats random but plateaus early — needs more steps or larger model (SIMD Phase 2 unlocks this). Logged to logs/run_20261111_141554.log."
    status: done
  - id: reward-model
    content: Implement an Outcome Reward Model (ORM) first (simpler — scores final answers only), then extend to a Process Reward Model (PRM) that scores intermediate steps using programmatically derived step-level labels from the synthetic dataset
    status: pending
  - id: mcts-reasoning
    content: Build MCTS over reasoning *steps* (not individual tokens) — expand top-k sampled step-candidates per node, score with PRM/ORM, backpropagate value up the tree; compare final accuracy against the greedy baseline
    status: pending
isProject: true
---

# reason.cs: From-Scratch Reasoning Transformer

This project combines the educational depth of Andrej Karpathy's `llm.c` / `micrograd` with the cutting-edge concepts of OpenAI's o1 (Test-Time Compute). We will build everything in pure C# to deeply understand the mechanics of both training and reasoning.

## 1. The Math & Tensor Foundation
Instead of relying on PyTorch, we will build a minimalist tensor library in C#.
*   Use 1D arrays with `Span<T>` and `Memory<T>` for cache-friendly memory access.
*   **Start with a naive (loop-based) correct MatMul.** Only add SIMD acceleration behind the same interface after correctness is locked in with tests. Mixing correctness work with `System.Runtime.Intrinsics` AVX2 intrinsics early introduces silent, hard-to-debug corruption.
*   SIMD target: `System.Runtime.Intrinsics.X86.Avx2` / `System.Numerics.Vector<T>` as a drop-in fast path.

## 2. Autograd: Manual Backward Passes
We will write **manual backward passes** for every core operation (not a tape-based autograd engine). This mirrors the `llm.c` approach — it is faster, less code, and more educational for a fixed Transformer architecture. A general autograd engine adds complexity without payoff at this scale.

Operations requiring both forward and backward:
*   MatMul, Softmax, LayerNorm, CrossEntropyLoss, Embedding lookup, RoPE

**Gradient sanity tests are mandatory before proceeding.** For every op, compare the analytical gradient against numerical finite-difference gradients. RoPE's backward pass is especially non-obvious and easy to get subtly wrong.

## 3. The Tiny Transformer Architecture
We will implement a modern, scaled-down Transformer model.
*   **Embeddings & Positional Encoding**: RoPE (Rotary Positional Embeddings). Note: RoPE has a non-trivial backward pass — unit tests against finite-difference gradients are required here before moving on.
*   **Self-Attention**: Multi-Head Attention (forward and backward pass).
*   **Feed Forward**: SwiGLU or GELU MLP block.
*   *Verification*: All gradient unit tests must pass before the training loop is built.

## 4. Training Loop & Synthetic Data
To test the model's ability to reason, we need it to learn multi-step logic.
*   Implement a simple AdamW optimizer in C#.
*   Create a synthetic dataset of **chained multi-step logic puzzles** requiring at least 2–3 reasoning steps (e.g., `A=2, B=A+1, C=B*3, C=?`). Simple single-step arithmetic (e.g., `2+3=?`) will not produce a measurable signal from MCTS vs. greedy decoding, since the base model can solve it in one step.
*   Hold out a test set before training. **Record greedy-decoding accuracy as the baseline** (see `eval-baseline`) — without this, it is impossible to demonstrate that MCTS/PRM actually helps.

## 5. Test-Time Compute (System 2 Thinking)

### 5a. Defining a "Reasoning Step"
MCTS over individual vocabulary tokens is computationally intractable (branching factor ~10k–50k). Real systems search over **reasoning steps**, not tokens. Before building anything:
*   Define a step as a fixed-length decoded sequence (e.g., 20–50 tokens representing one logical sub-conclusion).
*   Set the branching factor: expand top-k (k=4–8) candidates per node via sampling (nucleus or top-k).
*   Set a search budget (e.g., max 50–100 node expansions per query).

### 5b. Reward Model — ORM First, then PRM
*   **Outcome Reward Model (ORM)**: Train a small linear head on top of the base model to score whether the *final answer* is correct. Simpler to train — step-level labels are not needed. Build this first.
*   **Process Reward Model (PRM)**: Extend to scoring *intermediate steps*. For the synthetic dataset, step-level correctness labels can be derived **programmatically** (each arithmetic sub-step is verifiable). This removes the need for human annotation — but this labeling strategy must be explicitly designed as part of the synthetic data pipeline.

### 5c. Monte Carlo Tree Search
Implement MCTS over reasoning steps:
1.  **Selection**: Traverse tree using UCB1 (or PUCT) to pick a promising node.
2.  **Expansion**: Decode k step-candidates from the current state using sampling.
3.  **Evaluation**: Score each candidate with the PRM/ORM.
4.  **Backpropagation**: Update node values up the tree.
5.  **Comparison**: Measure final accuracy on the held-out test set vs. greedy baseline.

## Tech Stack
*   **Language**: C# (.NET 10)
*   **Libraries**: Minimal external dependencies. `System.Numerics` and `System.Runtime.Intrinsics` for SIMD.
*   **Testing**: xUnit or NUnit — mandatory for gradient verification.

## 6. Key Academic References
*   **Transformers**: *"Attention Is All You Need"* (Vaswani et al., 2017)
*   **RoPE**: *"RoFormer: Enhanced Transformer with Rotary Position Embedding"* (Su et al., 2021)
*   **Optimizer**: *"Decoupled Weight Decay Regularization"* (Loshchilov & Hutter, 2017) — AdamW
*   **Process Reward Models**: *"Let's Verify Step by Step"* (Lightman et al., 2023)
*   **Outcome Reward Models**: *"Training Verifiers to Solve Math Word Problems"* (Cobbe et al., 2021)
*   **Test-Time Compute / Reasoning**: *"Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"* (Zelikman et al., 2024)
