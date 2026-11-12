using ReasonCs.Core.Transformer;

namespace ReasonCs.Core.Reasoning;

/// <summary>
/// Monte Carlo Tree Search over reasoning <em>steps</em> (not individual tokens).
///
/// A "step" is a decoded sequence of <see cref="StepTokenLen"/> tokens appended
/// to the current node's token prefix.  Because our answers are short (≤4 chars),
/// one step = one complete candidate answer.
///
/// Algorithm — each simulation
/// ───────────────────────────
///   1. Selection   — walk the tree via UCB1 until a leaf node is reached.
///   2. Expansion   — decode <see cref="TopK"/> distinct step-candidates from the
///                    leaf via temperature sampling; score each with the ORM.
///   3. Evaluation  — backup value = max ORM score among new children
///                    (optimistic backup; conservative alternative: mean).
///   4. Backprop    — increment Visits and accumulate TotalValue from leaf to root.
///
/// After <see cref="SearchBudget"/> simulations, return the token sequence of the
/// highest-Q child of the root.
///
/// Parameters
/// ──────────
///   StepTokenLen  — tokens generated per expansion step (default 4 = max answer len)
///   TopK          — branching factor; candidates tried per expansion (default 4)
///   SearchBudget  — number of simulations (default 16)
///   Temperature   — sampling temperature; > 1 → more random, &lt; 1 → more peaked
///   ExplorationC  — UCB1 exploration constant √2 ≈ 1.41 (standard)
///   Seed          — RNG seed for reproducibility; applies on first Rng access
///
/// Why step-level (not token-level)?
/// ─────────────────────────────────
/// Token-level MCTS has a branching factor equal to vocabulary size (~10k–50k),
/// making the search intractable.  Step-level search keeps branching to TopK=4–8
/// and uses the ORM to prune unpromising branches without full rollouts.
///
/// PyTorch equivalent:
///   # MCTS is a pure search algorithm — no PyTorch equivalent.
///   # The neural network is used only inside Expand (sampling candidates)
///   # and Evaluate (scoring them with the ORM).
///   def search(model, reward, prompt, budget=16, topk=4):
///       root = MctsNode(prompt)
///       for _ in range(budget):
///           leaf     = select(root)              # UCB1 tree walk
///           children = expand(model, leaf, topk) # sample topk completions
///           value    = max(reward(c) for c in children)
///           backprop(leaf, value)
///       return max(root.children, key=lambda c: c.q).tokens
/// </summary>
public sealed class MctsSearch
{
    // ── Configuration (set via object initializer) ────────────────────────────
    public int   StepTokenLen  { get; init; } = 2;     // tokens per step — must match RewardTrainer.ContinuationLen
    public int   TopK          { get; init; } = 4;     // branching factor
    public int   SearchBudget  { get; init; } = 16;    // simulations per query
    public float Temperature   { get; init; } = 0.8f;  // diversity of sampling
    public float ExplorationC  { get; init; } = 1.41f; // UCB1 constant
    public int   Seed          { get; init; } = 0;
    /// <summary>
    /// Maximum tree depth (levels below root).  Nodes at this depth are
    /// not expanded further; the UCB1 re-selection loop uses their
    /// PriorScore directly.  Default 1 keeps the tree flat (best-of-N
    /// with UCB1 budget allocation) so the ORM always scores
    /// prompt + StepTokenLen tokens — matching its training distribution.
    /// Set &gt; 1 only when the ORM is trained on multi-step sequences.
    /// </summary>
    public int   MaxDepth      { get; init; } = 1;
    /// <summary>
    /// When true (default), the first candidate added to each expansion is
    /// the greedy (argmax) decode.  This guarantees MCTS accuracy ≥ greedy
    /// accuracy: even if all temperature-sampled candidates are wrong, the
    /// ORM can still identify the greedy answer as the best option.
    /// The remaining <see cref="TopK"/>-1 slots are filled by temperature
    /// sampling, giving MCTS a chance to find a <em>better</em> answer.
    /// </summary>
    public bool  IncludeGreedyCandidate { get; init; } = true;

    private readonly TransformerModel _model;
    private readonly RewardModel      _reward;

    // Lazy because init-only Seed is set by the object initializer after the ctor.
    private Random? _rng;
    private Random  Rng => _rng ??= new Random(Seed);

    public MctsSearch(TransformerModel model, RewardModel reward)
    {
        _model  = model;
        _reward = reward;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Run MCTS from <paramref name="promptTokens"/> and return the best
    /// full reasoning sequence found within <see cref="SearchBudget"/> simulations.
    ///
    /// The returned array starts with all prompt tokens, followed by the
    /// best-found continuation.
    /// </summary>
    public int[] Search(int[] promptTokens)
    {
        var root = new MctsNode(promptTokens);

        for (int sim = 0; sim < SearchBudget; sim++)
        {
            // 1. Selection — stops at a leaf OR at MaxDepth
            var leaf = Select(root);

            // 2. Expansion + Evaluation
            float value;
            if (leaf.Depth >= MaxDepth)
            {
                // At max depth: this node was scored at expansion time.
                // Re-use its PriorScore so the budget refines UCB1 estimates
                // without going deeper (which would produce sequences longer
                // than what the ORM was trained on).
                value = leaf.PriorScore;
            }
            else
            {
                var newChildren = leaf.IsLeaf ? Expand(leaf) : leaf.Children;
                value = Evaluate(newChildren);
            }

            // 4. Backpropagation
            Backpropagate(leaf, value);
        }

        // Return the highest-Q child's token sequence; fall back to prompt
        // if no children were created (e.g. budget = 0).
        if (root.Children.Count == 0)
            return promptTokens;

        return root.Children.MaxBy(c => c.Q)!.Tokens;
    }

    // ── Phase 1: Selection ────────────────────────────────────────────────────

    /// <summary>
    /// Walk the tree from <paramref name="root"/> via UCB1 until reaching a leaf node
    /// (a node with no children yet).
    ///
    /// UCB1:  Q(n) + C · √(ln(parent.Visits) / (n.Visits + 1))
    ///
    /// Ties are broken by the order of <see cref="MctsNode.Children"/>.
    /// </summary>
    private MctsNode Select(MctsNode root)
    {
        var node = root;
        // Descend while the node has children AND hasn't reached MaxDepth.
        // Stopping at MaxDepth prevents expansion beyond the ORM's training
        // distribution (prompt + StepTokenLen tokens).
        while (!node.IsLeaf && node.Depth < MaxDepth)
            node = node.Children.MaxBy(c => c.UCB(ExplorationC))!;
        return node;
    }

    // ── Phase 2: Expansion ────────────────────────────────────────────────────

    /// <summary>
    /// Decode <see cref="TopK"/> distinct step-candidates from the leaf's token
    /// sequence via temperature sampling, score each with the ORM, and attach
    /// them as children of <paramref name="leaf"/>.
    ///
    /// When <see cref="IncludeGreedyCandidate"/> is true, the first slot is
    /// always filled with the greedy (T→0) decode so that MCTS accuracy is
    /// always ≥ greedy accuracy.  The remaining slots use <see cref="Temperature"/>.
    ///
    /// Duplicate continuations (same new tokens) are discarded; at most
    /// 4 × TopK attempts are made to avoid an infinite loop on very low-entropy
    /// distributions.
    /// </summary>
    private List<MctsNode> Expand(MctsNode leaf)
    {
        var newChildren       = new List<MctsNode>(TopK);
        var seenContinuations = new HashSet<string>(TopK * 2);
        int attempts          = 0;
        int maxAttempts       = TopK * 4;

        // Optionally seed the expansion with the greedy (argmax) decode so
        // the search always has access to the model's best single-pass answer.
        if (IncludeGreedyCandidate)
        {
            int[] greedyTokens = TemperatureSample(leaf.Tokens, Rng, overrideTemp: 0f);
            string greedyKey   = string.Join(",", greedyTokens.Skip(leaf.Tokens.Length));
            seenContinuations.Add(greedyKey);   // mark so temp-sampling doesn't duplicate it
            float greedyScore  = _reward.Score(greedyTokens);
            leaf.Children.Add(new MctsNode(greedyTokens, parent: leaf, priorScore: greedyScore));
            newChildren.Add(leaf.Children[^1]);
        }

        while (newChildren.Count < TopK && attempts < maxAttempts)
        {
            attempts++;

            // Sample a continuation of StepTokenLen new tokens
            int[] candidateTokens = TemperatureSample(leaf.Tokens, Rng);

            // Build a key from only the NEW portion to detect duplicates
            string contKey = string.Join(",", candidateTokens.Skip(leaf.Tokens.Length));
            if (!seenContinuations.Add(contKey))
                continue;   // duplicate — try again

            // Score the candidate sequence.
            // Candidates have exactly leaf.Tokens.Length + StepTokenLen tokens,
            // matching the ORM training distribution (prompt + ContinuationLen=2).
            float score = _reward.Score(candidateTokens);

            var child = new MctsNode(candidateTokens, parent: leaf, priorScore: score);
            leaf.Children.Add(child);
            newChildren.Add(child);
        }

        return newChildren;
    }

    // ── Phase 3: Evaluation ───────────────────────────────────────────────────

    /// <summary>
    /// Compute the backup value from a list of newly expanded children.
    ///
    /// Strategy: optimistic — return the <em>maximum</em> ORM score.
    /// This rewards any path that reaches a promising child, driving the
    /// search to allocate more budget to that branch.
    ///
    /// Alternative: mean score (more conservative, better when ORM is noisy).
    /// </summary>
    private static float Evaluate(IReadOnlyList<MctsNode> children)
    {
        if (children.Count == 0) return 0f;
        float max = float.NegativeInfinity;
        foreach (var c in children)
            if (c.PriorScore > max) max = c.PriorScore;
        return max;
    }

    // ── Phase 4: Backpropagation ──────────────────────────────────────────────

    /// <summary>
    /// Walk from <paramref name="node"/> up to the root, incrementing
    /// <see cref="MctsNode.Visits"/> and adding <paramref name="value"/> to
    /// <see cref="MctsNode.TotalValue"/> at every ancestor (inclusive).
    /// </summary>
    private static void Backpropagate(MctsNode node, float value)
    {
        var current = node;
        while (current is not null)
        {
            current.Visits++;
            current.TotalValue += value;
            current = current.Parent;
        }
    }

    // ── Temperature sampling ─────────────────────────────────────────────────

    /// <summary>
    /// Autoregressively decode <see cref="StepTokenLen"/> new tokens from
    /// <paramref name="inputTokens"/> using temperature sampling.
    ///
    /// Temperature scaling:
    ///   p_i = exp((logit_i − max) / T) / Σ_j exp((logit_j − max) / T)
    ///
    /// T → 0 : greedy argmax (pass <paramref name="overrideTemp"/> = 0).
    /// T = 1 : pure softmax sampling.
    /// T > 1 : more uniform / exploratory.
    /// </summary>
    /// <param name="overrideTemp">
    /// When ≥ 0 overrides <see cref="Temperature"/> for this call.
    /// Pass 0f to get greedy argmax decoding.  Default −1 uses
    /// <see cref="Temperature"/>.
    /// </param>
    private int[] TemperatureSample(int[] inputTokens, Random rng, float overrideTemp = -1f)
    {
        float temp    = overrideTemp >= 0f ? overrideTemp : Temperature;
        var tokens    = new List<int>(inputTokens.Length + StepTokenLen);
        tokens.AddRange(inputTokens);

        int vocabSize = _model.Config.VocabSize;

        for (int step = 0; step < StepTokenLen; step++)
        {
            // Forward pass — [seqLen, VocabSize]
            var logits = _model.Forward([.. tokens]);

            // Extract last-position logits [VocabSize]
            int lastOffset = (tokens.Count - 1) * vocabSize;
            float[] lastLogits = new float[vocabSize];
            Array.Copy(logits.Data, lastOffset, lastLogits, 0, vocabSize);

            // Temperature sampling (or greedy argmax when temp ≈ 0)
            int token = temp < 1e-6f
                ? ArgMax(lastLogits)
                : SampleFromLogits(lastLogits, temp, rng);
            tokens.Add(token);
        }

        return [.. tokens];
    }

    /// <summary>Returns the index of the maximum value (greedy argmax).</summary>
    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /// <summary>
    /// Sample one token index from raw logits using temperature.
    /// Numerically stable: subtracts max before exp.
    /// </summary>
    private static int SampleFromLogits(float[] logits, float temperature, Random rng)
    {
        // Find max for numerical stability
        float maxVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxVal) maxVal = logits[i];

        // Compute temperature-scaled softmax
        float sumExp = 0f;
        float[] probs = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp((logits[i] - maxVal) / temperature);
            sumExp  += probs[i];
        }
        for (int i = 0; i < logits.Length; i++)
            probs[i] /= sumExp;

        // Multinomial sample via inverse CDF
        float u      = (float)rng.NextDouble();
        float cumsum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cumsum += probs[i];
            if (u < cumsum) return i;
        }
        return probs.Length - 1;   // floating-point rounding fallback
    }
}
