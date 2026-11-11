namespace ReasonCs.Core.Data;

/// <summary>
/// Minimal character-level tokenizer for the synthetic dataset.
/// Vocabulary is built from the characters that appear in the training corpus.
///
/// PyTorch equivalent:
///   # Python has no built-in tokenizer in PyTorch itself.
///   # Common alternatives:
///
///   # 1. Character-level (equivalent to this class):
///   chars = sorted(set(corpus))
///   char_to_id = {c: i for i, c in enumerate(chars)}
///   encode = lambda s: [char_to_id[c] for c in s]
///   decode = lambda ids: ''.join(chars[i] for i in ids)
///
///   # 2. BPE / subword (used by real LLMs):
///   import tiktoken
///   enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
///   enc.encode("hello world")
///
///   # 3. HuggingFace:
///   from transformers import AutoTokenizer
///   tokenizer = AutoTokenizer.from_pretrained("gpt2")
///
/// We use character-level because the synthetic dataset's vocabulary
/// (digits, letters, operators, spaces) is tiny and fully enumerable.
/// </summary>
public sealed class Tokenizer
{
    private readonly Dictionary<char, int> _charToId = new();
    private readonly Dictionary<int, char> _idToChar = new();

    public int VocabSize => _charToId.Count;

    public static Tokenizer BuildFromCorpus(IEnumerable<string> texts)
    {
        var t = new Tokenizer();
        int id = 0;
        foreach (var text in texts)
            foreach (var ch in text)
                if (!t._charToId.ContainsKey(ch))
                {
                    t._charToId[ch] = id;
                    t._idToChar[id] = ch;
                    id++;
                }
        return t;
    }

    public int[] Encode(string text) =>
        text.Select(c => _charToId[c]).ToArray();

    public string Decode(int[] ids) =>
        new string(ids.Select(i => _idToChar[i]).ToArray());
}
