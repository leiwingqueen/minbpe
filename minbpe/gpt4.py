"""
Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
Note that this is a pretrained tokenizer. By default and inside init(), it
loads the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken.

===============================================================================
练习 Step 3：加载 GPT-4 的 merges，让自己的 tokenizer 和 tiktoken 完全对齐
===============================================================================

和前两步的最大区别：
    Step 1/2 是【自己训练】出 merges；
    Step 3 是【直接加载】OpenAI 训练好的 merges，一个字都不能差，
    encode/decode 的结果必须和 tiktoken 逐 token 相同。

所以 GPT4Tokenizer 本质上只是 RegexTokenizer 的一层薄壳：
    - 切分规则：复用 GPT4_SPLIT_PATTERN（Step 2 已经用过）
    - BPE 逻辑：完全复用父类 RegexTokenizer 的 _encode_chunk / encode
    - 你要做的只有两件事：把 merges/vocab 灌进去 + 处理字节置换

会踩的两个坑（exercise.md 里也提到了）：

坑 1：tiktoken 不直接给你 merges，只给你 `enc._mergeable_ranks`
    mergeable_ranks 是 bytes -> rank(int) 的字典，比如 b"in" -> 258。
    它只记录了【合并后的结果】，没记录“它是由哪两个 token 合并来的”。
    恢复办法：对每个 token，在【只允许使用 rank 更小的合并】的前提下
    重跑一遍 BPE，最后剩下的两块就是它的父节点。
    这就是下面 bpe() + recover_merges() 干的事 —— 这两个函数
    exercise.md 明确说了可以直接抄，本文件已经保留完整实现，
    读懂即可，不要求你重写。

坑 2：GPT-4 把 0..255 这 256 个单字节 token 的顺序打乱了
    也就是字节 b 对应的 token id 不是 b，而是 mergeable_ranks[bytes([b])]。
    这纯粹是历史遗留，没有任何道理，但你必须照做：
        byte_shuffle         = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        inverse_byte_shuffle = {v: k for k, v in byte_shuffle.items()}
    encode 时：原始字节 -> 先过 byte_shuffle -> 再做 BPE
    decode 时：BPE 还原成字节 -> 再过 inverse_byte_shuffle -> 才能 utf-8 解码
    两个方向必须成对出现，漏一个就会得到乱码。

需要你实现的方法：
    1. __init__(self)                 # 灌 merges / vocab / byte_shuffle / special
    2. _encode_chunk(self, text_bytes)  # 编码方向的字节置换
    3. decode(self, ids)                # 解码方向的字节反置换

不需要实现的：
    train / save / load 都是 raise NotImplementedError —— 这是个预训练
    tokenizer，不支持再训练；save/load 因为多了 byte_shuffle 也不好存。
    save_vocab 已保留实现，只是给你 dump 出词表看看，顺便可以参考它
    是怎么用 inverse_byte_shuffle 重建词表的。

自测：
    python -m pytest tests/test_tokenizer.py -k "gpt4 or GPT4" -v
    # 全部做完再跑全量：python -m pytest tests/ -v
参考答案（实在卡住再看）：
    git show master:minbpe/gpt4.py
===============================================================================
"""

import tiktoken
from .regex import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    # 思路：把 token 拆成单字节，然后反复找“rank 最小且 < max_rank”的相邻 pair 合并。
    # max_rank 就是 token 自己的 rank —— 只允许用比它更早（更小 rank）的合并，
    # 这样最后剩下的两块，就一定是它的直接父节点。
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    # 返回值形状和你自己训练出来的 merges 一致：{(int, int): int}
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self):
        """
        TODO(Step 3): 从 tiktoken 加载 cl100k_base，把状态灌进本对象。

        步骤提示：
          1. super().__init__(pattern=GPT4_SPLIT_PATTERN)
             —— 必须先调父类构造，它会准备 compiled_pattern / special_tokens 等
          2. enc = tiktoken.get_encoding("cl100k_base")
             mergeable_ranks = enc._mergeable_ranks      # bytes -> int
          3. self.merges = recover_merges(mergeable_ranks)   # 见上面的坑 1
          4. 由 merges 重建 vocab（顺序很重要，merges 是按 rank 递增排列的，
             所以下面这样顺序遍历时，pair 的两个 id 一定已经在 vocab 里了）：
               vocab = {idx: bytes([idx]) for idx in range(256)}
               for (p0, p1), idx in self.merges.items():
                   vocab[idx] = vocab[p0] + vocab[p1]
               self.vocab = vocab
          5. 建立字节置换表（见上面的坑 2）：
               self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
               self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
          6. self.register_special_tokens(GPT4_SPECIAL_TOKENS)
             —— 父类的方法，会顺带把 inverse_special_tokens 也建好

        小坑：
          - 第 4 步的 vocab 是【置换后空间】的词表：vocab[i] == bytes([i])，
            它只在 BPE 内部用；真正还原成原文的那一步在 decode 里做反置换。
          - _mergeable_ranks 是 tiktoken 的私有属性，跨版本可能改名，
            但这里就是拿它，没有公开 API。
        """
        raise NotImplementedError("TODO Step 3: implement GPT4Tokenizer.__init__")

    def _encode_chunk(self, text_bytes):
        """
        TODO(Step 3): 编码单个 chunk，比父类多一步“字节置换”。

        步骤提示：
          1. text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
             —— 先把每个原始字节映射成 GPT-4 给它分配的 token id
          2. ids = super()._encode_chunk(text_bytes)
             —— 剩下的 BPE 合并逻辑完全复用 Step 2 写好的父类实现
          3. return ids

        为什么置换后还能塞回 bytes？
          因为 0..255 的置换仍然落在 0..255 里，是个双射，
          所以结果依然是一个合法的 bytes 对象。
        """
        raise NotImplementedError("TODO Step 3: implement GPT4Tokenizer._encode_chunk")

    def decode(self, ids):
        """
        TODO(Step 3): 解码，比父类多一步“字节反置换”。

        步骤提示：
          1. text_bytes = b"".join(self.vocab[idx] for idx in ids)
          2. text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
             —— 把置换空间里的字节还原成真实字节，顺序不能和上一步反过来
          3. return text_bytes.decode("utf-8", errors="replace")

        小坑：
          - 这里没有走父类 decode，因为父类不知道 byte_shuffle 的存在。
          - errors="replace" 的理由和 Step 2 一样：单个 token 可能是某个
            多字节 UTF-8 字符的一半。
        """
        raise NotImplementedError("TODO Step 3: implement GPT4Tokenizer.decode")

    # this is a pretrained tokenizer, it is not intended to be trained
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    # save/load would require some thought.
    # we'd have to change save/load of base to add support for byte_shuffle...
    # alternatively, we could move byte_shuffle to base class, but that would
    # mean that we're making ugly our beautiful Tokenizer just to support
    # the GPT-4 tokenizer and its weird historical quirks around byte_shuffle.
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file):
        # just for visualization purposes let's output the GPT-4 tokens
        # in the exact same format as the base class would.
        # simple run as:
        # python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')"
        from .base import render_token
        # build vocab being mindful of the byte shuffle
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # now merge the shuffled bytes and write to file
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
