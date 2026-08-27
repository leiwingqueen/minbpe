"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.

===============================================================================
练习 Step 2：把 BasicTokenizer 改造成 RegexTokenizer
===============================================================================

核心思想（和 BasicTokenizer 的唯一区别）：
    先用正则把文本切成若干 chunk（数字 / 字母 / 标点 / 空白 各成一类），
    每个 chunk 独立做 BPE，chunk 之间【绝不允许】合并。
    这样训练出来的 token 不会跨类别（比如不会出现 "dog." 这种 token）。

数据结构上的变化：
    BasicTokenizer:  ids = [104, 101, ...]              # 一维 list[int]
    RegexTokenizer:  ids = [[104, 101], [32, 119], ...] # 二维 list[list[int]]
                     每个子 list 是一个 chunk 的字节序列

需要你实现的方法（按建议顺序）：
    1. train(text, vocab_size, verbose=False)
    2. decode(ids)
    3. _encode_chunk(text_bytes)      # 和 BasicTokenizer.encode 几乎一样
    4. encode_ordinary(text)          # 切 chunk -> 逐个 _encode_chunk -> 拼接
    5. encode(text, allowed_special)  # Step 4（可选）：特殊 token 处理

可以直接复用 base.py 里的两个工具函数：
    get_stats(ids, counts=None) -> dict[(int,int), int]
        注意第二个参数 counts：传进去会【原地累加】，这正是跨 chunk 统计需要的
    merge(ids, pair, idx) -> list[int]
        把 ids 中所有连续出现的 pair 替换成 idx

自测：
    python -m pytest tests/test_tokenizer.py -k "RegexTokenizer" -v
    # Step 3/4 相关（GPT4Tokenizer）实现完再跑全量
参考答案（实在卡住再看）：
    git show HEAD:minbpe/regex.py
===============================================================================
"""

import regex as re
from .base import Tokenizer, get_stats, merge

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        """
        (Step 2): 训练 BPE。
        和 BasicTokenizer.train 的流程完全一致，只是多了"先切 chunk"这一步。

        步骤提示：
          1. assert vocab_size >= 256; num_merges = vocab_size - 256
          2. 用 re.findall(self.compiled_pattern, text) 把文本切成 text_chunks
             （findall 返回 list[str]，每个元素是一个 chunk 的原文）
          3. ids = [list(ch.encode("utf-8")) for ch in text_chunks]
             注意这里是【二维】的：list[list[int]]
          4. merges = {}                                    # (int,int) -> int
             vocab  = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
          5. 循环 num_merges 次：
             a. stats = {}；对每个 chunk_ids 调 get_stats(chunk_ids, stats)
                —— 把 stats 传进去，让计数在所有 chunk 上【累加】
             b. pair = max(stats, key=stats.get)            # 出现次数最多的 pair
             c. idx = 256 + i                               # 新 token 的 id
             d. ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
                —— 对每个 chunk 分别 merge，chunk 之间不会被合并，这就是关键
             e. merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]   # bytes 拼接
             f. verbose 时打印：merge {i+1}/{num_merges}: {pair} -> {idx} ...
          6. self.merges = merges   # encode 用
             self.vocab  = vocab    # decode 用

        小坑：
          - 不要用 self.vocab = self._build_vocab()，那是给 load() 用的
          - vocab[pair[0]] 是 bytes，相加是 bytes 拼接，不是数字相加
        """
        min_vocab_size = 256
        assert vocab_size >= min_vocab_size
        num_merges = vocab_size - min_vocab_size
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            ids.append(list(chunk.encode("uft-8")))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        token_id = min_vocab_size
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            max_pair = ()
            max_cnt = 0
            for pair, cnt in stats.items():
                if cnt > max_cnt:
                    max_cnt = cnt
                    max_pair = (pair[0], pair[1])
            # python这写法总感觉很骚
            ids = [merge(chunk_ids, max_pair, token_id) for chunk_ids in ids]
            merges[max_pair] = token_id
            vocab[token_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {max_pair} -> {token_id} ({vocab[token_id]}) had {stats[max_pair]} occurrences")
            token_id += 1
        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """
        TODO(Step 2): given ids (list of integers), return Python string

        步骤提示：
          1. 遍历 ids，逐个 idx 取出对应的 bytes 片段，收集到 part_bytes 列表：
             - if idx in self.vocab:                 -> self.vocab[idx]
             - elif idx in self.inverse_special_tokens:  # Step 4 用得上
                   -> self.inverse_special_tokens[idx].encode("utf-8")
             - else: raise ValueError(f"invalid token id: {idx}")
          2. text_bytes = b"".join(part_bytes)
          3. return text_bytes.decode("utf-8", errors="replace")

        小坑：为什么必须 errors="replace"？
          因为单个 token 可能是某个多字节 UTF-8 字符的一半，直接 decode 会抛异常。
        """
        raise NotImplementedError("TODO Step 2: implement RegexTokenizer.decode")

    def _encode_chunk(self, text_bytes):
        """
        TODO(Step 2): 对【单个 chunk 的字节串】做 BPE 编码，返回 token ids。
        这部分和 BasicTokenizer.encode 的循环体一模一样，可以直接搬过来。

        步骤提示：
          1. ids = list(text_bytes)                 # 0..255 的整数
          2. while len(ids) >= 2:
               stats = get_stats(ids)
               pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
               # 关键：按 merges 里的先后顺序（merge index 最小）来合并，
               #      必须和训练时的顺序一致，否则结果对不上
               if pair not in self.merges: break   # 已经没有可合并的了
               ids = merge(ids, pair, self.merges[pair])
          3. return ids

        小坑：min 在没有任何可用 merge 时，所有 pair 的 key 都是 inf，
              它会随便返回第一个 pair —— 所以必须靠 `pair not in self.merges` 跳出。
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            min_pair = ()
            min_token_id = float("inf")
            for pair, _ in stats.items():
                if pair in self.merges and self.merges[pair] < min_token_id:
                    min_token_id = self.merges[pair]
                    min_pair = (pair[0], pair[1])
            if len(min_pair) == 0:
                break
            ids = merge(ids, min_pair, min_token_id)
        return ids

    def encode_ordinary(self, text):
        """
        (Step 2): Encoding that ignores any special tokens.

        步骤提示：
          1. text_chunks = re.findall(self.compiled_pattern, text)
          2. 对每个 chunk：chunk.encode("utf-8") -> self._encode_chunk(...)
          3. 把各个 chunk 的结果依次 extend 到一个 ids 列表里返回
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk.encode("utf-8"))
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun

        TODO(Step 4，可选)：先做 Step 2 时，可以直接 `return self.encode_ordinary(text)`
        跑通测试，等做 Step 4 再回来补完整逻辑。

        步骤提示：
          1. 解析 allowed_special，得到本次生效的 special 字典：
             - "all"        -> self.special_tokens
             - "none"       -> {}
             - "none_raise" -> {}，并 assert 文本里不含任何 special token
             - set          -> self.special_tokens 中 key 在该 set 里的子集
             - 其他         -> raise ValueError
          2. if not special: return self.encode_ordinary(text)   # 快路径
          3. 用 special token 把文本切开，且保留分隔符本身：
             special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
             special_chunks = re.split(special_pattern, text)
             —— 用括号做捕获组，re.split 才会把 special token 一起返回
          4. 遍历 special_chunks：
             - part in special -> ids.append(special[part])      # 直接用它的 id
             - 否则            -> ids.extend(self.encode_ordinary(part))
        """
        raise NotImplementedError("TODO Step 2/4: implement RegexTokenizer.encode")
