"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.

-------------------------------------------------------------------------------
练习说明（exercise.md Step 1）
-------------------------------------------------------------------------------
请自己实现下面三个核心方法：train / encode / decode。

背景知识：
- 任何字符串都可以用 text.encode("utf-8") 转成 bytes，每个 byte 是 0..255 的整数，
  所以 id 0..255 天然就是最初的 256 个 token。
- BPE 的做法：反复找出出现次数最多的相邻 token 对 (pair)，把它合并成一个新 token，
  新 token 的 id 从 256 开始依次往上加。合并 num_merges 次后，词表大小就是 vocab_size。

你需要维护两个字典（基类 Tokenizer.__init__ 里已经初始化好了）：
- self.merges: dict[(int, int) -> int]，记录 "哪两个 id 合并成了哪个新 id"，
  插入顺序即合并顺序，encode 时要按这个顺序来。
- self.vocab:  dict[int -> bytes]，记录 "每个 id 对应的原始字节串"，decode 时用。

可以直接使用 base.py 里已经写好的两个工具函数：
- get_stats(ids)          -> {(id1, id2): count}    统计相邻 pair 出现次数
- merge(ids, pair, idx)   -> new_ids                把 ids 中所有 pair 替换成 idx

验证：python -m pytest tests/test_tokenizer.py -k Basic
      或 python train.py（会在 tests/taylorswift.txt 上训练并保存到 models/）
参考答案：git show master:minbpe/basic.py
-------------------------------------------------------------------------------
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        """
        在 text 上训练出 vocab_size 大小的词表。

        实现步骤提示：
        1. assert vocab_size >= 256；num_merges = vocab_size - 256
        2. 把 text 编码成 utf-8 bytes，再转成 list(int)，得到初始的 ids
        3. 初始化 merges = {}，vocab = {idx: bytes([idx]) for idx in range(256)}
        4. 循环 num_merges 次，每次：
           - stats = get_stats(ids) 统计所有相邻 pair 的出现次数
           - 取出现次数最多的 pair（提示：max(stats, key=stats.get)）
           - 新 token 的 id 是 256 + i
           - ids = merge(ids, pair, idx)，把文本里所有该 pair 替换掉
           - 记录 merges[pair] = idx
           - 记录 vocab[idx] = vocab[pair[0]] + vocab[pair[1]]（两段 bytes 拼接）
           - verbose 为 True 时打印一下这次合并的信息，方便观察学到的 token
        5. 把 merges / vocab 存回 self.merges / self.vocab
        """
        min_vocab_size = 256
        assert vocab_size >= min_vocab_size
        # 因为每合并一次，就会增加一个新的vocab
        num_merges = vocab_size - min_vocab_size
        ids = list(text.encode('utf-8'))
        token_id = min_vocab_size
        merges = {}
        vocab = {}
        for i in range(min_vocab_size):
            vocab[i] = bytes([i])
        for i in range(num_merges):
            stats = get_stats(ids)
            # 获取次数最多的pair
            max_count = 0
            max_pair = ()
            for pair, count in stats.items():
                if count > max_count:
                    max_count = count
                    max_pair = (pair[0], pair[1])
            ids = merge(ids, max_pair, token_id)
            merges[max_pair] = token_id
            vocab[token_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            if verbose:
                print(f"add new merge rule:{max_pair}={token_id}.vocab:{token_id}:{vocab[token_id]}")
            token_id += 1
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        """
        ids (list[int]) -> str

        实现步骤提示：
        1. 用 self.vocab 把每个 id 查成 bytes，再 b"".join 拼成完整的字节串
        2. .decode("utf-8", errors="replace") 转回字符串
           （必须带 errors="replace"，因为任意 id 序列拼出来的字节不一定是合法 utf-8）
        """
        raise NotImplementedError("TODO: implement BasicTokenizer.decode")

    def encode(self, text):
        """
        str -> ids (list[int])

        核心：必须严格按照训练时的合并顺序来合并，否则结果和训练不一致。
        merges 里的 value（新 id）越小 = 越早被合并 = 优先级越高。

        实现步骤提示：
        1. text 编码成 utf-8 bytes，转成 list(int)
        2. while len(ids) >= 2:
           - stats = get_stats(ids)
           - 在当前所有 pair 中，挑一个 merges 里"最早出现"的：
             pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
           - 坑点：如果所有 pair 都不在 merges 里，上面的 key 全是 inf，
             min 会随便返回第一个 pair。所以要判断 if pair not in self.merges: break
           - 否则 idx = self.merges[pair]；ids = merge(ids, pair, idx)
        3. 返回 ids
        """
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = get_stats(ids)
            min_token_id = float("inf")
            for pair, _ in stats.items():
                if pair in self.merges

        raise NotImplementedError("TODO: implement BasicTokenizer.encode")
