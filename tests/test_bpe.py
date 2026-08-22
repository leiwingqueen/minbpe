# https://huggingface.co/learn/llm-course/en/chapter6/5
# 离线版：不依赖 transformers / 网络，用 minbpe 自带的 GPT-2 切分正则 +
# GPT-2 的 byte-level 编码，复现 tokenizer.backend_tokenizer.pre_tokenizer 的输出。

import os
import sys
from collections import defaultdict

import regex as re

# 支持直接 `python tests/test_bpe.py` 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minbpe.regex import GPT2_SPLIT_PATTERN

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


def bytes_to_unicode():
    """
    GPT-2 的 byte <-> unicode 可逆映射，见
    https://github.com/openai/gpt-2/blob/master/src/encoder.py
    把 256 个字节映射到可打印字符，这样空格变成 'Ġ'、换行变成 'Ċ'。
    """
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_ENCODER = bytes_to_unicode()
compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)


def pre_tokenize_str(text):
    """等价于 HF GPT2 的 ByteLevel pre-tokenizer：返回 [(token, (start, end)), ...]"""
    return [(("".join(BYTE_ENCODER[b] for b in m.group().encode("utf-8"))), m.span())
            for m in compiled_pattern.finditer(text)]


word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)
