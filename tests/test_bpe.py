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

print('word_freqs:', word_freqs)

alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print('alphabet:', alphabet)

vocab = ["<|endoftext|>"] + alphabet.copy()

splits = {word: [c for c in word] for word in word_freqs.keys()}

print('splits:', splits)


def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

# finding the most frequent pair only takes a quick loop:
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print('best_pair and max_freq:', best_pair, max_freq)

merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")


def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits


# start train tokenizer
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

print('merges:', merges)
print('vocab:', vocab)


def tokenize(text):
    pre_tokenize_result = pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])


token_text = tokenize("This is not a token.")
print('token_text:', token_text)
