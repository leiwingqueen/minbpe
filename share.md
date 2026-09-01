# nn zero to hero系列-tokenizer实现



## 00 · 前言

近几年基于大模型的各种智能体和工具百花齐放，在感慨大模型发展迅猛的同时，又感慨工具的发展过于迅猛，有一种一个工具还没学透，新的工具又冒出来了。我个人倾向于学习一些比较底层的原理，一是这方面的原理一般不会轻易变化，另外一方面也方便让我理解上层的一些设计。最近看完karpathy大神的课程发现不少的内容还是会比较模糊，计划跟着教程从零开始训练一个LLM，那么第一篇就是先尝试实现一个tokenizer。



> 手写 BPE 分词器的分享记录 —— 两个分支、四个方法、以及一路踩到的坑。
>
> - 仓库：[leiwingqueen/minbpe](https://github.com/leiwingqueen/minbpe) （原仓库 https://github.com/karpathy/minbpe）
> - 实现分支：`exercise/basic-tokenizer`、`exercise/regex-tokenizer`
> - 测试：22 passed
> - 在线 BPE 实验台：https://claude.ai/code/artifact/d8e4605f-dbb1-4b30-b770-40d9886ef307

| | |
|---|---|
| **256** | 起始词表大小，就是全部字节值 |
| **~40** | BPE 核心代码行数（train + encode + decode） |
| **4.10×** | 压缩率：185,768 字节 → 45,338 token |
| **127×** | encode 提速，切 chunk 带来的意外收获 |

---

## 01 · 为什么值得花时间讲分词器

分词器是模型之外的一个独立模块：它有自己的训练数据、自己的训练过程，和神经网络完全解耦。但很多被归咎于「模型不够聪明」的现象，根子其实在这里。

GPT-4 (cl100k_base) 的实际切分：

```
"1234567890"  ->  123 | 456 | 789 | 0
"你好，世界"    ->  你 | 好 | ， | � | � | 界
```

数字被切成三位一组，所以模型做加法时看到的根本不是逐位对齐的数。「世」这个字被拆成两个不完整的字节 token —— 它在模型眼里甚至不是一个完整字符。

| 文本 | 字符 | UTF-8 字节 | Token | 每字符成本 |
|---|---:|---:|---:|---:|
| `hello world` | 11 | 11 | 2 | 0.18 |
| `你好，世界` | 5 | 15 | 6 | **1.20** |
| `人工智能大模型` | 7 | 21 | 8 | **1.14** |
| `1234567890` | 10 | 10 | 4 | 0.40 |

同样一句话，中文的 token 成本是英文的六倍以上。上下文窗口、API 计费、有效记忆长度，全部按 token 算。

> 分词器不是预处理的琐事，它是模型和世界之间唯一的接口。接口设计得糟，后面再大的网络也补不回来。

---

## 02 · 核心思想：在字节上做 BPE

字符级分词词表太小、序列太长；词级分词又必然遇到未登录词。BPE 走中间路线，而「字节级」解决了 Unicode 的开放性问题 —— **任何字符串 UTF-8 编码后都是 0–255 的整数序列，所以 id 0…255 天然就是最初的 256 个 token，永远不会有 OOV。**

剩下的事就是反复做一件事：找出出现次数最多的相邻 token 对，把它合并成一个新 token，新 id 从 256 开始递增。合并 `vocab_size - 256` 次，训练就结束了。

### 只需要两个字典

| 字典 | 类型 | 作用 |
|---|---|---|
| `merges` | `(int, int) -> int` | 「哪两个 id 合并成了哪个新 id」。**插入顺序即合并顺序**，encode 时必须按这个顺序重放。给 encode 用。 |
| `vocab` | `int -> bytes` | 「每个 id 对应哪一段原始字节」。由 merges 完全确定，`vocab[idx] = vocab[p0] + vocab[p1]` 是字节拼接。给 decode 用。 |

train / encode / decode 三个方法围绕这两个字典展开，加上 `base.py` 里现成的两个工具函数 `get_stats`（统计相邻对频次）和 `merge`（把所有 pair 替换成新 id），核心逻辑就只剩四十行。

```python
# BasicTokenizer.train —— 去掉注释后的骨架
num_merges = vocab_size - 256
ids = list(text.encode('utf-8'))
vocab = {i: bytes([i]) for i in range(256)}

for i in range(num_merges):
    stats = get_stats(ids)              # {(id1,id2): count}
    max_pair = … 取 count 最大的 pair …
    ids = merge(ids, max_pair, token_id)          # 全文替换
    merges[max_pair] = token_id
    vocab[token_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
    token_id += 1
```

> `minbpe/basic.py` · `exercise/basic-tokenizer`

---

## 03 · 手推一遍：`aaabdaaabac`

这是 Wikipedia 上的经典例子，也是仓库里的单元测试。a=97, b=98, c=99, d=100（ASCII），做三次合并：

| 步 | 最高频 pair | 次数 | 新 id | 合并后的序列 |
|---:|---|---:|---:|---|
| — | — | — | — | `97 97 97 98 100 97 97 97 98 97 99` |
| 1 | `(97, 97)` = `aa` | 4 | 256 | `256 97 98 100 256 97 98 97 99` |
| 2 | `(256, 97)` = `aaa` | 2 | 257 | `257 98 100 257 98 97 99` |
| 3 | `(257, 98)` = `aaab` | 2 | 258 | **`258 100 258 97 99`** |

注意第 3 步：合并出的新 token 自己又参与了下一轮统计。这就是 BPE 能长出长 token 的原因 —— 它是递归的。

值得强调的是，训练产物只有 merges 这张有序的规则表。**词表是规则表的副产品，不是独立存在的东西** —— 这也是为什么 `save()` 只需要写 merges，`load()` 再用 `_build_vocab()` 重建 vocab。

---

## 04 · 实验台

做了一个可交互的实验台：输入文本 → 正则切分 → BPE 合并 → token 序列，可以拖动合并次数、开关正则切分、显示 token id。

**https://claude.ai/code/artifact/d8e4605f-dbb1-4b30-b770-40d9886ef307**

三件值得试的事：

- 英文段落拉到 12 次合并，看 `·the` `·cat` `·sat` 怎么长出来 —— 空格是被合并进词里的，不是分隔符。
- 关掉「启用正则切分」再把次数拉高，token 会开始跨越单词边界，长出带标点的怪东西。
- 切到中文，满屏的 `�` 不是 bug —— 每个 token 只是一个字节，单独解码就是半个汉字。这就是第 05 节坑 3 的样子。

本地想跑同样的东西：

```bash
python train.py     # 在 tests/taylorswift.txt 上训练，输出 models/*.vocab
```

---

## 05 · 我踩的四个坑

### 坑 1 · encode 必须按合并顺序，不是按频次

train 里挑的是**出现次数最多**的 pair，encode 里挑的却必须是 **merges 中最早出现**的 pair（新 id 最小 = 优先级最高）。两者规则不同，很容易顺手写成一样的。一旦顺序对不上，encode 的结果就和训练时的分布不一致，模型收到的是它没见过的 token 组合。

```python
# train:  谁最频繁
max_pair = max(stats, key=stats.get)
# encode: 谁最早被学会
min_pair = min(stats, key=lambda p: merges.get(p, float("inf")))
```

### 坑 2 · `min()` 在全是 inf 时会返回第一个 pair

用 `min(..., key=lambda p: merges.get(p, float("inf")))` 时，如果当前所有 pair 都不在 merges 里，key 全部相等，`min` 会静默返回第一个 pair，然后你就用一个不存在的规则去 merge 了。**必须显式判断 `if pair not in merges: break`。**

我的写法是手写循环、只在命中 merges 时更新候选，最后靠 `if len(min_pair) == 0: break` 跳出，等价且更难写错。

### 坑 3 · decode 必须带 `errors="replace"`

单个 token 完全可能是某个多字节 UTF-8 字符的一半（前面「世」被拆成两半就是活例子）。任意 id 序列拼出来的字节串不保证是合法 UTF-8，直接 `decode("utf-8")` 会抛 `UnicodeDecodeError`。

这也解释了为什么 `.vocab` 文件只能给人看、不能拿来 `load()` —— 渲染是有损的。

### 坑 4 · train 里别用 `self._build_vocab()`

`_build_vocab()` 是给 `load()` 用的，它从已有的 merges 重建。train 过程中要自己维护 vocab 的增量拼接。

另外 `vocab[pair[0]] + vocab[pair[1]]` 是 **bytes 拼接**不是数字相加 —— 类型对了，语义才对。

---

## 06 · BasicTokenizer 的病：跨类别 token

在 `tests/taylorswift.txt`（185,768 字节的维基百科条目）上训练 512 大小的词表，前 20 个合并规则里就已经能闻到味道：

```
'e ' ', ' 'd ' '. ' 'r ' '20' 's ' 'in' 'on' 'ri'
't ' 'th' 'ed ' ', 20' 'an' 'ar' 'er ' 'y ' 'al' 'the '
```

最高频的合并全都是「字母 + 空格」和「标点 + 空格」。BPE 只认统计，它不知道空格是词的边界。继续训练下去，词表里会长出这种东西（512 词表中共 **32 个**跨类别 token）：

```
'. Retrieved '    'Archived from the original on '    'Taylor Swift '
', 2023'          'from the original '                '\'s "'
'. Re'            '), "'                              ', 201'
'Taylor Swif'     's, '                               ', 2012'
```

这些 token 在语言学上毫无意义，纯粹是这份语料的统计伪影。危害是双重的：**宝贵的词表槽位被浪费**，而且 `Archived from the original on ` 这种 token 一旦被学会，模型在其他上下文里看到「Archived」时就得走一条完全不同的编码路径。同一个词有多种切法，语义就被打散了。

> **更本质的问题：** 模型对 `dog`、`dog.`、`dog!`、` dog` 会学出四份独立的表示。它们本该共享同一个词根的语义。

---

## 07 · RegexTokenizer：先划好边界，再让 BPE 干活

GPT-2 论文给出的解法很直接：**先用正则把文本切成若干 chunk，每个 chunk 独立做 BPE，chunk 之间绝不允许合并。** 这是把语言学先验硬编码进预处理，而不是指望统计自己发现它。

```python
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

看着吓人，拆开就是七条按优先级排列的规则：

| 规则 | 匹配什么 | 例子 |
|---|---|---|
| `'(?i:[sdmt]\|ll\|ve\|re)` | 英文缩写后缀，不区分大小写 | `I` + `'ve` / `don` + `'t` |
| `[^\r\n\p{L}\p{N}]?+\p{L}+` | 一串字母，可带一个前导非字母字符（通常是空格） | `" world"` |
| `\p{N}{1,3}` | 数字，**最多三位** | `123` `456` `789` `0` |
| ` ?[^\s\p{L}\p{N}]++[\r\n]*` | 一串标点符号 | `"!!!?"` |
| `\s*[\r\n]` | 换行（连同前面的空白） | `"  \n"` |
| `\s+(?!\S)` | 结尾空白：把最后一个空格留给下一个词 | `"  "` + `" world"` |
| `\s+` | 兜底的空白 | `" "` |

`\p{N}{1,3}` 是刻意的：把数字硬性切成一到三位，模型至少能看到一致的数字分块方式。`\s+(?!\S)` 这条负向先行断言也很妙 —— 它保证一串空格里的**最后一个**空格会被留下来，附到后面那个词的前面，于是 ` world` 才能成为一个 token。

实测切分效果：

```
"hello world!!!? (안녕하세요!) lol123 😉"
  -> ['hello', ' world', '!!!?', ' (', '안녕하세요', '!)', ' lol', '123', ' 😉']
"I've don't"     -> ['I', "'ve", ' don', "'t"]
"hello   world"  -> ['hello', '  ', ' world']
```

### 实现上只有两处变化

算法本身一个字都不用改，只是数据结构从一维变成二维：

```python
# BasicTokenizer
ids = [104, 101, 108, 108, 111, ...]              # list[int]

# RegexTokenizer
ids = [[104, 101, 108, 108, 111], [32, 119, ...]] # list[list[int]]

# 统计时跨 chunk 累加，合并时逐 chunk 进行
stats = {}
for chunk_ids in ids:
    get_stats(chunk_ids, stats)        # 第二个参数原地累加
ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
```

> `minbpe/regex.py` · `exercise/regex-tokenizer`

`get_stats(ids, counts)` 的第二个参数就是为这个场景准备的：传一个字典进去让它原地累加，就能在全部 chunk 上得到全局频次；而 `merge` 逐 chunk 调用，物理上保证了跨边界的 pair 根本不会进入统计。

### 效果

| 512 词表 · taylorswift.txt | 跨类别 token | 压缩率 | 前 5 个 merge |
|---|---:|---:|---|
| BasicTokenizer | **32** | 2.36× | `'e '` `', '` `'d '` `'. '` `'r '` |
| RegexTokenizer | **1** | 2.13× | `'er'` `'20'` `'or'` `'in'` `'ed'` |

唯一剩下的那个「跨类别」token 是 `'s`，来自 pattern 第一条规则 —— 那是设计者故意保留的。

> **压缩率反而降了，这是对的。** 正则切分是在牺牲纯粹的压缩效率，换取 token 的语义一致性。BPE 的目标从来不是最优压缩，而是学出对下游语言模型最有用的切分单元。

---

## 08 · 一个意外收获：encode 快了 127 倍

写完两版之后跑单元测试，我发现 basic 分支的测试要跑 33 秒，regex 分支只要 5 秒。定位下去，慢的全在 `encode` 上。

| 词表 | Basic train | Basic encode | Regex train | Regex encode |
|---|---:|---:|---:|---:|
| 512 | 3.03s | 3.13s | 4.76s | **0.11s** |
| 1024 | 7.41s | 7.93s | — | — |
| 2048 | 15.55s | **16.49s** | 25.65s | **0.13s** |

> 全文 185,768 字节，Apple Silicon MacBook。Basic 的 encode 时间随词表线性增长，Regex 的几乎是常数。

原因在于 encode 的循环结构。`BasicTokenizer.encode` 每完成一次合并，就要对**整篇文档**重新 `get_stats` 一遍，复杂度是 `O(合并次数 × 文本长度)` —— 词表翻倍，encode 时间就翻倍。

`RegexTokenizer` 是在每个 chunk 内部跑同一个循环，而 chunk 平均只有几个字节长。内层循环的次数被 chunk 长度而不是文档长度限制住了，于是总耗时退化成对文本的一次线性扫描。

> 正则切分的初衷是提升 token 质量，性能是白送的。这类「约束反而带来加速」的情况在工程里很常见 —— 把大问题切成互不影响的小问题，往往顺手就把复杂度降下来了。

---

## 09 · Special token：一个安全问题

`<|endoftext|>` 这类特殊 token 不参与 BPE，它们在 encode 前就被切出来、直接映射到预留 id（GPT-4 里是 100257 起）。实现是用捕获组做 split，这样分隔符本身会被保留在结果里：

```python
special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
special_chunks = re.split(special_pattern, text)   # 括号 = 捕获组，分隔符会被保留

for part in special_chunks:
    if part in special:
        ids.append(special[part])          # 直接查表
    else:
        ids.extend(self.encode_ordinary(part))
```

真正值得讲的是 `allowed_special` 这个参数，默认值是 `"none_raise"` —— 文本里出现任何特殊 token 就直接断言失败。tiktoken 也是这个行为，看着烦人，但它防的是一类真实攻击：

> **为什么默认要报错**
>
> 如果用户输入里的 `<|endoftext|>` 被当成真的分隔符编码，攻击者就能在你的 prompt 中间伪造一个「文档结束」信号，从而越过前面的系统指令。这是 prompt injection 的一种底层形式。**要解析特殊 token，必须显式声明意图** —— 安全默认值应该让危险的事变难，而不是让它悄悄发生。

---

## 10 · 对齐 GPT-4：两个历史包袱

练习的第三步是加载 GPT-4 的 merges，让自己的实现和 tiktoken 逐 token 对齐。这一步我直接用了仓库里现成的 `gpt4.py`，但两个坑本身很有意思：

**merges 得反推出来。** tiktoken 只存了 `_mergeable_ranks`（每个 token 的字节串 → rank），没存「谁和谁合并成了它」。`recover_merges` 的思路是：对每个 token，枚举所有切分点，找出那个「两半的 rank 都比自己小」的唯一切法。父节点加 rank 足以还原整棵合并树。

**字节被打乱过。** GPT-4 把最初 256 个字节做了一次置换，原因不明，大概率是历史遗留。`byte_shuffle = {i: ranks[bytes([i])] for i in range(256)}` 可以恢复它，encode 和 decode 两头都得记得转换。

这一节的教训不是技术性的：**生产环境的分词器格式充满了不可推导的历史决策。** 想复现别人的分词器，光懂算法不够，还得读对方的代码。

---

## 11 · 回头看那些「模型的怪癖」

| 现象 | 分词器层面的解释 |
|---|---|
| 数不清 strawberry 里有几个 r | 模型看到的是几个 token，不是字母序列。token 内部的字符构成对它是不透明的。 |
| 算术容易错位 | `\p{N}{1,3}` 把数字按三位切块，不同长度的数字对齐方式完全不同。 |
| 中文/日文成本高 | 训练语料以英文为主，非拉丁字符没有被合并成长 token，一个字往往要 2–3 个 token。 |
| Python 缩进敏感 | 连续空格的切分方式取决于数量。GPT-2 时代每个空格一个 token，GPT-4 才把空白折叠成块。 |
| 某些罕见词让模型行为异常 | 分词器语料里出现过、但语言模型训练时几乎没见过的 token（如著名的 `SolidGoldMagikarp`），其 embedding 基本没被训练过。 |

> 这些都不是「模型不够聪明」。它们是接口层的物理限制，向上传导到了行为层。

---

## 12 · 自己练一遍

两个分支都保留了带中文注释的练习骨架 —— 核心方法被挖空，注释里写清了每一步该做什么和会踩什么坑。

```bash
# Step 1 — BasicTokenizer：train / encode / decode
git checkout exercise/basic-tokenizer
pytest tests/test_tokenizer.py -k Basic

# Step 2 — RegexTokenizer：正则切分 + 二维 ids + special token
git checkout exercise/regex-tokenizer
pytest tests/test_tokenizer.py -q          # 22 passed

# 看看学到的词表长什么样
python train.py                            # -> models/basic.vocab, models/regex.vocab
```

推荐顺序是先只做 Step 1 和 Step 2，把 `encode(decode(x)) == x` 跑通，再回头看 `models/*.vocab` 里逐行的合并树。那个文件比任何解释都直观。Step 3 的 GPT-4 对齐可以留到最后 —— 它更多是考古而不是算法。

> **一句话总结：** BPE 本身简单到四十行就能写完，难的是理解它的每一个约束（合并顺序、类别边界、字节安全）分别在防什么。写完这两个分支，再看 LLM 的各种「怪行为」，很多都不神秘了。

---

*所有数据实测于 `tests/taylorswift.txt`（185,561 字符 / 185,768 字节），Apple Silicon MacBook。*
*参考：GPT-2 paper · Sennrich et al. 2015 · tiktoken*
