# 构建领域特定的DStack-Tokenizer分词器

# 1. 基于train_info_month4.txt构建语料，保存至文件fine-tuning_corpus.txt

f_data = open("../data/train_info.txt", "r", encoding="utf-8") # D_T
f_corpus = open("../data/fine-tuning_corpus.txt", "w", encoding="utf-8")

for line in f_data:
    coupus =  line.split("\t")[2]
    f_corpus.write(coupus + "\n")

f_data.close()
f_corpus.close()


# 2. 对训练语料进行清洗，仅保留ASCII可打印字符
f_corpus = open("../data/fine-tuning_corpus.txt", "r", encoding="utf-8")
f_clean = open("../data/fine-tuning_corpus_clean.txt", "w", encoding="utf-8")

def clean_to_printable_ascii(text: str) -> str:
    # 保留ASCII范围内32~126的字符
    return ''.join(c for c in text if 32 <= ord(c) <= 126)

for line in f_corpus:
    line = clean_to_printable_ascii(line)
    f_clean.write(line + "\n")

f_corpus.close()
f_clean.close()

# 3. 基于BPE算法训练分词器，使用fine-tuning_corpus_clean.txt作为训练语料构建新的token

from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC
from pathlib import Path

# 新建 tokenizer 实例，使用 BPE 模型
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.BPEDecoder()

# 添加特殊符号
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=10000, min_frequency=5, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

# 用于扩展训练的领域数据（纯文本文件）
training_corpus = ["../data/fine-tuning_corpus_clean.txt"]

# 训练 tokenizer
tokenizer.train(files=training_corpus, trainer=trainer)

# 保存 tokenizer 文件
save_dir = "../data/new_tokenizer"
Path(save_dir).mkdir(parents=True, exist_ok=True)
tokenizer.model.save(save_dir)



# 4. 将新构建的token扩展至BERT现有的tokenizer中
from transformers import BertTokenizerFast
import json

# 原始 BERT tokenizer（用于初始化）
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_vocab = bert_tokenizer.get_vocab()

with open("../data/new_tokenizer/vocab.json", "r", encoding="utf-8") as f:
    bpe_vocab = json.load(f)
bpe_tokens = list(bpe_vocab.keys())

new_tokens = [tok for tok in bpe_tokens if tok not in bert_vocab]
print(f"新 token 数量: {len(new_tokens)}")

bert_tokenizer.add_tokens(new_tokens)

bert_tokenizer.save_pretrained("../data/expended_tokenizer")