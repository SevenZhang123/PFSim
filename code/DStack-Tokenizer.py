f_data = open("../data/train_info.txt", "r", encoding="utf-8")
f_corpus = open("../data/fine-tuning_corpus.txt", "w", encoding="utf-8")

for line in f_data:
    coupus =  line.split("\t")[2]
    f_corpus.write(coupus + "\n")

f_data.close()
f_corpus.close()


f_corpus = open("../data/fine-tuning_corpus.txt", "r", encoding="utf-8")
f_clean = open("../data/fine-tuning_corpus_clean.txt", "w", encoding="utf-8")

def clean_to_printable_ascii(text: str) -> str:
    return ''.join(c for c in text if 32 <= ord(c) <= 126)

for line in f_corpus:
    line = clean_to_printable_ascii(line)
    f_clean.write(line + "\n")

f_corpus.close()
f_clean.close()

from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC
from pathlib import Path

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.BPEDecoder()

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=10000, min_frequency=5, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

training_corpus = ["../data/fine-tuning_corpus_clean.txt"]

tokenizer.train(files=training_corpus, trainer=trainer)

save_dir = "../data/new_tokenizer"
Path(save_dir).mkdir(parents=True, exist_ok=True)
tokenizer.model.save(save_dir)


from transformers import BertTokenizerFast
import json

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_vocab = bert_tokenizer.get_vocab()

with open("../data/new_tokenizer/vocab.json", "r", encoding="utf-8") as f:
    bpe_vocab = json.load(f)
bpe_tokens = list(bpe_vocab.keys())

new_tokens = [tok for tok in bpe_tokens if tok not in bert_vocab]
print(f"New token number: {len(new_tokens)}")

bert_tokenizer.add_tokens(new_tokens)

bert_tokenizer.save_pretrained("../data/expended_tokenizer")