# Building a domain-specific encoder DStack-BERT

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertTokenizerFast, AdamW
from tqdm import tqdm
import os

# 1. Load the expanded word segmenter and use the pre-trained bert-base-uncased model.
tokenizer = BertTokenizerFast.from_pretrained("../data/expended_tokenizer")
base_model = "bert-base-uncased"

# 2. Custom dataset
class SentencePairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                splits = line.strip().split('\t')
                if len(splits) != 3:
                    continue
                text1, text2, label = splits
                self.samples.append((text1, text2, int(label)))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text1, text2, label = self.samples[idx]
        encoded1 = self.tokenizer(text1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoded2 = self.tokenizer(text2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return encoded1, encoded2, torch.tensor(label, dtype=torch.float)

# 3. Custom Sentence-BERT Model
class SentenceBERT(nn.Module):
    def __init__(self, model_name):
        super(SentenceBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0]
        return cls_embedding

# 4. Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        distance = torch.norm(embedding1 - embedding2, dim=1)
        loss = (1 - label) * 0.5 * torch.pow(distance, 2) + \
               label * 0.5 * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()

# 5. Preparing for training
file_path = "../data/augmented_data_clean.txt"
dataset = SentencePairDataset(file_path, tokenizer)

# Divide the training set and validation set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceBERT(model_name=base_model).to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 6. Early stopping configuration
epochs = 10
best_acc = 0
patience = 2
counter = 0
save_path = "../model"
os.makedirs(save_path, exist_ok=True)

# Accuracy calculation function
def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for encoded1, encoded2, labels in val_loader:
            input_ids1 = encoded1['input_ids'].squeeze(1).to(device)
            attention_mask1 = encoded1['attention_mask'].squeeze(1).to(device)
            input_ids2 = encoded2['input_ids'].squeeze(1).to(device)
            attention_mask2 = encoded2['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            emb1 = model(input_ids1, attention_mask1)
            emb2 = model(input_ids2, attention_mask2)

            distances = torch.norm(emb1 - emb2, dim=1)
            preds = (distances > 0.5).float()  # 距离小认为相似（标签0）
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 7. Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for encoded1, encoded2, labels in loop:
        input_ids1 = encoded1['input_ids'].squeeze(1).to(device)
        attention_mask1 = encoded1['attention_mask'].squeeze(1).to(device)
        input_ids2 = encoded2['input_ids'].squeeze(1).to(device)
        attention_mask2 = encoded2['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        emb1 = model(input_ids1, attention_mask1)
        emb2 = model(input_ids2, attention_mask2)

        loss = criterion(emb1, emb2, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    val_acc = evaluate(model, val_loader)
    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        model.bert.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break
