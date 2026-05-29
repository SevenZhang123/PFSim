import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertModel
import math
from collections import defaultdict
import time
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("../model/expended_tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("../model")
# model = BertModel.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(len(tokenizer))
model.eval()



def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return embedding.squeeze(0)  # [hidden_size]


def euclidean_distance(vec1, vec2):
    return torch.norm(vec1 - vec2, p=2).item()


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


def is_same_host(ipv4_fp: dict, ipv6_fp: dict, label: int, threshold: float = 0.05) -> int:
    ports4 = set(ipv4_fp.keys())
    ports6 = set(ipv6_fp.keys())
    common_ports = ports4 & ports6

    jaccard = jaccard_similarity(ports4, ports6)
    penalty = math.log(1 + 1 / (jaccard + 0.001))

    distances = []
    for port in common_ports:
        emb4 = get_embedding(ipv4_fp[port])
        emb6 = get_embedding(ipv6_fp[port])
        dist = torch.norm(emb4 - emb6, dim=0)
        distances.append(dist)

    avg_dist = np.mean(distances)
    score = avg_dist * penalty

    return 1 if score <= threshold else 0


f_ground = open("../data/test.txt", "r", encoding="utf-8") # Test dataset
f_service = open("../data/test_info.txt", "r", encoding="utf-8")
ip_fingerprint = defaultdict(dict)
sample_pairs_list = []


true_samples = []
pred_samples = []
time_costs = []

for line in f_service:
    if line[-1] == "\n":
        line = line[:-1]
    ip = line.split("\t")[0]
    port = int(line.split("\t")[1])
    service = line.split("\t")[2]
    ip_fingerprint[ip][port] = service
f_service.close()

for line in f_ground:
    if line[-1] == "\n":
        line = line[:-1]
    sample_pairs_list.append(line)
f_ground.close()

for sample in tqdm(sample_pairs_list):
    ipv4 = sample.split("\t")[0]
    ipv6 = sample.split("\t")[1]
    label = int(sample.split("\t")[2])
    ipv4_fingerprint = ip_fingerprint[ipv4]
    ipv6_fingerprint = ip_fingerprint[ipv6]

    start_time = time.time()
    result = is_same_host(ipv4_fingerprint, ipv6_fingerprint, label, threshold=0.05)

    end_time = time.time()
    cost_ms = (end_time - start_time) * 1000
    time_costs.append(cost_ms)

    true_samples.append(label)
    pred_samples.append(result)


acc = accuracy_score(true_samples, pred_samples)
recall = recall_score(true_samples, pred_samples, average='binary')
f1 = f1_score(true_samples, pred_samples, average='binary')
avg_time_cost = np.mean(time_costs) if time_costs else 0.0

print("Accuracy:{0}. Recall:{1}. F1:{2}".format(acc, recall, f1))
print("Average Time Overhead per Pair: {0:.2f} ms".format(avg_time_cost))



