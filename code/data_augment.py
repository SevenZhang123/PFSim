import random
import re
from tqdm import tqdm

service_names = set()
tls_ciphers = set()
tls_versions = {"TLSv1_1", "TLSv1_2", "TLSv1_3"}
vector_list = set()
product_list = set()
version_list = set()

positive_pairs = []
negative_pairs = []


def parse_fields(service_info):
    fields = dict()
    for item in service_info.split("&"):
        if "=" in item:
            k, v = item.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def modify_banner(banner):
    words = re.split(r'[-_ /,:;]', banner)
    # print(words)
    # words = banner.split()
    if not words:
        return banner
    for _ in range(random.randint(1, min(len(words), 3))):
        if len(words) == 1:
            idx = 0
        else:
            idx = random.randint(0, len(words) - 1)
        op = random.choice(["delete", "replace", "char_delete", "char_replace"])
        if op == "delete":
            words.pop(idx)
        elif op == "replace":
            words[idx] = random.choice(["unknown", "error", "OK", "wer", "vaild", "good", "zty"])
        elif op == "char_delete":
            if len(words[idx]) > 1:
                pos = random.randint(0, len(words[idx]) - 1)
                words[idx] = words[idx][:pos] + words[idx][pos+1:]
        elif op == "char_replace":
            if words[idx]:
                pos = random.randint(0, len(words[idx]) - 1)
                words[idx] = words[idx][:pos] + random.choice("xyz123") + words[idx][pos+1:]
    return " ".join(words)


def replace_software_field(software):
    if not software:
        return software
    softwares = software.split(",")
    new_softwares = []
    for s in softwares:
        parts = s.strip().split()
        if not parts:
            continue
        idx = random.choice([0, 1, 2])
        if idx == 0 and len(parts) >= 1:
            new_val = random.choice(list(vector_list - {parts[0]}))
            parts[0] = new_val
        elif idx == 1 and len(parts) >= 2:
            new_val = random.choice(list(product_list - {parts[1]}))
            parts[1] = new_val
        elif idx == 2 and len(parts) >= 3:
            new_val = random.choice(list(version_list - {parts[2]}))
            parts[2] = new_val
        new_softwares.append(" ".join(parts))
    return ",".join(new_softwares)


def delete_random_field(fields):
    choices = [k for k in fields.keys() if k in {'software', 'banner', 'tls_version', 'tls_cipher'}]
    if not choices:
        return fields
    field_to_delete = random.choice(choices)
    if field_to_delete == 'banner':
        banner = fields.get('banner', '')
        fields['banner'] = modify_banner(banner)
    elif field_to_delete == 'software':
        software = fields.get('software', '')
        if "," in software:
            parts = software.split(",")
            parts.pop(random.randint(0, len(parts) - 1))
            fields['software'] = ",".join(parts)
        else:
            fields.pop('software')
    else:
        fields.pop(field_to_delete, None)
    return fields


def fields_to_string(fields):
    return "&".join([f"{k}={v}" for k, v in fields.items()])


def create_pair(original_line, modified_info, label):
    ip, port, original_info = original_line.split("\t")
    # modified_line = f"{ip}\t{port}\t{modified_info}"
    return f"{original_info}\t{modified_info}\t{label}"

with open('../data/train_info.txt', 'r', encoding='utf-8') as f:
    raw_data = [line.strip() for line in f if line.strip()]

for line in raw_data:
    parts = line.split("\t")
    if len(parts) != 3:
        continue
    _, _, service_info = parts
    fields = parse_fields(service_info)
    if 'service_name' in fields:
        service_names.add(fields['service_name'])
    if 'tls_cipher' in fields:
        tls_ciphers.add(fields['tls_cipher'])
    if 'software' in fields:
        softwares = fields['software'].split(",")
        for s in softwares:
            parts = s.strip().split()
            if len(parts) >= 1:
                vector_list.add(parts[0])
            if len(parts) >= 2:
                product_list.add(parts[1])
            if len(parts) >= 3:
                version_list.add(parts[2])


# 1. service_name replacement
print("service_name replace start!")
for line in random.sample(raw_data, 40000):
    # print(line)
    ip, port, service_info = line.split("\t")
    fields = parse_fields(service_info)
    if 'service_name' in fields:
        original_name = fields['service_name']
        new_name = random.choice(list(service_names - {original_name}))
        fields['service_name'] = new_name
        new_info = fields_to_string(fields)
        negative_pairs.append(create_pair(line, new_info, 1))

# 2. banner obfuscation
print("banner modify start!")
for line in random.sample([l for l in raw_data if "banner=" in l], 80000):
    # print(line)
    ip, port, service_info = line.split("\t")
    fields = parse_fields(service_info)
    if 'banner' in fields:
        fields['banner'] = modify_banner(fields['banner'])
        new_info = fields_to_string(fields)
        positive_pairs.append(create_pair(line, new_info, 0))

# 3. software replacement
print("software replace start!")
for line in random.sample([l for l in raw_data if "software=" in l], 40000):
    ip, port, service_info = line.split("\t")
    fields = parse_fields(service_info)
    fields['software'] = replace_software_field(fields['software'])
    new_info = fields_to_string(fields)
    negative_pairs.append(create_pair(line, new_info, 1))

# 4. tls_version replacement
print("tls_version replace start!")
for line in random.sample([l for l in raw_data if "tls_version=" in l], 40000):
    ip, port, service_info = line.split("\t")
    fields = parse_fields(service_info)
    original_version = fields['tls_version']
    new_version = random.choice(list(tls_versions - {original_version}))
    fields['tls_version'] = new_version
    new_info = fields_to_string(fields)
    negative_pairs.append(create_pair(line, new_info, 1))

# 5. tls_cipher replacement
print("tls_cipher replace start!")
for line in random.sample([l for l in raw_data if "tls_cipher=" in l], 40000):
    ip, port, service_info = line.split("\t")
    fields = parse_fields(service_info)
    original_cipher = fields['tls_cipher']
    new_cipher = random.choice(list(tls_ciphers - {original_cipher}))
    fields['tls_cipher'] = new_cipher
    new_info = fields_to_string(fields)
    negative_pairs.append(create_pair(line, new_info, 1))

# 6. field deletion or obfuscation
print("field delete start!")
for line in random.sample(raw_data, 80000):
    ip, port, service_info = line.split("\t")
    fields = parse_fields(service_info)
    fields = delete_random_field(fields)
    new_info = fields_to_string(fields)
    positive_pairs.append(create_pair(line, new_info, 0))


all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

with open('../data/augmented_data.txt', 'w', encoding='utf-8') as f:
    for pair in tqdm(all_pairs):
        f.write(pair + "\n")


f_data = open("../data/augmented_data.txt", "r", encoding="utf-8")
f_clean = open("../data/augmented_data_clean.txt", "w", encoding="utf-8")

def clean_to_printable_ascii(text: str) -> str:
    return ''.join(c for c in text if 32 <= ord(c) <= 126)

for line in f_data:
    data1 = line.split("\t")[0]
    data2 = line.split("\t")[1]
    label = line.split("\t")[2]

    data1 = clean_to_printable_ascii(data1)
    data2 = clean_to_printable_ascii(data2)
    f_clean.write(data1 + "\t" + data2 + "\t" + label)

f_data.close()
f_clean.close()