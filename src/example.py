import gzip
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm, trange

from transformers import RobertaForSequenceClassification
from transformers.data.metrics import acc_and_f1, simple_accuracy


logging.basicConfig(level=logging.INFO)


CODEBERTA_PRETRAINED = "huggingface/CodeBERTa-small-v1"

LANGUAGES = [
    "go",
    "java",
    "javascript",
    "php",
    "python",
    "ruby",
]
FILES_PER_LANGUAGE = 1
EVALUATE = True

# Set up tokenizer
tokenizer = ByteLevelBPETokenizer("./pretrained/vocab.json", "./pretrained/merges.txt",)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

# Set up Tensorboard
tb_writer = SummaryWriter()


class CodeSearchNetDataset(Dataset):
    examples: List[Tuple[List[int], int]]

    def __init__(self, split: str = "train"):
        """
        train | valid | test
        """

        self.examples = []

        src_files = []
        for language in LANGUAGES:
            src_files += list(
                Path("../CodeSearchNet/resources/data/").glob(f"{language}/final/jsonl/{split}/*.jsonl.gz")
            )[:FILES_PER_LANGUAGE]
        for src_file in src_files:
            label = src_file.parents[3].name
            label_idx = LANGUAGES.index(label)
            print("🔥", src_file, label)
            lines = []
            fh = gzip.open(src_file, mode="rt", encoding="utf-8")
            for line in fh:
                o = json.loads(line)
                lines.append(o["code"])
            examples = [(x.ids, label_idx) for x in tokenizer.encode_batch(lines)]
            self.examples += examples
        print("🔥🔥")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return self.examples[i]


model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_PRETRAINED, num_labels=len(LANGUAGES))

train_dataset = CodeSearchNetDataset(split="train")
eval_dataset = CodeSearchNetDataset(split="test")


def collate(examples):
    input_ids = pad_sequence([torch.tensor(x[0]) for x in examples], batch_first=True, padding_value=1)
    labels = torch.tensor([x[1] for x in examples])
    # ^^  uncessary .unsqueeze(-1)
    return input_ids, labels


train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate)

batch = next(iter(train_dataloader))


model.to("cuda")
model.train()
for param in model.roberta.parameters():
    param.requires_grad = False
## ^^ Only train final layer.

print(f"num params:", model.num_parameters())
print(f"num trainable params:", model.num_parameters(only_trainable=True))


def evaluate():
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = np.empty((0), dtype=np.int64)
    out_label_ids = np.empty((0), dtype=np.int64)

    model.eval()

    eval_dataloader = DataLoader(eval_dataset, batch_size=512, collate_fn=collate)
    for step, (input_ids, labels) in enumerate(tqdm(eval_dataloader, desc="Eval")):
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to("cuda"), labels=labels.to("cuda"))
            loss = outputs[0]
            logits = outputs[1]
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
        preds = np.append(preds, logits.argmax(dim=1).detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    acc = simple_accuracy(preds, out_label_ids)
    f1 = f1_score(y_true=out_label_ids, y_pred=preds, average="macro")
    print("=== Eval: loss ===", eval_loss)
    print("=== Eval: acc. ===", acc)
    print("=== Eval: f1 ===", f1)
    # print(acc_and_f1(preds, out_label_ids))
    tb_writer.add_scalars("eval", {"loss": eval_loss, "acc": acc, "f1": f1}, global_step)


### Training loop

global_step = 0
train_iterator = trange(0, 4, desc="Epoch")
optimizer = torch.optim.AdamW(model.parameters())
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, (input_ids, labels) in enumerate(epoch_iterator):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.to("cuda"), labels=labels.to("cuda"))
        loss = outputs[0]
        loss.backward()
        tb_writer.add_scalar("training_loss", loss.item(), global_step)
        optimizer.step()
        global_step += 1
        if EVALUATE and global_step % 50 == 0:
            evaluate()
            model.train()


evaluate()

os.makedirs("./models/CodeBERT-language-id", exist_ok=True)
model.save_pretrained("./models/CodeBERT-language-id")