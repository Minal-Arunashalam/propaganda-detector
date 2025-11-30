# eval.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from dataset import PropagandaDataset
from model import PropagandaModel
from configs import Config


def eval_checkpoint(checkpoint_path):
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    #load test set
    test_ds = PropagandaDataset(cfg.test_csv, tokenizer, cfg.max_len)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # load model and checkpoint
    model = PropagandaModel(cfg.model_name, cfg.num_labels)
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()

    all_preds = []
    all_labels = []

    # loop over test data and collect predictions and true labels
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            probs = torch.sigmoid(logits)          # [batch, 14], values in [0,1]
            preds = (probs >= 0.5).int()           # threshold to 0/1

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    #micro metrics (overall performance)
    precision_micro = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    # macro metrics (average over labels)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("Micro (overall)")
    print("Precision_micro: ", precision_micro)
    print("Recall_micro: ", recall_micro)
    print("F1_micro: ", f1_micro)

    print("\nMacro (per-label avg)")
    print("Precision_macro: ", precision_macro)
    print("Recall_macro: ", recall_macro)
    print("F1_macro: ", f1_macro)


def main():
    #choose best checkpoint model version (lowest val loss)
    checkpoint_path = "checkpoint_epoch3.pt"
    eval_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()