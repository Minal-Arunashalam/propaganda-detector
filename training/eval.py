import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from dataset import PropagandaDataset
from model import PropagandaModel
from configs import Config


def collect_logits_and_labels(checkpoint_path):
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    test_ds = PropagandaDataset(cfg.test_csv, tokenizer, cfg.max_len)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = PropagandaModel(cfg.model_name, cfg.num_labels)
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()

    all_logits = []
    all_labels = []

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

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels


def eval_at_threshold(all_logits, all_labels, threshold):
    probs = torch.sigmoid(all_logits)           # [N, num_labels]
    preds = (probs >= threshold).int()

    preds_np = preds.numpy()
    labels_np = all_labels.numpy()

    precision_micro = precision_score(labels_np, preds_np, average="micro", zero_division=0)
    recall_micro    = recall_score(labels_np, preds_np, average="micro", zero_division=0)
    f1_micro        = f1_score(labels_np, preds_np, average="micro", zero_division=0)

    precision_macro = precision_score(labels_np, preds_np, average="macro", zero_division=0)
    recall_macro    = recall_score(labels_np, preds_np, average="macro", zero_division=0)
    f1_macro        = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    return {
        "threshold": threshold,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def main():
    cfg = Config()
    safe_name = cfg.model_name.replace("/", "_")
    checkpoint_path = f"{safe_name}_checkpoint_epoch2.pt"  #best epoch for RoBERTa

    #nun model once and collect logits (Raw predictions) and labels
    all_logits, all_labels = collect_logits_and_labels(checkpoint_path)

    #try multiple thresholds
    thresholds = [i / 10.0 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9
    results = []

    for t in thresholds:
        metrics = eval_at_threshold(all_logits, all_labels, t)
        results.append(metrics)
        print(f"Threshold {t:.1f} | "
              f"Micro F1: {metrics['f1_micro']:.4f} | "
              f"Micro P: {metrics['precision_micro']:.4f} | "
              f"Micro R: {metrics['recall_micro']:.4f}")

    #find best threshold by micro F1
    best = max(results, key=lambda m: m["f1_micro"])
    print("\nBest threshold by micro F1")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Micro Precision: {best['precision_micro']:.4f}")
    print(f"Micro Recall: {best['recall_micro']:.4f}")
    print(f"Micro F1: {best['f1_micro']:.4f}")
    print(f"Macro F1: {best['f1_macro']:.4f}")


if __name__ == "__main__":
    main()