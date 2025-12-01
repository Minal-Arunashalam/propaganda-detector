import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

from dataset import PropagandaDataset
from model import PropagandaModel
from configs import RobertaConfig, DebertaConfig, LlamaConfig


def get_device(cfg):
    """Auto-detect device, works in Colab"""
    if cfg.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


# Helper class for Llama with LoRA (same as in train.py)
class PropagandaModelWithLoRA(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return loss, logits

        return logits


def collect_logits_and_labels_roberta(checkpoint_path, epoch=None):
    """Collect logits and labels for RoBERTa model"""
    cfg = RobertaConfig()
    device = get_device(cfg)
    safe_name = cfg.model_name.replace("/", "_")

    # Determine checkpoint path
    if checkpoint_path is None:
        if epoch is None:
            epoch = 2  # Default to epoch 2
        checkpoint_path = os.path.join(
            "checkpoints", f"{safe_name}_checkpoint_epoch{epoch}.pt"
        )
    elif not os.path.isabs(checkpoint_path) and not checkpoint_path.startswith(
        "checkpoints"
    ):
        checkpoint_path = os.path.join("checkpoints", checkpoint_path)

    print(f"Loading RoBERTa checkpoint from: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    test_ds = PropagandaDataset(cfg.test_csv, tokenizer, cfg.max_len)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = PropagandaModel(cfg.model_name, cfg.num_labels)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels


def collect_logits_and_labels_deberta(checkpoint_path, epoch=None):
    """Collect logits and labels for DeBERTa model"""
    cfg = DebertaConfig()
    device = get_device(cfg)
    safe_name = cfg.model_name.replace("/", "_")

    # Determine checkpoint path
    if checkpoint_path is None:
        if epoch is None:
            epoch = 2  # Default to epoch 2
        checkpoint_path = os.path.join(
            "checkpoints", f"{safe_name}_checkpoint_epoch{epoch}.pt"
        )
    elif not os.path.isabs(checkpoint_path) and not checkpoint_path.startswith(
        "checkpoints"
    ):
        checkpoint_path = os.path.join("checkpoints", checkpoint_path)

    print(f"Loading DeBERTa checkpoint from: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    test_ds = PropagandaDataset(cfg.test_csv, tokenizer, cfg.max_len)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = PropagandaModel(cfg.model_name, cfg.num_labels)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels


def collect_logits_and_labels_llama(checkpoint_path, epoch=None, use_lora=True):
    """Collect logits and labels for Llama model"""
    from peft import PeftModel

    cfg = LlamaConfig()
    device = get_device(cfg)
    safe_name = cfg.model_name.replace("/", "_")

    # Determine checkpoint paths
    lora_checkpoint_dir = None
    classifier_path = None

    if checkpoint_path is None:
        if epoch is None:
            epoch = 2  # Default to epoch 2
        if use_lora:
            lora_checkpoint_dir = os.path.join(
                "checkpoints", f"{safe_name}_lora_checkpoint_epoch{epoch}"
            )
            classifier_path = os.path.join(
                "checkpoints", f"{safe_name}_classifier_epoch{epoch}.pt"
            )
        else:
            checkpoint_path = os.path.join(
                "checkpoints", f"{safe_name}_checkpoint_epoch{epoch}.pt"
            )
    else:
        # If path provided, assume full checkpoint (not LoRA)
        use_lora = False
        if not os.path.isabs(checkpoint_path) and not checkpoint_path.startswith(
            "checkpoints"
        ):
            checkpoint_path = os.path.join("checkpoints", checkpoint_path)

    print(f"Loading Llama checkpoint (LoRA={use_lora})...")

    # Llama tokenizer requires padding token
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    test_ds = PropagandaDataset(cfg.test_csv, tokenizer, cfg.max_len)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    if use_lora:
        print(f"  LoRA adapter: {lora_checkpoint_dir}")
        print(f"  Classifier: {classifier_path}")

        # Load base model and LoRA adapters
        base_model = AutoModel.from_pretrained(cfg.model_name)
        lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)

        # Create full model
        config = AutoConfig.from_pretrained(cfg.model_name)
        model = PropagandaModelWithLoRA(lora_model, config.hidden_size, cfg.num_labels)

        # Load classifier
        classifier_state = torch.load(classifier_path, map_location=device)
        model.classifier.load_state_dict(classifier_state)
    else:
        print(f"  Full checkpoint: {checkpoint_path}")
        model = PropagandaModel(cfg.model_name, cfg.num_labels)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels


def eval_at_threshold(all_logits, all_labels, threshold):
    probs = torch.sigmoid(all_logits)  # [N, num_labels]
    preds = (probs >= threshold).int()

    preds_np = preds.numpy()
    labels_np = all_labels.numpy()

    precision_micro = precision_score(
        labels_np, preds_np, average="micro", zero_division=0
    )
    recall_micro = recall_score(labels_np, preds_np, average="micro", zero_division=0)
    f1_micro = f1_score(labels_np, preds_np, average="micro", zero_division=0)

    precision_macro = precision_score(
        labels_np, preds_np, average="macro", zero_division=0
    )
    recall_macro = recall_score(labels_np, preds_np, average="macro", zero_division=0)
    f1_macro = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    return {
        "threshold": threshold,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def eval_roberta(checkpoint_path=None, epoch=None):
    """Evaluate RoBERTa model"""
    print("=" * 70)
    print("Evaluating RoBERTa Model")
    print("=" * 70)

    # Run model once and collect logits (Raw predictions) and labels
    all_logits, all_labels = collect_logits_and_labels_roberta(checkpoint_path, epoch)

    # Try multiple thresholds
    thresholds = [i / 10.0 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9
    results = []

    for t in thresholds:
        metrics = eval_at_threshold(all_logits, all_labels, t)
        results.append(metrics)
        print(
            f"Threshold {t:.1f} | "
            f"Micro F1: {metrics['f1_micro']:.4f} | "
            f"Micro P: {metrics['precision_micro']:.4f} | "
            f"Micro R: {metrics['recall_micro']:.4f}"
        )

    # Find best threshold by micro F1
    best = max(results, key=lambda m: m["f1_micro"])
    print("\n" + "=" * 70)
    print("Best threshold by micro F1")
    print("=" * 70)
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Micro Precision: {best['precision_micro']:.4f}")
    print(f"Micro Recall: {best['recall_micro']:.4f}")
    print(f"Micro F1: {best['f1_micro']:.4f}")
    print(f"Macro Precision: {best['precision_macro']:.4f}")
    print(f"Macro Recall: {best['recall_macro']:.4f}")
    print(f"Macro F1: {best['f1_macro']:.4f}")

    return best, results


def eval_deberta(checkpoint_path=None, epoch=None):
    """Evaluate DeBERTa model"""
    print("=" * 70)
    print("Evaluating DeBERTa Model")
    print("=" * 70)

    # Run model once and collect logits (Raw predictions) and labels
    all_logits, all_labels = collect_logits_and_labels_deberta(checkpoint_path, epoch)

    # Try multiple thresholds
    thresholds = [i / 10.0 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9
    results = []

    for t in thresholds:
        metrics = eval_at_threshold(all_logits, all_labels, t)
        results.append(metrics)
        print(
            f"Threshold {t:.1f} | "
            f"Micro F1: {metrics['f1_micro']:.4f} | "
            f"Micro P: {metrics['precision_micro']:.4f} | "
            f"Micro R: {metrics['recall_micro']:.4f}"
        )

    # Find best threshold by micro F1
    best = max(results, key=lambda m: m["f1_micro"])
    print("\n" + "=" * 70)
    print("Best threshold by micro F1")
    print("=" * 70)
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Micro Precision: {best['precision_micro']:.4f}")
    print(f"Micro Recall: {best['recall_micro']:.4f}")
    print(f"Micro F1: {best['f1_micro']:.4f}")
    print(f"Macro Precision: {best['precision_macro']:.4f}")
    print(f"Macro Recall: {best['recall_macro']:.4f}")
    print(f"Macro F1: {best['f1_macro']:.4f}")

    return best, results


def eval_llama(checkpoint_path=None, epoch=None, use_lora=True):
    """Evaluate Llama model"""
    print("=" * 70)
    print(f"Evaluating Llama Model (LoRA={use_lora})")
    print("=" * 70)

    # Run model once and collect logits (Raw predictions) and labels
    all_logits, all_labels = collect_logits_and_labels_llama(
        checkpoint_path, epoch, use_lora
    )

    # Try multiple thresholds
    thresholds = [i / 10.0 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9
    results = []

    for t in thresholds:
        metrics = eval_at_threshold(all_logits, all_labels, t)
        results.append(metrics)
        print(
            f"Threshold {t:.1f} | "
            f"Micro F1: {metrics['f1_micro']:.4f} | "
            f"Micro P: {metrics['precision_micro']:.4f} | "
            f"Micro R: {metrics['recall_micro']:.4f}"
        )

    # Find best threshold by micro F1
    best = max(results, key=lambda m: m["f1_micro"])
    print("\n" + "=" * 70)
    print("Best threshold by micro F1")
    print("=" * 70)
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Micro Precision: {best['precision_micro']:.4f}")
    print(f"Micro Recall: {best['recall_micro']:.4f}")
    print(f"Micro F1: {best['f1_micro']:.4f}")
    print(f"Macro Precision: {best['precision_macro']:.4f}")
    print(f"Macro Recall: {best['recall_macro']:.4f}")
    print(f"Macro F1: {best['f1_macro']:.4f}")

    return best, results


def main():
    import sys

    if len(sys.argv) < 2:
        print("Please specify a model type:")
        print("  python eval.py roberta [epoch]")
        print("  python eval.py deberta [epoch]")
        print("  python eval.py llama [epoch] [--no-lora]")
        print("\nExample:")
        print("  python eval.py roberta 2")
        print("  python eval.py llama 3")
        print("  python eval.py llama 2 --no-lora  # For full fine-tuning")
        return

    model_type = sys.argv[1].lower()
    epoch = None
    use_lora = True

    # Parse epoch if provided
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        epoch = int(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[2] != "--no-lora":
        print(f"Warning: '{sys.argv[2]}' is not a valid epoch number, using default")

    # Check for --no-lora flag
    if "--no-lora" in sys.argv:
        use_lora = False

    if model_type == "roberta":
        eval_roberta(epoch=epoch)
    elif model_type == "deberta":
        eval_deberta(epoch=epoch)
    elif model_type == "llama":
        eval_llama(epoch=epoch, use_lora=use_lora)
    else:
        print(f"Unknown model type: {model_type}")
        print("Usage: python eval.py [roberta|deberta|llama] [epoch] [--no-lora]")


if __name__ == "__main__":
    main()
