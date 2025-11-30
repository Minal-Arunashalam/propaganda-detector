#ties everything together and is the file to run
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import PropagandaDataset
from model import PropagandaModel
from configs import Config

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    cfg = Config()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = PropagandaDataset(cfg.train_csv, tokenizer, cfg.max_len)
    val_ds = PropagandaDataset(cfg.val_csv, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = PropagandaModel(cfg.model_name, cfg.num_labels).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        loss = train_epoch(model, train_loader, optimizer, cfg.device)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {loss:.4f}")

        # TODO: add val later

        torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pt")

if __name__ == "__main__":
    main()
