class Config:
    model_name = "roberta-base"
    num_labels = 14

    train_csv = "../data/processed/train.csv"
    val_csv = "../data/processed/val.csv"
    test_csv = "../data/processed/test.csv"

    max_len = 256
    batch_size = 8
    lr = 2e-5
    epochs = 5
    device = "cuda"
