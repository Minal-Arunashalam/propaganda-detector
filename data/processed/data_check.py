import pandas as pd, json

# this file does basic data checks on the processed train/val/test csv files to make sure
#everythings good. it makes sur theres no data leakage between splits, and that labels are one hot encoded correctly

df_train = pd.read_csv("train.csv")
df_val   = pd.read_csv("val.csv")
df_test  = pd.read_csv("test.csv")

# 1. Label length = 14 and only 0/1
def check_labels(df, name):
    lengths = set()
    bad = set()
    total_pos = 0
    for s in df["labels"]:
        vec = json.loads(s)
        lengths.add(len(vec))
        for x in vec:
            if x not in (0, 1):
                bad.add(x)
        total_pos += sum(vec)
    print(name, "lengths:", lengths, "| bad values:", bad, "| total positives:", total_pos)

check_labels(df_train, "train")
check_labels(df_val, "val")
check_labels(df_test, "test")

# 2. No article overlap between splits
train_ids = set(df_train["article_id"])
val_ids   = set(df_val["article_id"])
test_ids  = set(df_test["article_id"])

print("train ∩ val:", train_ids & val_ids)
print("train ∩ test:", train_ids & test_ids)
print("val ∩ test:", val_ids & test_ids)