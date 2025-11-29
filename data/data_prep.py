import os
import json
import random
from collections import defaultdict
import pandas as pd

BASE_DIR = "raw/datasets"
ARTICLES_DIR = os.path.join(BASE_DIR, "train-articles")
#we're doing task 2, not task 1
TRAIN_LABELS_FILE = os.path.join(BASE_DIR, "train-task2-TC.labels")
OUTPUT_DIR = "processed"
RANDOM_SEED = 42
#80/10/10 train/val/test split
TRAIN_PROP = 0.8
VAL_PROP = 0.1
TEST_PROP = 0.1

def read_task2_labels(path):
    #read train-task2-TC.labels
    #each line in the labels file should be: article_id<TAB>technique_label<TAB>span_start<TAB>span_end 
    #return list of (article_id, technique_label, span_start, span_end)
    #the span is where the propogana technique occurs in the article text

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            parts = line.split()
            if len(parts) != 4:
                #skip if line doesnt have 4 columsn
                print("skipping label line:", line)
                continue

            article_id = parts[0]
            tech_label = parts[1]
            span_start = int(parts[2])
            span_end = int(parts[3])
            records.append((article_id, tech_label, span_start, span_end))

    return records


def read_article(article_id):
    #read article text from article<article_id>.txt as a string
    filename = f"article{article_id}.txt"
    path = os.path.join(ARTICLES_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find article file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def get_line_spans(article_text):
    #split article into lines and compute the character spans for each line
    #characters spans are (start character, end character) for each line
    #return list of (line id, start char, end char, line text without newline)
    #doing this so we can get sentence, label pairs

    #split article into lines, keeping newlines
    lines = article_text.splitlines(keepends=True)

    spans = []
    current_pos = 0
    #go through each line and compute spans
    for i, line in enumerate(lines):
        start = current_pos
        end = start + len(line) #include newline in end pos, but remove in the stored text (model doesnt need newlines)
        text_no_newline = line.rstrip("\n")
        spans.append((i, start, end, text_no_newline))
        current_pos = end

    return spans


def overlaps(a_start, a_end, b_start, b_end):
    #check if two spans overlap
    return not (a_end <= b_start or b_end <= a_start)

def main():
    print("reading task 2 labels from:", TRAIN_LABELS_FILE)
    label_records = read_task2_labels(TRAIN_LABELS_FILE)
    print("Number of labeled spans:", len(label_records))

    #group labels by article id so we can process each article once
    labels_by_article = defaultdict(list)
    for article_id, tech_label, span_start, span_end in label_records:
        labels_by_article[article_id].append((tech_label, span_start, span_end))

    # this will store the set of techniques for each sentence
    #key is (article id, line id), value is set of prop technique strings
    sentence_labels = defaultdict(set)

    #this will store line spans for each article
    #k is article id, value is list of (line id, start, end, text)
    article_line_spans = {}

    #for each article: read text, compute line spans, then map each label span to corresponding sentece
    #just mapping labels to sentences
    for article_id, spans_list in labels_by_article.items():
        #read article text
        article_text = read_article(article_id)

        #get (lineid, start, end, text)
        line_spans = get_line_spans(article_text)
        article_line_spans[article_id] = line_spans

        #for each span (tech_label, span_start, span_end)
        for tech_label, span_start, span_end in spans_list:
            #check which lines this span overlaps
            for (line_idx, line_start, line_end, line_text) in line_spans:
                if overlaps(span_start, span_end, line_start, line_end):
                    #add the proopganda technique label to this sentence
                    sentence_labels[(article_id, line_idx)].add(tech_label)

    #get all unique technique labels
    all_labels = set()
    for s in sentence_labels.values():
        for tech in s:
            all_labels.add(tech)
    #create label to id mapping
    label_list = sorted(list(all_labels))
    label2id = {label: i for i, label in enumerate(label_list)}

    print("Number of unique propoganda techniques:", len(label_list))
    for i, label in enumerate(label_list):
        print(f"{i}: {label}")


    rows = []
    for article_id, line_spans in article_line_spans.items():
        for (line_idx, start, end, text) in line_spans:
            if text.strip() == "":
                continue

            #get set of techniques for this sentence (maybe empty)
            tech_set = sentence_labels.get((article_id, line_idx), set())

            #make one hot encoding vector for labels
            #start with all zeros
            vec = [0] * len(label_list)
            #set ones for each technique present
            #each index is the label id
            for tech in tech_set:
                vec[label2id[tech]] = 1

            #save labels as json string so we can parse later
            vec_str = json.dumps(vec)
            rows.append({
                "article_id": article_id,
                "sentence_id": line_idx,
                "text": text,
                "labels": vec_str
            })

    df_all = pd.DataFrame(rows)
    print("Total sentence rows:", len(df_all))

    #split into train/val/test by article_id
    random.seed(RANDOM_SEED)
    article_ids = df_all["article_id"].unique().tolist()
    #shuffle article ids randomly
    random.shuffle(article_ids)

    n_total = len(article_ids)
    n_train = int(n_total * TRAIN_PROP)
    n_val = int(n_total * VAL_PROP)
    #rest go to test
    n_test = n_total - n_train - n_val
    #split articles ids into train val and test
    train_ids = set(article_ids[:n_train])
    val_ids = set(article_ids[n_train:n_train + n_val])
    test_ids = set(article_ids[n_train + n_val:])

    print("Total articles:", n_total)
    print("Train/val/test articles:", len(train_ids), len(val_ids), len(test_ids))
    #make the train, val, test dataframes
    df_train = df_all[df_all["article_id"].isin(train_ids)].reset_index(drop=True)
    df_val = df_all[df_all["article_id"].isin(val_ids)].reset_index(drop=True)
    df_test = df_all[df_all["article_id"].isin(test_ids)].reset_index(drop=True)

    print("train/val/test sentences:", len(df_train), len(df_val), len(df_test))

    #save csvs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("Saved train/val/test csv files to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()