import os
import time
import csv
from copy import deepcopy
import itertools

import numpy as np
from pytorch_lightning import seed_everything

import logging
logging.disable(logging.INFO)

from demo.kgrs import KGRS


def load_data():
    train_pos = np.load("data/train_pos.npy")
    train_neg = np.load("data/train_neg.npy")
    np.random.shuffle(train_pos)
    np.random.shuffle(train_neg)

    all_users = set(train_pos[:, 0]) | set(train_neg[:, 0])
    all_items = set(train_pos[:, 1]) | set(train_neg[:, 1])

    n_user = max(all_users) + 1
    n_item = max(all_items) + 1

    train_pos_len = int(len(train_pos) * 0.8)
    train_neg_len = int(len(train_neg) * 0.8)

    test_pos = train_pos[train_pos_len:]
    test_neg = train_neg[train_neg_len:]
    train_pos = train_pos[:train_pos_len]
    train_neg = train_neg[:train_neg_len]

    return train_pos, train_neg, test_pos, test_neg, n_user, n_item


def run_experiment(config_overrides, seed=42):
    # load data
    train_pos, train_neg, test_pos, test_neg, n_user, n_item = load_data()

    # further split train into train/validation to avoid tuning on test
    val_ratio = 0.1
    tp_val_len = int(len(train_pos) * val_ratio)
    tn_val_len = int(len(train_neg) * val_ratio)

    # take last portion as validation
    val_pos = train_pos[-tp_val_len:] if tp_val_len > 0 else np.empty((0, train_pos.shape[1]), dtype=train_pos.dtype)
    val_neg = train_neg[-tn_val_len:] if tn_val_len > 0 else np.empty((0, train_neg.shape[1]), dtype=train_neg.dtype)
    train_pos = train_pos[:-tp_val_len] if tp_val_len > 0 else train_pos
    train_neg = train_neg[:-tn_val_len] if tn_val_len > 0 else train_neg

    # build KGRS with overrides
    cfg = dict(config_overrides)
    # ensure reproducibility
    cfg.setdefault("seed", seed)

    kgrs = KGRS(
        train_pos=deepcopy(train_pos),
        train_neg=deepcopy(train_neg),
        kg_lines=open('data/kg.txt', encoding='utf-8').readlines(),
        n_user=n_user,
        n_item=n_item,
        config=cfg
    )

    # Prepare validation pairs and labels for early stopping
    val_data = None
    if val_pos.shape[0] + val_neg.shape[0] > 0:
        val_concat = np.concatenate((deepcopy(val_neg), deepcopy(val_pos)), axis=0)
        np.random.shuffle(val_concat)
        val_labels = val_concat[:, 2]
        val_pairs = val_concat[:, :2]
        val_data = (val_pairs, val_labels)

    start = time.time()
    # Use early stopping on validation to avoid overfitting
    kgrs.training(val_data=val_data, early_stopping_patience=3, output_log=False)
    train_time = time.time() - start

    # CTR evaluation
    test_data = np.concatenate((deepcopy(test_neg), deepcopy(test_pos)), axis=0)
    np.random.shuffle(test_data)
    test_label = test_data[:, 2]
    test_pairs = test_data[:, :2]
    scores = kgrs.eval_ctr((test_pairs, test_label))

    # topk evaluation
    # build user -> pos sets
    user_pos_items = {}
    user_train_pos_items = {}
    for record in train_pos:
        u, it = record[0], record[1]
        user_train_pos_items.setdefault(u, set()).add(it)
    for record in test_pos:
        u, it = record[0], record[1]
        user_pos_items.setdefault(u, set()).add(it)

    user_item_lists = kgrs.eval_topk((list(user_pos_items.keys()), user_pos_items))

    # compute AUC and nDCG@5 locally (reuse evaluate logic lightly)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true=test_label, y_score=scores)

    # compute nDCG@5
    import math

    def nDCG(sorted_items, pos_item, train_pos_item, k=5):
        dcg = 0
        train_pos_item = set(train_pos_item)
        filter_item = set(filter(lambda item: item not in train_pos_item, pos_item))
        max_correct = min(len(filter_item), k)
        train_hit_num = 0
        valid_num = 0
        recommended_items = set()

        for index in range(len(sorted_items)):
            if sorted_items[index] in train_pos_item:
                train_hit_num += 1
            else:
                valid_num += 1

            if sorted_items[index] in filter_item and sorted_items[index] not in recommended_items:
                dcg += 1 / math.log2(index - train_hit_num + 2)
                recommended_items.add(sorted_items[index])

            if valid_num >= k:
                break

        idcg = sum([1 / math.log2(i + 2) for i in range(max_correct)])
        return dcg / idcg if idcg > 0 else 0.0

    ndcg5 = np.mean([
        nDCG(user_item_lists[index], user_pos_items[user], user_train_pos_items.get(user, set()))
        for index, user in enumerate(list(user_pos_items.keys()))
    ])

    return {
        "auc": float(auc),
        "ndcg5": float(ndcg5),
        "train_time": float(train_time),
        **config_overrides
    }


def grid_search(out_csv='tune_results.csv', max_runs=None):
    # define search space (small default grid)
    emb_dims = [16, 32]
    lrs = [2e-3, 1e-3]
    margins = [10, 30]
    neg_rates = [1.0, 1.5]
    epoch_nums = [10, 20]

    combos = list(itertools.product(emb_dims, lrs, margins, neg_rates, epoch_nums))
    if max_runs is not None:
        combos = combos[:max_runs]

    # header
    header = ["emb_dim", "learning_rate", "margin", "neg_rate", "epoch_num", "auc", "ndcg5", "train_time"]

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for emb_dim, lr, margin, neg_rate, epoch_num in combos:
            print(f"Running: emb_dim={emb_dim}, lr={lr}, margin={margin}, neg_rate={neg_rate}, epoch={epoch_num}")
            cfg = {
                "emb_dim": emb_dim,
                "learning_rate": lr,
                "margin": margin,
                "neg_rate": neg_rate,
                "epoch_num": epoch_num,
            }
            try:
                res = run_experiment(cfg)
                writer.writerow([emb_dim, lr, margin, neg_rate, epoch_num, res['auc'], res['ndcg5'], res['train_time']])
                f.flush()
            except Exception as e:
                print(f"Run failed for {cfg}: {e}")

    print(f"Grid search finished. Results saved to {out_csv}")


if __name__ == '__main__':
    # set global seed for reproducibility
    seed_everything(1088, workers=True)
    # Run a small grid (use max_runs to limit runs if desired)
    grid_search(out_csv='tune_results.csv', max_runs=12)
