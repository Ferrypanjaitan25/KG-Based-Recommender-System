import logging
import time
from copy import deepcopy
import math
import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score

import os
import sys

# os.path.join(..., '..') -> go up one level to (.../KG-BASED-RECOMMENDER-SYSTEM)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.kgrs import KGRS


def nDCG(sorted_items, pos_item, train_pos_item, k=5):
    """
    sorted_items : list of recommended item ids (ordered by score desc)
    pos_item     : set of test positive items for user
    train_pos_item: set of train positive items for user (to filter)
    """
    dcg = 0.0
    train_pos_item = set(train_pos_item)
    pos_item = set(pos_item)
    # filter out test positives that appear in train positives
    filter_item = pos_item - train_pos_item
    max_correct = min(len(filter_item), k)
    train_hit_num = 0
    valid_num = 0
    recommended_items = set()

    for index in range(len(sorted_items)):
        item = sorted_items[index]
        if item in train_pos_item:
            train_hit_num += 1
        else:
            valid_num += 1
            if item in filter_item and item not in recommended_items:
                # index - train_hit_num is the rank among valid items (0-based)
                dcg += 1.0 / math.log2(index - train_hit_num + 2.0)
                recommended_items.add(item)
            if valid_num >= k:
                break

    idcg = sum([1.0 / math.log2(i + 2.0) for i in range(max_correct)]) if max_correct > 0 else 1.0
    return (dcg / idcg) if idcg > 0 else 0.0


def load_data():
    # ensure working dir is repository root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(base_dir)

    train_pos = np.load("data/train_pos.npy")
    train_neg = np.load("data/train_neg.npy")
    np.random.shuffle(train_pos)
    np.random.shuffle(train_neg)

    all_users = set(train_pos[:, 0]) | set(train_neg[:, 0])
    all_items = set(train_pos[:, 1]) | set(train_neg[:, 1])
    n_user = int(max(all_users)) + 1
    n_item = int(max(all_items)) + 1

    train_pos_len, train_neg_len = int(len(train_pos) * 0.8), int(len(train_neg) * 0.8)
    test_pos, test_neg = train_pos[train_pos_len:], train_neg[train_neg_len:]
    train_pos, train_neg = train_pos[:train_pos_len], train_neg[:train_neg_len]

    return train_pos, train_neg, test_pos, test_neg, n_user, n_item


def get_user_pos_items(train_pos, test_pos):
    """
    Returns:
      user_pos_items: dict[user] -> set(test positive items)
      user_train_pos_items: dict[user] -> set(train positive items)
    """
    user_pos_items = {}
    user_train_pos_items = {}

    for record in train_pos:
        user, item = int(record[0]), int(record[1])
        user_train_pos_items.setdefault(user, set()).add(item)

    for record in test_pos:
        user, item = int(record[0]), int(record[1])
        user_pos_items.setdefault(user, set()).add(item)

    return user_pos_items, user_train_pos_items


def evaluate():
    train_pos, train_neg, test_pos, test_neg, n_user, n_item = load_data()
    user_pos_items, user_train_pos_items = get_user_pos_items(train_pos=train_pos, test_pos=test_pos)

    logging.disable(logging.INFO)
    seed_everything(1088, workers=True)
    torch.set_num_threads(8)

    auc, ndcg5 = 0.0, 0.0
    init_timeout = train_timeout = ctr_timeout = topk_timeout = False
    start_time = time.time()

    # Inisialisasi KGRS
    kgrs = KGRS(
        train_pos=deepcopy(train_pos),
        train_neg=deepcopy(train_neg),
        kg_lines=open("data/kg.txt", encoding="utf-8").readlines(),
        n_user=n_user,
        n_item=n_item
    )

    init_time = time.time() - start_time
    kgrs.training()
    train_time = time.time() - start_time - init_time

    # Siapkan data test: gabungkan negative dan positive, label ada di kolom ke-2 (index 2)
    test_data = np.concatenate((deepcopy(test_neg), deepcopy(test_pos)), axis=0)
    np.random.shuffle(test_data)

    # Pastikan test_data punya kolom label (index 2). Jika tidak, user harus menyesuaikan data.
    if test_data.shape[1] < 3:
        raise ValueError("test_data harus memiliki 3 kolom: [user, item, label]. Periksa file data/test_pos.npy dan test_neg.npy")

    test_label = test_data[:, 2].astype(float)
    test_data_pairs = test_data[:, :2].astype(int)

    scores = kgrs.eval_ctr(test_data=test_data_pairs)
    auc = float(roc_auc_score(y_true=test_label, y_score=scores))
    ctr_time = time.time() - start_time - init_time - train_time

    users = list(user_pos_items.keys())
    if len(users) == 0:
        raise ValueError("Tidak ada user untuk evaluasi Top-K (user_pos_items kosong). Periksa data test_pos.npy")

    user_item_lists = kgrs.eval_topk(users=users)
    ndcg5 = np.mean([
        nDCG(user_item_lists[index], user_pos_items[user], user_train_pos_items.get(user, set()))
        for index, user in enumerate(users)
    ])

    topk_time = time.time() - start_time - init_time - train_time - ctr_time

    return auc, ndcg5, init_timeout, train_timeout, ctr_timeout, topk_timeout, init_time, train_time, ctr_time, topk_time


if __name__ == '__main__':
    seed_everything(1088, workers=True)
    print("Training started...")
    start_time = time.time()

    auc, ndcg5, init_timeout, train_timeout, ctr_timeout, topk_timeout, init_time, train_time, ctr_time, topk_time = evaluate()

    print("\n--- Result ---")
    print(f"AUC Score       : {auc:.4f}")
    print(f"nDCG@5 Score    : {ndcg5:.4f}")
    print("------------------------")
    print(f"⏱ Initialization Time : {init_time:.2f} seconds")
    print(f"⏱ Training Time       : {train_time:.2f} seconds")
    print(f"⏱ CTR Evaluation Time : {ctr_time:.2f} seconds")
    print(f"⏱ TopK Evaluation Time: {topk_time:.2f} seconds")
    print("------------------------")
    total_time = time.time() - start_time
    print(f"⏱ Execution Time      : {total_time:.2f} seconds")
