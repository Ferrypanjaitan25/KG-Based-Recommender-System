import os
import sys
from typing import List
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# pastikan bisa import dari root (kalau dibutuhkan)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===================== DATALOADER =====================
class Dataloader:
    def __init__(self, train_pos, train_neg, kg_lines, n_user, n_item,
                 train_batch_size: int = 512, neg_rate: float = 4.0):
        self.kg, self.rel_dict, self.n_entity = self._convert_kg(kg_lines)
        self.train_pos, self.train_neg = train_pos, train_neg
        self.n_user = n_user
        self.n_item = n_item
        self._load_ratings()
        self.known_neg_dict = []
        self._add_recsys_to_kg()
        self.train_batch_size = train_batch_size
        self.neg_rate = neg_rate
        self.ent_num = self.n_entity + self.n_user
        self.rel_num = len(self.rel_dict)

    def _add_recsys_to_kg(self):
        self.rel_dict['feedback_recsys'] = max(self.rel_dict.values()) + 1 if self.rel_dict else 0
        for interaction in self.train_pos:
            self.kg.append((int(interaction[0]), self.rel_dict['feedback_recsys'], int(interaction[1])))
        for interaction in self.train_neg:
            self.known_neg_dict.append((int(interaction[0]), self.rel_dict['feedback_recsys'], int(interaction[1])))

    def _load_ratings(self):
        self.n_entity = max(self.n_item, self.n_entity)
        for i in range(len(self.train_pos)):
            self.train_pos[i][0] = int(self.train_pos[i][0]) + self.n_entity
            self.train_pos[i][1] = int(self.train_pos[i][1])
        for i in range(len(self.train_neg)):
            self.train_neg[i][0] = int(self.train_neg[i][0]) + self.n_entity
            self.train_neg[i][1] = int(self.train_neg[i][1])

    def _convert_kg(self, lines):
        entity_set, kg = set(), []
        rel_dict = {}
        rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "relation2id.txt"))
        with open(rel_path, encoding="utf8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    rel_dict[parts[0]] = int(parts[1])

        for line in lines:
            a = line.strip().split("\t")
            if len(a) < 3:
                continue
            h, r, t = int(a[0]), rel_dict[a[1]], int(a[2])
            kg.append((h, r, t))
            entity_set.add(h); entity_set.add(t)

        print("number of entities (containing items): %d" % len(entity_set))
        print("number of relations: %d" % len(rel_dict))
        return kg, rel_dict, (max(list(entity_set)) + 1) if entity_set else 0

    def get_user_pos_item_list(self):
        train_user_pos_item = {}
        all_record = np.concatenate([self.train_pos, self.train_neg], axis=0)
        for record in self.train_pos:
            u, it = int(record[0]) - self.n_entity, int(record[1])
            train_user_pos_item.setdefault(u, set()).add(it)
        item_list = list(set(all_record[:, 1].astype(int).tolist()))
        return item_list, train_user_pos_item

    def get_training_batch(self):
        pos_data = [f for f in self.kg]
        neg_data = [f for f in self.known_neg_dict]
        hr_tail_set, rt_head_set = {}, {}
        for f in pos_data + neg_data:
            hr_tail_set.setdefault((f[0], f[1]), set()).add(f[2])
            rt_head_set.setdefault((f[1], f[2]), set()).add(f[0])

        sample_failed_time = 0
        sample_failed_max = int(len(self.kg) * self.neg_rate) + 1

        while len(neg_data) < len(self.kg) * self.neg_rate and sample_failed_time < sample_failed_max:
            for f in self.kg:
                if len(neg_data) >= len(self.kg) * self.neg_rate:
                    break
                if random.random() > 0.5:
                    # sample tail
                    if f[0] >= self.n_entity:
                        tail = random.randint(0, self.n_item - 1)
                        while tail in hr_tail_set[(f[0], f[1])] and sample_failed_time < sample_failed_max:
                            sample_failed_time += 1
                            tail = random.randint(0, self.n_item - 1)
                    else:
                        tail = random.randint(0, self.n_entity - 1)
                        while tail in hr_tail_set[(f[0], f[1])] and sample_failed_time < sample_failed_max:
                            sample_failed_time += 1
                            tail = random.randint(0, self.n_entity - 1)
                    if sample_failed_time < sample_failed_max:
                        hr_tail_set[(f[0], f[1])].add(tail)
                        neg_data.append((f[0], f[1], tail))
                else:
                    # sample head
                    if f[0] >= self.n_entity:
                        head = random.randint(self.n_entity, self.n_entity + self.n_user - 1)
                        while head in rt_head_set[(f[1], f[2])] and sample_failed_time < sample_failed_max:
                            sample_failed_time += 1
                            head = random.randint(self.n_entity, self.n_entity + self.n_user - 1)
                    else:
                        head = random.randint(0, self.n_entity - 1)
                        while head in rt_head_set[(f[1], f[2])] and sample_failed_time < sample_failed_max:
                            sample_failed_time += 1
                            head = random.randint(0, self.n_entity - 1)
                    if sample_failed_time < sample_failed_max:
                        rt_head_set[(f[1], f[2])].add(head)
                        neg_data.append((head, f[1], f[2]))

        random.shuffle(pos_data); random.shuffle(neg_data)
        pos_batches = np.array_split(pos_data, max(1, len(pos_data) // self.train_batch_size))
        neg_batches = np.array_split(neg_data, len(pos_batches))
        pos_batches = [b.transpose() for b in pos_batches]
        neg_batches = [b.transpose() for b in neg_batches]
        return [[pos_batches[i], neg_batches[i]] for i in range(len(pos_batches))]

# ===================== MODEL (TransE) =====================
class TransE(torch.nn.Module):
    def __init__(self, ent_num: int, rel_num: int, dataloader: Dataloader,
                 dim: int = 128, l1: bool = True, margin: float = 20,
                 learning_rate: float = 7e-4, weight_decay: float = 1e-5,
                 device_index: int = -1, grad_clip: float = 1.0):
        super().__init__()
        # auto CPU/GPU
        if device_index >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_index}")
        else:
            self.device = torch.device("cpu")

        self.ent_num, self.rel_num = ent_num, rel_num
        self.dataloader = dataloader
        self.dim, self.l1, self.margin = dim, l1, margin
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.grad_clip = grad_clip

        self.ent_embedding = torch.nn.Embedding(self.ent_num, self.dim, device=self.device)
        self.rel_embedding = torch.nn.Embedding(self.rel_num, self.dim, device=self.device)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, head, rel, tail) -> torch.Tensor:
        head = torch.as_tensor(head, dtype=torch.int64, device=self.device)
        rel  = torch.as_tensor(rel,  dtype=torch.int64, device=self.device)
        tail = torch.as_tensor(tail, dtype=torch.int64, device=self.device)

        h = self.dropout(self.ent_embedding(head))
        r = self.dropout(self.rel_embedding(rel))
        t = self.dropout(self.ent_embedding(tail))

        if self.l1:
            score = torch.sum(torch.abs(h + r - t), dim=-1, keepdim=True)
        else:
            score = torch.sum((h + r - t) ** 2, dim=-1, keepdim=True)
        return -score

    def optimize(self, pos, neg):
        pos_score = self.forward(pos[0], pos[1], pos[2])
        neg_score = self.forward(neg[0], neg[1], neg[2])
        pos_m = torch.matmul(pos_score, torch.t(torch.ones_like(neg_score)))
        neg_m = torch.t(torch.matmul(neg_score, torch.t(torch.ones_like(pos_score))))
        loss = torch.mean(torch.clamp(neg_m - pos_m + self.margin, min=0))
        return loss

    @torch.no_grad()
    def _normalize_embeddings(self):
        self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, p=2, dim=1)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, p=2, dim=1)

    def ctr_eval(self, eval_batches: List[np.array]):
        eval_batches = [b.transpose() for b in eval_batches]
        scores = []
        for b in eval_batches:
            rel = [self.dataloader.rel_dict['feedback_recsys'] for _ in range(len(b[0]))]
            s = torch.squeeze(self.forward(b[0] + self.dataloader.n_entity, rel, b[1]), dim=-1)
            scores.append(s.detach().cpu().numpy())
        return np.concatenate(scores, axis=0)

    def top_k_eval(self, users: List[int], k: int = 5):
        item_list, train_user_pos_item = self.dataloader.get_user_pos_item_list()
        out = []
        for u in users:
            head = [u + self.dataloader.n_entity for _ in range(len(item_list))]
            rel  = [self.dataloader.rel_dict['feedback_recsys'] for _ in range(len(item_list))]
            tail = item_list
            scores = torch.squeeze(self.forward(head, rel, tail), dim=-1)
            order = np.argsort(scores.detach().cpu().numpy())[::-1]
            chosen = []
            for idx in order:
                if len(chosen) >= k: break
                if u not in train_user_pos_item or item_list[idx] not in train_user_pos_item[u]:
                    chosen.append(item_list[idx])
            out.append(chosen)
        return out

    def train_TransE(self, epoch_num: int, output_log=False):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4,)

        best = float("inf")
        wait, patience = 0, 10

        for ep in tqdm(range(epoch_num)):
            self.train(True)
            batches = self.dataloader.get_training_batch()
            losses = []
            for pos, neg in batches:
                loss = self.optimize(pos, neg)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                opt.step()
                losses.append(loss.detach().cpu().item())

            mean_loss = float(np.mean(losses)) if losses else 0.0
            sched.step(mean_loss)
            self._normalize_embeddings()

            if output_log:
                lr = opt.param_groups[0]["lr"]
                print(f"epoch {ep+1}/{epoch_num} loss={mean_loss:.6f} lr={lr:.2e}")

            if mean_loss + 1e-7 < best:
                best = mean_loss; wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if output_log: print("Early stopping.")
                    break

# ===================== WRAPPER =====================
class KGRS:
    def __init__(self, train_pos: np.array, train_neg: np.array, kg_lines: List[str], n_user: int, n_item=int):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        config = {
            "batch_size": 512,
            "eval_batch_size": 1024,
            "neg_rate": 4.0,
            "emb_dim": 128,
            "l1": True,
            "margin": 20,
            "learning_rate": 7e-4,
            "weight_decay": 1e-5,
            "epoch_num": 180
        }
        self.batch_size = config["batch_size"]
        self.eval_batch_size = config["eval_batch_size"]
        self.neg_rate = config["neg_rate"]
        self.emb_dim = config["emb_dim"]
        self.l1 = config["l1"]
        self.margin = config["margin"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.epoch_num = config["epoch_num"]
        self.device_index = 0 if torch.cuda.is_available() else -1

        self.kg = kg_lines
        self.dataloader = Dataloader(
            train_pos, train_neg, self.kg, n_user=n_user, n_item=n_item,
            neg_rate=self.neg_rate, train_batch_size=self.batch_size
        )
        self.model = TransE(
            ent_num=self.dataloader.ent_num,
            rel_num=self.dataloader.rel_num,
            dataloader=self.dataloader,
            margin=self.margin,
            dim=self.emb_dim,
            l1=self.l1,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device_index=self.device_index,
            grad_clip=1.0
        )

    def training(self):
        self.model.train_TransE(epoch_num=self.epoch_num)
        self.model.train(False)
        self.model.eval()

    def eval_ctr(self, test_data: np.array) -> np.array:
        self.model.eval()
        eval_batches = np.array_split(test_data, max(1, len(test_data) // self.eval_batch_size))
        with torch.no_grad():
            return self.model.ctr_eval(eval_batches)

    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        self.model.eval()
        with torch.no_grad():
            return self.model.top_k_eval(users, k=k)
