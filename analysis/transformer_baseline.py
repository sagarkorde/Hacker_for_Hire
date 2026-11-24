"""Fine-tuned transformer baseline on the paper's vendor-disjoint folds (one seed per run).

Uses results/logs/folds_vendor_disjoint.csv written by classify_extra.py, so the folds are exactly
those of the other models. Inside each training fold, a vendor-disjoint 20% validation split is held
out; the model is trained on the remaining 80% and the decision threshold is the one that maximises
F1 on the validation split. The test fold is never used for training or for the threshold.

Writes results/logs/transformer_seed{seed}.json and transformer_oof_seed{seed}.csv.
Run (CPU is fine, about 1 h per seed for distilroberta-base on 4 cores):
    python analysis/transformer_baseline.py --seed 1
Merge seeds and compare with the rule: python analysis/transformer_merge.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--folds", default=str(LOG / "folds_vendor_disjoint.csv"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = pd.read_csv(a.folds, keep_default_na=False)
    assert d["fold"].nunique() == 5 and int(d["label"].sum()) == 451, "unexpected fold file"
    X, y, g = d["text"].astype(str).to_numpy(), d["label"].to_numpy(), d["vendor"].astype(str).to_numpy()
    tok = AutoTokenizer.from_pretrained(a.model)

    def enc(texts):
        return tok(list(texts), truncation=True, max_length=a.max_len, padding="max_length", return_tensors="pt")

    def proba(model, texts):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                e = {k: v.to(dev) for k, v in enc(texts[i:i + 256]).items()}
                out.append(torch.softmax(model(**e).logits, -1)[:, 1].float().cpu().numpy())
        return np.concatenate(out)

    def train(idx):
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)
        e = enc(X[idx])
        yt = torch.tensor(y[idx])
        pos = max(int(y[idx].sum()), 1)
        lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, (len(idx) - pos) / pos], dtype=torch.float).to(dev))
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps = a.epochs * int(np.ceil(len(idx) / a.batch))
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        for _ in range(a.epochs):
            model.train()
            perm = np.random.permutation(len(idx))
            for i in range(0, len(idx), a.batch):
                b = torch.as_tensor(perm[i:i + a.batch])
                out = model(input_ids=e["input_ids"][b].to(dev), attention_mask=e["attention_mask"][b].to(dev))
                loss = lossf(out.logits, yt[b].to(dev))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
        return model

    prob = np.zeros(len(d))
    pred = np.zeros(len(d), dtype=int)
    folds, t0, infer_s, infer_n = [], time.time(), 0.0, 0
    for k in range(5):
        tr = np.flatnonzero(d["fold"].to_numpy() != k)
        te = np.flatnonzero(d["fold"].to_numpy() == k)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed)
        i_fit, i_val = next(gss.split(tr, y[tr], g[tr]))
        fit, val = tr[i_fit], tr[i_val]
        model = train(fit)
        pv = proba(model, X[val])
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        thr = float(grid[int(np.argmax([f1_score(y[val], pv >= t, zero_division=0) for t in grid]))])
        s = time.time()
        prob[te] = proba(model, X[te])
        infer_s += time.time() - s
        infer_n += len(te)
        pred[te] = (prob[te] >= thr).astype(int)
        folds.append({"fold": k, "threshold": thr, "f1": float(f1_score(y[te], pred[te])),
                      "n_fit": int(len(fit)), "n_val": int(len(val)), "n_test": int(len(te))})
        print(json.dumps(folds[-1]), flush=True)
        del model
    res = {"model": a.model, "seed": a.seed, "max_len": a.max_len, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
           "device": dev, "torch_threads": torch.get_num_threads(), "train_minutes_total": (time.time() - t0) / 60,
           "inference_listings_per_s": infer_n / infer_s,
           "precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred)),
           "f1": float(f1_score(y, pred)), "ap": float(average_precision_score(y, prob)),
           "fold_f1_sd": float(np.std([f["f1"] for f in folds], ddof=1)), "folds": folds}
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / f"transformer_seed{a.seed}.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame({"row": d["row"], "prob": prob, "pred": pred}).to_csv(LOG / f"transformer_oof_seed{a.seed}.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}, indent=1))


if __name__ == "__main__":
    main()
