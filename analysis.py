import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from matching import MatchEngine, learn_weights, DEFAULT_W
from generate_data import hidden_preferences, pair_features

DARK, MOSS, BEIGE, ROSY, MID = "#0A3323", "#839958", "#F7F4D5", "#D3968C", "#105666"
os.makedirs("results", exist_ok=True)
sns.set_theme(style="whitegrid", rc={"axes.facecolor": BEIGE, "figure.facecolor": BEIGE, "grid.color": "#DDD9AE",
                                     "axes.edgecolor": DARK, "text.color": DARK, "axes.labelcolor": DARK,
                                     "xtick.color": DARK, "ytick.color": DARK})

eng, FEAT = pair_features("users.csv")
users, idx = eng.users, eng.idx
feedback = pd.read_csv("feedback.csv")
N = len(users)
_, arch, TRUE_W, THR, P_TRUE = hidden_preferences(FEAT)


def train_data(fb, u):
    d = fb[fb.user_id == u]
    return np.array([FEAT[idx[u], idx[m]] for m in d.matched_user_id]), d.action.values


# hold-out AUC
fb_sorted = feedback.sort_values(["user_id", "timestamp"])
tr, te = [], []
for u, g in fb_sorted.groupby("user_id"):
    k = int(round(len(g) * 0.6)); tr += list(g.index[:k]); te += list(g.index[k:])
train, test = feedback.loc[tr], feedback.loc[te]
learned_tr = {u: learn_weights(*train_data(train, u)) for u in users.user_id}


def mean_auc(weight_fn):
    a = []
    for u, g in test.groupby("user_id"):
        if g.action.nunique() < 2:
            continue
        X = np.array([FEAT[idx[u], idx[m]] for m in g.matched_user_id])
        a.append(roc_auc_score(g.action, X @ weight_fn(u)))
    return float(np.mean(a)), len(a)


auc_default, n_eval = mean_auc(lambda u: DEFAULT_W)
auc_learned, _ = mean_auc(lambda u: learned_tr[u])
auc_oracle, _ = mean_auc(lambda u: TRUE_W[idx[u]])

# expected top-5 acceptance
USER_WEIGHTS = {u: learn_weights(*train_data(feedback, u)) for u in users.user_id}


def top5_expected(weight_fn, k=5):
    out = []
    for u in users.user_id:
        i = idx[u]; seen = {idx[m] for m in feedback[feedback.user_id == u].matched_user_id}
        cand = np.array([j for j in range(N) if j != i and j not in seen])
        top = cand[np.argsort(-(FEAT[i, cand] @ weight_fn(u)))[:k]]
        out.append(P_TRUE[i, top].mean())
    return float(np.mean(out))


rand_rate = float(np.mean([np.delete(P_TRUE[i], i).mean() for i in range(N)]))
acc_default = top5_expected(lambda u: DEFAULT_W)
acc_learned = top5_expected(lambda u: USER_WEIGHTS[u])
acc_oracle = top5_expected(lambda u: TRUE_W[idx[u]])

# online simulation
POOL, ROUNDS = 25, 8


def simulate(adaptive, seed, k=5):
    rs = np.random.default_rng(seed)
    hX = {u: [] for u in users.user_id}; hy = {u: [] for u in users.user_id}
    W = {u: DEFAULT_W.copy() for u in users.user_id}; curve = []
    for _ in range(ROUNDS):
        exp = []
        for u in users.user_id:
            i = idx[u]
            cand = rs.choice(np.array([j for j in range(N) if j != i]), size=POOL, replace=False)
            top = cand[np.argsort(-(FEAT[i, cand] @ W[u]))[:k]]
            p = P_TRUE[i, top]; exp.append(p.mean())
            y = (rs.random(k) < p).astype(int)
            hX[u] += list(FEAT[i, top]); hy[u] += list(y)
            if adaptive:
                W[u] = learn_weights(np.array(hX[u]), np.array(hy[u]))
        curve.append(np.mean(exp))
    return np.array(curve)


adaptive = np.mean([simulate(True, s) for s in range(5)], axis=0)
static = np.mean([simulate(False, s) for s in range(5)], axis=0)

# charts
fig, ax = plt.subplots(figsize=(7.5, 4))
r = np.arange(1, ROUNDS + 1)
ax.plot(r, 100 * static, "o--", color=ROSY, label="Static default weights")
ax.plot(r, 100 * adaptive, "o-", color=MID, label="Adaptive (learns from feedback)")
ax.axhline(100 * rand_rate, ls=":", color=DARK, lw=1, label="Random baseline")
ax.set_xlabel("Feedback round (5 recommendations each)"); ax.set_ylabel("Top-5 acceptance rate (%)")
ax.set_title("Feedback loop improves acceptance over time"); ax.legend()
plt.tight_layout(); plt.savefig("results/acceptance_curve.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
vals = [auc_default, auc_learned, auc_oracle]
ax.bar(["Default\nweights", "Learned\nweights", "Oracle\n(upper bound)"], vals, color=[ROSY, MID, MOSS], width=.55)
ax.set_ylim(.5, 1); ax.set_ylabel("Mean hold-out ROC-AUC (per user)")
for i, v in enumerate(vals):
    ax.text(i, v + .008, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig("results/holdout_auc.png", dpi=150); plt.close()

W_df = pd.DataFrame(USER_WEIGHTS, index=["Text", "MBTI", "Location"]).T
W_df["archetype"] = arch
summary_w = W_df.groupby("archetype").mean().round(3)
fig, ax = plt.subplots(figsize=(7.5, 4))
plot_df = summary_w.reset_index().melt(id_vars="archetype", var_name="weight", value_name="value")
sns.barplot(data=plot_df, x="archetype", y="value", hue="weight", ax=ax, palette=[MID, ROSY, MOSS])
for k, v in enumerate(DEFAULT_W):
    ax.axhline(v, ls=":", lw=.8, color=[MID, ROSY, MOSS][k])
ax.set_title("Learned weights per user type (dotted = default weights)"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig("results/learned_weights.png", dpi=150); plt.close()


# data for the dashboard's report tab
pd.DataFrame({"round": r, "static": 100 * static, "adaptive": 100 * adaptive}).to_csv("results/acceptance_curve.csv", index=False)
summary_w.rename(columns={"Text": "Text", "MBTI": "MBTI", "Location": "Location"}).to_csv("results/weights_by_archetype.csv")
# How predictive is each signal on its own? Mean per-user AUC of the raw signal against Accept/Reject (no training involved).
feat_auc = {}
for k, name in enumerate(["Text similarity", "MBTI pairing", "Location"]):
    a = []
    for u, g in feedback.groupby("user_id"):
        if g.action.nunique() < 2:
            continue
        x = np.array([FEAT[idx[u], idx[m]][k] for m in g.matched_user_id])
        a.append(roc_auc_score(g.action, x))
    feat_auc[name] = float(np.mean(a))
pd.DataFrame({"signal": list(feat_auc), "auc": list(feat_auc.values())}).to_csv("results/feature_auc.csv", index=False)

# report
metrics = dict(auc_default=auc_default, auc_learned=auc_learned, auc_oracle=auc_oracle, users_evaluated=n_eval,
               random=rand_rate, top5_default=acc_default, top5_learned=acc_learned, top5_oracle=acc_oracle,
               sim_round1=float(adaptive[0]), sim_round8_adaptive=float(adaptive[-1]),
               sim_round8_static=float(static[-1]))
json.dump(metrics, open("results/metrics.json", "w"), indent=1)
report = f"""# Performance analysis

Data: `users.csv` (120 users) and `feedback.csv` ({len(feedback)} Accept/Reject actions), both simulated with hidden
user preferences (see `generate_data.py`).

## Hold-out ROC-AUC (first 60% of each user's feedback trains, last 40% tests; {n_eval} users with both outcomes)

| Weights | Mean per-user AUC |
|---|---|
| Default 0.5 / 0.3 / 0.2 | {auc_default:.3f} |
| Learned per user | {auc_learned:.3f} |
| Oracle (true hidden weights, upper bound) | {auc_oracle:.3f} |

## Top-5 acceptance rate, round by round (cold start, 8 rounds, 5 seeds)

| | Round 1 | Round 8 |
|---|---|---|
| Static default weights | {100*static[0]:.0f}% | {100*static[-1]:.0f}% |
| Adaptive (learns from feedback) | {100*adaptive[0]:.0f}% | {100*adaptive[-1]:.0f}% |

Random recommendations would be accepted {100*rand_rate:.0f}% of the time.

Expected top-5 acceptance on unseen candidates, using weights trained on the whole feedback log:
default {100*acc_default:.1f}%, learned {100*acc_learned:.1f}%, oracle {100*acc_oracle:.1f}%.

## Caveat
Feedback is simulated from hidden preferences built from the same three features the model uses, so this shows the
model can recover per-user preferences, not that it works on real users.

Charts: `acceptance_curve.png`, `holdout_auc.png`, `learned_weights.png`.
"""
open("results/performance_report.md", "w").write(report)
print(report)
