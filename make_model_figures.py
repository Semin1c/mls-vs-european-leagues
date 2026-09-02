"""
How MLS differs from Europe's big five on the pitch
===================================================
Companion to make_balance_figure.py. Everything here is computed from the two
committed datasets, so the figures regenerate from scratch:

  data/combined_league_standings_2010_2023.csv   final tables, 2010-2023
  data/final_merged_fixtures_2010-2024.csv       team-season performance, 1,680 rows

Writes five figures:

  figures/parity_gini.png        Gini of points by league - a second, schedule-free
                                 check on the balance result
  figures/home_advantage.png     home win % minus away win %, and share of one-goal
                                 matches, by league
  figures/counts_vs_rates.png    why season length has to be divided out before
                                 any "MLS scores less" claim
  figures/accuracy_is_the_wrong_metric.png
                                 85.6% of team-seasons are European, so accuracy is
                                 nearly free; MLS recall shows what the model misses
  figures/model_auc.png          the same three feature sets judged on AUC with
                                 class weighting, which is the honest comparison

    python3 make_model_figures.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PITCH, NAVY, GREY, LGREY = "#D62828", "#0B2545", "#B9BEC4", "#ECEEF0"  # MLS crest red on navy
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlecolor": NAVY, "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "axes.titlepad": 12,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.edgecolor": GREY,
})
os.makedirs("figures", exist_ok=True)

# Matches per season. MLS and the Bundesliga play 34; the rest play 38. Every
# counting stat below has to be divided by this before leagues are comparable.
SEASON_LENGTH = {"MLS": 34, "BUNDESLIGA": 34, "LA LIGA": 38,
                 "LIGUE 1": 38, "PREMIER LEAGUE": 38, "SERIE A": 38}
PRETTY = {"MLS": "MLS", "BUNDESLIGA": "Bundesliga", "LA LIGA": "La Liga",
          "LIGUE 1": "Ligue 1", "PREMIER LEAGUE": "Premier League", "SERIE A": "Serie A"}


def bar_colors(labels):
    return [PITCH if l == "MLS" else GREY for l in labels]


def strip(ax, keep_left=True):
    for s in ["top", "right"] + ([] if keep_left else ["left"]):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Gini of points -- an independent check on the balance headline
# ---------------------------------------------------------------------------
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if x.sum() == 0:
        return 0.0
    return np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum())


st = pd.read_csv("data/combined_league_standings_2010_2023.csv", low_memory=False)
st = st.dropna(subset=["Pts", "MP", "League", "Season", "Rk", "Team"])

per_season = st.groupby(["League", "Season"])["Pts"].apply(gini).rename("gini").reset_index()
mean_gini = per_season.groupby("League")["gini"].mean().sort_values()
lowest = per_season.pivot(index="Season", columns="League", values="gini").idxmin(axis=1)
mls_lowest, n_seasons = int((lowest == "MLS").sum()), lowest.size

print("mean Gini of points by league (lower = more equal):")
print(mean_gini.round(4).to_string())
print(f"MLS is the most equal league in {mls_lowest} of {n_seasons} seasons\n")

fig, ax = plt.subplots(figsize=(8.6, 4.3))
ax.barh(range(len(mean_gini)), mean_gini.values,
        color=bar_colors(mean_gini.index), height=0.62, zorder=3)
ax.set_yticks(range(len(mean_gini)))
ax.set_yticklabels(mean_gini.index, fontsize=11)
ax.set_xlim(0, mean_gini.max() * 1.25)
ax.set_xlabel("Gini coefficient of end-of-season points  (lower = more evenly matched)",
              fontsize=11)
ax.set_title("Gini coefficient of end-of-season points, by league", loc="left")
ax.xaxis.grid(True, color=LGREY, zorder=0)
strip(ax, keep_left=False)
for i, v in enumerate(mean_gini.values):
    ax.text(v + mean_gini.max() * 0.02, i, f"{v:.3f}", va="center", fontsize=11.5,
            fontweight="bold", color=PITCH if mean_gini.index[i] == "MLS" else NAVY)
fig.tight_layout()
fig.savefig("figures/parity_gini.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Home advantage -- what actually separates MLS on the pitch
# ---------------------------------------------------------------------------
fx = pd.read_csv("data/final_merged_fixtures_2010-2024.csv", low_memory=False)
fx["lg"] = fx["league"].str.upper().str.strip()
fx["home_adv"] = fx["home_win_pct"] - fx["away_win_pct"]
fx["close_pct"] = (fx["home_close_match_pct"] + fx["away_close_match_pct"]) / 2

adv = (fx.groupby("lg")[["home_win_pct", "away_win_pct", "home_adv", "close_pct"]]
       .mean().rename(index=PRETTY).sort_values("home_adv"))

m, e = fx[fx.lg == "MLS"], fx[fx.lg != "MLS"]
t_adv, p_adv = stats.ttest_ind(m["home_adv"].dropna(), e["home_adv"].dropna(), equal_var=False)
t_cls, p_cls = stats.ttest_ind(m["close_pct"].dropna(), e["close_pct"].dropna(), equal_var=False)
print("home advantage (home win % - away win %) by league:")
print(adv.round(2).to_string())
print(f"\nhome advantage  MLS {m.home_adv.mean():.1f} vs Europe {e.home_adv.mean():.1f}  p={p_adv:.1e}")
print(f"close matches   MLS {m.close_pct.mean():.1f}% vs Europe {e.close_pct.mean():.1f}%  p={p_cls:.1e}\n")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

a1.barh(range(len(adv)), adv["home_adv"].values,
        color=bar_colors(adv.index), height=0.6, zorder=3)
a1.set_yticks(range(len(adv)))
a1.set_yticklabels(adv.index, fontsize=11)
a1.set_xlim(0, adv["home_adv"].max() * 1.28)
a1.set_xlabel("Home win rate minus away win rate (percentage points)", fontsize=11)
a1.set_title("Home advantage by league", loc="left")
a1.xaxis.grid(True, color=LGREY, zorder=0)
strip(a1, keep_left=False)
for i, v in enumerate(adv["home_adv"].values):
    a1.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=11.5, fontweight="bold",
            color=PITCH if adv.index[i] == "MLS" else NAVY)

cl = adv["close_pct"].sort_values()
a2.barh(range(len(cl)), cl.values, color=bar_colors(cl.index), height=0.6, zorder=3)
a2.set_yticks(range(len(cl)))
a2.set_yticklabels(cl.index, fontsize=11)
a2.set_xlim(0, cl.max() * 1.3)
a2.set_xlabel("Share of matches decided by one goal or fewer (%)", fontsize=11)
a2.set_title("Matches decided by one goal or fewer", loc="left")
a2.xaxis.grid(True, color=LGREY, zorder=0)
strip(a2, keep_left=False)
for i, v in enumerate(cl.values):
    a2.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=11.5, fontweight="bold",
            color=PITCH if cl.index[i] == "MLS" else NAVY)

fig.tight_layout()
fig.savefig("figures/home_advantage.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Counts vs rates -- the season-length correction
# ---------------------------------------------------------------------------
fx["mp"] = fx["lg"].map(SEASON_LENGTH)
fx["goals_per_match"] = fx["goals"] / fx["mp"]
fx["yellows_per_match"] = fx["yellow_cards"] / fx["mp"]

# Cards as well as goals, since the README quotes both.
fx["yellows_per_match"] = fx["yellow_cards"] / fx["mp"]
cards = pd.DataFrame({"per season": fx.groupby("lg")["yellow_cards"].mean(),
                      "per match": fx.groupby("lg")["yellows_per_match"].mean()}
                     ).rename(index=PRETTY).sort_values("per match", ascending=False)
print("yellow cards per season vs per match:")
print(cards.round(3).to_string())
print(f"  MLS {cards.loc['MLS', 'per match']:.3f} per match vs European average "
      f"{cards.drop('MLS')['per match'].mean():.3f}\n")

raw = fx.groupby("lg")["goals"].mean().rename(index=PRETTY)
per = fx.groupby("lg")["goals_per_match"].mean().rename(index=PRETTY)
order = raw.sort_values().index
raw, per = raw[order], per[order]
print("goals per season vs goals per match:")
print(pd.DataFrame({"per season": raw, "per match": per}).round(3).to_string(), "\n")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.4), sharey=True)
for ax, series, title, xlabel, fmt in [
    (a1, raw, "Goals per season", "Goals scored per season", "{:.1f}"),
    (a2, per, "Goals per match", "Goals scored per match", "{:.3f}"),
]:
    ax.barh(range(len(series)), series.values, color=bar_colors(series.index),
            height=0.6, zorder=3)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(series.index, fontsize=11)
    ax.set_xlim(0, series.max() * 1.3)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, loc="left", fontsize=13)
    ax.xaxis.grid(True, color=LGREY, zorder=0)
    strip(ax, keep_left=False)
    for i, v in enumerate(series.values):
        ax.text(v + series.max() * 0.025, i, fmt.format(v), va="center", fontsize=11.5,
                fontweight="bold", color=PITCH if series.index[i] == "MLS" else NAVY)
a1.text(0.02, -0.30, "MLS and the Bundesliga play 34 matches a season; the other four play 38.",
        transform=a1.transAxes, fontsize=10.5, color=NAVY)
fig.tight_layout()
fig.savefig("figures/counts_vs_rates.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Can a model tell the leagues apart, and on what?
# ---------------------------------------------------------------------------
fx["is_european"] = (fx["lg"] != "MLS").astype(int)
drop = ["league", "season", "team", "lg", "is_european", "mp",
        "home_adv", "close_pct", "goals_per_match", "yellows_per_match"]
features = [c for c in fx.select_dtypes(include=["int64", "float64"]).columns if c not in drop]

# Season-length-sensitive totals, versus everything already expressed as a rate.
COUNTS = ["goals", "assists", "goals_and_assists", "goals_excluding_pk",
          "penalty_kicks", "penalty_attempts", "yellow_cards", "red_cards"]
RATES = [c for c in features if c not in COUNTS]

model_df = fx.dropna(subset=features + ["is_european"])
y = model_df["is_european"]
cv = StratifiedKFold(5, shuffle=True, random_state=42)

# Only 14% of team-seasons are MLS, so plain accuracy is nearly free: a model that
# answers "European" every single time already scores the majority-class rate.
# Class weighting plus AUC is what actually shows whether the leagues separate.
baseline = y.mean()


def evaluate(cols, balanced):
    lr = LogisticRegression(max_iter=5000, random_state=42,
                            class_weight="balanced" if balanced else None)
    pipe = make_pipeline(StandardScaler(), lr)
    score = lambda m, target: cross_val_score(pipe, model_df[cols], target, cv=cv,
                                              scoring=m, n_jobs=-1).mean()
    return {"accuracy": score("accuracy", y), "auc": score("roc_auc", y),
            "mls_recall": score("recall", 1 - y)}


# A navy ramp here rather than the pitch green: green means MLS everywhere else in
# this project, and these bars are feature sets, not leagues.
SETC = ["#9FB3C8", "#4A6C8C", NAVY]
sets = [("Rate statistics only", RATES, SETC[0]),
        ("Season totals only", COUNTS, SETC[1]),
        ("Everything together", features, SETC[2])]
plain = {n: evaluate(c, False) for n, c, _ in sets}
weighted = {n: evaluate(c, True) for n, c, _ in sets}

# The same club appears in up to 14 seasons, so random folds can put Atlanta United
# 2019 in train and Atlanta United 2020 in test. Hold out whole clubs, and whole
# seasons, to check the result is not just recognising familiar teams.
def grouped_check(cols):
    out = {}
    for label, cvobj, groups in [
        ("random folds", cv, None),
        ("held-out clubs", GroupKFold(5), model_df["team"]),
        ("held-out seasons", GroupKFold(5), model_df["season"]),
    ]:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=5000, random_state=42, class_weight="balanced"))
        out[label] = (
            cross_val_score(pipe, model_df[cols], y, cv=cvobj, groups=groups,
                            scoring="roc_auc", n_jobs=-1).mean(),
            cross_val_score(pipe, model_df[cols], 1 - y, cv=cvobj, groups=groups,
                            scoring="recall", n_jobs=-1).mean())
    return out


robust = grouped_check(features)
print(f"class balance: European {int(y.sum())} ({baseline:.1%}), MLS {int((1 - y).sum())}")
print(f"distinct clubs {model_df.team.nunique()} (MLS: "
      f"{model_df[model_df.lg == 'MLS'].team.nunique()})")
print("class-weighted model, tested three ways:")
for label, (auc, rec) in robust.items():
    print(f"  {label:17s} AUC {auc:.3f} | MLS recall {rec:.3f}")
print("league classification, 5-fold cross-validated:")
for n, _, _ in sets:
    p, w = plain[n], weighted[n]
    print(f"  {n:22s} accuracy {p['accuracy']:.3f} | AUC {p['auc']:.3f} | "
          f"MLS recall {p['mls_recall']:.3f} -> class-weighted {w['mls_recall']:.3f}")
print(f"  {'always guess European':22s} accuracy {baseline:.3f} | AUC 0.500 | MLS recall 0.000\n")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 4.7))
ys = np.arange(len(sets))
names = [n for n, _, _ in sets]
cols = [c for _, _, c in sets]

for ax, key, title, xlabel in [
    (a1, "accuracy", "Accuracy", "Accuracy"),
    (a2, "mls_recall", "MLS team-seasons identified",
     "Share of MLS team-seasons correctly identified"),
]:
    vals = [plain[n][key] for n in names]
    ax.barh(ys, vals, color=cols, height=0.58, zorder=3)
    ax.set_xlim(0, 1.16)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, loc="left", fontsize=13.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=LGREY, zorder=0)
    strip(ax, keep_left=False)
    for yy, v, c in zip(ys, vals, cols):
        ax.text(v + 0.016, yy, f"{v:.1%}", va="center", fontsize=12.5,
                fontweight="bold", color=c)

a1.axvline(baseline, color=PITCH, lw=1.6, ls=":", zorder=4)
a1.annotate(f"answering \"European\"\nevery time scores {baseline:.1%}",
            xy=(baseline, 2.42), xytext=(baseline - 0.06, 2.62),
            ha="right", va="center", fontsize=10.5, color=PITCH, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PITCH, lw=1.3))
a1.set_ylim(2.95, -0.6)
a2.set_ylim(2.95, -0.6)

fig.tight_layout()
fig.savefig("figures/accuracy_is_the_wrong_metric.png", dpi=200)
plt.close(fig)

# The corrected model: class weighting, judged on AUC.
fig, ax = plt.subplots(figsize=(9.6, 4.3))
aucs = [weighted[n]["auc"] for n in names]
ax.barh(ys, aucs, color=cols, height=0.58, zorder=3)
ax.axvline(0.5, color=NAVY, lw=1.4, ls=":", zorder=4)
ax.text(0.5 - 0.012, len(sets) - 0.42, "coin flip", ha="right", va="center",
        fontsize=10.5, color=NAVY, fontweight="bold")
ax.set_ylim(len(sets) - 0.2, -0.6)
ax.set_yticks(ys)
ax.set_yticklabels(names, fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0, 1.1)
ax.set_xlabel("Area under the ROC curve, class-weighted model", fontsize=11)
ax.set_title("Classification AUC by feature set, class-weighted model", loc="left")
ax.xaxis.grid(True, color=LGREY, zorder=0)
strip(ax, keep_left=False)
for yy, v, c in zip(ys, aucs, cols):
    ax.text(v + 0.013, yy, f"{v:.3f}", va="center", fontsize=13, fontweight="bold", color=c)
fig.tight_layout()
fig.savefig("figures/model_auc.png", dpi=200)
plt.close(fig)

print("wrote figures/parity_gini.png, home_advantage.png, counts_vs_rates.png,")
print("      accuracy_is_the_wrong_metric.png, model_auc.png")
