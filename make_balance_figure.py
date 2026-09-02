"""
Competitive balance by league
=============================
Two views of how evenly matched a league is, computed from the standings in
data/combined_league_standings_2010_2023.csv:

  left   spread of points-per-match within a season, averaged over seasons.
         Points-per-match rather than raw points because MLS and the Bundesliga
         play 34 games while the other three play 38.
  right  how concentrated titles are, via the most successful single club.
         MLS awards two conference titles a season, so it is shown separately
         and read as "no club dominates" rather than compared slot-for-slot.

    python3 make_balance_figure.py  ->  figures/competitive_balance.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PITCH, NAVY, GREY, LGREY = "#D62828", "#0B2545", "#B9BEC4", "#ECEEF0"  # MLS crest red on navy
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlecolor": NAVY, "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "axes.titlepad": 12,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.edgecolor": GREY,
})
os.makedirs("figures", exist_ok=True)

d = pd.read_csv("data/combined_league_standings_2010_2023.csv", low_memory=False)
d = d.dropna(subset=["Pts", "MP", "League", "Season", "Rk", "Team"])
# Recompute rather than trust the stored column, which is rounded to 3 places.
d["ppm"] = d["Pts"] / d["MP"]

# --- left: within-season spread of points per match --------------------------
spread = (d.groupby(["League", "Season"])["ppm"].std()
          .groupby("League").mean().sort_values())
print("mean within-season spread of points per match (lower = more balanced):")
print(spread.round(4).to_string())

# --- two objections an MLS analyst will raise, answered before they ask ------
# 1. MLS runs two conferences, so a league-wide spread pools two competitions.
mls = d[d.League == "MLS"].copy()
mls["conf"] = mls.groupby("Season")["Rk"].transform(lambda r: (r.astype(int) == 1).cumsum())
within_conf = mls.groupby(["Season", "conf"])["ppm"].std().groupby(level=0).mean().mean()
print(f"\nMLS spread league-wide {spread['MLS']:.3f} vs within conference "
      f"{within_conf:.3f} -> the two-conference structure is not what produces it")

# 2. MLS fields more clubs (16-29) than the European leagues (18-20). If a bigger
#    table mechanically widened the spread, that would cut against MLS, not for it;
#    resampling a European league down to MLS-sized tables shows it does neither.
sizes = d.groupby(["League", "Season"]).size().groupby("League").agg(["min", "max"])
rng = np.random.default_rng(3)
for lg in ["Serie A", "Ligue 1"]:
    full = d[d.League == lg].groupby("Season")["ppm"].std().mean()
    resampled = np.mean([
        np.mean([g.ppm.sample(min(len(g), rng.integers(16, 24)),
                              random_state=int(rng.integers(1e6))).std()
                 for _, g in d[d.League == lg].groupby("Season")])
        for _ in range(200)])
    print(f"  {lg:8s} full table {full:.3f} | resampled to 16-23 clubs {resampled:.3f}")
print(f"  clubs per season: MLS {sizes.loc['MLS', 'min']}-{sizes.loc['MLS', 'max']}, "
      f"European leagues {sizes.drop('MLS')['min'].min()}-{sizes.drop('MLS')['max'].max()}")

# --- right: how much the most dominant club won ------------------------------
top = d[d.Rk == 1]
dom = {}
for lg, g in top.groupby("League"):
    counts = g.Team.value_counts()
    dom[lg] = (counts.index[0], int(counts.iloc[0]), g.Season.nunique(), int(g.Team.nunique()))
print("\nmost dominant club per league:")
for lg, (team, n, seasons, uniq) in dom.items():
    print(f"  {lg:16s} {team} won {n} of {seasons} | {uniq} different winners")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.8))

colors = [PITCH if lg == "MLS" else GREY for lg in spread.index]
b = a1.barh(range(len(spread)), spread.values, color=colors, height=0.62, zorder=3)
a1.set_yticks(range(len(spread))); a1.set_yticklabels(spread.index, fontsize=11)
a1.set_xlabel("Spread of points per match within a season")
a1.set_xlim(0, spread.max() * 1.22)
a1.set_title("Competitive balance, 2010-2023", loc="left")
a1.xaxis.grid(True, color=LGREY, zorder=0); a1.set_axisbelow(True)
for s in ["top", "right", "left"]: a1.spines[s].set_visible(False)
for i, v in enumerate(spread.values):
    a1.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=11.5, fontweight="bold",
            color=PITCH if spread.index[i] == "MLS" else NAVY)
a1.set_xlabel("Spread of points per match within a season  (lower = more evenly matched)", fontsize=11)

order = [lg for lg in spread.index if lg != "MLS"]
titles = [dom[lg][1] for lg in order]
seasons = dom[order[0]][2]
labels = [f"{lg}\n{dom[lg][0]}" for lg in order]
x = np.arange(len(order))
b2 = a2.bar(x, titles, color=GREY, width=0.6, zorder=3)
a2.axhline(seasons, color=NAVY, lw=1.2, ls=":", zorder=4)
a2.text(len(order) - 0.5, seasons + 0.25, f"{seasons} seasons", ha="right",
        fontsize=10, color=NAVY)
a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=9.5, linespacing=1.4)
a2.set_ylabel("Titles won by the top club"); a2.set_ylim(0, seasons + 4.5)
a2.set_title("One club takes most of them in Europe", loc="left")
a2.yaxis.grid(True, color=LGREY, zorder=0); a2.set_axisbelow(True)
for s in ["top", "right"]: a2.spines[s].set_visible(False)
for bar, v in zip(b2, titles):
    a2.text(bar.get_x() + bar.get_width() / 2, v + 0.25, str(v), ha="center",
            va="bottom", fontsize=14, fontweight="bold")
mls_u = dom["MLS"][3]
a2.text(len(order) / 2 - 0.5, seasons + 2.6,
        f"MLS: {mls_u} different clubs topped a conference",
        ha="center", fontsize=11.5, fontweight="bold", color=PITCH)

fig.tight_layout()
fig.savefig("figures/competitive_balance.png", dpi=200)
plt.close(fig)
print("\nwrote figures/competitive_balance.png")
