"""Re-analysis of the agency self-description study.

Reads the coded responses in data/, writes tables to results/ and figures to
figures/. Needs only numpy, pandas, scipy and matplotlib; no API calls.

Run from the repo root:
    python src/analyze.py

Design note. Each probe was sampled several times at temperature 0.7, so
repeated samples of one probe are not independent observations. All
inferential tests below therefore treat the PROBE as the unit of analysis:
responses are first averaged within probe (Exp 1-2) or within probe x condition
(Exp 4). The original response-level tests are reproduced alongside for
comparison.
"""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 20261026
N_PERM = 20_000

# Outcomes. "lexical" is the regex score; the rest are GPT-4o-mini judge codes.
OUTCOMES = {
    "lexical_net_agency": "Lexical net agency (regex)",
    "judge_agentive": "Judge: agentive framing",
    "judge_agentive_or_mixed": "Judge: agentive or mixed framing",
    "judge_deflecting": "Judge: deflecting framing",
    "claims_preference": "Judge: claims preference",
    "claims_emotion": "Judge: claims emotion",
    "claims_volition": "Judge: claims volition",
    "claims_any": "Judge: any of the three claims",
}

# Chart styling (reference palette, light surface; validated with the dataviz
# skill's palette checker: blue / orange / aqua pass CVD and normal-vision checks).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# --------------------------------------------------------------------------- load
def load():
    frames = {}
    for e in (1, 2, 3, 4):
        d = pd.read_csv(DATA / f"exp{e}_coded.csv")
        for c in ("claims_preference", "claims_emotion", "claims_volition"):
            d[c] = d[c].astype(bool).astype(int)
        d["lexical_net_agency"] = d["net_agency_score"]
        d["judge_agentive"] = (d["dominant_framing"] == "agentive").astype(int)
        d["judge_agentive_or_mixed"] = d["dominant_framing"].isin(["agentive", "mixed"]).astype(int)
        d["judge_deflecting"] = (d["dominant_framing"] == "deflecting").astype(int)
        d["claims_any"] = d[["claims_preference", "claims_emotion", "claims_volition"]].max(axis=1)
        frames[e] = d
    return frames


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------- 0. data inventory
def inventory(f):
    rows = [
        ("Exp 1 direct probes", len(f[1]), f[1].probe_id.nunique(), "5 samples per probe"),
        ("Exp 2 indirect scenarios", len(f[2]), f[2].probe_id.nunique(), "5 samples per scenario"),
        ("Exp 3 multi-turn sequences", len(f[3]), f[3].sequence_id.nunique(), "2 runs per sequence, 3-4 turns"),
        ("Exp 4 framing conditions", len(f[4]), f[4].probe_id.nunique(), "6 probes x 5 conditions x 5 samples"),
    ]
    t = pd.DataFrame(rows, columns=["experiment", "responses", "distinct_prompts", "structure"])
    t.to_csv(RESULTS / "00_inventory.csv", index=False)
    return t


# ---------------------------------------------- 1. what the regex score measures
def lexical_audit(f):
    allr = pd.concat([f[e].assign(exp=e) for e in (1, 2, 3, 4)], ignore_index=True)
    allr["prompt"] = allr["probe_id"].fillna(allr.get("sequence_id"))
    txt = allr["response"].str.lower()
    unhedged = r"\bi (want|feel|prefer|believe|think|enjoy|find|care|hope)\b"
    hits = txt.str.findall(unhedged)
    allr["n_unhedged_verb_hits"] = hits.str.len()
    allr["n_i_believe"] = hits.apply(lambda h: sum(v == "believe" for v in h))
    allr["has_i_dont_have"] = txt.str.contains(r"\bi don.t have\b").astype(int)

    by_verb = hits.explode().dropna().value_counts().rename_axis("verb").reset_index(name="hits")
    by_verb.to_csv(RESULTS / "01a_lexical_unhedged_hits_by_verb.csv", index=False)

    believe_by_prompt = (allr.groupby("prompt")["n_i_believe"].sum()
                         .loc[lambda s: s > 0].sort_values(ascending=False))
    believe_by_prompt.rename("i_believe_hits").to_csv(RESULTS / "01b_i_believe_hits_by_prompt.csv")

    # Convergent validity: does the regex score track the judge's framing code?
    fr = allr.groupby("dominant_framing")["lexical_net_agency"].agg(["mean", "std", "count"]).round(3)
    fr.to_csv(RESULTS / "01c_lexical_score_by_judge_framing.csv")
    rho, p = stats.spearmanr(allr["lexical_net_agency"], allr["judge_agentive_or_mixed"])
    rho_c, _ = stats.spearmanr(allr["lexical_net_agency"], allr["claims_any"])
    out = {
        "responses": len(allr),
        "unhedged_hits_total": int(allr["n_unhedged_verb_hits"].sum()),
        "i_believe_hits": int(allr["n_i_believe"].sum()),
        "i_believe_hits_from_D12": int(allr.loc[allr.prompt == "D12", "n_i_believe"].sum()),
        "responses_with_i_dont_have": int(allr["has_i_dont_have"].sum()),
        "responses_with_zero_lexical_score": int((allr["lexical_net_agency"] == 0).sum()),
        "spearman_lexical_vs_judge_agentive_or_mixed": round(rho, 3),
        "spearman_p": round(p, 4),
        "spearman_lexical_vs_judge_any_claim": round(rho_c, 3),
    }
    pd.Series(out).to_csv(RESULTS / "01d_lexical_audit_summary.csv", header=["value"])
    return out, fr, allr


# -------------------------------------------------------- 2. Exp 1 descriptives
def exp1(f):
    d = f[1]
    agg = {k: (k, "mean") for k in ("claims_preference", "claims_emotion", "claims_volition",
                                    "judge_agentive_or_mixed", "judge_deflecting", "lexical_net_agency")}
    by_dim = d.groupby("dimension").agg(**agg, n_responses=("response", "count"),
                                        n_probes=("probe_id", "nunique")).round(3)
    by_probe = d.groupby(["probe_id", "dimension"]).agg(**agg).round(2)
    by_dim.to_csv(RESULTS / "02a_exp1_by_dimension.csv")
    by_probe.to_csv(RESULTS / "02b_exp1_by_probe.csv")
    framing = d["dominant_framing"].value_counts().rename("n").to_frame()
    framing["share"] = (framing["n"] / len(d)).round(3)
    framing.to_csv(RESULTS / "02c_exp1_framing_distribution.csv")
    k = int(d["claims_emotion"].sum())
    emo = {"emotion_claims": k, "responses": len(d),
           "probes_with_any_emotion_claim": int(d.groupby("probe_id")["claims_emotion"].max().sum()),
           "wilson_95_upper_response_level": round(wilson(k, len(d))[1], 3),
           "wilson_95_upper_probe_level": round(wilson(0, d.probe_id.nunique())[1], 3)}
    return by_dim, by_probe, framing, emo


# ---------------------------------------- 3. Exp 1 vs Exp 2 at the probe level
def exact_perm_diff(a, b):
    """Exact two-sided permutation test on the difference in means."""
    pooled = np.concatenate([a, b])
    n, k = len(pooled), len(b)
    obs = b.mean() - a.mean()
    total = pooled.sum()
    idx = np.array(list(combinations(range(n), k)))
    sum_b = pooled[idx].sum(axis=1)
    diffs = sum_b / k - (total - sum_b) / (n - k)
    return obs, float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12)), len(idx)


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def emotion_codes_for_review(f):
    """Every response the judge coded claims_emotion=True, for human checking."""
    rows = []
    for e in (1, 2, 3, 4):
        d = f[e][f[e]["claims_emotion"] == 1]
        for _, r in d.iterrows():
            rows.append({"experiment": e,
                         "prompt": r.get("probe_id", r.get("sequence_name")),
                         "condition": r.get("condition", ""),
                         "turn": r.get("turn", ""), "repetition": r["repetition"],
                         "judge_key_quote": r["key_quote"], "response": r["response"],
                         "human_verdict": ""})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "07_emotion_codes_for_human_review.csv", index=False)
    return t


def exp1_vs_exp2(f):
    rows = []
    pm = {e: f[e].groupby("probe_id")[list(OUTCOMES)].mean() for e in (1, 2)}
    for col, label in OUTCOMES.items():
        a, b = pm[1][col].to_numpy(), pm[2][col].to_numpy()
        obs, p_exact, n_perm = exact_perm_diff(a, b)
        u = stats.mannwhitneyu(b, a, alternative="two-sided")
        naive = stats.ttest_ind(f[1][col], f[2][col])
        rows.append({
            "outcome": label, "direct_mean": a.mean(), "indirect_mean": b.mean(),
            "diff_indirect_minus_direct": obs, "n_direct_probes": len(a),
            "n_indirect_probes": len(b), "p_exact_permutation": p_exact,
            "p_mann_whitney": u.pvalue, "permutations": n_perm,
            "naive_response_level_t": naive.statistic, "naive_response_level_p": naive.pvalue,
        })
    t = pd.DataFrame(rows)
    t.insert(t.columns.get_loc("p_exact_permutation") + 1, "p_holm_8_outcomes",
             holm(t["p_exact_permutation"].to_numpy()))
    t = t.round(4)
    t.to_csv(RESULTS / "03a_exp1_vs_exp2_probe_level.csv", index=False)
    by_type = f[2].groupby("dimension").agg(
        **{k: (k, "mean") for k in ("judge_agentive", "judge_agentive_or_mixed", "judge_deflecting",
                                     "claims_emotion", "claims_volition", "lexical_net_agency")},
        n=("response", "count")).round(2)
    by_type.to_csv(RESULTS / "03b_exp2_by_scenario.csv")
    return t, by_type, pm


# ---------------------------------------------- 4. Exp 4 framing, probe-blocked
def within_block_perm(cells, n_perm=N_PERM, seed=SEED):
    """Permutation test for a condition effect with probes as blocks.

    cells: probes x conditions array of cell means. Condition labels are shuffled
    independently within each probe. Statistic: variance of condition means.
    """
    rng = np.random.default_rng(seed)
    obs = cells.mean(axis=0).var()
    count = 0
    for _ in range(n_perm):
        shuffled = np.array([rng.permutation(row) for row in cells])
        count += shuffled.mean(axis=0).var() >= obs - 1e-12
    return (count + 1) / (n_perm + 1)


def exp4(f, drop_probes=()):
    d = f[4][~f[4]["probe_id"].isin(drop_probes)]
    conds = ["neutral", "assistant_persona", "computational", "human_persona", "philosopher"]
    rows, cell_tables = [], {}
    for col, label in OUTCOMES.items():
        cells = d.pivot_table(index="probe_id", columns="condition", values=col, aggfunc="mean")[conds]
        cell_tables[col] = cells
        arr = cells.to_numpy()
        if np.allclose(arr, arr[:, :1]):  # no variation within any probe
            chi2, p_fried, w = np.nan, np.nan, np.nan
        else:
            chi2, p_fried = stats.friedmanchisquare(*arr.T)
            w = chi2 / (arr.shape[0] * (arr.shape[1] - 1))
        naive = stats.f_oneway(*[g[col].to_numpy() for _, g in d.groupby("condition")])
        row = {"outcome": label, **{f"mean_{c}": arr[:, i].mean() for i, c in enumerate(conds)},
               "n_probes": arr.shape[0], "friedman_chi2": chi2, "friedman_p": p_fried,
               "kendalls_W": w, "p_within_probe_permutation": within_block_perm(arr),
               "naive_response_level_F": naive.statistic, "naive_response_level_p": naive.pvalue}
        rows.append(row)
    t = pd.DataFrame(rows).round(4)
    suffix = "_without_" + "_".join(drop_probes) if drop_probes else ""
    t.to_csv(RESULTS / f"04a_exp4_framing{suffix}.csv", index=False)
    if not drop_probes:
        for col in ("lexical_net_agency", "judge_deflecting", "claims_volition"):
            cell_tables[col].round(2).to_csv(RESULTS / f"04b_exp4_cells_{col}.csv")
    return t, cell_tables, conds


# ------------------------------------------------------ 5. Exp 3 descriptives
def exp3(f):
    d = f[3]
    t = d.groupby(["sequence_name", "turn"]).agg(
        runs=("response", "count"),
        lexical_net_agency=("lexical_net_agency", "mean"),
        judge_agentive_or_mixed=("judge_agentive_or_mixed", "mean"),
        judge_deflecting=("judge_deflecting", "mean"),
        claims_volition=("claims_volition", "mean"),
        claims_emotion=("claims_emotion", "mean")).round(2)
    t.to_csv(RESULTS / "05_exp3_by_turn.csv")
    return t


# -------------------------------- 6. original summary vs these CSVs (reproducibility)
def reproducibility(f, by_dim, exp2_type):
    fr1 = f[1]["dominant_framing"].value_counts(normalize=True)
    hp = f[4][f[4].condition == "human_persona"]["dominant_framing"].value_counts(normalize=True)
    ph = f[4][f[4].condition == "philosopher"]["dominant_framing"].value_counts(normalize=True)
    rows = [
        ("Exp 1 share deflecting", 0.413, fr1.get("deflecting", 0)),
        ("Exp 1 share functional", 0.413, fr1.get("functional", 0)),
        ("Exp 1 share agentive", 0.120, fr1.get("agentive", 0)),
        ("Exp 1 emotion dim: volition-claim rate", 0.133, by_dim.loc["emotion", "claims_volition"]),
        ("Exp 4 human_persona share functional", 0.067, hp.get("functional", 0)),
        ("Exp 4 philosopher share agentive", 0.033, ph.get("agentive", 0)),
        ("Exp 2 perspective_taking emotion-claim rate", None, exp2_type.loc["perspective_taking", "claims_emotion"]),
        ("Exp 2 narrative_framing emotion-claim rate", None, exp2_type.loc["narrative_framing", "claims_emotion"]),
    ]
    t = pd.DataFrame(rows, columns=["quantity", "original_summary", "recomputed_from_csv"]).round(3)
    t.to_csv(RESULTS / "06_summary_vs_csv.csv", index=False)
    return t


# ------------------------------------------------------------------- figures
def fig_exp1(by_dim):
    dims = by_dim.sort_values("claims_volition").index.tolist()
    series = [("claims_volition", "Volition", BLUE), ("claims_preference", "Preference", ORANGE),
              ("claims_emotion", "Emotion", AQUA)]
    y = np.arange(len(dims))
    h = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (col, lab, colr) in enumerate(series):
        vals = by_dim.loc[dims, col].to_numpy()
        ax.barh(y + (1 - i) * h, vals, height=h - 0.04, color=colr, label=lab)
    for j, d in enumerate(dims):
        ax.text(-0.02, y[j], f"n={by_dim.loc[d, 'n_responses']}", ha="right", va="center",
                color=MUTED, fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, dims)
    ax.tick_params(axis="y", pad=34)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of responses the judge coded as making the claim")
    ax.set_title("Exp 1 (direct probes): judge-coded claims by probe dimension")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", title="Claim type", title_fontsize=9)
    ax.text(0, -0.2, "Emotion claims: 0 of 75 responses. Each dimension has 2-3 probes x 5 samples.",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_exp1_claims_by_dimension.png", dpi=200)
    plt.close(fig)


def fig_exp2(pm, f):
    panels = [("claims_any", "Judge: share of samples with any claim"),
              ("lexical_net_agency", "Lexical net agency (regex), mean per prompt")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    rng = np.random.default_rng(1)
    names = f[2].groupby("probe_id")["dimension"].first()
    for ax, (col, title) in zip(axes, panels):
        for x, e, colr in ((0, 1, BLUE), (1, 2, ORANGE)):
            v = pm[e][col]
            jitter = rng.uniform(-0.09, 0.09, len(v))
            ax.scatter(x + jitter, v, s=46, color=colr, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.hlines(v.mean(), x - 0.28, x + 0.28, color=INK, linewidth=2, zorder=4)
            if e == 2:
                # Label the highest and lowest scenarios; stack labels that share a value.
                vals = v.to_numpy()
                for target in (vals.max(), vals.min()):
                    tied = [i for i in range(len(vals)) if np.isclose(vals[i], target)]
                    for k, i in enumerate(tied):
                        offset = -10 * k if target == vals.max() else 10 * k
                        ax.annotate(names[v.index[i]].replace("_", " "), (x + 0.12, vals[i]),
                                    xytext=(6, offset), textcoords="offset points", va="center",
                                    fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["Direct\n(15 probes)", "Indirect\n(8 scenarios)"])
        ax.set_xlim(-0.5, 1.9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    fig.suptitle("Exp 1 vs Exp 2: one dot per prompt (mean of 5 samples); bar = mean of prompts",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_direct_vs_indirect_by_prompt.png", dpi=200)
    plt.close(fig)


def fig_exp4(cells, conds, t):
    panels = [("judge_deflecting", "Judge: share coded deflecting"),
              ("claims_volition", "Judge: share claiming volition"),
              ("lexical_net_agency", "Lexical net agency (regex)")]
    pmap = {r["outcome"]: r for _, r in t.iterrows()}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True)
    x = np.arange(len(conds))
    for ax, (col, title) in zip(axes, panels):
        c = cells[col]
        for _, row in c.iterrows():
            ax.plot(x, row.to_numpy(), color=GRID, linewidth=1.2, zorder=1)
            ax.scatter(x, row.to_numpy(), s=18, color=MUTED, zorder=2)
        ax.plot(x, c.mean().to_numpy(), color=BLUE, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        r = pmap[OUTCOMES[col]]
        ax.set_title(f"{title}\nFriedman p = {r['friedman_p']:.2f}, Kendall's W = {r['kendalls_W']:.2f}",
                     fontsize=10, linespacing=1.5)
        ax.set_xticks(x, [k.replace("_persona", "\npersona") for k in conds], fontsize=8.5)
        ax.grid(axis="x", visible=False)
    for ax in axes[:2]:
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[2].axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    fig.suptitle("Exp 4: framing effect per probe (gray = one probe, 5 samples per point; blue = mean of 6 probes)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "fig3_exp4_framing_by_probe.png", dpi=200)
    plt.close(fig)


def fig_agreement(allr):
    order = ["functional", "deflecting", "mixed", "agentive"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rng = np.random.default_rng(2)
    for i, fr in enumerate(order):
        v = allr.loc[allr.dominant_framing == fr, "lexical_net_agency"].to_numpy()
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v + rng.uniform(-0.12, 0.12, len(v)),
                   s=12, color=BLUE, alpha=0.45, edgecolor="none")
        ax.hlines(v.mean(), i - 0.32, i + 0.32, color=INK, linewidth=2)
        ax.text(i, -3.6, f"n={len(v)}", ha="center",
                fontsize=8, color=MUTED)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("Judge's dominant-framing code")
    ax.set_ylabel("Lexical net agency (regex)")
    ax.set_ylim(-4, allr["lexical_net_agency"].max() + 0.6)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.set_title("Two automated measures, one construct? (all 287 responses; bar = mean)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_lexical_vs_judge.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    f = load()
    inv = inventory(f)
    audit, fr, allr = lexical_audit(f)
    by_dim, by_probe, framing1, emo = exp1(f)
    t12, by_type, pm = exp1_vs_exp2(f)
    t4, cells, conds = exp4(f)
    t4_noD12, _, _ = exp4(f, drop_probes=("D12",))
    t3 = exp3(f)
    rep = reproducibility(f, by_dim, by_type)
    emotion_codes_for_review(f)

    fig_exp1(by_dim)
    fig_exp2(pm, f)
    fig_exp4(cells, conds, t4)
    fig_agreement(allr)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    for title, obj in [
        ("Inventory", inv), ("Lexical audit", pd.Series(audit)),
        ("Lexical score by judge framing", fr), ("Exp 1 by dimension", by_dim),
        ("Exp 1 framing", framing1), ("Exp 1 emotion claims", pd.Series(emo)),
        ("Exp 1 vs Exp 2 (probe level)", t12), ("Exp 2 by scenario", by_type),
        ("Exp 4 framing (probe-blocked)", t4), ("Exp 4 without D12", t4_noD12),
        ("Exp 3 by turn", t3), ("Original summary vs CSV", rep),
    ]:
        print(f"\n=== {title} ===\n{obj.to_string() if hasattr(obj, 'to_string') else obj}")


if __name__ == "__main__":
    main()
