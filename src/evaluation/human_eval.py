"""Human-evaluation analysis for the revision (one or two raters).

Pure functions over the exported ratings CSV and the frozen study package. No
ticket text is emitted; the module works with item IDs and the private key.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

CORRECTNESS = ("correct", "partially_correct", "incorrect", "cannot_judge")
PREFERENCE = ("A", "B", "equal", "both_incorrect", "cannot_judge")
STRICT = {"correct": 1.0, "partially_correct": 0.0, "incorrect": 0.0, "cannot_judge": np.nan}
LENIENT = {"correct": 1.0, "partially_correct": 1.0, "incorrect": 0.0, "cannot_judge": np.nan}
ORDINAL = {"incorrect": 0, "partially_correct": 1, "correct": 2}
Z_ALPHA = 1.959963984540054
Z_POWER = 0.8416212335729143


@dataclass(frozen=True)
class Study:
    package: Path
    items: dict[str, dict]
    key: dict[str, dict]
    manifest: dict
    automatic: dict[tuple[str, str, str], float]
    package_id: str


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_id(package: Path) -> str:
    digest = hashlib.sha256()
    for name in ("items.json", "key.json", "manifest.json"):
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def load_study(package: Path, verify_runs: bool = True) -> Study:
    package = Path(package)
    items = {item["item_id"]: item for item in json.loads(
        (package / "items.json").read_text(encoding="utf-8"))}
    key = json.loads((package / "key.json").read_text(encoding="utf-8"))
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    automatic: dict[tuple[str, str, str], float] = {}
    for label, run in manifest["runs"].items():
        panel, source = label.split("/")
        details = Path(run["run_dir"]) / run["details"]
        if verify_runs and _sha(details) != run["details_sha256"]:
            raise ValueError(f"Frozen details changed for {label}: {details}")
        for record in json.loads(details.read_text(encoding="utf-8")):
            automatic[(panel, source, record["ticket_id"])] = float(
                record["deltas"]["delta_cosine"])
    return Study(package, items, key, manifest, automatic, package_id(package))


def load_ratings(path: Path, study: Study) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"rater", "item_id", "panel", "ticket_id", "source", "block",
                "a_correct", "b_correct", "preference", "a_side", "b_side",
                "package_id"}
    if missing := required - set(df.columns):
        raise ValueError(f"Ratings file is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Ratings file is empty")
    if df.item_id.duplicated().any():
        raise ValueError("Duplicate item_id in ratings")
    unknown = set(df.item_id) - set(study.items)
    if unknown:
        raise ValueError(f"Ratings contain items outside the package: {len(unknown)}")
    packages = set(df.package_id.astype(str))
    if packages != {study.package_id}:
        raise ValueError(
            f"Ratings package_id does not match the package: {sorted(packages)} vs {study.package_id}"
        )
    if not df.a_correct.isin(CORRECTNESS).all() or not df.b_correct.isin(CORRECTNESS).all():
        raise ValueError("Unexpected correctness labels")
    if not df.preference.isin(PREFERENCE).all():
        raise ValueError("Unexpected preference labels")
    if not set(df.a_side.unique()) <= {"baseline", "feedback"}:
        raise ValueError("a_side must be baseline or feedback")
    if not set(df.b_side.unique()) <= {"baseline", "feedback"}:
        raise ValueError("b_side must be baseline or feedback")
    if not (df.a_side != df.b_side).all():
        raise ValueError("a_side and b_side must differ")
    return df


def orient(df: pd.DataFrame, study: Study) -> pd.DataFrame:
    out = df.copy()

    def side(row, column, side):
        return row[f"a_{column}"] if row["a_side"] == side else row[f"b_{column}"]

    out["baseline_correct"] = [side(row, "correct", "baseline") for _, row in out.iterrows()]
    out["feedback_correct"] = [side(row, "correct", "feedback") for _, row in out.iterrows()]
    out["baseline_strict"] = out["baseline_correct"].map(STRICT)
    out["feedback_strict"] = out["feedback_correct"].map(STRICT)
    out["baseline_lenient"] = out["baseline_correct"].map(LENIENT)
    out["feedback_lenient"] = out["feedback_correct"].map(LENIENT)
    for flag in ("unsafe", "missing"):
        if f"a_{flag}" in out and f"b_{flag}" in out:
            out[f"baseline_{flag}"] = [side(row, flag, "baseline") for _, row in out.iterrows()]
            out[f"feedback_{flag}"] = [side(row, flag, "feedback") for _, row in out.iterrows()]

    def preferred(row):
        if row["preference"] in {"A", "B"}:
            return row["a_side"] if row["preference"] == "A" else row["b_side"]
        return row["preference"]

    out["preferred"] = out.apply(preferred, axis=1)
    out["baseline_len"] = [
        len(study.items[row["item_id"]]["A"]) if row["a_side"] == "baseline"
        else len(study.items[row["item_id"]]["B"]) for _, row in out.iterrows()]
    out["feedback_len"] = [
        len(study.items[row["item_id"]]["A"]) if row["a_side"] == "feedback"
        else len(study.items[row["item_id"]]["B"]) for _, row in out.iterrows()]
    out["len_diff"] = out["feedback_len"] - out["baseline_len"]
    out["arm"] = out["panel"] + "/" + out["source"]
    out["auto_delta"] = [
        study.automatic.get((row["panel"], row["source"], row["ticket_id"]), np.nan)
        for _, row in out.iterrows()]
    return out


def bootstrap_mean(values: Iterable[float], n_resamples: int = 10000, seed: int = 42) -> dict:
    values = np.asarray(list(values), dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"n": 0, "mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[draws].mean(axis=1)
    return {"n": int(len(values)), "mean": float(values.mean()),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5))}


def mcnemar_exact(baseline_only: int, feedback_only: int) -> float:
    total = baseline_only + feedback_only
    if total == 0:
        return 1.0
    return float(min(1.0, stats.binomtest(min(baseline_only, feedback_only), total, 0.5).pvalue * 2))


def mde_proportion(n: int, discordant_rate: float) -> dict:
    if n <= 0 or discordant_rate <= 0:
        return {"detectable_advantage": np.nan, "mde_delta": np.nan}
    advantage = 0.5 + np.sqrt((Z_ALPHA + Z_POWER) ** 2 / (4 * n * discordant_rate))
    return {"detectable_advantage": float(min(1.0, advantage)),
            "mde_delta": float((min(1.0, advantage) - 0.5) * discordant_rate)}


def cohen_kappa(a: list, b: list) -> float:
    a, b = list(a), list(b)
    if len(a) == 0:
        return np.nan
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return np.nan
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)))
    for left, right in zip(a, b):
        matrix[index[left], index[right]] += 1
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / total ** 2)
    if expected >= 1:
        return np.nan
    return float((observed - expected) / (1 - expected))


def weighted_kappa(a: list, b: list) -> float:
    pairs = [(ORDINAL[x], ORDINAL[y]) for x, y in zip(a, b)
             if x in ORDINAL and y in ORDINAL]
    if not pairs:
        return np.nan
    n = len(pairs)
    categories = 3
    observed = np.zeros((categories, categories))
    for left, right in pairs:
        observed[left, right] += 1
    observed /= n
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)
    weights = np.array([[abs(i - j) / (categories - 1) for j in range(categories)]
                        for i in range(categories)])
    denominator = (weights * expected).sum()
    if denominator == 0:
        return np.nan
    return float(1 - (weights * observed).sum() / denominator)


def pabak(a: list, b: list) -> float:
    if len(a) == 0:
        return np.nan
    observed = np.mean([x == y for x, y in zip(a, b)])
    return float(2 * observed - 1)


def gwet_ac1(a: list, b: list) -> float:
    a, b = list(a), list(b)
    if len(a) == 0:
        return np.nan
    labels = sorted(set(a) | set(b))
    q = len(labels)
    if q < 2:
        return np.nan
    observed = np.mean([x == y for x, y in zip(a, b)])
    counts = {label: 0 for label in labels}
    for x, y in zip(a, b):
        counts[x] += 1
        counts[y] += 1
    total = 2 * len(a)
    chance = sum((counts[label] / total) * (1 - counts[label] / total) for label in labels) / (q - 1)
    if chance >= 1:
        return np.nan
    return float((observed - chance) / (1 - chance))


def bootstrap_kappa(a: list, b: list, fn, n_resamples: int = 2000, seed: int = 42) -> dict:
    if len(a) == 0:
        return {"kappa": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    point = fn(a, b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(a), size=len(a))
        draws.append(fn([a[i] for i in idx], [b[i] for i in idx]))
    draws = np.asarray([d for d in draws if np.isfinite(d)], dtype=float)
    if len(draws) == 0:
        return {"kappa": point, "ci_lower": np.nan, "ci_upper": np.nan}
    return {"kappa": point, "ci_lower": float(np.percentile(draws, 2.5)),
            "ci_upper": float(np.percentile(draws, 97.5))}


def _auto_alignment(group: pd.DataFrame) -> dict:
    valid = group.dropna(subset=["auto_delta", "feedback_strict", "baseline_strict"]).copy()
    if valid.empty:
        return {"n": 0}
    human = valid["feedback_strict"].to_numpy() - valid["baseline_strict"].to_numpy()
    automatic = valid["auto_delta"].to_numpy()
    signs = np.sign(human) == np.sign(automatic)
    nonzero = (human != 0) & (automatic != 0)
    rho = p = np.nan
    if len(valid) > 2 and np.unique(human).size > 1 and np.unique(automatic).size > 1:
        rho, p = stats.spearmanr(automatic, human)
    preference = valid[valid["preferred"].isin(["baseline", "feedback"])]
    auc = np.nan
    if preference["preferred"].nunique() == 2:
        auc = float(roc_auc_score((preference["preferred"] == "feedback").astype(int),
                                  preference["auto_delta"]))
    return {
        "n": int(len(valid)),
        "auto_mean": float(automatic.mean()),
        "human_mean": float(human.mean()),
        "sign_agreement": float(signs[nonzero].mean()) if nonzero.any() else np.nan,
        "sign_n": int(nonzero.sum()),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_p": float(p) if np.isfinite(p) else np.nan,
        "auc_preference": auc,
        "n_preference": int(len(preference)),
        "auto_negative_human_positive": int(((automatic < 0) & (human > 0)).sum()),
        "auto_positive_human_negative": int(((automatic > 0) & (human < 0)).sum()),
    }


def rater_tables(df: pd.DataFrame, study: Study) -> dict[str, pd.DataFrame]:
    summary, preferences, flags, length, alignment, sensitivity = [], [], [], [], [], []
    for arm, group in df.groupby("arm", sort=True):
        panel, source = arm.split("/")
        changed_share = study.manifest["sample"].get(arm, {}).get("changed_share")
        for kind, baseline_col, feedback_col in (
                ("strict", "baseline_strict", "feedback_strict"),
                ("lenient", "baseline_lenient", "feedback_lenient")):
            base = group[baseline_col].to_numpy()
            feed = group[feedback_col].to_numpy()
            mask = ~np.isnan(base) & ~np.isnan(feed)
            b, f = base[mask], feed[mask]
            delta = f - b
            ci = bootstrap_mean(delta)
            baseline_only = int(((b == 1) & (f == 0)).sum())
            feedback_only = int(((b == 0) & (f == 1)).sum())
            mde = mde_proportion(len(delta), (baseline_only + feedback_only) / max(len(delta), 1))
            row = {
                "panel": panel, "source": source, "metric": kind, "n": len(delta),
                "baseline_rate": float(b.mean()) if len(b) else np.nan,
                "feedback_rate": float(f.mean()) if len(f) else np.nan,
                "delta": ci["mean"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
                "baseline_only": baseline_only, "feedback_only": feedback_only,
                "n_discordant": baseline_only + feedback_only,
                "discordant_rate": (baseline_only + feedback_only) / max(len(delta), 1),
                "mcnemar_p": mcnemar_exact(baseline_only, feedback_only),
                **mde,
            }
            if kind == "strict":
                row["changed_share"] = changed_share
                row["population_delta"] = (ci["mean"] * changed_share
                                           if changed_share is not None else np.nan)
                row["population_ci_lower"] = (ci["ci_lower"] * changed_share
                                              if changed_share is not None else np.nan)
                row["population_ci_upper"] = (ci["ci_upper"] * changed_share
                                              if changed_share is not None else np.nan)
                row["cannot_judge"] = int(group["preferred"].eq("cannot_judge").sum())
            summary.append(row)

        counts = group["preferred"].value_counts()
        net = counts.get("feedback", 0) - counts.get("baseline", 0)
        net_values = group["preferred"].map({"feedback": 1, "baseline": -1}).fillna(0).to_numpy()
        net_ci = bootstrap_mean(net_values)
        preferences.append({
            "panel": panel, "source": source, "n": len(group),
            "feedback_better": int(counts.get("feedback", 0)),
            "baseline_better": int(counts.get("baseline", 0)),
            "equal": int(counts.get("equal", 0)),
            "both_incorrect": int(counts.get("both_incorrect", 0)),
            "cannot_judge": int(counts.get("cannot_judge", 0)),
            "net_feedback_minus_baseline": int(net),
            "net_ci_lower": net_ci["ci_lower"], "net_ci_upper": net_ci["ci_upper"],
        })

        flag_row = {"panel": panel, "source": source, "n": len(group)}
        for flag in ("unsafe", "missing"):
            if f"baseline_{flag}" in group:
                flag_row[f"baseline_{flag}_rate"] = float(group[f"baseline_{flag}"].mean())
                flag_row[f"feedback_{flag}_rate"] = float(group[f"feedback_{flag}"].mean())
        flags.append(flag_row)

        rho = rho_p = np.nan
        valid_length = group.dropna(subset=["len_diff", "feedback_strict", "baseline_strict"])
        human_delta = valid_length["feedback_strict"] - valid_length["baseline_strict"]
        if (len(valid_length) > 2 and np.unique(valid_length["len_diff"]).size > 1
                and np.unique(human_delta).size > 1):
            rho, rho_p = stats.spearmanr(valid_length["len_diff"], human_delta)
        small = group[group["len_diff"].abs() <= 100]
        large = group[group["len_diff"].abs() > 100]
        length.append({
            "panel": panel, "source": source, "n": len(group),
            "baseline_len": float(group["baseline_len"].mean()),
            "feedback_len": float(group["feedback_len"].mean()),
            "len_diff": float(group["len_diff"].mean()),
            "spearman_len_delta": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p": float(rho_p) if np.isfinite(rho_p) else np.nan,
            "n_matched": len(small),
            "delta_matched": float((small["feedback_strict"] - small["baseline_strict"]).mean())
            if len(small) else np.nan,
            "n_unmatched": len(large),
            "delta_unmatched": float((large["feedback_strict"] - large["baseline_strict"]).mean())
            if len(large) else np.nan,
        })

        alignment.append({"panel": panel, "source": source, **_auto_alignment(group)})

        for name, subset in (
                ("strict", group),
                ("lenient", group),
                ("exclude_cannot", group[~group["preferred"].eq("cannot_judge")]),
                ("length_matched", group[group["len_diff"].abs() <= 100])):
            values = (subset["feedback_lenient" if name == "lenient" else "feedback_strict"]
                      - subset["baseline_lenient" if name == "lenient" else "baseline_strict"]).to_numpy()
            ci = bootstrap_mean(values)
            sensitivity.append({"panel": panel, "source": source, "analysis": name,
                                "n": ci["n"], "delta": ci["mean"],
                                "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"]})

    return {
        "summary": pd.DataFrame(summary),
        "preferences": pd.DataFrame(preferences),
        "flags": pd.DataFrame(flags),
        "length": pd.DataFrame(length),
        "alignment": pd.DataFrame(alignment),
        "sensitivity": pd.DataFrame(sensitivity),
    }


def agreement_tables(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, pd.DataFrame]:
    merged = first.merge(second, on="item_id", suffixes=("_1", "_2"))
    if len(merged) != len(first) or len(merged) != len(second):
        raise ValueError("Raters do not cover the same items")
    rows, correctness, consensus = [], [], []
    for arm, group in merged.groupby("arm_1", sort=True):
        panel, source = arm.split("/")
        preference_1 = group["preferred_1"].astype(str).tolist()
        preference_2 = group["preferred_2"].astype(str).tolist()
        preference_agreement = float(np.mean([a == b for a, b in zip(preference_1, preference_2)]))
        kappa = bootstrap_kappa(preference_1, preference_2, cohen_kappa)
        side_rows = {}
        for side in ("baseline", "feedback"):
            a = group[f"{side}_correct_1"].astype(str).tolist()
            b = group[f"{side}_correct_2"].astype(str).tolist()
            k = bootstrap_kappa(a, b, cohen_kappa)
            w = weighted_kappa([x for x in a if x != "cannot_judge"],
                               [y for x, y in zip(a, b) if x != "cannot_judge"])
            side_rows[side] = {"agreement": float(np.mean([x == y for x, y in zip(a, b)])),
                               "kappa": k["kappa"], "kappa_ci_lower": k["ci_lower"],
                               "kappa_ci_upper": k["ci_upper"], "weighted_kappa": w}
        pair_agreement = float(np.mean(
            [(a, b) == (c, d) for a, b, c, d in zip(
                group["baseline_correct_1"], group["feedback_correct_1"],
                group["baseline_correct_2"], group["feedback_correct_2"])]))
        rows.append({
            "panel": panel, "source": source, "n_items": len(group),
            "preference_agreement": preference_agreement,
            "preference_kappa": kappa["kappa"],
            "preference_kappa_ci_lower": kappa["ci_lower"],
            "preference_kappa_ci_upper": kappa["ci_upper"],
            "preference_pabak": pabak(preference_1, preference_2),
            "preference_ac1": gwet_ac1(preference_1, preference_2),
            "baseline_correct_agreement": side_rows["baseline"]["agreement"],
            "baseline_correct_kappa": side_rows["baseline"]["kappa"],
            "baseline_correct_weighted_kappa": side_rows["baseline"]["weighted_kappa"],
            "feedback_correct_agreement": side_rows["feedback"]["agreement"],
            "feedback_correct_kappa": side_rows["feedback"]["kappa"],
            "feedback_correct_weighted_kappa": side_rows["feedback"]["weighted_kappa"],
            "pair_agreement": pair_agreement,
        })
        consensus_group = group[(group["preferred_1"] == group["preferred_2"])]
        values = ((consensus_group["feedback_strict_1"] - consensus_group["baseline_strict_1"])
                  + (consensus_group["feedback_strict_2"] - consensus_group["baseline_strict_2"])) / 2
        ci = bootstrap_mean(values)
        consensus.append({"panel": panel, "source": source, "n_consensus": len(consensus_group),
                          "delta_consensus": ci["mean"], "ci_lower": ci["ci_lower"],
                          "ci_upper": ci["ci_upper"]})
        for side in ("baseline", "feedback"):
            correctness.append({"panel": panel, "source": source, "side": side,
                                **side_rows[side]})
    return {"agreement": pd.DataFrame(rows), "correctness": pd.DataFrame(correctness),
            "consensus": pd.DataFrame(consensus)}


def combined_effect(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    merged = first.merge(second, on="item_id", suffixes=("_1", "_2"))
    rows = []
    for arm, group in merged.groupby("arm_1", sort=True):
        panel, source = arm.split("/")
        delta_1 = group["feedback_strict_1"] - group["baseline_strict_1"]
        delta_2 = group["feedback_strict_2"] - group["baseline_strict_2"]
        combined = (delta_1 + delta_2) / 2
        ci = bootstrap_mean(combined)
        rows.append({"panel": panel, "source": source, "n_items": len(group),
                     "rater_1_delta": float(delta_1.mean()),
                     "rater_2_delta": float(delta_2.mean()),
                     "combined_delta": ci["mean"], "ci_lower": ci["ci_lower"],
                     "ci_upper": ci["ci_upper"]})
    return pd.DataFrame(rows)
