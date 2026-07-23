from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .registry import metric_registry_frame


def _roc_auc(y_true: pd.Series, scores: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy()
    score = pd.to_numeric(scores, errors="coerce").to_numpy()
    levels = np.unique(y[np.isfinite(y)])
    if len(levels) != 2:
        return np.nan
    positive = y == levels[-1]
    negative = y == levels[0]
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = stats.rankdata(score, method="average")
    mann_whitney = float(ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
    return mann_whitney / (n_positive * n_negative)


def cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int = 2000,
    seed: int = 20260713,
) -> tuple[float, float, int]:
    """Percentile CI by resampling participants and retaining all their recordings."""
    clusters = frame[cluster_col].dropna().unique()
    if len(clusters) < 2:
        return np.nan, np.nan, 0
    grouped = {cluster: frame.loc[frame[cluster_col] == cluster] for cluster in clusters}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [grouped[cluster].assign(_bootstrap_cluster=index) for index, cluster in enumerate(sampled)]
        estimate = statistic(pd.concat(pieces, ignore_index=True))
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return np.nan, np.nan, 0
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high), len(estimates)


def describe_metrics(
    frame: pd.DataFrame,
    *,
    subject_col: str = "SubjectID",
    bootstrap_replicates: int = 2000,
    seed: int = 20260713,
) -> pd.DataFrame:
    rows = []
    registry = metric_registry_frame()
    for _, spec in registry.iterrows():
        feature = spec["feature"]
        if feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        work = frame[[subject_col]].assign(value=values).dropna()
        median = float(work["value"].median()) if len(work) else np.nan
        low, high, successful = cluster_bootstrap(
            work,
            cluster_col=subject_col,
            statistic=lambda sample: float(sample["value"].median()),
            replicates=bootstrap_replicates,
            seed=seed,
        )
        rows.append(
            {
                "feature": feature,
                "family": spec["family"],
                "role": spec["role"],
                "recordings_nonmissing": int(values.notna().sum()),
                "participants_nonmissing": int(work[subject_col].nunique()),
                "missing_fraction": float(values.isna().mean()),
                "median": median,
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "zero_fraction_nonmissing": float((values.dropna() == 0).mean()) if values.notna().any() else np.nan,
                "median_cluster_bootstrap_ci_low": low,
                "median_cluster_bootstrap_ci_high": high,
                "bootstrap_successful": successful,
            }
        )
    return pd.DataFrame(rows)


def pairwise_clustered_spearman(
    frame: pd.DataFrame,
    *,
    subject_col: str = "SubjectID",
    features: list[str] | None = None,
    minimum_participants: int = 20,
    bootstrap_replicates: int = 500,
    seed: int = 20260713,
) -> pd.DataFrame:
    """Pairwise Spearman structure with participant-clustered confidence intervals."""
    if features is None:
        registry = metric_registry_frame()
        features = registry.loc[registry["role"].str.startswith("primary"), "feature"].tolist()
    registry = metric_registry_frame().set_index("feature")
    rows = []
    pair_index = 0
    for left_index, left in enumerate(features[:-1]):
        for right in features[left_index + 1 :]:
            pair_index += 1
            if left not in frame or right not in frame:
                continue
            work = frame[[subject_col]].assign(
                left=pd.to_numeric(frame[left], errors="coerce"),
                right=pd.to_numeric(frame[right], errors="coerce"),
            ).dropna()
            participants = work[subject_col].nunique()
            if participants < minimum_participants or work["left"].nunique() < 3 or work["right"].nunique() < 3:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "n_recordings": len(work),
                        "n_participants": participants,
                        "rho": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "status": "insufficient_support",
                    }
                )
                continue
            statistic = lambda sample: float(stats.spearmanr(sample["left"], sample["right"]).statistic)
            rho = statistic(work)
            low, high, successful = cluster_bootstrap(
                work,
                cluster_col=subject_col,
                statistic=statistic,
                replicates=bootstrap_replicates,
                seed=seed + pair_index,
            )
            rows.append(
                {
                    "feature_left": left,
                    "family_left": registry.loc[left, "family"] if left in registry.index else None,
                    "feature_right": right,
                    "family_right": registry.loc[right, "family"] if right in registry.index else None,
                    "n_recordings": len(work),
                    "n_participants": participants,
                    "rho": rho,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_successful": successful,
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def _rank_normal_scores(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.notna()
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    if observed.sum() < 3:
        return output
    ranks = stats.rankdata(numeric[observed], method="average")
    probabilities = (ranks - 0.5) / len(ranks)
    output.loc[observed] = stats.norm.ppf(probabilities)
    return output


def participant_persistence(
    frame: pd.DataFrame,
    *,
    subject_col: str = "SubjectID",
    features: list[str] | None = None,
    minimum_participants: int = 20,
    minimum_repeated_participants: int = 10,
) -> pd.DataFrame:
    """Random-intercept variance partition, explicitly labeled persistence—not reliability.

    The fixed transform is a rank-normal score for all eligible continuous measures. Features
    with >80% zeros are skipped because a Gaussian random-intercept model is inappropriate.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("participant_persistence requires statsmodels") from exc
    if features is None:
        registry = metric_registry_frame()
        features = registry.loc[registry["role"].str.startswith("primary"), "feature"].tolist()
    rows = []
    for feature in features:
        if feature not in frame:
            continue
        work = frame[[subject_col]].assign(raw=pd.to_numeric(frame[feature], errors="coerce")).dropna()
        counts = work.groupby(subject_col).size()
        zero_fraction = float((work["raw"] == 0).mean()) if len(work) else np.nan
        base = {
            "feature": feature,
            "n_recordings": len(work),
            "n_participants": work[subject_col].nunique(),
            "n_repeated_participants": int((counts >= 2).sum()),
            "zero_fraction": zero_fraction,
            "between_participant_variance": np.nan,
            "within_participant_variance": np.nan,
            "persistence_icc": np.nan,
            "interpretation": "participant rank persistence; not test-retest or inter-rater reliability",
            "status": "",
        }
        if (
            base["n_participants"] < minimum_participants
            or base["n_repeated_participants"] < minimum_repeated_participants
        ):
            base["status"] = "insufficient_repeated_support"
            rows.append(base)
            continue
        if zero_fraction > 0.80 or work["raw"].nunique() < 5:
            base["status"] = "skipped_sparse_or_low_variance"
            rows.append(base)
            continue
        work["value"] = _rank_normal_scores(work["raw"])
        try:
            fit = smf.mixedlm("value ~ 1", work, groups=work[subject_col]).fit(
                reml=True, method="lbfgs", disp=False
            )
            between = float(fit.cov_re.iloc[0, 0])
            within = float(fit.scale)
            base.update(
                between_participant_variance=between,
                within_participant_variance=within,
                persistence_icc=between / (between + within) if between + within > 0 else np.nan,
                status="ok" if fit.converged else "not_converged",
            )
        except Exception as exc:
            base["status"] = f"model_failed:{type(exc).__name__}"
        rows.append(base)
    return pd.DataFrame(rows)


def one_recording_per_participant(
    frame: pd.DataFrame, *, subject_col: str = "SubjectID", seed: int = 20260713
) -> pd.DataFrame:
    """Pre-specified sensitivity sample; random selection is reproducible and audited."""
    rng = np.random.default_rng(seed)
    chosen = []
    for _, participant_rows in frame.groupby(subject_col, sort=True):
        chosen.append(participant_rows.iloc[int(rng.integers(0, len(participant_rows)))])
    return pd.DataFrame(chosen).reset_index(drop=True)


def participant_level_group_contrasts(
    frame: pd.DataFrame,
    *,
    group_col: str = "diagnosis_reported",
    subject_col: str = "SubjectID",
    group_a: str = "ALS",
    group_b: str = "CONTROLS",
    features: list[str] | None = None,
    bootstrap_replicates: int = 2000,
    seed: int = 20260713,
    minimum_per_group: int = 10,
) -> pd.DataFrame:
    """Participant-level exploratory contrasts robust to unequal recording/group counts.

    Each participant contributes one median per metric. Bootstrap sampling is stratified by
    group, so the larger ALS group cannot dominate uncertainty through extra recordings.
    """
    if features is None:
        registry_frame = metric_registry_frame()
        features = registry_frame.loc[registry_frame["role"].str.startswith("primary"), "feature"].tolist()
    registry = metric_registry_frame().set_index("feature")
    rng = np.random.default_rng(seed)
    rows = []
    for feature in features:
        if feature not in frame:
            continue
        work = frame[[subject_col, group_col]].assign(value=pd.to_numeric(frame[feature], errors="coerce")).dropna()
        group_conflicts = work.groupby(subject_col)[group_col].nunique()
        work = work.loc[~work[subject_col].isin(group_conflicts[group_conflicts > 1].index)]
        participant = work.groupby([subject_col, group_col], as_index=False)["value"].median()
        a = participant.loc[participant[group_col] == group_a, "value"].to_numpy()
        b = participant.loc[participant[group_col] == group_b, "value"].to_numpy()
        row = {
            "feature": feature,
            "family": registry.loc[feature, "family"] if feature in registry.index else None,
            "group_a": group_a,
            "group_b": group_b,
            "n_group_a_participants": len(a),
            "n_group_b_participants": len(b),
            "median_difference_a_minus_b": np.nan,
            "median_difference_ci_low": np.nan,
            "median_difference_ci_high": np.nan,
            "cliffs_delta_a_vs_b": np.nan,
            "cliffs_delta_ci_low": np.nan,
            "cliffs_delta_ci_high": np.nan,
            "status": "",
        }
        if len(a) < minimum_per_group or len(b) < minimum_per_group:
            row["status"] = "insufficient_participants_in_one_group"
            rows.append(row)
            continue

        def effects(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
            median_difference = float(np.median(left) - np.median(right))
            comparisons = left[:, None] - right[None, :]
            cliffs_delta = float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)
            return median_difference, cliffs_delta

        median_difference, cliffs_delta = effects(a, b)
        median_boot = []
        delta_boot = []
        for _ in range(bootstrap_replicates):
            sampled_a = rng.choice(a, size=len(a), replace=True)
            sampled_b = rng.choice(b, size=len(b), replace=True)
            median_value, delta_value = effects(sampled_a, sampled_b)
            median_boot.append(median_value)
            delta_boot.append(delta_value)
        row.update(
            median_difference_a_minus_b=median_difference,
            median_difference_ci_low=float(np.percentile(median_boot, 2.5)),
            median_difference_ci_high=float(np.percentile(median_boot, 97.5)),
            cliffs_delta_a_vs_b=cliffs_delta,
            cliffs_delta_ci_low=float(np.percentile(delta_boot, 2.5)),
            cliffs_delta_ci_high=float(np.percentile(delta_boot, 97.5)),
            status="ok",
        )
        rows.append(row)
    return pd.DataFrame(rows)


def perceptual_links(
    features: pd.DataFrame,
    consensus: pd.DataFrame,
    category_metric_map: dict[str, list[str]],
    *,
    subject_col: str = "SubjectID",
    file_col: str = "file_name",
    bootstrap_replicates: int = 2000,
    seed: int = 20260713,
) -> pd.DataFrame:
    """Category-specific links with participant-clustered uncertainty and no resampling/SMOTE."""
    registry = metric_registry_frame().set_index("feature")
    merged = features.merge(consensus, on=file_col, how="inner", validate="one_to_many")
    rows = []
    for category, metrics in category_metric_map.items():
        category_frame = merged.loc[merged["category"] == category].copy()
        y_numeric = pd.to_numeric(category_frame["consensus_rating"], errors="coerce")
        for feature in metrics:
            if feature not in category_frame or feature not in registry.index:
                continue
            x = pd.to_numeric(category_frame[feature], errors="coerce")
            work = category_frame[[subject_col]].assign(x=x, y=y_numeric).dropna()
            unique_y = sorted(work["y"].unique())
            row = {
                "category": category,
                "feature": feature,
                "family": registry.loc[feature, "family"],
                "n_recordings": len(work),
                "n_participants": work[subject_col].nunique(),
                "outcome_levels": len(unique_y),
                "effect_type": None,
                "effect": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "estimable": False,
                "reason": "",
            }
            if len(work) < 10 or work[subject_col].nunique() < 5:
                row["reason"] = "insufficient_clustered_support"
                rows.append(row)
                continue
            if len(unique_y) == 2:
                counts = work["y"].value_counts()
                if counts.min() < 5:
                    row["reason"] = "fewer_than_5_in_one_human_qc_class"
                    rows.append(row)
                    continue
                direction = registry.loc[feature, "worse"]
                oriented = -work["x"] if direction == "lower" else work["x"]
                work = work.assign(oriented=oriented)
                statistic = lambda sample: float(_roc_auc(sample["y"], sample["oriented"]))
                effect = statistic(work)
                low, high, _ = cluster_bootstrap(
                    work,
                    cluster_col=subject_col,
                    statistic=statistic,
                    replicates=bootstrap_replicates,
                    seed=seed,
                )
                row.update(effect_type="roc_auc", effect=effect, ci_low=low, ci_high=high, estimable=True)
            elif len(unique_y) >= 3:
                statistic = lambda sample: float(stats.spearmanr(sample["x"], sample["y"]).statistic)
                effect = statistic(work)
                low, high, _ = cluster_bootstrap(
                    work,
                    cluster_col=subject_col,
                    statistic=statistic,
                    replicates=bootstrap_replicates,
                    seed=seed,
                )
                row.update(effect_type="spearman_rho", effect=effect, ci_low=low, ci_high=high, estimable=True)
            else:
                row["reason"] = "outcome_has_fewer_than_2_levels"
            rows.append(row)
    return pd.DataFrame(rows)


def _percentile_score(values: pd.Series) -> pd.Series:
    """Empirical 0–1 percentile with missing values retained and ties averaged."""
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.notna()
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    n = int(observed.sum())
    if n == 1:
        result.loc[observed] = 0.5
    elif n > 1:
        ranks = numeric.loc[observed].rank(method="average")
        result.loc[observed] = (ranks - 1) / (n - 1)
    return result


def direction_oriented_family_indices(
    frame: pd.DataFrame,
    *,
    id_columns: tuple[str, ...] = ("file_name", "SubjectID"),
    minimum_fraction_metrics: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create analysis-specific family indices whose larger values always mean worse.

    Only registry metrics with an explicit ``worse`` direction are eligible. Each metric
    is converted to an empirical percentile, lower-worse measures are reversed, and the
    within-family median is used when at least the pre-specified fraction of eligible
    metrics is present. These are secondary formative summaries, not latent factor scores
    and not a global quality scale.
    """
    if not 0 < minimum_fraction_metrics <= 1:
        raise ValueError("minimum_fraction_metrics must be in (0, 1]")
    registry = metric_registry_frame()
    eligible = registry.loc[
        registry["worse"].isin(["higher", "lower"])
        & registry["role"].str.startswith("primary")
        & registry["feature"].isin(frame.columns)
    ].copy()
    output = frame[[column for column in id_columns if column in frame.columns]].copy()
    audit_rows = []
    for family, specs in eligible.groupby("family", sort=True):
        oriented_columns = []
        for spec in specs.itertuples():
            score = _percentile_score(frame[spec.feature])
            if spec.worse == "lower":
                score = 1 - score
            column = f"_oriented__{spec.feature}"
            output[column] = score
            oriented_columns.append(column)
            audit_rows.append(
                {
                    "family": family,
                    "feature": spec.feature,
                    "registry_role": spec.role,
                    "registry_worse_direction": spec.worse,
                    "orientation_applied": (
                        "reverse_percentile" if spec.worse == "lower" else "percentile"
                    ),
                    "nonmissing_recordings": int(score.notna().sum()),
                }
            )
        minimum_metrics = max(1, int(np.ceil(len(oriented_columns) * minimum_fraction_metrics)))
        support = output[oriented_columns].notna().sum(axis=1)
        output[f"qfamily__{family}"] = output[oriented_columns].median(axis=1, skipna=True).where(
            support >= minimum_metrics
        )
        output[f"qfamily_n_metrics__{family}"] = support
        output[f"qfamily_required_metrics__{family}"] = minimum_metrics
        output.drop(columns=oriented_columns, inplace=True)
    return output, pd.DataFrame(audit_rows)


def _average_precision_binary(y_true: pd.Series, scores: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy()
    score = pd.to_numeric(scores, errors="coerce").to_numpy()
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = score[mask]
    levels = np.unique(y)
    if len(levels) != 2:
        return np.nan
    positive = y == levels[-1]
    if positive.sum() == 0:
        return np.nan
    order = np.argsort(-score, kind="stable")
    positive_ordered = positive[order]
    precision = np.cumsum(positive_ordered) / np.arange(1, len(positive_ordered) + 1)
    return float(np.sum(precision * positive_ordered) / positive.sum())


def family_alignment_matrix(
    family_indices: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_system: str,
    subject_col: str = "SubjectID",
    file_col: str = "file_name",
    minimum_class_recordings: int = 5,
    minimum_participants: int = 5,
    bootstrap_replicates: int = 2000,
    seed: int = 20260713,
) -> pd.DataFrame:
    """Cross every objective family index with every perceptual family label.

    Binary labels use ROC AUC and rank-biserial effect (2*AUC-1); prevalence and
    average precision are retained to expose class imbalance. All objective indices and
    human labels are oriented so larger values mean more artifact burden.
    """
    family_column = "category" if "category" in labels.columns else "family"
    rating_column = (
        "consensus_rating" if "consensus_rating" in labels.columns else "rating"
    )
    merged = family_indices.merge(
        labels[[file_col, family_column, rating_column]],
        on=file_col,
        how="inner",
        validate="one_to_many",
    )
    objective_columns = {
        column.removeprefix("qfamily__"): column
        for column in family_indices.columns
        if column.startswith("qfamily__") and not column.startswith("qfamily_n_")
    }
    rows = []
    pair_number = 0
    for human_family, human_frame in merged.groupby(family_column, sort=True):
        y = pd.to_numeric(human_frame[rating_column], errors="coerce")
        for objective_family, score_column in objective_columns.items():
            pair_number += 1
            work = human_frame[[subject_col]].assign(
                y=y, score=pd.to_numeric(human_frame[score_column], errors="coerce")
            ).dropna()
            levels = sorted(work["y"].unique())
            counts = work["y"].value_counts()
            row = {
                "label_system": label_system,
                "human_family": human_family,
                "objective_family": objective_family,
                "matched_family": human_family == objective_family,
                "n_recordings": len(work),
                "n_participants": work[subject_col].nunique(),
                "outcome_levels": len(levels),
                "positive_recordings": int((work["y"] == levels[-1]).sum()) if levels else 0,
                "negative_recordings": int((work["y"] == levels[0]).sum()) if levels else 0,
                "positive_prevalence": (
                    float((work["y"] == levels[-1]).mean()) if len(levels) == 2 else np.nan
                ),
                "effect_type": None,
                "effect": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "roc_auc": np.nan,
                "average_precision": np.nan,
                "estimable": False,
                "reason": "",
                "objective_direction": "higher_is_worse",
                "human_direction": "higher_is_worse",
            }
            if len(work) < 2 * minimum_class_recordings:
                row["reason"] = "insufficient_recordings"
            elif work[subject_col].nunique() < minimum_participants:
                row["reason"] = "insufficient_participants"
            elif len(levels) == 2 and counts.min() < minimum_class_recordings:
                row["reason"] = "under_supported_human_class"
            elif len(levels) == 2:
                auc = float(_roc_auc(work["y"], work["score"]))
                statistic = lambda sample: float(
                    2 * _roc_auc(sample["y"], sample["score"]) - 1
                )
                effect = statistic(work)
                low, high, _ = cluster_bootstrap(
                    work,
                    cluster_col=subject_col,
                    statistic=statistic,
                    replicates=bootstrap_replicates,
                    seed=seed + pair_number,
                )
                row.update(
                    effect_type="rank_biserial_from_auc",
                    effect=effect,
                    ci_low=low,
                    ci_high=high,
                    roc_auc=auc,
                    average_precision=_average_precision_binary(work["y"], work["score"]),
                    estimable=True,
                )
            elif len(levels) >= 3:
                statistic = lambda sample: float(
                    stats.spearmanr(sample["score"], sample["y"]).statistic
                )
                effect = statistic(work)
                low, high, _ = cluster_bootstrap(
                    work,
                    cluster_col=subject_col,
                    statistic=statistic,
                    replicates=bootstrap_replicates,
                    seed=seed + pair_number,
                )
                row.update(
                    effect_type="spearman_rho",
                    effect=effect,
                    ci_low=low,
                    ci_high=high,
                    estimable=True,
                )
            else:
                row["reason"] = "outcome_has_fewer_than_2_levels"
            rows.append(row)
    return pd.DataFrame(rows)


def matched_family_specificity(
    family_indices: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_system: str,
    subject_col: str = "SubjectID",
    file_col: str = "file_name",
    minimum_class_recordings: int = 5,
    bootstrap_replicates: int = 2000,
    seed: int = 20260713,
) -> pd.DataFrame:
    """Test whether matched family effects exceed off-diagonal family effects."""
    family_column = "category" if "category" in labels.columns else "family"
    rating_column = (
        "consensus_rating" if "consensus_rating" in labels.columns else "rating"
    )
    merged = family_indices.merge(
        labels[[file_col, family_column, rating_column]],
        on=file_col,
        how="inner",
        validate="one_to_many",
    )
    objective_columns = {
        column.removeprefix("qfamily__"): column
        for column in family_indices.columns
        if column.startswith("qfamily__")
    }

    def effects(sample: pd.DataFrame) -> tuple[list[float], list[float]]:
        matched = []
        mismatched = []
        for human_family, human_frame in sample.groupby(family_column):
            y = pd.to_numeric(human_frame[rating_column], errors="coerce")
            for objective_family, score_column in objective_columns.items():
                work = pd.DataFrame(
                    {
                        "y": y,
                        "score": pd.to_numeric(human_frame[score_column], errors="coerce"),
                    }
                ).dropna()
                counts = work["y"].value_counts()
                if len(counts) != 2 or counts.min() < minimum_class_recordings:
                    continue
                effect = float(2 * _roc_auc(work["y"], work["score"]) - 1)
                if not np.isfinite(effect):
                    continue
                (matched if human_family == objective_family else mismatched).append(effect)
        return matched, mismatched

    matched, mismatched = effects(merged)
    point = (
        float(np.mean(matched) - np.mean(mismatched))
        if matched and mismatched
        else np.nan
    )

    def statistic(sample: pd.DataFrame) -> float:
        bootstrap_matched, bootstrap_mismatched = effects(sample)
        if not bootstrap_matched or not bootstrap_mismatched:
            return np.nan
        return float(np.mean(bootstrap_matched) - np.mean(bootstrap_mismatched))

    low, high, successful = cluster_bootstrap(
        merged,
        cluster_col=subject_col,
        statistic=statistic,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    return pd.DataFrame(
        [
            {
                "label_system": label_system,
                "mean_matched_effect": float(np.mean(matched)) if matched else np.nan,
                "mean_mismatched_effect": (
                    float(np.mean(mismatched)) if mismatched else np.nan
                ),
                "matched_minus_mismatched": point,
                "ci_low": low,
                "ci_high": high,
                "matched_pairs_estimable": len(matched),
                "mismatched_pairs_estimable": len(mismatched),
                "n_recordings_with_labels": merged[file_col].nunique(),
                "n_participants_with_labels": merged[subject_col].nunique(),
                "bootstrap_successful": successful,
                "status": (
                    "ok"
                    if np.isfinite(point) and successful > 0
                    else "under_supported_specificity_estimand"
                ),
            }
        ]
    )


def compare_binary_label_systems(
    family_indices: pd.DataFrame,
    labels_a: pd.DataFrame,
    labels_b: pd.DataFrame,
    *,
    label_a_name: str,
    label_b_name: str,
    shared_families: list[str],
    subject_col: str = "SubjectID",
    file_col: str = "file_name",
    minimum_class_recordings: int = 5,
    bootstrap_replicates: int = 2000,
    seed: int = 20260713,
) -> pd.DataFrame:
    """Paired, shared-recording comparison of two binary perceptual label systems.

    The estimand is ΔAUC on exactly the same recordings for each family. A positive
    difference favors label system A. This does not correct for unavailable rater
    reliability and must not be described as an intrinsic ranking of annotation systems.
    """
    def wide(labels: pd.DataFrame, prefix: str) -> pd.DataFrame:
        family_column = "category" if "category" in labels.columns else "family"
        rating_column = (
            "consensus_rating" if "consensus_rating" in labels.columns else "rating"
        )
        subset = labels.loc[labels[family_column].isin(shared_families)]
        result = subset.pivot(index=file_col, columns=family_column, values=rating_column)
        return result.add_prefix(prefix).reset_index()

    merged = (
        family_indices.merge(wide(labels_a, "a__"), on=file_col, how="inner")
        .merge(wide(labels_b, "b__"), on=file_col, how="inner")
    )
    rows = []
    for index, family in enumerate(shared_families):
        score_column = f"qfamily__{family}"
        a_column = f"a__{family}"
        b_column = f"b__{family}"
        if any(column not in merged for column in [score_column, a_column, b_column]):
            rows.append(
                {
                    "family": family,
                    "label_a": label_a_name,
                    "label_b": label_b_name,
                    "status": "missing_family_or_label_column",
                }
            )
            continue
        work = merged[[subject_col]].assign(
            score=pd.to_numeric(merged[score_column], errors="coerce"),
            a=pd.to_numeric(merged[a_column], errors="coerce"),
            b=pd.to_numeric(merged[b_column], errors="coerce"),
        ).dropna()
        a_counts = work["a"].value_counts()
        b_counts = work["b"].value_counts()
        row = {
            "family": family,
            "label_a": label_a_name,
            "label_b": label_b_name,
            "n_shared_recordings": len(work),
            "n_shared_participants": work[subject_col].nunique(),
            "prevalence_a": float(work["a"].mean()) if len(work) else np.nan,
            "prevalence_b": float(work["b"].mean()) if len(work) else np.nan,
            "auc_a": np.nan,
            "auc_b": np.nan,
            "delta_auc_a_minus_b": np.nan,
            "delta_ci_low": np.nan,
            "delta_ci_high": np.nan,
            "bootstrap_successful": 0,
            "status": "",
            "scale_comparison": "binary_presence_0_absent_1_present",
            "direction_comparison": "all_higher_is_worse",
        }
        if (
            len(a_counts) != 2
            or len(b_counts) != 2
            or a_counts.min() < minimum_class_recordings
            or b_counts.min() < minimum_class_recordings
        ):
            row["status"] = "under_supported_class_on_shared_recordings"
            rows.append(row)
            continue
        auc_a = float(_roc_auc(work["a"], work["score"]))
        auc_b = float(_roc_auc(work["b"], work["score"]))
        statistic = lambda sample: float(
            _roc_auc(sample["a"], sample["score"])
            - _roc_auc(sample["b"], sample["score"])
        )
        low, high, successful = cluster_bootstrap(
            work,
            cluster_col=subject_col,
            statistic=statistic,
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        row.update(
            auc_a=auc_a,
            auc_b=auc_b,
            delta_auc_a_minus_b=auc_a - auc_b,
            delta_ci_low=low,
            delta_ci_high=high,
            bootstrap_successful=successful,
            status="ok",
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_familywise_fdr(
    frame: pd.DataFrame, *, p_column: str = "p_value", family_column: str = "family"
) -> pd.DataFrame:
    result = frame.copy()
    result["q_value"] = np.nan
    for _, indices in result.groupby(family_column).groups.items():
        valid = result.loc[indices, p_column].notna()
        valid_indices = result.loc[indices].index[valid]
        if len(valid_indices):
            p_values = result.loc[valid_indices, p_column].astype(float).to_numpy()
            order = np.argsort(p_values)
            ranked = p_values[order]
            adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
            adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
            adjusted = np.empty_like(adjusted_ranked)
            adjusted[order] = np.clip(adjusted_ranked, 0, 1)
            result.loc[valid_indices, "q_value"] = adjusted
    return result
