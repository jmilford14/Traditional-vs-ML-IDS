"""
CIC-IDS2017 Intrusion Detection: Rule-Based vs Machine Learning
===============================================================

Uses the CIC-IDS2017 MachineLearningCSV files.

Expected files:
    datasets/cic_ids2017/*.csv

Run:
    python ids_cic_ids2017.py

Notes:
    CIC-IDS2017 is large, so this version uses a memory-safe stratified sample.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# ═════════════════════════════════════════════════════════════════════════════
# 1. CIC-IDS2017 CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

DATASET_DIR = Path("datasets/cic_ids2017")

# Start with 300,000.
# If your laptop still runs out of memory, reduce this to 150_000.
# If it runs smoothly, you can later try 500_000.
MAX_SAMPLES = 300_000

CLASS_NAMES = [
    "Normal",
    "BruteForce",
    "DoS",
    "DDoS",
    "WebAttack",
    "Infiltration",
    "Botnet",
    "PortScan",
    "Heartbleed",
]

LABEL_MAP_CIC = {
    # Normal traffic
    "BENIGN": "Normal",

    # Brute force attacks
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",

    # DoS attacks
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",

    # DDoS
    "DDoS": "DDoS",

    # Web attacks - different downloads sometimes encode the dash differently
    "Web Attack � Brute Force": "WebAttack",
    "Web Attack � XSS": "WebAttack",
    "Web Attack � Sql Injection": "WebAttack",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",

    # Other attack types
    "Infiltration": "Infiltration",
    "Bot": "Botnet",
    "PortScan": "PortScan",
    "Heartbleed": "Heartbleed",
}


# ═════════════════════════════════════════════════════════════════════════════
# 2. CIC-IDS2017 LOADER
# ═════════════════════════════════════════════════════════════════════════════

def optimise_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric columns.
    """
    for col in df.columns:
        if col in {"Label", "attack_type"}:
            continue

        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def load_cic_ids2017() -> pd.DataFrame:
    """
    Load all CIC-IDS2017 CSV files from the dataset folder.

    Returns:
        Full cleaned DataFrame.
    """
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Could not find dataset folder: {DATASET_DIR}\n"
            "Create this folder and place the 8 CIC-IDS2017 CSV files inside it."
        )

    csv_files = sorted(DATASET_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {DATASET_DIR}\n"
            "Place the CIC-IDS2017 MachineLearningCSV files inside this folder."
        )

    print(f"    Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"      - {file.name}")

    dataframes = []

    for file in csv_files:
        print(f"\n    Loading: {file.name}")

        df = pd.read_csv(file, low_memory=False)

        # CIC-IDS2017 column names usually contain leading spaces.
        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            raise ValueError(
                f"Could not find 'Label' column in {file.name}.\n"
                f"Available columns: {df.columns.tolist()}"
            )

        # Clean labels
        df["Label"] = df["Label"].astype(str).str.strip()

        # Replace infinity values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Map raw labels into grouped project labels
        df["attack_type"] = df["Label"].map(LABEL_MAP_CIC)

        unknown_count = df["attack_type"].isna().sum()
        if unknown_count > 0:
            unknown_labels = sorted(df.loc[df["attack_type"].isna(), "Label"].unique().tolist())
            print(f"    [!] Dropping {unknown_count:,} rows with unknown labels: {unknown_labels}")
            df.dropna(subset=["attack_type"], inplace=True)

        df = optimise_numeric_types(df)
        dataframes.append(df)

        print(f"    Rows loaded after cleaning: {len(df):,}")

    full_df = pd.concat(dataframes, ignore_index=True)

    print("\n    Removing duplicate rows...")
    before = len(full_df)
    full_df.drop_duplicates(inplace=True)
    after = len(full_df)
    print(f"    Duplicates removed: {before - after:,}")

    # Encode labels to integers
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    full_df["target"] = full_df["attack_type"].map(class_to_idx).astype(int)

    full_df.reset_index(drop=True, inplace=True)

    return full_df


# ═════════════════════════════════════════════════════════════════════════════
# 3. RULE-BASED IDS FOR CIC-IDS2017
# ═════════════════════════════════════════════════════════════════════════════

class RuleBasedIDSCIC:
    """
    Simple threshold-based IDS baseline for CIC-IDS2017.

    This is intentionally lightweight and interpretable.
    It is not expected to outperform machine learning.
    """

    def __init__(self):
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []

        for _, row in X.iterrows():
            preds.append(self._classify(row))

        return np.array(preds, dtype=int)

    def _classify(self, row: pd.Series) -> int:
        dst_port = float(row.get("Destination Port", 0) or 0)
        flow_duration = float(row.get("Flow Duration", 0) or 0)

        total_fwd_packets = float(row.get("Total Fwd Packets", 0) or 0)
        total_bwd_packets = float(row.get("Total Backward Packets", 0) or 0)

        total_len_fwd = float(row.get("Total Length of Fwd Packets", 0) or 0)
        total_len_bwd = float(row.get("Total Length of Bwd Packets", 0) or 0)

        flow_bytes_s = float(row.get("Flow Bytes/s", 0) or 0)
        flow_packets_s = float(row.get("Flow Packets/s", 0) or 0)

        syn_flag_count = float(row.get("SYN Flag Count", 0) or 0)
        psh_flag_count = float(row.get("PSH Flag Count", 0) or 0)

        packet_len_mean = float(row.get("Packet Length Mean", 0) or 0)
        avg_packet_size = float(row.get("Average Packet Size", 0) or 0)

        # 1. DDoS:
        # Very high packet/flow rate or large packet bursts.
        if (
            flow_packets_s > 10_000
            or flow_bytes_s > 10_000_000
            or total_fwd_packets > 1000
        ):
            return self.class_to_idx["DDoS"]

        # 2. DoS:
        # Sustained one-way or imbalanced flow behaviour.
        if (
            flow_duration > 1_000_000
            and total_fwd_packets > 20
            and total_bwd_packets <= 5
            and total_len_fwd > 1000
        ):
            return self.class_to_idx["DoS"]

        # 3. PortScan:
        # Many short flows, small packet counts, often to non-standard ports.
        if (
            flow_duration < 2_000_000
            and total_fwd_packets <= 3
            and total_bwd_packets <= 3
            and total_len_fwd < 500
            and dst_port not in {80, 443, 53}
        ):
            return self.class_to_idx["PortScan"]

        # 4. BruteForce:
        # FTP/SSH ports with repeated login-like traffic.
        if (
            dst_port in {21, 22}
            and total_fwd_packets >= 3
            and flow_duration > 1_000
        ):
            return self.class_to_idx["BruteForce"]

        # 5. WebAttack:
        # HTTP/HTTPS traffic with abnormal request/response behaviour.
        if (
            dst_port in {80, 443, 8080}
            and total_fwd_packets >= 3
            and total_len_fwd > 500
            and total_len_bwd > 0
            and packet_len_mean > 50
        ):
            return self.class_to_idx["WebAttack"]

        # 6. Botnet:
        # Long-running repeated communication pattern.
        if (
            flow_duration > 10_000_000
            and total_fwd_packets > 10
            and total_bwd_packets > 5
            and avg_packet_size > 40
        ):
            return self.class_to_idx["Botnet"]

        # 7. Infiltration:
        # Long flow with payload transfer and push flags.
        if (
            flow_duration > 5_000_000
            and total_len_fwd > 1000
            and total_len_bwd > 1000
            and psh_flag_count > 0
        ):
            return self.class_to_idx["Infiltration"]

        # 8. Heartbleed:
        # Rare SSL-related attack. Approximate rule only.
        if (
            dst_port in {443, 444}
            and flow_duration > 1_000_000
            and total_len_bwd > 5000
        ):
            return self.class_to_idx["Heartbleed"]

        # 9. Normal by default
        return self.class_to_idx["Normal"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    labels = list(range(len(CLASS_NAMES)))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0
    )
    rec = recall_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    f1pc = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0
    )

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")

    print("\n  Per-class F1:")
    for cls, score in zip(CLASS_NAMES, f1pc):
        bar = "█" * int(score * 20)
        print(f"    {cls:<14} {score:.3f}  {bar}")

    print("\n  Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        labels=labels,
        zero_division=0
    ))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "f1_per_class": f1pc,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "Rule-Based IDS": "#e74c3c",
    "Decision Tree": "#f39c12",
    "Random Forest": "#27ae60",
    "Gradient Boosting": "#2980b9",
}

BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#c9d1d9"
GRID = "#21262d"


def _style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

    for s in ax.spines.values():
        s.set_edgecolor(GRID)


def plot_results(results: list, output_path: str):
    models = [r["model"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    mlabels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = [PALETTE.get(m, "#8b949e") for m in models]

    fig = plt.figure(figsize=(22, 18), facecolor=BG)
    fig.suptitle(
        "CIC-IDS2017  |  Rule-Based IDS vs Machine Learning",
        fontsize=17,
        color=TEXT,
        fontweight="bold",
        y=0.98
    )

    gs = GridSpec(
        3,
        4,
        figure=fig,
        hspace=0.55,
        wspace=0.38,
        top=0.93,
        bottom=0.07,
        left=0.06,
        right=0.97
    )

    # Row 0 — metric bars
    for col, (metric, lbl) in enumerate(zip(metrics, mlabels)):
        ax = fig.add_subplot(gs[0, col])
        vals = [r[metric] for r in results]
        bars = ax.bar(models, vals, color=colors, width=0.55, zorder=3)

        ax.set_ylim(0, 1.12)
        ax.set_title(lbl, fontsize=11, pad=6)
        ax.set_ylabel("Score", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        _style(ax)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=TEXT
            )

    # Row 1 — confusion matrices
    for col, result in enumerate(results):
        ax = fig.add_subplot(gs[1, col])
        cm = result["confusion_matrix"]
        cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(
            cmn,
            annot=True,
            fmt=".2f",
            cmap=sns.color_palette("mako", as_cmap=True),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            cbar=False,
            linewidths=0.3,
            linecolor=BG,
            annot_kws={"size": 6},
            vmin=0,
            vmax=1
        )

        ax.set_title(result["model"], fontsize=9, pad=4)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Actual", fontsize=7)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)
        _style(ax)

    # Row 2 left — per-class F1
    ax_f1 = fig.add_subplot(gs[2, :2])
    x = np.arange(len(CLASS_NAMES))
    width = 0.18

    for i, result in enumerate(results):
        offset = (i - len(results) / 2 + 0.5) * width
        ax_f1.bar(
            x + offset,
            result["f1_per_class"],
            width,
            label=result["model"],
            color=colors[i],
            alpha=0.88,
            zorder=3
        )

    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax_f1.set_ylabel("F1-Score", fontsize=9)
    ax_f1.set_title("Per-Class F1 Score Comparison", fontsize=11)
    ax_f1.set_ylim(0, 1.12)
    ax_f1.legend(
        fontsize=7,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        loc="lower right"
    )
    ax_f1.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    _style(ax_f1)

    # Row 2 right — radar
    ax_r = fig.add_subplot(gs[2, 2:], polar=True)
    ax_r.set_facecolor(PANEL)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(mlabels, color=TEXT, fontsize=8)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#8b949e", fontsize=6)
    ax_r.grid(color=GRID, linewidth=0.4)
    ax_r.spines["polar"].set_edgecolor(GRID)
    ax_r.set_title("Radar: All Metrics", fontsize=11, color=TEXT, pad=12)

    for i, result in enumerate(results):
        vals = [result[m] for m in metrics] + [result[metrics[0]]]
        ax_r.plot(angles, vals, linewidth=1.8, color=colors[i], label=result["model"])
        ax_r.fill(angles, vals, alpha=0.07, color=colors[i])

    ax_r.legend(
        loc="upper right",
        bbox_to_anchor=(1.38, 1.18),
        fontsize=7,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=BG,
        edgecolor="none"
    )
    plt.close()

    print(f"  [✓] Comparison chart → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(rf_pipeline, feature_names: list, output_path: str):
    rf_model = rf_pipeline.named_steps["model"]
    importances = rf_model.feature_importances_

    top_n = min(15, len(importances))
    top_idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.barh(
        range(top_n),
        importances[top_idx][::-1],
        color=plt.cm.viridis(np.linspace(0.2, 0.85, top_n)),
        edgecolor=BG,
        linewidth=0.5,
    )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(
        [feature_names[i] for i in top_idx[::-1]],
        fontsize=8,
        color=TEXT
    )

    ax.set_xlabel("Mean Decrease in Impurity", fontsize=9, color=TEXT)
    ax.set_title(
        "Random Forest — Top Feature Importances (CIC-IDS2017)",
        fontsize=11,
        color=TEXT,
        pad=10
    )

    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.tick_params(colors=TEXT)

    for s in ax.spines.values():
        s.set_edgecolor(GRID)

    fig.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=BG,
        edgecolor="none"
    )
    plt.close()

    print(f"  [✓] Feature importance → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    OUT = Path("outputs_cic_ids2017")
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  IDS COMPARISON  ·  CIC-IDS2017 dataset")
    print("=" * 60)

    # 7.1 Load dataset
    print("\n[1] Loading CIC-IDS2017 …")
    full_df = load_cic_ids2017()

    print(f"\n    Total samples after cleaning : {len(full_df):,}")
    print(f"    Total columns after cleaning : {len(full_df.columns):,}")

    print("\n    Grouped class distribution:")
    for cls, cnt in full_df["attack_type"].value_counts().items():
        print(f"      {cls:<14} {cnt:8,}  ({cnt / len(full_df) * 100:.2f}%)")

    # 7.2 Prepare features
    drop_cols = ["Label", "attack_type", "target"]
    feature_cols = [c for c in full_df.columns if c not in drop_cols]

    X_df = full_df[feature_cols].copy()
    y = full_df["target"].values

    print(f"\n    Feature cols : {len(feature_cols)}")

    # 7.3 Memory-safe stratified sample
    print("\n    Creating memory-safe stratified sample...")

    if len(X_df) > MAX_SAMPLES:
        X_sample_df, _, y_sample, _ = train_test_split(
            X_df,
            y,
            train_size=MAX_SAMPLES,
            random_state=42,
            stratify=y
        )
    else:
        X_sample_df = X_df
        y_sample = y

    print(f"    Using samples : {len(X_sample_df):,}")

    print("\n    Sampled class distribution:")
    sampled_labels = pd.Series(y_sample).map({i: c for i, c in enumerate(CLASS_NAMES)})
    for cls, cnt in sampled_labels.value_counts().items():
        print(f"      {cls:<14} {cnt:8,}  ({cnt / len(sampled_labels) * 100:.2f}%)")

    # 7.4 Train/test split
    print("\n    Removing extremely rare classes from sampled data...")

    # Stratified train/test split requires at least 2 samples per class.
    # Very rare classes such as Heartbleed may appear only once in the sample.
    sample_counts = pd.Series(y_sample).value_counts()
    valid_classes = sample_counts[sample_counts >= 2].index.tolist()

    rare_classes = sample_counts[sample_counts < 2].index.tolist()

    if rare_classes:
        rare_class_names = [CLASS_NAMES[i] for i in rare_classes]
        print(f"    [!] Removing rare sampled classes with fewer than 2 records: {rare_class_names}")

        keep_mask = pd.Series(y_sample).isin(valid_classes).values
        X_sample_df = X_sample_df.loc[keep_mask].copy()
        y_sample = y_sample[keep_mask]

    print(f"    Samples after rare-class removal : {len(X_sample_df):,}")

    print("\n    Final sampled class distribution:")
    sampled_labels = pd.Series(y_sample).map({i: c for i, c in enumerate(CLASS_NAMES)})
    for cls, cnt in sampled_labels.value_counts().items():
        print(f"      {cls:<14} {cnt:8,}  ({cnt / len(sampled_labels) * 100:.2f}%)")

    print("\n    Creating train/test split...")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_sample_df,
        y_sample,
        test_size=0.25,
        random_state=42,
        stratify=y_sample
    )

    print(f"    Train samples : {len(X_train_df):,}")
    print(f"    Test samples  : {len(X_test_df):,}")

    # 7.5 Rule-based IDS
    print("\n[2] Running Rule-Based IDS …")

    rule_ids = RuleBasedIDSCIC()
    y_pred_rule = rule_ids.predict(X_test_df)
    r_rule = compute_metrics(y_test, y_pred_rule, "Rule-Based IDS")

    # 7.6 Machine learning models
    print("\n[3] Training ML models …")

    ml_models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
    }

    ml_results = []
    rf_pipeline = None

    for name, model in ml_models.items():
        print(f"\n    Training {name} …")

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model)
        ])

        pipe.fit(X_train_df, y_train)

        y_pred = pipe.predict(X_test_df)
        result = compute_metrics(y_test, y_pred, name)
        ml_results.append(result)

        if name == "Random Forest":
            rf_pipeline = pipe

    # 7.7 Summary
    all_results = [r_rule] + ml_results

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 58)

    for r in all_results:
        print(
            f"  {r['model']:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f}"
        )

    best = max(all_results, key=lambda x: x["f1"])
    print(f"\n  ★  Best: {best['model']}  (F1 = {best['f1']:.4f})")

    # 7.8 Export JSON
    export = []

    for r in all_results:
        export.append({
            "model": r["model"],
            "accuracy": round(r["accuracy"], 4),
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "f1_per_class": {
                cls: round(float(s), 4)
                for cls, s in zip(CLASS_NAMES, r["f1_per_class"])
            },
        })

    with open(OUT / "metrics_summary.json", "w") as f:
        json.dump(export, f, indent=2)

    print(f"\n  [✓] Metrics JSON → {OUT / 'metrics_summary.json'}")

    # 7.9 Generate plots
    print("\n[4] Generating visualisations …")

    plot_results(
        all_results,
        str(OUT / "ids_comparison.png")
    )

    if rf_pipeline is not None:
        plot_feature_importance(
            rf_pipeline,
            feature_cols,
            str(OUT / "feature_importance.png")
        )

    print("\n[✓] Pipeline complete.\n")

    return all_results


if __name__ == "__main__":
    main()