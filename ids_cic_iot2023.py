import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════

TRAIN_PATH = "CICIoT2023/train.csv"
TEST_PATH = "CICIoT2023/test.csv"

MAX_TRAIN_SAMPLES = 200000
MAX_TEST_SAMPLES = 50000

CLASS_NAMES = [
    "Normal",
    "DDoS",
    "Recon",
    "BruteForce",
]

LABEL_MAP = {
    # Normal
    "BenignTraffic": "Normal",
    "Benign": "Normal",
    "BENIGN": "Normal",
    "Normal": "Normal",

    # DDoS / DoS
    "DDoS": "DDoS",
    "DoS": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_FLOOD": "DDoS",
    "DDoS-RSTFINFLOOD": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DoS-HTTP_Flood": "DDoS",
    "DoS-SYN_Flood": "DDoS",
    "DoS-TCP_Flood": "DDoS",
    "DoS-UDP_Flood": "DDoS",

    # Recon
    "Recon": "Recon",
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",

    # Brute Force
    "DictionaryBruteForce": "BruteForce",
    "BruteForce": "BruteForce",
}


def normalise_text(value):
    return str(value).strip().replace(" ", "_")


# ═════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════

def find_label_column(df: pd.DataFrame) -> str:
    possible_cols = [
        "label",
        "Label",
        "Attack",
        "attack",
        "Class",
        "class",
        "Category",
        "category",
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find label column. "
        "Print df.columns.tolist() and add the correct column name."
    )


def prepare_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()

    normalised_map = {
        normalise_text(k).lower(): v
        for k, v in LABEL_MAP.items()
    }

    df["raw_label"] = df[label_col].apply(normalise_text)

    df["attack_type"] = (
        df["raw_label"]
        .str.lower()
        .map(normalised_map)
    )

    unknown = df["attack_type"].isna().sum()

    if unknown > 0:
        print(f"\n    [!] Dropping {unknown:,} rows with unwanted/unmapped labels.")
        print("    Unmapped labels found:")
        print(df.loc[df["attack_type"].isna(), "raw_label"].unique())

    df = df.dropna(subset=["attack_type"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Dataset is empty after label mapping. "
            "Check the label values in your train/test CSV."
        )

    le = LabelEncoder()
    le.classes_ = np.array(CLASS_NAMES)

    df["label"] = le.transform(df["attack_type"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        c for c in numeric_cols
        if c != "label"
    ]

    df = df[feature_cols + ["attack_type", "label"]]

    return df


def load_cic_iot2023_train_test(train_path: str, test_path: str):
    print("\n    Loading CIC-IoT2023 train/test files...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"    Raw train rows: {len(train_df):,}")
    print(f"    Raw test rows : {len(test_df):,}")

    train_label_col = find_label_column(train_df)
    test_label_col = find_label_column(test_df)

    print(f"    Train label column: {train_label_col}")
    print(f"    Test label column : {test_label_col}")

    train_df = prepare_dataframe(train_df, train_label_col)
    test_df = prepare_dataframe(test_df, test_label_col)

    SAMPLES_PER_CLASS_TRAIN = 30000
    SAMPLES_PER_CLASS_TEST = 8000

    print("\n    Applying balanced sampling...")

    train_df = (
        train_df
        .groupby("attack_type", group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), SAMPLES_PER_CLASS_TRAIN),
                random_state=42
            )
        )
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .groupby("attack_type", group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), SAMPLES_PER_CLASS_TEST),
                random_state=42
            )
        )
        .reset_index(drop=True)
    )

    print(f"\n    Balanced train rows: {len(train_df):,}")
    print(f"    Balanced test rows : {len(test_df):,}")

    shared_features = [
        c for c in train_df.columns
        if c in test_df.columns and c not in ("attack_type", "label")
    ]

    train_df = train_df[shared_features + ["attack_type", "label"]]
    test_df = test_df[shared_features + ["attack_type", "label"]]

    print(f"\n    Final train rows: {len(train_df):,}")
    print(f"    Final test rows : {len(test_df):,}")
    print(f"    Feature columns : {len(shared_features)}")

    return train_df, test_df, shared_features


# ═════════════════════════════════════════════════════════════
# RULE-BASED IDS
# ═════════════════════════════════════════════════════════════

class RuleBasedIDS:
    T = {
        "ddos_rate_min": 1000,
        "recon_rate_max": 100,
        "bruteforce_ssh_min": 1,
    }

    _COLS = {
        "rate": "Rate",
        "syn_flag": "syn_flag_number",
        "rst_flag": "rst_flag_number",
        "icmp": "ICMP",
        "tcp": "TCP",
        "udp": "UDP",
        "ssh": "SSH",
        "dns": "DNS",
    }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._resolved = {
            key: col if col in X.columns else None
            for key, col in self._COLS.items()
        }

        return np.array([
            self._classify(row)
            for _, row in X.iterrows()
        ])

    def _get(self, row, key, default=0.0):
        col = self._resolved.get(key)

        if col is None:
            return default

        value = row.get(col, default)

        return default if pd.isna(value) else float(value)

    def _classify(self, row) -> int:
        rate = self._get(row, "rate")
        syn = self._get(row, "syn_flag")
        rst = self._get(row, "rst_flag")
        icmp = self._get(row, "icmp")
        tcp = self._get(row, "tcp")
        udp = self._get(row, "udp")
        ssh = self._get(row, "ssh")
        dns = self._get(row, "dns")

        if rate > self.T["ddos_rate_min"] and (
            syn > 0 or rst > 0 or icmp > 0 or udp > 0
        ):
            return CLASS_NAMES.index("DDoS")

        if ssh >= self.T["bruteforce_ssh_min"]:
            return CLASS_NAMES.index("BruteForce")

        if rate < self.T["recon_rate_max"] and (
            tcp > 0 or icmp > 0 or dns > 0
        ):
            return CLASS_NAMES.index("Recon")

        return CLASS_NAMES.index("Normal")


# ═════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    labels = list(range(len(CLASS_NAMES)))

    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0,
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    f1pc = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0,
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
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            labels=labels,
            zero_division=0,
        )
    )

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "f1_per_class": f1pc,
    }


# ═════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═════════════════════════════════════════════════════════════

from matplotlib.gridspec import GridSpec

PALETTE = {
    "Rule-Based IDS":    "#e74c3c",
    "Decision Tree":     "#f39c12",
    "Random Forest":     "#27ae60",
    "Gradient Boosting": "#2980b9",
}

BG    = "#0d1117"
PANEL = "#161b22"
TEXT  = "#c9d1d9"
GRID  = "#21262d"


def _style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def plot_results(results: list, output_path: str):
    models = [r["model"] for r in results]

    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    colors = [PALETTE.get(model, "#8b949e") for model in models]

    fig = plt.figure(figsize=(20, 16), facecolor=BG)

    fig.suptitle(
        "CIC-IoT2023 | Rule-Based IDS vs Machine Learning",
        fontsize=17,
        color=TEXT,
        fontweight="bold",
        y=0.98,
    )

    gs = GridSpec(
        3,
        4,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        top=0.93,
        bottom=0.05,
        left=0.06,
        right=0.97,
    )

    # Row 1 — metric bar charts
    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = fig.add_subplot(gs[0, col])

        values = [r[metric] for r in results]

        bars = ax.bar(
            models,
            values,
            color=colors,
            width=0.55,
            zorder=3,
        )

        ax.set_ylim(0, 1.12)
        ax.set_title(label, fontsize=11, pad=6)
        ax.set_ylabel("Score", fontsize=8)

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.2f}")
        )

        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

        _style(ax)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=TEXT,
            )

    # Row 2 — normalised confusion matrices
    for col, result in enumerate(results):
        ax = fig.add_subplot(gs[1, col])

        cm = result["confusion_matrix"]

        cm_normalised = cm.astype(float) / (
            cm.sum(axis=1, keepdims=True) + 1e-9
        )

        sns.heatmap(
            cm_normalised,
            annot=True,
            fmt=".2f",
            cmap=sns.color_palette("mako", as_cmap=True),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            cbar=False,
            linewidths=0.3,
            linecolor=BG,
            annot_kws={"size": 8},
            vmin=0,
            vmax=1,
        )

        ax.set_title(result["model"], fontsize=9, pad=4)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Actual", fontsize=7)

        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

        _style(ax)

    # Row 3 left — per-class F1 grouped bar chart
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
            zorder=3,
        )

    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax_f1.set_ylabel("F1-Score", fontsize=9)
    ax_f1.set_title("Per-Class F1 Score Comparison", fontsize=11)
    ax_f1.set_ylim(0, 1.12)

    ax_f1.legend(
        fontsize=7,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        loc="lower right",
    )

    ax_f1.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    _style(ax_f1)

    # Row 3 right — radar chart
    ax_radar = fig.add_subplot(gs[2, 2:], polar=True)

    ax_radar.set_facecolor(PANEL)

    angles = np.linspace(
        0,
        2 * np.pi,
        len(metrics),
        endpoint=False,
    ).tolist()

    angles += angles[:1]

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_labels, color=TEXT, fontsize=8)

    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(
        ["0.25", "0.50", "0.75", "1.00"],
        color="#8b949e",
        fontsize=6,
    )

    ax_radar.grid(color=GRID, linewidth=0.4)
    ax_radar.spines["polar"].set_edgecolor(GRID)

    ax_radar.set_title(
        "Radar: All Metrics",
        fontsize=11,
        color=TEXT,
        pad=12,
    )

    for i, result in enumerate(results):
        values = [result[m] for m in metrics]
        values += values[:1]

        ax_radar.plot(
            angles,
            values,
            linewidth=1.8,
            color=colors[i],
            label=result["model"],
        )

        ax_radar.fill(
            angles,
            values,
            alpha=0.07,
            color=colors[i],
        )

    ax_radar.legend(
        loc="upper right",
        bbox_to_anchor=(1.38, 1.18),
        fontsize=7,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=BG,
        edgecolor="none",
    )

    plt.close()

    print(f"  [✓] Comparison dashboard saved to {output_path}")


def plot_feature_importance(rf_model, feature_names: list, output_path: str):
    importance = rf_model.feature_importances_

    top_n = min(15, len(feature_names))
    top_idx = np.argsort(importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)

    ax.set_facecolor(PANEL)

    ax.barh(
        range(top_n),
        importance[top_idx][::-1],
        color=plt.cm.viridis(np.linspace(0.2, 0.85, top_n)),
        edgecolor=BG,
        linewidth=0.5,
    )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(
        [feature_names[i] for i in top_idx[::-1]],
        fontsize=8,
        color=TEXT,
    )

    ax.set_xlabel(
        "Mean Decrease in Impurity",
        fontsize=9,
        color=TEXT,
    )

    ax.set_title(
        "Random Forest — Top 15 Feature Importances (CIC-IoT2023)",
        fontsize=11,
        color=TEXT,
        pad=10,
    )

    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.tick_params(colors=TEXT)

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    fig.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=BG,
        edgecolor="none",
    )

    plt.close()

    print(f"  [✓] Feature importance saved to {output_path}")
# ═════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════

def main():
    OUT = Path("outputs_cic_iot2023")
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CIC-IoT2023 IDS COMPARISON")
    print("  Classes: Normal, DDoS, Recon, BruteForce")
    print("=" * 60)

    print("\n[1] Loading train/test dataset...")

    train_df, test_df, feature_cols = load_cic_iot2023_train_test(
        TRAIN_PATH,
        TEST_PATH,
    )

    print("\n    Train class distribution:")
    for cls, count in train_df["attack_type"].value_counts().items():
        print(
            f"      {cls:<15} {count:>8,} "
            f"({count / len(train_df) * 100:.1f}%)"
        )

    print("\n    Test class distribution:")
    for cls, count in test_df["attack_type"].value_counts().items():
        print(
            f"      {cls:<15} {count:>8,} "
            f"({count / len(test_df) * 100:.1f}%)"
        )

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    X_test_df = pd.DataFrame(
        X_test,
        columns=feature_cols,
    )

    print(
        f"\n    Train: {len(X_train):,} "
        f"| Test: {len(X_test):,}"
    )

    # Rule-based IDS
    print("\n[2] Running Rule-Based IDS...")

    rule_ids = RuleBasedIDS()
    y_pred_rule = rule_ids.predict(X_test_df)

    r_rule = compute_metrics(
        y_test,
        y_pred_rule,
        "Rule-Based IDS",
    )

    # ML classifiers
    print("\n[3] Training ML classifiers...")

    ml_models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
        ),
    }

    ml_results = []
    rf_model = None

    for name, model in ml_models.items():
        print(f"\n    Training {name}...")

        model.fit(X_train_sc, y_train)

        y_pred = model.predict(X_test_sc)

        result = compute_metrics(
            y_test,
            y_pred,
            name,
        )

        ml_results.append(result)

        try:
            cv = cross_val_score(
                model,
                X_train_sc,
                y_train,
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
            )

            print(
                f"    5-fold CV F1: "
                f"{cv.mean():.4f} ± {cv.std():.4f}"
            )

        except Exception as e:
            print(f"    [!] Cross-validation skipped: {e}")

        if name == "Random Forest":
            rf_model = model

    all_results = [r_rule] + ml_results

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print(
        f"  {'Model':<22} "
        f"{'Accuracy':>9} "
        f"{'Precision':>10} "
        f"{'Recall':>8} "
        f"{'F1':>8}"
    )

    print("  " + "-" * 62)

    for r in all_results:
        print(
            f"  {r['model']:<22} "
            f"{r['accuracy']:>9.4f} "
            f"{r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f}"
        )

    best = max(all_results, key=lambda x: x["f1"])

    print(
        f"\n  Best model: {best['model']} "
        f"(F1 = {best['f1']:.4f})"
    )

    export = []

    for r in all_results:
        export.append({
            "model": r["model"],
            "accuracy": round(r["accuracy"], 4),
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "f1_per_class": {
                cls: round(float(score), 4)
                for cls, score in zip(CLASS_NAMES, r["f1_per_class"])
            },
        })

    with open(OUT / "metrics_summary.json", "w") as f:
        json.dump(export, f, indent=2)

    print(f"\n  [✓] Metrics JSON saved to {OUT / 'metrics_summary.json'}")

    print("\n[4] Generating visualisations...")

    plot_results(
        all_results,
        str(OUT / "ids_comparison.png"),
    )

    if rf_model is not None:
        plot_feature_importance(
            rf_model,
            feature_cols,
            str(OUT / "feature_importance.png"),
        )

    print("\n[✓] Pipeline complete.\n")

    return all_results


if __name__ == "__main__":
    main()