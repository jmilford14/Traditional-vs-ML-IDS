"""
IoT Intrusion Detection: Rule-Based vs Machine Learning Comparison
=================================================================
Uses the N-BaIoT / UNSW-NB15 inspired synthetic IoT dataset structure.
We simulate a realistic IoT attack dataset since direct downloads
require external network access.
"""

import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Scikit-learn ────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# ── Plotting ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC IoT DATASET GENERATOR
#     Mimics N-BaIoT / CICIDS feature space
# ═══════════════════════════════════════════════════════════════════════════

def generate_iot_dataset(n_samples: int = 15_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic IoT network traffic dataset with labelled attacks.

    Attack categories (matching real IoT threat landscape):
      - Normal          : legitimate device traffic
      - DoS             : Denial-of-Service flood
      - Mirai_Botnet    : Mirai-style botnet C&C / scan traffic
      - Port_Scan       : reconnaissance scanning
      - MITM            : Man-in-the-Middle / ARP spoofing
    """
    rng = np.random.RandomState(random_state)

    class_config = {
        "Normal":       {"n": int(n_samples * 0.45), "label": 0},
        "DoS":          {"n": int(n_samples * 0.20), "label": 1},
        "Mirai_Botnet": {"n": int(n_samples * 0.15), "label": 2},
        "Port_Scan":    {"n": int(n_samples * 0.12), "label": 3},
        "MITM":         {"n": int(n_samples * 0.08), "label": 4},
    }

    records = []

    for attack_type, cfg in class_config.items():
        n = cfg["n"]

        if attack_type == "Normal":
            data = {
                "flow_duration":        rng.exponential(5.0,  n),
                "fwd_packets":          rng.poisson(12,       n).astype(float),
                "bwd_packets":          rng.poisson(10,       n).astype(float),
                "fwd_bytes":            rng.lognormal(7, 1.5, n),
                "bwd_bytes":            rng.lognormal(6, 1.5, n),
                "packet_rate":          rng.uniform(1, 50,    n),
                "byte_rate":            rng.uniform(100, 5000,n),
                "fwd_iat_mean":         rng.exponential(0.5,  n),
                "bwd_iat_mean":         rng.exponential(0.5,  n),
                "psh_flag_count":       rng.poisson(2,        n).astype(float),
                "ack_flag_count":       rng.poisson(10,       n).astype(float),
                "syn_flag_count":       rng.poisson(1,        n).astype(float),
                "fin_flag_count":       rng.poisson(1,        n).astype(float),
                "rst_flag_count":       rng.poisson(0.1,      n).astype(float),
                "unique_dest_ports":    rng.randint(1, 5,     n).astype(float),
                "avg_packet_size":      rng.normal(400, 150,  n),
                "flow_iat_std":         rng.exponential(0.3,  n),
                "active_mean":          rng.exponential(2,    n),
                "idle_mean":            rng.exponential(1,    n),
            }

        elif attack_type == "DoS":
            data = {
                "flow_duration":        rng.exponential(0.2,  n),
                "fwd_packets":          rng.poisson(200,      n).astype(float),
                "bwd_packets":          rng.poisson(5,        n).astype(float),
                "fwd_bytes":            rng.lognormal(8, 0.5, n),
                "bwd_bytes":            rng.lognormal(3, 0.5, n),
                "packet_rate":          rng.uniform(500, 5000,n),
                "byte_rate":            rng.uniform(50000,500000,n),
                "fwd_iat_mean":         rng.exponential(0.001,n),
                "bwd_iat_mean":         rng.exponential(0.1,  n),
                "psh_flag_count":       rng.poisson(0.1,      n).astype(float),
                "ack_flag_count":       rng.poisson(1,        n).astype(float),
                "syn_flag_count":       rng.poisson(50,       n).astype(float),
                "fin_flag_count":       rng.poisson(0.1,      n).astype(float),
                "rst_flag_count":       rng.poisson(5,        n).astype(float),
                "unique_dest_ports":    rng.randint(1, 3,     n).astype(float),
                "avg_packet_size":      rng.normal(60, 20,    n),
                "flow_iat_std":         rng.exponential(0.001,n),
                "active_mean":          rng.exponential(0.1,  n),
                "idle_mean":            rng.exponential(0.01, n),
            }

        elif attack_type == "Mirai_Botnet":
            data = {
                "flow_duration":        rng.exponential(0.5,  n),
                "fwd_packets":          rng.poisson(30,       n).astype(float),
                "bwd_packets":          rng.poisson(2,        n).astype(float),
                "fwd_bytes":            rng.lognormal(5, 0.8, n),
                "bwd_bytes":            rng.lognormal(2, 0.5, n),
                "packet_rate":          rng.uniform(50, 300,  n),
                "byte_rate":            rng.uniform(1000,20000,n),
                "fwd_iat_mean":         rng.exponential(0.05, n),
                "bwd_iat_mean":         rng.exponential(0.5,  n),
                "psh_flag_count":       rng.poisson(0.5,      n).astype(float),
                "ack_flag_count":       rng.poisson(3,        n).astype(float),
                "syn_flag_count":       rng.poisson(10,       n).astype(float),
                "fin_flag_count":       rng.poisson(2,        n).astype(float),
                "rst_flag_count":       rng.poisson(2,        n).astype(float),
                "unique_dest_ports":    rng.randint(1, 3,     n).astype(float),
                "avg_packet_size":      rng.normal(100, 40,   n),
                "flow_iat_std":         rng.exponential(0.02, n),
                "active_mean":          rng.exponential(0.3,  n),
                "idle_mean":            rng.exponential(0.1,  n),
            }

        elif attack_type == "Port_Scan":
            data = {
                "flow_duration":        rng.exponential(0.05, n),
                "fwd_packets":          rng.poisson(2,        n).astype(float),
                "bwd_packets":          rng.poisson(1,        n).astype(float),
                "fwd_bytes":            rng.lognormal(3, 0.5, n),
                "bwd_bytes":            rng.lognormal(2, 0.5, n),
                "packet_rate":          rng.uniform(20, 200,  n),
                "byte_rate":            rng.uniform(100, 2000,n),
                "fwd_iat_mean":         rng.exponential(0.01, n),
                "bwd_iat_mean":         rng.exponential(0.01, n),
                "psh_flag_count":       rng.poisson(0.1,      n).astype(float),
                "ack_flag_count":       rng.poisson(0.5,      n).astype(float),
                "syn_flag_count":       rng.poisson(1,        n).astype(float),
                "fin_flag_count":       rng.poisson(0.1,      n).astype(float),
                "rst_flag_count":       rng.poisson(0.5,      n).astype(float),
                "unique_dest_ports":    rng.randint(50, 1024, n).astype(float),  # KEY indicator
                "avg_packet_size":      rng.normal(50, 15,    n),
                "flow_iat_std":         rng.exponential(0.005,n),
                "active_mean":          rng.exponential(0.05, n),
                "idle_mean":            rng.exponential(0.01, n),
            }

        elif attack_type == "MITM":
            data = {
                "flow_duration":        rng.exponential(8.0,  n),
                "fwd_packets":          rng.poisson(15,       n).astype(float),
                "bwd_packets":          rng.poisson(15,       n).astype(float),
                "fwd_bytes":            rng.lognormal(6, 1.0, n),
                "bwd_bytes":            rng.lognormal(6, 1.0, n),
                "packet_rate":          rng.uniform(5, 40,    n),
                "byte_rate":            rng.uniform(200, 4000,n),
                "fwd_iat_mean":         rng.exponential(1.0,  n),
                "bwd_iat_mean":         rng.exponential(1.0,  n),
                "psh_flag_count":       rng.poisson(5,        n).astype(float),
                "ack_flag_count":       rng.poisson(15,       n).astype(float),
                "syn_flag_count":       rng.poisson(2,        n).astype(float),
                "fin_flag_count":       rng.poisson(2,        n).astype(float),
                "rst_flag_count":       rng.poisson(1,        n).astype(float),
                "unique_dest_ports":    rng.randint(1, 4,     n).astype(float),
                "avg_packet_size":      rng.normal(350, 100,  n),
                "flow_iat_std":         rng.exponential(0.8,  n),
                "active_mean":          rng.exponential(3,    n),
                "idle_mean":            rng.exponential(2,    n),
            }

        df_chunk = pd.DataFrame(data)
        df_chunk["attack_type"] = attack_type
        df_chunk["label"] = cfg["label"]
        records.append(df_chunk)

    df = pd.concat(records, ignore_index=True)

    # clip negatives that can arise from normal distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  RULE-BASED IDS  (traditional / baseline)
# ═══════════════════════════════════════════════════════════════════════════

class RuleBasedIDS:
    """
    Signature / threshold-based Intrusion Detection System.
    Rules are hand-crafted from domain knowledge about IoT attacks.
    """

    # Thresholds (tunable per deployment)
    THRESHOLDS = {
        "dos_packet_rate":     300,    # packets/s → DoS
        "dos_syn_flags":       20,     # SYN flood indicator
        "dos_avg_pkt_size":    100,    # tiny packets → flood
        "scan_dest_ports":     20,     # many ports → scan
        "scan_fwd_packets":    5,      # few pkts per flow
        "botnet_fwd_iat":      0.1,    # rapid periodic sends
        "botnet_packet_rate":  40,     # moderate burst
        "mitm_iat_ratio":      2.0,    # symmetric IAT → relay
    }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self._classify_row(row))
        return np.array(predictions)

    def _classify_row(self, row) -> int:
        T = self.THRESHOLDS

        # ── DoS rule ────────────────────────────────────────────────────
        if (row["packet_rate"]   > T["dos_packet_rate"] or
            row["syn_flag_count"] > T["dos_syn_flags"]  or
            row["avg_packet_size"] < T["dos_avg_pkt_size"] and
            row["packet_rate"]   > 100):
            return 1  # DoS

        # ── Port Scan rule ───────────────────────────────────────────────
        if (row["unique_dest_ports"] > T["scan_dest_ports"] and
            row["fwd_packets"]       < T["scan_fwd_packets"]):
            return 3  # Port_Scan

        # ── Mirai Botnet rule ────────────────────────────────────────────
        if (row["fwd_iat_mean"]  < T["botnet_fwd_iat"] and
            row["packet_rate"]   > T["botnet_packet_rate"] and
            row["bwd_packets"]   < 5):
            return 2  # Mirai_Botnet

        # ── MITM rule ────────────────────────────────────────────────────
        # Symmetric bi-directional flow with long duration
        fwd_iat = row["fwd_iat_mean"] + 1e-9
        bwd_iat = row["bwd_iat_mean"] + 1e-9
        iat_ratio = max(fwd_iat, bwd_iat) / min(fwd_iat, bwd_iat)
        if (iat_ratio < T["mitm_iat_ratio"] and
            row["flow_duration"] > 3.0 and
            row["ack_flag_count"] > 10):
            return 4  # MITM

        return 0  # Normal


# ═══════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["Normal", "DoS", "Mirai_Botnet", "Port_Scan", "MITM"]

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    # per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None,
                            labels=list(range(len(CLASS_NAMES))),
                            zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"\n  Per-class F1:")
    for cls, score in zip(CLASS_NAMES, f1_per_class):
        bar = "█" * int(score * 20)
        print(f"    {cls:<14} {score:.3f}  {bar}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                zero_division=0))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "f1_per_class": f1_per_class,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

PALETTE = {
    "Rule-Based IDS":          "#e74c3c",
    "Decision Tree":           "#f39c12",
    "Random Forest":           "#27ae60",
    "Gradient Boosting":       "#2980b9",
}
BG      = "#0d1117"
PANEL   = "#161b22"
TEXT    = "#c9d1d9"
GRID    = "#21262d"


def style_axes(ax):
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
    colors = [PALETTE.get(m, "#8b949e") for m in models]

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    fig.suptitle("IoT IDS: Rule-Based vs Machine Learning",
                 fontsize=18, color=TEXT, fontweight="bold", y=0.98)

    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.05, left=0.06, right=0.97)

    # ── Row 0: metric bars ──────────────────────────────────────────────
    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = fig.add_subplot(gs[0, col])
        vals = [r[metric] for r in results]
        bars = ax.bar(models, vals, color=colors, width=0.55, zorder=3)
        ax.set_ylim(0, 1.08)
        ax.set_title(label, fontsize=11, pad=6)
        ax.set_ylabel("Score", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        style_axes(ax)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, color=TEXT)

    # ── Row 1: Confusion matrices ───────────────────────────────────────
    for col, result in enumerate(results):
        ax = fig.add_subplot(gs[1, col])
        cm = result["confusion_matrix"]
        # normalise rows
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        cmap = sns.color_palette("mako", as_cmap=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar=False, linewidths=0.3, linecolor=BG,
                    annot_kws={"size": 6}, vmin=0, vmax=1)
        ax.set_title(result["model"], fontsize=9, pad=4)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Actual", fontsize=7)
        ax.tick_params(axis="x", rotation=35, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)
        style_axes(ax)

    # ── Row 2, col 0-1: per-class F1 grouped bar ─────────────────────
    ax_f1 = fig.add_subplot(gs[2, :2])
    x = np.arange(len(CLASS_NAMES))
    width = 0.18
    for i, result in enumerate(results):
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax_f1.bar(x + offset, result["f1_per_class"],
                         width, label=result["model"],
                         color=colors[i], alpha=0.88, zorder=3)
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax_f1.set_ylabel("F1-Score", fontsize=9)
    ax_f1.set_title("Per-Class F1 Score Comparison", fontsize=11)
    ax_f1.set_ylim(0, 1.1)
    ax_f1.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID,
                 labelcolor=TEXT, loc="lower right")
    ax_f1.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    style_axes(ax_f1)

    # ── Row 2, col 2-3: radar / spider chart ────────────────────────
    ax_radar = fig.add_subplot(gs[2, 2:], polar=True)
    ax_radar.set_facecolor(PANEL)
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_labels, color=TEXT, fontsize=8)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["0.25","0.5","0.75","1.0"],
                              color="#8b949e", fontsize=6)
    ax_radar.grid(color=GRID, linewidth=0.4)
    ax_radar.spines["polar"].set_edgecolor(GRID)
    ax_radar.set_title("Radar: All Metrics", fontsize=11, color=TEXT, pad=12)

    for i, result in enumerate(results):
        vals_r = [result[m] for m in metrics] + [result[metrics[0]]]
        ax_radar.plot(angles, vals_r, linewidth=1.5,
                      color=colors[i], label=result["model"])
        ax_radar.fill(angles, vals_r, alpha=0.08, color=colors[i])

    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                    fontsize=7, facecolor=PANEL, edgecolor=GRID,
                    labelcolor=TEXT)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"\n  [✓] Figure saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FEATURE IMPORTANCE PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(rf_model, feature_names: list, output_path: str):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 12
    top_idx = indices[:top_n]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    bars = ax.barh(range(top_n),
                   importances[top_idx][::-1],
                   color=plt.cm.viridis(np.linspace(0.2, 0.85, top_n)),
                   edgecolor=BG, linewidth=0.5)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]],
                       fontsize=9, color=TEXT)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=9, color=TEXT)
    ax.set_title("Random Forest – Top Feature Importances",
                 fontsize=12, color=TEXT, pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  [✓] Feature importance saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
from pathlib import Path

def main():
    OUT = Path('outputs_rt_iot2022')
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  IoT INTRUSION DETECTION: Rule-Based vs ML")
    print("="*60)

    # ── 6.1  Generate dataset ───────────────────────────────────────────
    print("\n[1] Generating IoT network traffic dataset …")
    df = generate_iot_dataset(n_samples=15_000)
    print(f"    Total samples : {len(df):,}")
    print(f"    Features      : {df.shape[1]-2}")
    print(f"    Class distribution:")
    for cls, cnt in df["attack_type"].value_counts().items():
        print(f"      {cls:<16} {cnt:5,}  ({cnt/len(df)*100:.1f}%)")

    # save dataset sample
    df.to_csv(OUT / "iot_dataset_sample.csv", index=False)

    # ── 6.2  Prepare features / labels ─────────────────────────────────
    feature_cols = [c for c in df.columns if c not in ("attack_type","label")]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 6.3  Rule-Based IDS ─────────────────────────────────────────────
    print("\n[2] Running Rule-Based IDS …")
    rule_ids = RuleBasedIDS()
    y_pred_rule = rule_ids.predict(X_test)
    r_rule = compute_metrics(y_test, y_pred_rule, "Rule-Based IDS")

    # ── 6.4  ML Models ──────────────────────────────────────────────────
    print("\n[3] Training ML models …")

    ml_models = {
        "Decision Tree":     DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=150, max_depth=20,
                                                     n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                         random_state=42),
    }

    ml_results = []
    rf_model   = None

    for name, model in ml_models.items():
        print(f"\n    Training {name} …")
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        result = compute_metrics(y_test, y_pred, name)
        ml_results.append(result)

        # cross-val on training set
        cv_scores = cross_val_score(model, X_train_sc, y_train,
                                    cv=5, scoring="f1_weighted", n_jobs=-1)
        print(f"    5-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if name == "Random Forest":
            rf_model = model

    # ── 6.5  Compare all approaches ─────────────────────────────────────
    all_results = [r_rule] + ml_results

    print("\n" + "="*60)
    print("  SUMMARY COMPARISON")
    print("="*60)
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("-" * 60)
    for r in all_results:
        print(f"{r['model']:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f}"
              f" {r['recall']:>8.4f} {r['f1']:>8.4f}")

    best = max(all_results, key=lambda x: x["f1"])
    print(f"\n  ★  Best model: {best['model']}  (F1 = {best['f1']:.4f})")

    # ── 6.6  Save metrics JSON ──────────────────────────────────────────
    metrics_export = []
    for r in all_results:
        metrics_export.append({
            "model":     r["model"],
            "accuracy":  round(r["accuracy"],  4),
            "precision": round(r["precision"], 4),
            "recall":    round(r["recall"],    4),
            "f1":        round(r["f1"],        4),
            "f1_per_class": {
                cls: round(float(score), 4)
                for cls, score in zip(CLASS_NAMES, r["f1_per_class"])
            }
        })
    with open(OUT / "metrics_summary.json", "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"\n  [✓] Metrics JSON → {OUT}/metrics_summary.json")

    # ── 6.7  Plots ──────────────────────────────────────────────────────
    print("\n[4] Generating visualisations …")
    plot_results(all_results,     str(OUT / "ids_comparison.png"))
    plot_feature_importance(rf_model, feature_cols,
                            str(OUT / "feature_importance.png"))

    print("\n[✓] Pipeline complete.\n")
    return all_results


if __name__ == "__main__":
    main()
