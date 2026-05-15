"""
UNSW-NB15 Intrusion Detection: Rule-Based vs Machine Learning
============================================================
Uses the official UNSW-NB15 training and testing CSV files.

Expected files:
    datasets/unsw_nb15/UNSW_NB15_training-set.csv
    datasets/unsw_nb15/UNSW_NB15_testing-set.csv

Run:
    python ids_unsw_nb15.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# ═════════════════════════════════════════════════════════════════════════════
# 1.  UNSW-NB15 LOADER
# ═════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = [
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]


def load_unsw_nb15() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load official UNSW-NB15 train/test CSV files from local paths.

    Returns:
        train_df, test_df
    """
    train_path = Path("datasets/unsw_nb15/UNSW_NB15_training-set.csv")
    test_path = Path("datasets/unsw_nb15/UNSW_NB15_testing-set.csv")

    if not train_path.exists():
        raise FileNotFoundError(
            f"Could not find training file: {train_path}\n"
            "Place UNSW_NB15_training-set.csv in datasets/unsw_nb15/"
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Could not find testing file: {test_path}\n"
            "Place UNSW_NB15_testing-set.csv in datasets/unsw_nb15/"
        )

    print(f"    Loading UNSW-NB15 training file: {train_path}")
    print(f"    Loading UNSW-NB15 testing file:  {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for df_name, df in [("train", train_df), ("test", test_df)]:
        # Clean label text
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

        # Remove rows with missing or unknown labels
        valid_mask = df["attack_cat"].isin(CLASS_NAMES)
        n_invalid = (~valid_mask).sum()
        if n_invalid:
            bad_vals = sorted(df.loc[~valid_mask, "attack_cat"].unique().tolist())
            print(f"    [!] Dropping {n_invalid:,} invalid rows from {df_name}: {bad_vals}")
            df.drop(df.index[~valid_mask], inplace=True)

        # Replace inf with nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    train_df["target"] = train_df["attack_cat"].map(class_to_idx).astype(int)
    test_df["target"] = test_df["attack_cat"].map(class_to_idx).astype(int)

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, test_df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  RULE-BASED IDS FOR UNSW-NB15
# ═════════════════════════════════════════════════════════════════════════════

class RuleBasedIDSUNSW:
    """
    Very simple threshold-based IDS baseline for UNSW-NB15.

    This baseline is intentionally lightweight and interpretable.
    It is only a rough heuristic and is not expected to outperform ML.
    """

    def __init__(self):
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            preds.append(self._classify(row))
        return np.array(preds, dtype=int)

    def _classify(self, row: pd.Series) -> int:
        proto = str(row.get("proto", "")).lower()
        service = str(row.get("service", "")).lower()
        state = str(row.get("state", "")).upper()

        dur = float(row.get("dur", 0) or 0)
        spkts = float(row.get("spkts", 0) or 0)
        dpkts = float(row.get("dpkts", 0) or 0)
        sbytes = float(row.get("sbytes", 0) or 0)
        dbytes = float(row.get("dbytes", 0) or 0)
        rate = float(row.get("rate", 0) or 0)
        sload = float(row.get("sload", 0) or 0)
        dload = float(row.get("dload", 0) or 0)
        sttl = float(row.get("sttl", 0) or 0)
        dttl = float(row.get("dttl", 0) or 0)

        ct_srv_src = float(row.get("ct_srv_src", 0) or 0)
        ct_srv_dst = float(row.get("ct_srv_dst", 0) or 0)
        ct_dst_ltm = float(row.get("ct_dst_ltm", 0) or 0)
        ct_src_ltm = float(row.get("ct_src_ltm", 0) or 0)
        ct_dst_src_ltm = float(row.get("ct_dst_src_ltm", 0) or 0)
        ct_src_dport_ltm = float(row.get("ct_src_dport_ltm", 0) or 0)

        trans_depth = float(row.get("trans_depth", 0) or 0)
        response_body_len = float(row.get("response_body_len", 0) or 0)
        is_ftp_login = float(row.get("is_ftp_login", 0) or 0)
        ct_ftp_cmd = float(row.get("ct_ftp_cmd", 0) or 0)
        ct_flw_http_mthd = float(row.get("ct_flw_http_mthd", 0) or 0)
        is_sm_ips_ports = float(row.get("is_sm_ips_ports", 0) or 0)

        # 1. Generic
        if (
            service == "dns"
            and state == "INT"
            and dpkts == 0
            and dbytes == 0
            and rate > 50000
            and ct_srv_src >= 10
        ):
            return self.class_to_idx["Generic"]

        # 2. DoS
        if (
            state == "INT"
            and dpkts == 0
            and dbytes == 0
            and (rate > 80000 or sload > 5e7 or spkts <= 2)
        ):
            return self.class_to_idx["DoS"]

        # 3. Reconnaissance
        if (
            (state == "INT" or state == "FIN")
            and dbytes == 0
            and (ct_dst_ltm <= 2 and ct_srv_dst <= 2)
            and (ct_src_dport_ltm <= 2 and ct_dst_src_ltm <= 2)
            and service in {"-", "http", "dns", "snmp"}
        ):
            return self.class_to_idx["Reconnaissance"]

        # 4. Analysis
        if (
            state == "INT"
            and dbytes == 0
            and ct_dst_src_ltm >= 4
            and ct_srv_dst >= 4
        ):
            return self.class_to_idx["Analysis"]

        # 5. Exploits
        if (
            state == "FIN"
            and dbytes > 0
            and dpkts > 0
            and service in {"http", "ftp", "ftp-data", "smtp", "-"}
            and (trans_depth > 0 or response_body_len > 0 or ct_flw_http_mthd > 0)
        ):
            return self.class_to_idx["Exploits"]

        # 6. Backdoor
        if (
            state in {"INT", "CON"}
            and sttl >= 200
            and dttl == 0
            and service in {"-", "irc", "ftp"}
        ):
            return self.class_to_idx["Backdoor"]

        # 7. Shellcode
        if (
            state in {"INT", "FIN"}
            and dpkts == 0
            and dbytes == 0
            and sbytes >= 500
            and service == "-"
        ):
            return self.class_to_idx["Shellcode"]

        # 8. Worms
        if (
            service == "http"
            and state == "FIN"
            and dbytes > 0
            and spkts >= 8
            and dpkts >= 4
        ):
            return self.class_to_idx["Worms"]

        # 9. Fuzzers
        if (
            state == "FIN"
            and service in {"-", "ftp", "dns", "http"}
            and dbytes > 0
            and dpkts > 0
            and dur > 0.1
            and rate < 200
        ):
            return self.class_to_idx["Fuzzers"]

        # 10. Normal
        return self.class_to_idx["Normal"]


# ═════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    labels = list(range(len(CLASS_NAMES)))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    f1pc = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print("\n  Per-class F1:")
    for cls, score in zip(CLASS_NAMES, f1pc):
        bar = "█" * int(score * 20)
        print(f"    {cls:<16} {score:.3f}  {bar}")

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
# 4.  VISUALISATION
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
        "UNSW-NB15  |  Rule-Based IDS vs Machine Learning",
        fontsize=17,
        color=TEXT,
        fontweight="bold",
        y=0.98
    )

    gs = GridSpec(
        3, 4,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        top=0.93,
        bottom=0.05,
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
            annot_kws={"size": 7},
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
        offset = (i - len(results)/2 + 0.5) * width
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
    ax_f1.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
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

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  [✓] Comparison chart → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(rf_pipeline, feature_names_after_preproc: list, output_path: str):
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
        [feature_names_after_preproc[i] for i in top_idx[::-1]],
        fontsize=8,
        color=TEXT
    )
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=9, color=TEXT)
    ax.set_title(
        "Random Forest — Top Feature Importances (UNSW-NB15)",
        fontsize=11,
        color=TEXT,
        pad=10
    )
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.tick_params(colors=TEXT)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  [✓] Feature importance → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    OUT = Path("outputs_unsw_nb15")
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  IDS COMPARISON  ·  UNSW-NB15 dataset")
    print("=" * 60)

    # 6.1 Load
    print("\n[1] Loading UNSW-NB15 …")
    train_df, test_df = load_unsw_nb15()

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"\n    Train samples : {len(train_df):,}")
    print(f"    Test samples  : {len(test_df):,}")
    print(f"    Total samples : {len(full_df):,}")

    drop_cols = ["id", "attack_cat", "label", "target"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    print(f"    Feature cols  : {len(feature_cols)}")

    print("\n    Class distribution:")
    for cls, cnt in full_df["attack_cat"].value_counts().items():
        print(f"      {cls:<16} {cnt:6,}  ({cnt / len(full_df) * 100:.1f}%)")

    # 6.2 Features
    X_train_df = train_df[feature_cols].copy()
    X_test_df = test_df[feature_cols].copy()
    y_train = train_df["target"].values
    y_test = test_df["target"].values

    categorical_cols = ["proto", "service", "state"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0))
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ],
        remainder="drop"
    )

    print(f"\n    Using official split  |  Train: {len(X_train_df):,}  Test: {len(X_test_df):,}")

    # 6.3 Rule-based IDS
    print("\n[2] Running Rule-Based IDS …")
    rule_ids = RuleBasedIDSUNSW()
    y_pred_rule = rule_ids.predict(X_test_df)
    r_rule = compute_metrics(y_test, y_pred_rule, "Rule-Based IDS")

    # 6.4 ML models
    print("\n[3] Training ML models …")
    ml_models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
    }

    ml_results = []
    rf_pipeline = None
    feature_names_after_preproc = None

    for name, model in ml_models.items():
        print(f"\n    Training {name} …")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train_df, y_train)
        y_pred = pipe.predict(X_test_df)
        result = compute_metrics(y_test, y_pred, name)
        ml_results.append(result)

        cv_scores = cross_val_score(
            pipe,
            X_train_df,
            y_train,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1
        )
        print(f"    5-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if name == "Random Forest":
            rf_pipeline = pipe
            feature_names_after_preproc = pipe.named_steps["preprocessor"].get_feature_names_out().tolist()

    # 6.5 Summary
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

    # 6.6 Export JSON
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

    # 6.7 Plots
    print("\n[4] Generating visualisations …")
    plot_results(all_results, str(OUT / "ids_comparison.png"))

    if rf_pipeline is not None and feature_names_after_preproc is not None:
        plot_feature_importance(
            rf_pipeline,
            feature_names_after_preproc,
            str(OUT / "feature_importance.png")
        )

    print("\n[✓] Pipeline complete.\n")
    return all_results


if __name__ == "__main__":
    main()