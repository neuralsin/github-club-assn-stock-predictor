from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

def four_way_comparison(y_true, preds: dict, probs: dict = None, out_path: str = None) -> pd.DataFrame:
    rows = []
    for model_name, pred_values in preds.items():
        auc_val = 0.5000
        pr_auc_val = round(float(pd.Series(y_true).mean()), 4)

        if probs and model_name in probs and probs[model_name] is not None:
            p = probs[model_name]
            try:
                auc_val = round(float(roc_auc_score(y_true, p)), 4)
                pr_auc_val = round(float(average_precision_score(y_true, p)), 4)
            except Exception:
                pass
        else:
            try:
                auc_val = round(float(roc_auc_score(y_true, pred_values)), 4)
                pr_auc_val = round(float(average_precision_score(y_true, pred_values)), 4)
            except Exception:
                pass

        rows.append(
            {
                "Model": model_name,
                "ROC-AUC": auc_val,
                "PR-AUC": pr_auc_val,
                "Accuracy": round(float(accuracy_score(y_true, pred_values)), 4),
                "Precision": round(float(precision_score(y_true, pred_values, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_true, pred_values, zero_division=0)), 4),
                "F1 Score": round(float(f1_score(y_true, pred_values, zero_division=0)), 4),
            }
        )
    df_table = pd.DataFrame(rows)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_table.to_csv(out_path, index=False)
    return df_table


def class_balance_report(labels: pd.Series, out_path: str = None) -> str:
    clean_labels = labels.dropna().astype(int)
    counts = clean_labels.value_counts().sort_index()
    props = clean_labels.value_counts(normalize=True).sort_index()

    table_df = pd.DataFrame(
        {
            "Class": ["0 (Down / Lower Barrier)", "1 (Up / Upper Barrier)"],
            "Count": [int(counts.get(0, 0)), int(counts.get(1, 0))],
            "Percentage": [f"{props.get(0, 0.0) * 100:.2f}%", f"{props.get(1, 0.0) * 100:.2f}%"],
        }
    )

    markdown_content = (
        "Class Balance Report\n\n"
        f"- **Total Samples Analyzed**: {len(clean_labels)}\n"
        f"- **Dominant Class**: Class {int(counts.idxmax())} ({props.max() * 100:.2f}%)\n\n"
        + table_df.to_markdown(index=False)
        + "\n"
    )

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
    return markdown_content

def plot_predicted_vs_actual(dates, actual, predicted, out_path: str = "outputs/prediction_plot.png") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    date_series = pd.to_datetime(dates)
    plot_len = min(120, len(date_series))
    d_sub = date_series[-plot_len:]
    a_sub = actual[-plot_len:]
    p_sub = predicted[-plot_len:]

    ax1.step(d_sub, a_sub, label="Ground Truth Direction", color="tab:blue", linewidth=1.8, where="post")
    ax1.step(d_sub, p_sub, label="STOCK PREDICTOR (Engineered Stack)", color="tab:orange", linestyle="--", linewidth=1.8, where="post")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Down (0)", "Up (1)"])
    ax1.set_ylabel("Directional State")
    ax1.set_title("STOCK PREDICTOR — Out-of-Sample Movement Classification (Trailing Test Window)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)

    agreement = (a_sub == p_sub).astype(int)
    colors = ["tab:green" if x == 1 else "tab:red" for x in agreement]
    ax2.bar(d_sub, agreement, color=colors, width=1.0, alpha=0.7)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Mismatch", "Correct"])
    ax2.set_ylabel("Hit / Miss")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
