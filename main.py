import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


from src import (
    data_ingest_01 as ingest,
    labeling_02 as label,
    features_03 as feat,
    baselines_04 as base,
    cpcv_05 as cpcv,
    model_tcn_06 as tcn_mod,
    model_meta_07 as meta,
    backtest_stats_08 as stats,
    report_09 as report,
)

def run():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    ticker = cfg["ticker"]
    print(f"STOCK PREDICTOR: Starting quantitative pipeline for {ticker}")
    df = ingest.fetch_ohlcv(
        ticker=ticker,
        start=cfg.get("start_date"),
        end=cfg.get("end_date"),
        data_source=cfg.get("data_source", "yfinance"),
        folder_path=cfg.get("folder_path"),
    )
    print(f"Ingested {len(df)} bars from {df.index.min()} to {df.index.max()}")

    labels_naive = label.naive_next_day_label(df)
    tb_df = label.triple_barrier_labels(df, **cfg["labeling"])
    labels_tb = tb_df["label_tb"]

    features, optimal_d = feat.build_features(df, **cfg["fracdiff"])
    print(f"Fractional differentiation optimal order d = {optimal_d:.2f}")

    label.leakage_audit(features, labels_tb, "outputs/leakage_audit.csv", n_rows=30)

    aligned = features.join(labels_tb).join(labels_naive).dropna()
    target = aligned["label_tb"].astype(int)
    features_clean = aligned[features.columns]

    raw_cols = [c for c in ["raw_close", "raw_volume", "raw_return_1d"] if c in features_clean.columns]
    split_idx = int(len(aligned) * 0.8)

    train_features = features_clean.iloc[:split_idx]
    test_features = features_clean.iloc[split_idx:]
    train_target = target.iloc[:split_idx]
    test_target = target.iloc[split_idx:]
    test_dates = aligned.index[split_idx:]

    pred_persist = base.persistence_baseline(df).loc[test_dates].values
    pred_majority = base.majority_class_baseline(target, range(split_idx)).loc[test_dates].values

    scaler_raw = StandardScaler()
    X_raw_train = scaler_raw.fit_transform(train_features[raw_cols])
    X_raw_test = scaler_raw.transform(test_features[raw_cols])

    raw_lgb = meta.train_meta_model(X_raw_train, train_target.values, cfg.get("lightgbm"))
    prob_raw = raw_lgb.predict_proba(X_raw_test)[:, 1]
    pred_raw = (prob_raw > 0.5).astype(int)

    scaler_all = StandardScaler()
    X_eng_train = scaler_all.fit_transform(train_features)
    X_eng_test = scaler_all.transform(test_features)

    window = cfg["tcn"]["input_window"]
    X_seq_train, y_seq_train = tcn_mod.make_windows(X_eng_train, train_target.values, window)
    X_seq_test, y_seq_test = tcn_mod.make_windows(X_eng_test, test_target.values, window)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcn = tcn_mod.TCN(
        n_features=X_eng_train.shape[1],
        channels=cfg["tcn"]["channels"],
        kernel_size=cfg["tcn"]["kernel_size"],
        dropout=cfg["tcn"]["dropout"],
    )

    val_split = int(len(X_seq_train) * 0.85)
    tcn_mod.train_tcn(
        tcn,
        X_seq_train[:val_split],
        y_seq_train[:val_split],
        X_seq_train[val_split:],
        y_seq_train[val_split:],
        epochs=cfg["tcn"]["epochs"],
        batch_size=cfg["tcn"]["batch_size"],
        lr=cfg["tcn"]["lr"],
        device=device,
    )

    tcn_train_probs = tcn_mod.predict_tcn(tcn, X_seq_train, device=device)
    tcn_test_probs = tcn_mod.predict_tcn(tcn, X_seq_test, device=device)

    tab_model = meta.train_meta_model(X_eng_train[window:], y_seq_train.numpy(), cfg.get("lightgbm"))
    lgb_test_probs = tab_model.predict_proba(X_eng_test[window:])[:, 1]

    tcn_val_probs = tcn_mod.predict_tcn(tcn, X_seq_train[val_split:], device=device)
    lgb_val_probs = tab_model.predict_proba(X_eng_train[window + val_split:])[:, 1]

    val_y = y_seq_train[val_split:].numpy().astype(int)

    weights = np.linspace(0.0, 1.0, 21)
    best_w = 0.5
    best_val_auc = 0.0
    for w in weights:
        val_blend = w * tcn_val_probs + (1.0 - w) * lgb_val_probs
        try:
            v_auc = roc_auc_score(val_y, val_blend)
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                best_w = w
        except Exception:
            pass

    val_stack_probs = best_w * tcn_val_probs + (1.0 - best_w) * lgb_val_probs
    prob_eng = best_w * tcn_test_probs + (1.0 - best_w) * lgb_test_probs

    thresholds = np.linspace(0.45, 0.55, 21)
    best_th = 0.5
    best_acc = 0
    for th in thresholds:
        acc = accuracy_score(val_y, (val_stack_probs > th).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    pred_eng = (prob_eng > best_th).astype(int)


    eval_target = y_seq_test.numpy().astype(int)
    eval_dates = test_dates[window:]

    preds_dict = {
        "Persistence Baseline": pred_persist[window:],
        "Majority Class": pred_majority[window:],
        "Raw Features (LightGBM)": pred_raw[window:],
        "Engineered Stack (TCN + LightGBM)": pred_eng,
    }

    probs_dict = {
        "Persistence Baseline": None,
        "Majority Class": np.full(len(eval_target), float(train_target.mean())),
        "Raw Features (LightGBM)": prob_raw[window:],
        "Engineered Stack (TCN + LightGBM)": prob_eng,
    }

    comp_table = report.four_way_comparison(eval_target, preds_dict, probs_dict, "outputs/comparison_table.csv")
    report.class_balance_report(target, "outputs/class_balance_report.md")
    report.plot_predicted_vs_actual(eval_dates, eval_target, pred_eng, "outputs/prediction_plot.png")

    splitter = cpcv.CombinatorialPurgedCV(
        n_groups=cfg["cpcv"]["n_groups"],
        n_test_groups=cfg["cpcv"]["n_test_groups"],
        embargo_pct=cfg["cpcv"]["embargo_pct"],
    )

    touch_indices = tb_df["touch_idx"].loc[aligned.index].values
    is_sharpes = []
    oos_sharpes = []
    fold_accs = []
    is_intraday = any(hasattr(d, "hour") and d.hour != 0 for d in aligned.index[:10])
    periods_year = 252 * 75 if is_intraday else 252

    for tr_idx, te_idx, grps in splitter.split(len(aligned), touch_indices):
        tr_y = target.iloc[tr_idx].values
        te_y = target.iloc[te_idx].values

        if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
            continue

        fold_scaler = StandardScaler()
        tr_X = fold_scaler.fit_transform(features_clean.iloc[tr_idx])
        te_X = fold_scaler.transform(features_clean.iloc[te_idx])

        fold_model = meta.train_meta_model(tr_X, tr_y, cfg.get("lightgbm"))
        is_pred = fold_model.predict(tr_X)
        oos_pred = fold_model.predict(te_X)

        tr_ret = features_clean["raw_return_1d"].iloc[tr_idx].values
        te_ret = features_clean["raw_return_1d"].iloc[te_idx].values

        is_strat_ret = tr_ret * np.where(is_pred == 1, 1.0, -1.0)
        oos_strat_ret = te_ret * np.where(oos_pred == 1, 1.0, -1.0)

        is_sharpes.append(stats.sharpe_ratio(is_strat_ret, periods_per_year=periods_year))
        oos_sharpes.append(stats.sharpe_ratio(oos_strat_ret, periods_per_year=periods_year))
        fold_accs.append(accuracy_score(te_y, oos_pred))

    best_sr = max(oos_sharpes) if oos_sharpes else 0.0
    dsr_val = stats.deflated_sharpe_ratio(best_sr, oos_sharpes, len(aligned))

    print("\n--- Four-Way Model Comparison Table ---")
    print(comp_table.to_string(index=False))
    print(f"\nCPCV Cross-Validation: {len(oos_sharpes)} Folds")
    print(f"Mean CPCV OOS Accuracy: {np.mean(fold_accs):.4f}")
    print(f"Maximum OOS Sharpe Ratio: {best_sr:.4f}")
    print(f"Deflated Sharpe Ratio (DSR): {dsr_val:.4f}")
    print("Deliverables generated in outputs/ directory.")

    counts = target.value_counts()
    props = target.value_counts(normalize=True)

    web_data = {
        "ticker": ticker,
        "regime": f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
        "total_bars": len(df),
        "analyzed_samples": len(aligned),
        "cpcv_mean_acc": round(float(np.mean(fold_accs)), 4),
        "optimal_d": round(float(optimal_d), 2),
        "max_oos_sharpe": round(float(best_sr), 4),
        "dsr": round(float(dsr_val), 4),
        "class_0_count": int(counts.get(0, 0)),
        "class_0_pct": round(float(props.get(0, 0.0) * 100), 2),
        "class_1_count": int(counts.get(1, 0)),
        "class_1_pct": round(float(props.get(1, 0.0) * 100), 2),
        "comparison_table": comp_table.to_dict(orient="records"),
    }


    web_dir = Path("web")
    if web_dir.exists():
        with open(web_dir / "data.json", "w") as f:
            json.dump(web_data, f, indent=2)
        assets_dir = web_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        shutil.copy("outputs/prediction_plot.png", assets_dir / "prediction_plot.png")

if __name__ == "__main__":
    run()

