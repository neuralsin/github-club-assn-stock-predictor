import importlib
import sys

modules_map = {
    "data_ingest_01": "src.01_data_ingest",
    "labeling_02": "src.02_labeling",
    "features_03": "src.03_features",
    "baselines_04": "src.04_baselines",
    "cpcv_05": "src.05_cpcv",
    "model_tcn_06": "src.06_model_tcn",
    "model_meta_07": "src.07_model_meta",
    "backtest_stats_08": "src.08_backtest_stats",
    "report_09": "src.09_report",
    "data_ingest": "src.01_data_ingest",
    "labeling": "src.02_labeling",
    "features": "src.03_features",
    "baselines": "src.04_baselines",
    "cpcv": "src.05_cpcv",
    "model_tcn": "src.06_model_tcn",
    "model_meta": "src.07_model_meta",
    "backtest_stats": "src.08_backtest_stats",
    "report": "src.09_report",
}

for export_name, mod_path in modules_map.items():
    mod = importlib.import_module(mod_path)
    globals()[export_name] = mod
    sys.modules[f"{__name__}.{export_name}"] = mod
