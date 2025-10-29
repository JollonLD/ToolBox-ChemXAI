"""
generate_predictions_from_optuna.py

Creates predictions (y_pred) for each QM9 property using:
 - Optuna best hyperparameters parsed from the optimization summary .txt
 - feature subset from the degradation/explanation CSV (minimal test_loss iteration)

Behavior and fallbacks:
 - Tries to read `Paper/results/optuna/optimization_summary_all_properties.txt` and parse best hyperparams.
 - Loads `Paper/results/test5/all_properties_degradation_results.csv` to find the iteration with minimal `test_loss` per property.
 - Attempts to load `Paper/results/test5/all_properties_explanation_results.csv` and extract `original_indices`/`features_selected` for that iteration; if missing, falls back to using all features and warns.
 - Loads descriptors from `Paper/desc_mordred_qm9.csv` if present; otherwise tries to use `chemxai.data.qm9_tabular` to compute descriptors.
 - Attempts to use `train_with_params` from `Paper.degradation_feature_selection_XAI` (re-using model architecture + hyperparams). If import or run fails, falls back to training a scikit-learn RandomForestRegressor.
 - For each property, trains a model (using the selected features), evaluates on a held-out test split, saves `predictions_{safe_prop}.csv` and `pred_vs_true_{safe_prop}.png` in `Paper/results/test5/`.

Note: This script is defensive and prints warnings when assumptions are required. It does not assume GPU availability.
"""

import ast
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent


def parse_optuna_txt(path):
    """Parse the provided optuna summary text and return a dict:
    { property_name: {param_name: value, ...}, ... }
    This parser is tolerant but expects the same structure as the provided file.
    """
    text = Path(path).read_text(encoding='utf-8')
    props = {}
    blocks = re.split(r"\nPropriedade:\s*", text)
    # The first block is header
    for b in blocks[1:]:
        # first line: property name (may contain ':'), get until newline
        lines = b.strip().splitlines()
        prop_name = lines[0].strip()
        params = {}
        # find lines like '  Melhor loss de validação: 0.000001' or '    n_layers: 3'
        for ln in lines[1:]:
            m1 = re.match(r"\s*Melhor loss de validação:\s*(\S+)", ln)
            if m1:
                params['best_val_loss'] = float(m1.group(1))
                continue
            m2 = re.match(r"\s*(\w[\w_\d]+):\s*(.+)$", ln)
            if m2:
                key = m2.group(1).strip()
                val = m2.group(2).strip()
                # try to convert numeric
                try:
                    if '.' in val or 'e' in val.lower():
                        v = float(val)
                    else:
                        v = int(val)
                except Exception:
                    v = val
                params[key] = v
        props[prop_name] = params
    return props


def safe_literal_eval(s):
    if pd.isna(s):
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        # try to parse simple comma-separated numbers
        try:
            nums = [float(x) for x in re.split(r"[,;\s]+", str(s).strip()) if x != '']
            return nums
        except Exception:
            return s


def load_descriptors():
    # 1) Try Paper/desc_mordred_qm9.csv
    csv_path = ROOT / 'Paper' / 'desc_mordred_qm9.csv'
    if csv_path.exists():
        print(f"Loading descriptors from {csv_path}")
        df = pd.read_csv(csv_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        X = df.values.astype(float)
        # Try to load targets from chemxai if available
        try:
            from chemxai.data import qm9_tabular
            qm = qm9_tabular()
            # load_qm9_dataset returns (coords, props, natoms)
            coords, props, natoms = qm.load_qm9_dataset()
            y_all = np.array(props)
            return X, y_all, props
        except Exception as e:
            print(f"Could not auto-load targets via qm9_tabular.load_qm9_dataset(): {e}. Targets will be None.")
            return X, None, None

    # 2) Fallback: try qm9_tabular.compute_descriptors
    try:
        from chemxai.data import qm9_tabular
        qm = qm9_tabular()
        print("Computing descriptors using qm9_tabular.compute_descriptors()...")
        res = qm.compute_descriptors()
        # expected: X, y, props or similar
        if isinstance(res, tuple) and len(res) >= 2:
            X = np.array(res[0])
            y = np.array(res[1])
            props = res[2] if len(res) > 2 else None
            return X, y, props
    except Exception as e:
        print(f"Failed computing descriptors via qm9_tabular: {e}")

    raise FileNotFoundError("Could not find or compute descriptors. Please provide 'Paper/desc_mordred_qm9.csv' or ensure qm9_tabular.compute_descriptors() is available.")


def safe_prop_to_fname(p):
    return re.sub(r"[^0-9a-zA-Z]+", '_', p).strip('_')[:120]


def main():
    optuna_txt = ROOT / 'results' / 'optuna' / 'optimization_summary_all_properties.txt'
    degradation_csv = RESULTS_DIR / 'all_properties_degradation_results.csv'
    explanation_csv = RESULTS_DIR / 'all_properties_explanation_results.csv'

    if not optuna_txt.exists():
        print(f"Optuna summary not found at {optuna_txt}. Exiting.")
        return

    print("Parsing optuna summary...")
    hyper_map = parse_optuna_txt(optuna_txt)

    print("Loading degradation results...")
    df_deg = pd.read_csv(degradation_csv)

    # find minimal test_loss row per property
    best_iters = df_deg.loc[df_deg.groupby('property')['test_loss'].idxmin()].reset_index(drop=True)

    # attempt to load explanation results (may be empty)
    if explanation_csv.exists():
        df_exp = pd.read_csv(explanation_csv)
    else:
        df_exp = pd.DataFrame()

    # load descriptors and targets
    try:
        X_all, y_all, props = load_descriptors()
    except Exception as e:
        print(f"Error loading descriptors: {e}")
        return

    if y_all is None:
        print("Targets (y) not available; cannot proceed to train/evaluate. Exiting.")
        return

    # For safety, ensure y_all is 2D with properties as columns
    if y_all.ndim == 1:
        y_all = y_all.reshape(-1, 1)

    os.makedirs(RESULTS_DIR / 'predictions', exist_ok=True)

    for _, row in best_iters.iterrows():
        prop = row['property']
        it = int(row['iteration'])
        prop_idx = int(row['property_idx']) if 'property_idx' in row else None
        print('\n' + '='*40)
        print(f"Processing property: {prop} (idx={prop_idx}), best iteration: {it}")

        # get selected features from explanation CSV
        selected_idx = None
        if not df_exp.empty:
            match = df_exp[(df_exp['property'] == prop) & (df_exp['iteration'] == it)]
            if not match.empty:
                # try fields 'original_indices' or 'features_selected'
                val = match.iloc[0].get('original_indices') if 'original_indices' in match.columns else None
                if pd.isna(val) or val is None:
                    val = match.iloc[0].get('features_selected') if 'features_selected' in match.columns else None
                parsed = safe_literal_eval(val)
                if parsed is not None:
                    selected_idx = np.array(parsed, dtype=int)

        if selected_idx is None:
            print(f"Warning: no selected features found for property {prop} iteration {it} in {explanation_csv}. Using all features as fallback.")
            selected_idx = np.arange(X_all.shape[1], dtype=int)

        X_sel = X_all[:, selected_idx]
        y = y_all[:, prop_idx] if prop_idx is not None else y_all[:, 0]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.1, random_state=42)

        # get hyperparams for property
        params = hyper_map.get(prop, {})
        # build model via train_with_params if possible
        model = None
        try:
            sys.path.append(str(ROOT))
            from Paper.degradation_feature_selection_XAI import train_with_params

            # build layers list from params if present
            layers = []
            if 'n_layers' in params:
                nl = int(params['n_layers'])
                for i in range(nl):
                    key = f'layer_{i}_size'
                    if key in params:
                        layers.append(int(params[key]))
            if not layers:
                layers = [512, 256, 128]

            # collect other args
            epochs = int(params.get('n_epochs', 100)) if isinstance(params.get('n_epochs', None), (int, float)) else 100
            lr = float(params.get('lr', 0.001))
            optimizer_name = params.get('optimizer', 'Adam')
            loss_fn = params.get('loss_function', 'L1Loss')
            batch_size = int(params.get('batch_size', 512))
            dropout_rate = float(params.get('dropout_rate', 0.0)) if 'dropout_rate' in params else 0.0
            weight_decay = float(params.get('weight_decay', 0.0)) if 'weight_decay' in params else 0.0

            print(f"Training MLP with layers={layers}, lr={lr}, opt={optimizer_name}, epochs={epochs}")
            model, history = train_with_params(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                              lr=lr, layers=layers, optimizer_name=optimizer_name,
                                              loss_function=loss_fn, dropout_rate=dropout_rate,
                                              weight_decay=weight_decay)

            # Predict on X_test
            model.eval()
            import torch
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                y_pred = model(X_test_t).cpu().numpy().flatten()

        except Exception as e:
            print(f"Could not use train_with_params / MLP (fallback). Reason: {e}")
            print("Falling back to RandomForestRegressor for quick predictions.")
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

        # Save predictions and plot
        fname_safe = safe_prop_to_fname(prop)
        out_csv = RESULTS_DIR / 'predictions' / f'predictions_{fname_safe}.csv'
        out_png = RESULTS_DIR / 'predictions' / f'pred_vs_true_{fname_safe}.png'

        df_out = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
        df_out.to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv}")

        # plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        mn = min(np.nanmin(y_test), np.nanmin(y_pred))
        mx = max(np.nanmax(y_test), np.nanmax(y_pred))
        plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title(f'Predicted vs True — {prop}')
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved plot to {out_png}")

    print('\nAll done.')


if __name__ == '__main__':
    main()
