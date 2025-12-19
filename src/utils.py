from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
)

@dataclass
class EvalResult:
    roc_auc: float
    pr_auc: float
    brier: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    precision: float
    recall: float
    fpr: float
    tnr: float
    accuracy: float
    balanced_acc: float

def evaluate_at_threshold(
    y_true: Union[np.ndarray, pd.Series],
    y_proba: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> EvalResult:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    balanced_acc = 0.5 * (recall + tnr)

    return EvalResult(
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        brier=float(brier),
        threshold=float(threshold),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        precision=float(precision),
        recall=float(recall),
        fpr=float(fpr),
        tnr=float(tnr),
        accuracy=float(accuracy),
        balanced_acc=float(balanced_acc),
    )

def pick_threshold_for_target_recall(
    y_true: Union[np.ndarray, pd.Series],
    y_proba: Union[np.ndarray, pd.Series],
    target_recall: float = 0.80,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds length = len(recall)-1; recall[0] has no threshold
    for i in range(1, len(recall)):
        if recall[i] >= target_recall:
            return float(thresholds[i - 1])
    return 1.0

def _group_series(group: Union[pd.Series, np.ndarray]) -> pd.Series:
    if isinstance(group, pd.Series):
        return group.astype("object")
    return pd.Series(group, dtype="object")

def group_rates_at_threshold(
    y_true: Union[np.ndarray, pd.Series],
    y_proba: Union[np.ndarray, pd.Series],
    group: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    g = _group_series(group)

    y_pred = (y_proba >= threshold).astype(int)

    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "g": g})
    out = []
    for grp, d in df.groupby("g", dropna=False):
        n = len(d)
        pos_rate = d["yhat"].mean() if n else np.nan

        d_pos = d[d["y"] == 1]
        d_neg = d[d["y"] == 0]

        tpr = d_pos["yhat"].mean() if len(d_pos) else np.nan
        fpr = d_neg["yhat"].mean() if len(d_neg) else np.nan

        out.append(
            {
                "группа": grp,
                "n": n,
                "n_дефолт": int((d["y"] == 1).sum()),
                "n_не_дефолт": int((d["y"] == 0).sum()),
                "доля_положит_решений": float(pos_rate) if pd.notna(pos_rate) else np.nan,
                "полнота_TPR": float(tpr) if pd.notna(tpr) else np.nan,
                "ложноположит_FPR": float(fpr) if pd.notna(fpr) else np.nan,
            }
        )
    return pd.DataFrame(out).sort_values("n", ascending=False).reset_index(drop=True)

def disparity_summary(group_table: pd.DataFrame) -> Dict[str, float]:
    gt = group_table.copy()

    def diff(col: str) -> float:
        s = gt[col].dropna()
        return float(s.max() - s.min()) if len(s) else float("nan")

    def ratio(col: str) -> float:
        s = gt[col].dropna()
        if len(s) == 0:
            return float("nan")
        mn, mx = float(s.min()), float(s.max())
        if mx == 0:
            return float("nan")
        return mn / mx

    return {
        "разница_долей_решений": diff("доля_положит_решений"),
        "отношение_долей_решений": ratio("доля_положит_решений"),
        "разница_TPR": diff("полнота_TPR"),
        "разница_FPR": diff("ложноположит_FPR"),
    }

def measure_inference_latency_ms(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    n_runs: int = 30,
    batch_size: int = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    import time
    rng = np.random.default_rng(random_state)

    n = len(X) if hasattr(X, "__len__") else X.shape[0]
    times = []
    for _ in range(n_runs):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        Xb = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        t0 = time.perf_counter()
        _ = model.predict_proba(Xb)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "среднее_мс": float(times.mean()),
        "p95_мс": float(np.percentile(times, 95)),
        "мин_мс": float(times.min()),
        "макс_мс": float(times.max()),
    }
