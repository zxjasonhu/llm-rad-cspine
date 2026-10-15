import numpy as np
import pandas as pd

from typing import List, Optional

def base_calculate_metrics(y_true, y_pred):
    """Binary classification metrics for one fracture label."""
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape:
        raise ValueError("Ground truth and prediction lengths differ")
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    ratio = lambda num, den: num / den if den else float("nan")
    sen = ratio(tp, tp + fn)
    spec = ratio(tn, tn + fp)
    ppv = ratio(tp, tp + fp)
    npv = ratio(tn, tn + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    n = len(true)
    observed = ratio(tp + tn, n)
    expected = ratio((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn), n * n)
    kappa = ratio(observed - expected, 1 - expected)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "ACC": observed, "SEN": sen, "SPEC": spec, "PPV": ppv,
            "NPV": npv, "F1": f1, "Kappa": kappa}

frac_cols = [
    "C1_Fracture",
    "C2_Fracture",
    "C3_Fracture",
    "C4_Fracture",
    "C5_Fracture",
    "C6_Fracture",
    "C7_Fracture",
    "Occ_Condyle_Fracture",
]

def _collapse_multi_col(
    gt: pd.DataFrame, pred: pd.DataFrame, cols: List[str], _name: str
) -> dict:
    """
    Collapse multiple columns to a single column
    Args:
        gt: ground truth dataframe
        pred: predicted dataframe
        cols: columns to collapse
        _name: name of the new column
    Returns:
        dict: evaluation metrics
    """
    pred[_name] = pred[cols].sum(axis=1)
    pred[_name] = pred[_name].apply(lambda x: 1 if x > 0 else 0)
    gt[_name] = gt[cols].sum(axis=1)
    gt[_name] = gt[_name].apply(lambda x: 1 if x > 0 else 0)

    gt_c = gt[_name].values
    pred_c = pred[_name].values

    return base_calculate_metrics(y_true=gt_c, y_pred=pred_c)


def regularize_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Regularize the dataframe by converting the column to binary values
    Args:
        df: dataframe to regularize
        col: column name to regularize
    Returns:
        pd.DataFrame: regularized dataframe
    """
    df = df.copy()
    df[col] = df[col].astype(str).apply(lambda x: x.lower())
    is_ambiguous = df[col].apply(
        lambda x: any(kw in x for kw in ["null", "none", "nan"])
    )
    df[col] = df[col].apply(lambda x: "1" if any(kw in x for kw in ["1.", "yes", "true"]) else x)
    df[col] = df[col].apply(lambda x: 0 if any(kw in x for kw in ["no evi", "no info", "no finding", "0", "no", "false"]) else 1)
    df[col] = df[col].astype(int)
    # Missing values are negative; unfamiliar nonempty values are positive.
    df.loc[is_ambiguous, col] = 0
    return df


def _eval(gt: pd.DataFrame, pred: pd.DataFrame, col: Optional[str | List[str]]) -> dict:
    """
    Evaluate the performance of the model
    Args:
        gt: ground truth dataframe
        pred: predicted dataframe
        col: column name to evaluate
    Returns:
        dict: evaluation metrics
    """
    # Create copies to avoid modifying original DataFrames
    gt = gt.copy().astype(str)
    pred = pred.copy().astype(str)

    if col is None:
        col = frac_cols
    elif isinstance(col, str):
        col = [col]

    metrics = {}

    for c in col:
        if c not in gt.columns:
            raise ValueError(f"Column {c} not in ground truth dataframe")

        if c not in pred.columns:
            raise ValueError(f"Column {c} not in predicted dataframe")

        # regularize the dataframe
        gt = regularize_df(gt, c)
        pred = regularize_df(pred, c)

        gt_c = gt[c].values
        pred_c = pred[c].values

        metrics[c] = base_calculate_metrics(y_true=gt_c, y_pred=pred_c)

    metrics["Any_Fracture"] = _collapse_multi_col(gt, pred, frac_cols, "Any_Fracture")

    return metrics


def _metrics2df(metrics: dict) -> pd.DataFrame:
    """
    Convert metrics to dataframe
    Args:
        metrics: metrics dict
    Returns:
        pd.DataFrame: metrics dataframe
    """
    df = pd.DataFrame(metrics).T
    # name the first column as "category"
    df.index.name = "category"
    df.reset_index(inplace=True)
    return df


def binary_eval(
    gt: pd.DataFrame, pred: pd.DataFrame, col: Optional[str | List[str]]
) -> pd.DataFrame:
    """
    Evaluate the performance of the model
    Args:
        gt: ground truth dataframe
        pred: predicted dataframe
        col: column name to evaluate
    Returns:
        pd.DataFrame: metrics dataframe
        str: report
    """
    metrics = _eval(gt, pred, col)
    metrics_df = _metrics2df(metrics)
    return metrics_df
