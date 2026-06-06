"""
Shared output helpers for the CoVaR model modules (COCAVIAR.py, DCCGARCH.py).

The (VaR, CoVaR) forecasters return one DataFrame per (reference, system)
pair with the following schema (Dimitriadis & Hoga 2026 notation):

    Date, X_loss, Y_loss, VaR_X, CoVaR_Y

DCC-GARCH additionally adds S_VaR and S_CoVaR (the per-observation
lexicographic-score components from D&H 2026 eq. 5); when these columns
are absent (as in COCAVIAR.py output), the helpers recompute them on the
fly so the summary table is comparable across models.

`report` is what each module calls from `__main__`, so that
`python COCAVIAR.py` or `python DCCGARCH.py` reproduces the notebook-style
output (printed summary, two CSVs, one PNG) on disk without needing
Jupyter.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from losses import s_var, s_covar


# =========================================================================
# 1. Combined long-format DataFrame
# =========================================================================


def _ensure_scores(df, beta=0.95, alpha=0.95):
    """Add S_VaR / S_CoVaR columns if they are not already present.

    Both COCAVIAR.py and DCCGARCH.py work in loss coordinates (positive
    values), so the standard tick-loss and CoVaR-score from D&H eq. 5
    apply directly.
    """
    out = df.copy()
    if "S_VaR" not in out.columns:
        out["S_VaR"] = s_var(out["VaR_X"].values, out["X_loss"].values, beta=beta)
    if "S_CoVaR" not in out.columns:
        out["S_CoVaR"] = s_covar(out["VaR_X"].values, out["CoVaR_Y"].values,
                                  out["X_loss"].values, out["Y_loss"].values,
                                  alpha=alpha)
    return out


def combine_results(results, best_specs=None, beta=0.95, alpha=0.95):
    """Concat the per-pair forecast frames into one long-format DataFrame.

    Adds a `Reference` column (the X asset) and, if `best_specs` is given,
    a `Spec` column with the lex-best CoCAViaR specification per pair.
    """
    parts = []
    for ref, df in results.items():
        d = _ensure_scores(df, beta=beta, alpha=alpha)
        d["Reference"] = ref
        if best_specs is not None and ref in best_specs:
            d["Spec"] = best_specs[ref]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# =========================================================================
# 2. Summary table
# =========================================================================


def summary_table(results, best_specs=None, beta=0.95, alpha=0.95):
    """Per-pair summary of (VaR, CoVaR) calibration and score components.

    Columns: Reference, [Spec], n, VaR_distress, CoVaR_cond_viol,
             Avg_S_VaR, Avg_S_CoVaR.
    """
    rows = []
    for ref, df in results.items():
        d = _ensure_scores(df, beta=beta, alpha=alpha)
        distress = d["X_loss"] > d["VaR_X"]
        n_d = max(1, int(distress.sum()))
        co_viol = (((d["Y_loss"] > d["CoVaR_Y"]) & distress).sum() / n_d)
        row = {
            "Reference":         ref,
            "n":                 int(len(d)),
            "VaR_distress":      float(distress.mean()),
            "CoVaR_cond_viol":   float(co_viol),
            "Avg_S_VaR":         float(d["S_VaR"].mean()),
            "Avg_S_CoVaR":       float(d["S_CoVaR"].mean()),
        }
        if best_specs is not None and ref in best_specs:
            row = {"Reference": ref, "Spec": best_specs[ref], **{k: v for k, v in row.items() if k != "Reference"}}
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
# 3. Plot grid
# =========================================================================


def plot_covar_grid(results, model_name, best_specs=None,
                    save_to=None, figsize=(15, 13), show=False,
                    beta=0.95, alpha=0.95, show_var=True):
    """Tall 5x2 portrait grid of CoVaR_Y forecasts per (reference, system) pair.

    Each panel: grey X-loss and Y-loss series, red VaR_X line, purple
    CoVaR_Y line. Title shows empirical VaR distress rate (target 5%)
    and conditional CoVaR violation rate (target 5%).

    Parameters
    ----------
    show_var : bool, default True
        If True (CoCAViaR-style), plot both VaR_X and CoVaR_Y, with the
        X-loss series and the VaR-distress rate in the title. If False
        (DCC-GARCH-style), plot only the CoVaR_Y line and the Y-loss
        series, with the conditional-violation rate in the title. The
        DCC-GARCH VaR forecast is uninformative as a stand-alone risk
        measure (the appropriate univariate VaR benchmark is GARCH-t),
        so showing only the CoVaR line keeps the figure aligned with
        the lex-loss philosophy of the multivariate analysis.
    """
    n = len(results)
    ncols = 2
    rows = max(1, -(-n // ncols))   # ceil division -> 5 rows for 9-10 pairs
    fig, axes = plt.subplots(rows, ncols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Shared x-axis range across all panels so a shorter-sample pair
    # (e.g. the Diageo ADR) aligns on a common timeline.
    all_dates = pd.concat([pd.to_datetime(df["Date"]) for df in results.values()])
    xmin, xmax = all_dates.min(), all_dates.max()

    handles = labels = None
    for i, (ref, df) in enumerate(results.items()):
        ax = axes[i]
        if show_var:
            ax.plot(df["Date"], df["X_loss"],   color="black", alpha=0.30, lw=0.7,
                    label="X loss (reference)")
        ax.plot(df["Date"], df["Y_loss"],   color="grey",  alpha=0.35, lw=0.7,
                label="Y loss (system)")
        if show_var:
            ax.plot(df["Date"], df["VaR_X"],    color="red", lw=1.5,
                    label=f"VaR_X ({int(beta*100)}%)")
        ax.plot(df["Date"], df["CoVaR_Y"],  color="purple", lw=1.6,
                label=f"CoVaR_Y|X ({int(alpha*100)}%|{int(beta*100)}%)")
        ax.set_xlim(xmin, xmax)

        distress = df["X_loss"] > df["VaR_X"]
        n_d = max(1, int(distress.sum()))
        cv  = ((df["Y_loss"] > df["CoVaR_Y"]) & distress).sum() / n_d
        vd  = float(distress.mean())

        spec_lbl = (f" ({best_specs[ref]})"
                    if best_specs is not None and ref in best_specs else "")
        if show_var:
            title = (f"{ref}{spec_lbl}\n"
                     f"distress={vd:.2%}  cond viol={cv:.2%}")
        else:
            # CoVaR-only view: the VaR is merely the conditioning device in the
            # two-step (VaR, CoVaR) estimator, not the object of interest, so
            # neither the VaR line nor the VaR distress rate is shown.
            title = f"{ref}{spec_lbl}\nCoVaR cond viol={cv:.2%}"
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # Legend: if there is an empty panel slot (odd number of pairs, e.g. 9 in a
    # 5x2 grid), host the legend there so it never overlaps a panel; otherwise
    # place a figure-level legend below the grid.
    if n < len(axes):
        leg_ax = axes[n]
        leg_ax.set_visible(True)
        leg_ax.axis("off")
        leg_ax.legend(handles, labels, loc="center", fontsize=16, framealpha=0.9)
        for j in range(n + 1, len(axes)):
            axes[j].set_visible(False)
    else:
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=15, framealpha=0.9, bbox_to_anchor=(0.5, -0.015))
    suptitle_obj = ("(VaR_X, CoVaR_Y)" if show_var else "CoVaR_Y")
    fig.suptitle(f"{model_name} -- out-of-sample {suptitle_obj} forecasts",
                 fontsize=22, y=1.005)
    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, dpi=200, bbox_inches="tight")
        print(f"  saved plot -> {save_to}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =========================================================================
# 4. End-to-end driver -- called from __main__ blocks
# =========================================================================


def report(results, model_name, best_specs=None, save_dir="output",
           show=False, beta=0.95, alpha=0.95, show_var=True):
    """End-to-end: print summary to stdout and save combined CSV, summary
    CSV, and the 5x2 plot to ``save_dir``.

    Parameters
    ----------
    show_var : bool, default True
        Forwarded to ``plot_covar_grid``. Set to False for DCC-GARCH
        family models, where the VaR forecast is not the primary
        quantity of interest -- only CoVaR is.
    """
    os.makedirs(save_dir, exist_ok=True)

    print()
    print(f"========== {model_name} -- SUMMARY ==========")
    summary = summary_table(results, best_specs=best_specs,
                            beta=beta, alpha=alpha)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    combined = combine_results(results, best_specs=best_specs,
                               beta=beta, alpha=alpha)
    combined_path = os.path.join(save_dir, f"{model_name.lower()}_forecasts.csv")
    summary_path  = os.path.join(save_dir, f"{model_name.lower()}_summary.csv")
    plot_path     = os.path.join(save_dir, f"{model_name.lower()}_grid.png")

    combined.to_csv(combined_path, index=False)
    print(f"\n  saved combined forecasts -> {combined_path}  ({len(combined):,} rows)")
    summary.to_csv(summary_path, index=False)
    print(f"  saved summary table      -> {summary_path}")

    plot_covar_grid(results, model_name, best_specs=best_specs,
                    save_to=plot_path, show=show, beta=beta, alpha=alpha,
                    show_var=show_var)

    return summary, combined
