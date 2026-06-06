"""
Shared output helpers for the univariate VaR / ES model modules.

All of QRF, QGB, GAS, GARCH, HISTSIM and EVT return the same per-stock
forecast schema (Date, Actual, VaR_1%, VaR_5%, Median (optional), ES_1%,
ES_5%, FZ_5%, FZ_1%). This module turns that dict into the three outputs
the corresponding Jupyter notebooks produce:

  - combine_results(results) -> long-format DataFrame across stocks
  - summary_table(results)   -> per-stock averages and violation rates
  - plot_forecast_grid(...)  -> 5x2 matplotlib grid (one panel per stock)
  - report(results, name)    -> writes both CSVs + the PNG to `output/`,
                                printable summary to stdout

`report` is the function each model module calls from `__main__`, so that
`python QRF.py` (or any of the other model modules) reproduces the full
notebook output on disk without needing Jupyter.
"""

import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def combine_results(results):
    """Concat the per-stock forecast frames into one long-format DataFrame.

    Adds a `Stock` column so the result can be filtered or grouped easily.
    """
    parts = []
    for stock, df in results.items():
        d = df.copy()
        d["Stock"] = stock
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def summary_table(results):
    """Per-stock summary mirroring the `*_summary_df` cell in each notebook.

    Returns a DataFrame with one row per stock:
      Stock, Avg_FZ5%, Avg_FZ1%, VaR5_Viol, VaR1_Viol, ES5_Viol
    """
    rows = []
    for stock, df in results.items():
        rows.append({
            "Stock":     stock,
            "Avg_FZ5%":  df["FZ_5%"].mean(),
            "Avg_FZ1%":  df["FZ_1%"].mean(),
            "VaR5_Viol": (df["Actual"] < df["VaR_5%"]).mean(),
            "VaR1_Viol": (df["Actual"] < df["VaR_1%"]).mean(),
            "ES5_Viol":  (df["Actual"] < df["ES_5%"]).mean(),
        })
    return pd.DataFrame(rows)


def plot_forecast_grid(results, model_name, save_to=None,
                       figsize=(13, 18), show=False):
    """Reproduce the forecast grid plot in a tall 5x2 portrait layout.

    A portrait layout keeps each panel large and legible once the figure is
    scaled to the text width in the thesis; a landscape grid shrinks the ten
    panels to an unreadable size. Per stock: black returns, red solid VaR_5%,
    dark-red dashed VaR_1%, purple solid ES_5%, indigo dashed ES_1%. The
    two-line title shows the stock and its empirical violation rates and
    average FZ losses.

    Parameters
    ----------
    results : dict {stock: forecast DataFrame}
    model_name : str
        Used in the suptitle and in the saved file name.
    save_to : str or None
        If given, also writes the figure to disk as PNG.
    figsize : tuple
        Forwarded to plt.subplots.
    show : bool
        If True, blocks on plt.show(). Default False (suitable for
        running .py from the terminal); set True if calling from a REPL
        or notebook.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import pandas as pd

    n = len(results)
    ncols = 2
    rows = max(1, -(-n // ncols))   # ceil division -> 5 rows for 10 assets
    fig, axes = plt.subplots(rows, ncols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Shared x-axis range across all panels, so assets with a shorter sample
    # (e.g. the Diageo ADR, which starts later) align on a common timeline
    # rather than getting their own auto-scaled axis with different ticks.
    all_dates = pd.concat([pd.to_datetime(res["Date"]) for res in results.values()])
    xmin, xmax = all_dates.min(), all_dates.max()

    handles = labels = None
    for i, (stock, res) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(res["Date"], res["Actual"], color="black",  alpha=0.45, lw=0.7, label="Returns")
        ax.plot(res["Date"], res["VaR_5%"], color="red",      lw=1.5,    label="VaR 5%")
        ax.plot(res["Date"], res["VaR_1%"], color="darkred",  lw=1.5,    label="VaR 1%", linestyle="--")
        ax.plot(res["Date"], res["ES_5%"],  color="purple",   lw=1.5,    label="ES 5%")
        ax.plot(res["Date"], res["ES_1%"],  color="indigo",   lw=1.5,    label="ES 1%", linestyle="--")
        ax.set_xlim(xmin, xmax)

        v5 = (res["Actual"] < res["VaR_5%"]).mean()
        v1 = (res["Actual"] < res["VaR_1%"]).mean()
        fz5 = res["FZ_5%"].mean()
        fz1 = res["FZ_1%"].mean()
        ax.set_title(f"{stock}\n"
                     f"viol 5%={v5:.2%}  1%={v1:.2%}\n"
                     f"FZ5={fz5:.2f}  FZ1={fz1:.2f}", fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # Hide unused panels.
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{model_name} -- out-of-sample VaR and ES forecasts",
                 fontsize=22, y=1.005)
    # Pack panels first, reserving a bottom band for the figure-level legend
    # so it never overlaps the bottom row's x-axis tick labels.
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=15, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    if save_to is not None:
        plt.savefig(save_to, dpi=200, bbox_inches="tight")
        print(f"  saved plot -> {save_to}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def report(results, model_name, save_dir="output", show=False):
    """End-to-end: print summary to stdout and save combined CSV, summary
    CSV, and the 5x2 plot to ``save_dir``.

    The model modules call this from their ``__main__`` block, so that
    running e.g. ``python QRF.py`` reproduces the full notebook output:
    a printed summary table, two CSVs, and a PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    print()
    print(f"========== {model_name} -- SUMMARY ==========")
    summary = summary_table(results)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    combined = combine_results(results)
    combined_path = os.path.join(save_dir, f"{model_name.lower()}_forecasts.csv")
    summary_path  = os.path.join(save_dir, f"{model_name.lower()}_summary.csv")
    plot_path     = os.path.join(save_dir, f"{model_name.lower()}_grid.png")

    combined.to_csv(combined_path, index=False)
    print(f"\n  saved combined forecasts -> {combined_path}  ({len(combined):,} rows)")
    summary.to_csv(summary_path, index=False)
    print(f"  saved summary table      -> {summary_path}")

    plot_forecast_grid(results, model_name, save_to=plot_path, show=show)

    return summary, combined
