"""
This module provides the following utilities:
    - Import and Dependency Management: Importing necessary libraries and modules.
    - Plot Outlier Numbers: Generates a scatter plot to compare the number of outliers removed by different methods.
    - Plot Dataset Inclusion: Generates a scatter plot to compare the number of users in different datasets.
    - Plot Model Comparison: Generates a scatter plot to compare user reputations predicted by different models.
    - Plot Reputation Comparison: Generates a scatter plot to compare original, predicted, and final reputations per user.
    - Plot Final Experience vs Final Reputation: Generates a scatter plot to compare user final experience against final reputation.
    - Plot All Visualizations: Orchestrates the generation of all visualizations for the project.
"""

# ========== Imports and Dependencies ===========
import plotly.express as px  #  Plotly Express for creating interactive plots.
import plotly.graph_objects as go  # Plotly Graph Objects for more complex visualizations.
import numpy as np  # NumPy for light jitter and density grids
from plotly.subplots import make_subplots  # Faceted layouts

import pandas as pd  # Pandas for DataFrame handling and CSV data processing.
from pathlib import Path  # Pathlib for file path handling and manipulation.

import modules.config as config  # Configuration settings for the project.
import modules.system_guard as system  # System guard module for managing system-level operations and configurations.
from modules.system_guard import logger  # Logger for logging messages and errors.
import modules.utils as utils  # File handling utilities for loading and saving data.


# ========== Helpers for clearer dataset inclusion visuals ==========
def _compute_majority_per_user_per_day(
    dfs: dict, labels: list, priority_order=None
) -> pd.DataFrame:
    """
    Given a dictionary of labeled DataFrames (each with Day, User_Id) and a list of labels to include
    (e.g., ["Normal (Supervised)", "Intentional (Supervised)", "Unintentional (Supervised)"]),
    return a DataFrame with columns ["Day", "Normal", "Intentional", "Unintentional"] where, for each day,
    each user is assigned to exactly ONE category based on the majority of their submissions.
    Ties are resolved by `priority_order` (default: Intentional > Unintentional > Normal).
    """
    if priority_order is None:
        priority_order = ["Intentional", "Unintentional", "Normal"]

    frames = []
    for label in labels:
        if label not in dfs:
            continue
        df = dfs[label].copy()
        if "Day" not in df.columns or "User_Id" not in df.columns:
            continue
        # Extract the base category name before the " (" suffix
        category = label.split(" (")[0]
        tmp = df.groupby(["Day", "User_Id"]).size().reset_index(name="count")
        tmp["Category"] = category
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["Day", "Normal", "Intentional", "Unintentional"])

    all_df = pd.concat(frames, ignore_index=True)
    # For each (Day, User_Id) keep the category with the maximum count; break ties by priority_order
    all_df["prio"] = all_df["Category"].apply(
        lambda c: (
            priority_order.index(c) if c in priority_order else len(priority_order)
        )
    )
    all_df = all_df.sort_values(
        ["Day", "User_Id", "count", "prio"], ascending=[True, True, False, True]
    )
    majority = all_df.groupby(["Day", "User_Id"]).first().reset_index()

    counts = (
        majority.groupby(["Day", "Category"])["User_Id"]
        .nunique()
        .reset_index()
        .pivot(index="Day", columns="Category", values="User_Id")
        .fillna(0.0)
        .reset_index()
    )
    # Ensure all expected columns exist
    for col in ["Normal", "Intentional", "Unintentional"]:
        if col not in counts.columns:
            counts[col] = 0.0
    counts = counts.sort_values("Day")
    # Ensure Day is datetime (not Period) for Plotly
    counts["Day"] = pd.to_datetime(counts["Day"].astype(str))
    return counts


def _compute_submission_share(dfs: dict, labels: list) -> pd.DataFrame:
    """
    Given labeled DataFrames (each with Day, User_Id rows representing submissions),
    compute the per-day percentage share of submissions per category.
    Returns columns: ["Day", "Normal", "Intentional", "Unintentional"] with values in [0, 100].
    """
    frames = []
    for label in labels:
        if label not in dfs:
            continue
        df = dfs[label].copy()
        if "Day" not in df.columns:
            continue
        category = label.split(" (")[0]
        tmp = df.groupby("Day").size().reset_index(name="count")
        tmp["Category"] = category
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["Day", "Normal", "Intentional", "Unintentional"])

    all_df = pd.concat(frames, ignore_index=True)
    totals = all_df.groupby("Day")["count"].sum().rename("total").reset_index()
    wide = (
        all_df.pivot_table(
            index="Day", columns="Category", values="count", aggfunc="sum"
        )
        .fillna(0.0)
        .reset_index()
    )
    wide = wide.merge(totals, on="Day", how="left")
    for col in ["Normal", "Intentional", "Unintentional"]:
        if col not in wide.columns:
            wide[col] = 0.0
    # Convert to percentages
    for col in ["Normal", "Intentional", "Unintentional"]:
        wide[col] = (wide[col] / wide["total"] * 100.0).fillna(0.0)
    wide = wide.drop(columns=["total"]).sort_values("Day")
    # Ensure Day is datetime (not Period) for Plotly
    wide["Day"] = pd.to_datetime(wide["Day"].astype(str))
    return wide


# ========== Plot ONLY Majority-Both Dataset Inclusion Figure ==========
@system.timer_decorator
def plot_dataset_inclusion_user_majority_both_only():
    """
    Generate ONLY the combined 'Users per Day by Majority Category' plot (Supervised & Unsupervised, six traces).
    Saves:
      - plots/dataset_inclusion_user_majority_both.html
      - plots/dataset_inclusion_user_majority_both.png
    """
    try:
        # Build split map (same mapping used elsewhere)
        split_types = ["supervised", "unsupervised"]
        split_names = ["Normal", "Intentional", "Unintentional"]
        split_map = []
        for stype in split_types:
            for sname in split_names:
                if stype == "unsupervised":
                    fname = f"{sname}_Data"
                else:
                    fname = f"{sname}_Data_supervised"
                label = f"{sname} ({stype.capitalize()})"
                split_map.append(
                    {"stype": stype, "sname": sname, "file": fname, "label": label}
                )

        # Load splits
        split_dfs = {}
        for entry in split_map:
            try:
                if entry["file"].endswith("_supervised"):
                    folder = config.GROUND_TRUTH_FOLDER
                else:
                    folder = config.SPLITTED_DATA
                df = utils.load_csv(folder_name=folder, file_name=entry["file"])
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                if "Timestamp" in df.columns and "User_Id" in df.columns:
                    df["Day"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
                    split_dfs[entry["label"]] = df
                else:
                    logger.warning(
                        f"Missing expected columns in {entry['file']}.csv: {df.columns}"
                    )
            except Exception as e:
                logger.warning(f"Could not load split {entry['file']}: {e}")

        supervised_labels = [
            "Normal (Supervised)",
            "Intentional (Supervised)",
            "Unintentional (Supervised)",
        ]
        unsupervised_labels = [
            "Normal (Unsupervised)",
            "Intentional (Unsupervised)",
            "Unintentional (Unsupervised)",
        ]

        # Compute majority per user/day for each pipeline
        supervised_majority = _compute_majority_per_user_per_day(
            split_dfs,
            supervised_labels,
            priority_order=["Intentional", "Unintentional", "Normal"],
        )
        unsupervised_majority = _compute_majority_per_user_per_day(
            split_dfs,
            unsupervised_labels,
            priority_order=["Intentional", "Unintentional", "Normal"],
        )

        if supervised_majority.empty and unsupervised_majority.empty:
            logger.warning("No data available to plot the majority-both figure.")
            return

        # Rolling mean smoothing
        def _roll(df_):
            if df_ is None or df_.empty:
                return df_
            out = df_.copy()
            for c in ["Normal", "Intentional", "Unintentional"]:
                out[c] = out[c].rolling(window=7).mean()
            return out

        maj_sup = _roll(supervised_majority)
        maj_uns = _roll(unsupervised_majority)

        # Six-color mapping
        color6 = {
            "Normal (Supervised)": "#2ca02c",
            "Intentional (Supervised)": "#d62728",
            "Unintentional (Supervised)": "#9467bd",
            "Normal (Unsupervised)": "#66c2a5",
            "Intentional (Unsupervised)": "#fc8d62",
            "Unintentional (Unsupervised)": "#8da0cb",
        }

        # ---- Build figure (paired legend order, y in [0,100], explanatory note) ----
        fig = go.Figure()

        # Desired legend/trace order: Normal (Sup) → Normal (Unsup) → Unintentional (Sup) → Unintentional (Unsup) → Intentional (Sup) → Intentional (Unsup)

        # 1) Normal
        if maj_sup is not None and not maj_sup.empty:
            fig.add_trace(
                go.Scatter(
                    x=maj_sup["Day"],
                    y=maj_sup["Normal"],
                    mode="lines",
                    name="Normal — Supervised",
                    line=dict(color=color6["Normal (Supervised)"]),
                )
            )
        if maj_uns is not None and not maj_uns.empty:
            fig.add_trace(
                go.Scatter(
                    x=maj_uns["Day"],
                    y=maj_uns["Normal"],
                    mode="lines",
                    name="Normal — Unsupervised",
                    line=dict(color=color6["Normal (Unsupervised)"], dash="dot"),
                )
            )

        # 2) Unintentional
        if maj_sup is not None and not maj_sup.empty:
            fig.add_trace(
                go.Scatter(
                    x=maj_sup["Day"],
                    y=maj_sup["Unintentional"],
                    mode="lines",
                    name="Unintentional — Supervised",
                    line=dict(color=color6["Unintentional (Supervised)"]),
                )
            )
        if maj_uns is not None and not maj_uns.empty:
            fig.add_trace(
                go.Scatter(
                    x=maj_uns["Day"],
                    y=maj_uns["Unintentional"],
                    mode="lines",
                    name="Unintentional — Unsupervised",
                    line=dict(color=color6["Unintentional (Unsupervised)"], dash="dot"),
                )
            )

        # 3) Intentional
        if maj_sup is not None and not maj_sup.empty:
            fig.add_trace(
                go.Scatter(
                    x=maj_sup["Day"],
                    y=maj_sup["Intentional"],
                    mode="lines",
                    name="Intentional — Supervised",
                    line=dict(color=color6["Intentional (Supervised)"]),
                )
            )
        if maj_uns is not None and not maj_uns.empty:
            fig.add_trace(
                go.Scatter(
                    x=maj_uns["Day"],
                    y=maj_uns["Intentional"],
                    mode="lines",
                    name="Intentional — Unsupervised",
                    line=dict(color=color6["Intentional (Unsupervised)"], dash="dot"),
                )
            )

        fig.update_layout(
            title=dict(
                text="Users per Day by Majority Category (Supervised & Unsupervised — Six Traces)",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Day",
            yaxis_title="Users (7‑day rolling mean; majority per user/day)",
            template="plotly_dark",
            font=dict(size=14),
            legend=dict(
                x=1.02, y=1, xanchor="left", yanchor="top", traceorder="normal"
            ),
            yaxis=dict(range=[0, 100]),
            annotations=[
                dict(
                    text="Each user/day belongs to a single category (majority; tie‑break: Intentional &gt; Unintentional &gt; Normal). Lines show users per pipeline (7‑day rolling mean).",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.18,
                    showarrow=False,
                    align="center",
                    font=dict(size=12),
                )
            ],
        )

        # --- improve line visibility & legend grouping ---
        fig.update_traces(line=dict(width=2.2))
        fig.update_layout(font=dict(size=16), margin=dict(l=80, r=260, t=70, b=70))

        # Tighter y ticks and a slight opacity tweak for dashed (unsupervised) traces
        fig.update_yaxes(dtick=10)
        for tr in fig.data:
            try:
                if (
                    getattr(tr, "line", None)
                    and getattr(tr.line, "dash", None) == "dot"
                ):
                    tr.opacity = 0.9
            except Exception:
                pass

        # Save only this figure
        fig.write_html(str(config.PLOTS / "dataset_inclusion_user_majority_both.html"))
        fig.write_image(
            str(config.PLOTS / "dataset_inclusion_user_majority_both.png"),
            width=1600,
            height=1000,
            scale=1,
        )
        logger.info(
            "Saved only the majority-both dataset inclusion figure (HTML & PNG)."
        )

    except Exception as e:
        logger.error(f"Failed to plot majority-both dataset inclusion figure: {e}")
        raise


# ========== Plot Outlier Numbers ==========
@system.timer_decorator
def plot_outlier_numbers():
    """
    Generate a **simplified, aggregated** visualization of outlier removals **per method** (no per‑chunk traces).

    What it does
    ---------------------------------
        - Loads 'outlier_numbers.csv' from 'processed_data'.
        - Aggregates across *all* columns and chunks to compute the **total outliers removed per method**.
        - Creates a **horizontal bar chart** (one bar per method) with value labels (count and % of total).
        - Saves:
            • plots/outlier_methods_comparison.html
            • plots/outlier_methods_comparison.png
            • processed_data/outlier_summary_by_method.csv  (helper table for the thesis appendix)

    Rationale
    ---------------------------------
        - Avoids cluttered legends (no per‑chunk layering).
        - Clear mapping: **color = method** only.
        - Easier to read in static PNGs (no hover needed).
    """
    try:
        # 1) Load metrics
        df_raw = utils.load_csv(
            folder_name=config.OUTLIER_DATA, file_name="outlier_numbers"
        )
        df = pd.DataFrame(df_raw) if not isinstance(df_raw, pd.DataFrame) else df_raw

        if df.empty:
            logger.warning("No data in outlier methods metrics file.")
            return

        # Ensure expected columns exist
        expected = {"method", "column", "chunk_id", "outliers_removed"}
        missing = expected - set(df.columns)
        if missing:
            logger.error(f"Missing columns in outlier_numbers.csv: {missing}")
            return

        # 2) Aggregate by method (sum over columns & chunks)
        agg = (
            df.groupby("method", as_index=False)["outliers_removed"]
            .sum()
            .rename(columns={"outliers_removed": "total_removed"})
        )

        if agg.empty:
            logger.warning("Aggregated outlier summary is empty.")
            return

        # Sort methods by total removed (descending) for readability
        agg = agg.sort_values(
            "total_removed", ascending=True
        )  # ascending for horizontal bars (bottom=largest after flip)
        grand_total = float(agg["total_removed"].sum())
        agg["percent"] = agg["total_removed"].apply(
            lambda v: 0.0 if grand_total == 0 else (v / grand_total) * 100.0
        )

        # 3) Persist a small summary CSV for reference
        summary_path = config.OUTLIER_DATA / "outlier_summary_by_method.csv"
        try:
            agg_sorted_desc = agg.sort_values("total_removed", ascending=False)
            agg_sorted_desc.to_csv(summary_path, index=False)
        except Exception as e_csv:
            logger.warning(f"Could not save outlier summary CSV: {e_csv}")

        # 4) Build a clean horizontal bar chart (one bar per method)
        # Robust color list (long palette)
        palette = (
            px.colors.qualitative.Dark24
            + px.colors.qualitative.Safe
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Set2
            + px.colors.qualitative.Pastel
        )
        colors = [palette[i % len(palette)] for i in range(len(agg))]

        text_labels = [
            f"{int(v):,}  ({p:.1f}%)"
            for v, p in zip(agg["total_removed"], agg["percent"])
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=agg["total_removed"],
                    y=agg["method"],
                    orientation="h",
                    text=text_labels,
                    textposition="auto",
                    marker=dict(color=colors, line=dict(width=0.5, color="black")),
                    hovertemplate="Method: %{y}<br>Total removed: %{x:,} (%{customdata:.1f}%)<extra></extra>",
                    customdata=agg["percent"],
                )
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            font=dict(size=14),
            title=dict(
                text="Total Outliers Removed by Method (Aggregated across columns & chunks)",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Outliers removed (count)",
            yaxis_title="Method",
            showlegend=False,
            margin=dict(l=140, r=40, t=80, b=60),
        )

        # --- visual polish ---
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(margin=dict(l=160, r=40, t=80, b=60), font=dict(size=16))
        fig.update_xaxes(
            tickformat="~s", zeroline=True, zerolinewidth=1, zerolinecolor="#666"
        )

        # 5) Save outputs (overwrite the previous filenames so the thesis picks them up)
        fig.write_html(str(config.PLOTS / "outlier_methods_comparison.html"))
        fig.write_image(
            str(config.PLOTS / "outlier_methods_comparison.png"),
            width=1600,
            height=1000,
            scale=1,
        )

        logger.info(
            "Outlier methods (aggregated by method) visualization saved successfully."
        )
    except Exception as e:
        logger.error(f"Failed to plot outlier numbers (aggregated): {str(e)}")
        raise


# ========== Plot Dataset Inclusion ==========
@system.timer_decorator
def plot_dataset_inclusion():
    """
    Function which generates a scatter plot to compare the number of users in different datasets.

    Features
    ---------------------------------
        - Loads user data from 'Normal_Data.csv', 'Intentional_Data.csv', and 'Unintentional_Data.csv'.
        - Aggregates the number of unique users in each dataset per day.
        - Creates a scatter plot to visualize the number of users in each dataset.
        - Assigns distinct colors and markers to different datasets for better visualization.
        - Saves the generated plot as 'dataset_inclusion.png'.

    Raises Exception
    ---------------------------------
        - If the required CSV files are missing or contain invalid data.
        - If the plot generation fails due to errors in data processing or visualization.
    """
    try:  # Try-except block to catch and handle exceptions
        split_types = [
            "supervised",
            "unsupervised",
        ]  # Define the split types to consider
        split_names = [
            "Normal",
            "Intentional",
            "Unintentional",
        ]  # Define the split names to consider
        split_map = []
        for stype in split_types:
            for sname in split_names:
                if stype == "unsupervised":
                    fname = f"{sname}_Data"
                else:
                    fname = f"{sname}_Data_supervised"
                label = f"{sname} ({stype.capitalize()})"
                split_map.append(
                    {"stype": stype, "sname": sname, "file": fname, "label": label}
                )

        split_dfs = {}
        for entry in split_map:
            try:
                if entry["file"].endswith("_supervised"):
                    folder = config.GROUND_TRUTH_FOLDER
                else:
                    folder = config.SPLITTED_DATA
                df = utils.load_csv(folder_name=folder, file_name=entry["file"])
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                if "Timestamp" in df.columns and "User_Id" in df.columns:
                    # Keep Day as timezone-naive datetime (not Period) for Plotly serialization
                    df["Day"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
                    split_dfs[entry["label"]] = df
                else:
                    logger.warning(
                        f"Missing columns in {entry['file']}.csv: {df.columns}"
                    )
            except Exception as e:
                logger.warning(f"Could not load split {entry['file']}: {e}")

        agg_per_day = {}
        for label, df in split_dfs.items():
            agg = df.groupby("Day")["User_Id"].nunique().reset_index(name=label)
            agg_per_day[label] = agg

        all_days = None
        for agg in agg_per_day.values():
            if all_days is None:
                all_days = agg[["Day"]].copy()
            else:
                all_days = pd.merge(all_days, agg[["Day"]], on="Day", how="outer")
        merged = None
        for label, agg in agg_per_day.items():
            if merged is None:
                merged = agg
            else:
                merged = pd.merge(merged, agg, on="Day", how="outer")
        if merged is None or merged.empty:
            logger.warning("No splits found to plot for dataset inclusion comparison.")
            return
        merged = merged.fillna(0)
        merged["Day"] = pd.to_datetime(merged["Day"].astype(str))
        merged = merged.sort_values("Day")
        for label in agg_per_day.keys():
            merged[label] = merged[label].rolling(window=7).mean()

        # Try to load the dataset CSV to get the user count
        try:
            dataset_df = utils.load_csv(config.DATA_FOLDER, config.DATASET_FILE_NAME)
            if not isinstance(dataset_df, pd.DataFrame):
                dataset_df = pd.DataFrame(dataset_df)
            dataset_df["User_Id"] = (
                dataset_df["User_Id"].astype(str).str.replace("User_", "").astype(int)
            )
            users = dataset_df["User_Id"].nunique()
        except Exception as e:
            logger.warning(f"Could not load Dataset.csv for user count: {e}")
            users = 0

        # Define styles and plot labels for all splits
        split_styles = {
            "Normal (Unsupervised)": {"color": "#66c2a5", "symbol": "circle"},
            "Intentional (Unsupervised)": {"color": "#fc8d62", "symbol": "triangle-up"},
            "Unintentional (Unsupervised)": {"color": "#8da0cb", "symbol": "square"},
            "Normal (Supervised)": {"color": "#2ca02c", "symbol": "circle-open"},
            "Intentional (Supervised)": {
                "color": "#d62728",
                "symbol": "triangle-up-open",
            },
            "Unintentional (Supervised)": {"color": "#9467bd", "symbol": "square-open"},
        }
        all_labels = [
            "Normal (Unsupervised)",
            "Intentional (Unsupervised)",
            "Unintentional (Unsupervised)",
            "Normal (Supervised)",
            "Intentional (Supervised)",
            "Unintentional (Supervised)",
        ]
        supervised_labels = [
            "Normal (Supervised)",
            "Intentional (Supervised)",
            "Unintentional (Supervised)",
        ]
        unsupervised_labels = [
            "Normal (Unsupervised)",
            "Intentional (Unsupervised)",
            "Unintentional (Unsupervised)",
        ]

        # --- Plot 1: Both supervised and unsupervised ---
        fig = go.Figure()
        for label in all_labels:
            if label in merged.columns:
                style = split_styles.get(label, {"color": None, "symbol": "circle"})
                fig.add_trace(
                    go.Scatter(
                        x=merged["Day"],
                        y=merged[label],
                        mode="markers+lines",
                        name=label,
                        marker=dict(color=style["color"], symbol=style["symbol"]),
                    )
                )
        fig.update_layout(
            title=dict(
                text="Aggregated Users per Dataset Split and Day (Supervised & Unsupervised)",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Day",
            yaxis_title="Number of Users (7-day rolling mean)",
            template="plotly_dark",
            legend_title=f"Splits (Max Users: {users})",
            font=dict(size=14),
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        )
        fig.write_html(
            str(config.PLOTS / "dataset_inclusion_aggregated_comparison.html")
        )
        fig.write_image(
            str(config.PLOTS / "dataset_inclusion_aggregated_comparison.png"),
            width=1600,
            height=1000,
            scale=1,
        )
        logger.info(
            "Aggregated dataset inclusion comparison (supervised and unsupervised) visualization saved."
        )

        # --- Plot 2: Supervised only ---
        # Filtering columns and labels for supervised splits
        # --- Supervised-only plot ---
        fig_supervised = go.Figure()
        # Only include supervised splits
        # --- Supervised split separation implemented here ---
        for label in supervised_labels:
            if label in merged.columns:
                style = split_styles.get(label, {"color": None, "symbol": "circle"})
                fig_supervised.add_trace(
                    go.Scatter(
                        x=merged["Day"],
                        y=merged[label],
                        mode="markers+lines",
                        name=label,
                        marker=dict(color=style["color"], symbol=style["symbol"]),
                    )
                )
        fig_supervised.update_layout(
            title=dict(
                text="Aggregated Users per Dataset Split and Day (Supervised Only)",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Day",
            yaxis_title="Number of Users (7-day rolling mean)",
            template="plotly_dark",
            legend_title=f"Supervised Splits (Max Users: {users})",
            font=dict(size=14),
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        )
        fig_supervised.write_html(
            str(config.PLOTS / "dataset_inclusion_supervised_comparison.html")
        )
        fig_supervised.write_image(
            str(config.PLOTS / "dataset_inclusion_supervised_comparison.png"),
            width=1600,
            height=1000,
            scale=1,
        )
        logger.info("Supervised-only dataset inclusion comparison visualization saved.")

        # --- Plot 3: Unsupervised only ---
        # Filtering columns and labels for unsupervised splits
        # --- Unsupervised split separation implemented here ---
        fig_unsupervised = go.Figure()
        for label in unsupervised_labels:
            if label in merged.columns:
                style = split_styles.get(label, {"color": None, "symbol": "circle"})
                fig_unsupervised.add_trace(
                    go.Scatter(
                        x=merged["Day"],
                        y=merged[label],
                        mode="markers+lines",
                        name=label,
                        marker=dict(color=style["color"], symbol=style["symbol"]),
                    )
                )
        fig_unsupervised.update_layout(
            title=dict(
                text="Aggregated Users per Dataset Split and Day (Unsupervised Only)",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Day",
            yaxis_title="Number of Users (7-day rolling mean)",
            template="plotly_dark",
            legend_title=f"Unsupervised Splits (Max Users: {users})",
            font=dict(size=14),
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        )
        fig_unsupervised.write_html(
            str(config.PLOTS / "dataset_inclusion_unsupervised_comparison.html")
        )
        fig_unsupervised.write_image(
            str(config.PLOTS / "dataset_inclusion_unsupervised_comparison.png"),
            width=1600,
            height=1000,
            scale=1,
        )
        logger.info(
            "Unsupervised-only dataset inclusion comparison visualization saved."
        )

        # --- New Plot 4: Majority category per user/day (Supervised & Unsupervised) ---
        # Supervised
        supervised_majority = _compute_majority_per_user_per_day(
            split_dfs,
            supervised_labels,
            priority_order=["Intentional", "Unintentional", "Normal"],
        )
        if not supervised_majority.empty:
            # Apply 7-day rolling mean on counts
            maj_sup = supervised_majority.copy()
            for c in ["Normal", "Intentional", "Unintentional"]:
                maj_sup[c] = maj_sup[c].rolling(window=7).mean()
            fig_maj_sup = go.Figure()
            fig_maj_sup.add_trace(
                go.Scatter(
                    x=maj_sup["Day"],
                    y=maj_sup["Normal"],
                    mode="lines",
                    name="Normal",
                    stackgroup="one",
                )
            )
            fig_maj_sup.add_trace(
                go.Scatter(
                    x=maj_sup["Day"],
                    y=maj_sup["Unintentional"],
                    mode="lines",
                    name="Unintentional",
                    stackgroup="one",
                )
            )
            fig_maj_sup.add_trace(
                go.Scatter(
                    x=maj_sup["Day"],
                    y=maj_sup["Intentional"],
                    mode="lines",
                    name="Intentional",
                    stackgroup="one",
                )
            )
            fig_maj_sup.update_layout(
                title=dict(
                    text="Users per Day by Majority Category (Supervised)",
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title="Day",
                yaxis_title="Users (7-day rolling mean)",
                template="plotly_dark",
                font=dict(size=14),
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
                annotations=[
                    dict(
                        text="Each user/day assigned to a single category (tie-break: Intentional > Unintentional > Normal). Sums equal total active users per day.",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.18,
                        showarrow=False,
                        align="center",
                        font=dict(size=12),
                    )
                ],
            )
            fig_maj_sup.write_html(
                str(config.PLOTS / "dataset_inclusion_user_majority_supervised.html")
            )
            fig_maj_sup.write_image(
                str(config.PLOTS / "dataset_inclusion_user_majority_supervised.png"),
                width=1600,
                height=1000,
                scale=1,
            )

        # Unsupervised
        unsupervised_majority = _compute_majority_per_user_per_day(
            split_dfs,
            unsupervised_labels,
            priority_order=["Intentional", "Unintentional", "Normal"],
        )
        if not unsupervised_majority.empty:
            maj_uns = unsupervised_majority.copy()
            for c in ["Normal", "Intentional", "Unintentional"]:
                maj_uns[c] = maj_uns[c].rolling(window=7).mean()
            fig_maj_uns = go.Figure()
            fig_maj_uns.add_trace(
                go.Scatter(
                    x=maj_uns["Day"],
                    y=maj_uns["Normal"],
                    mode="lines",
                    name="Normal",
                    stackgroup="one",
                )
            )
            fig_maj_uns.add_trace(
                go.Scatter(
                    x=maj_uns["Day"],
                    y=maj_uns["Unintentional"],
                    mode="lines",
                    name="Unintentional",
                    stackgroup="one",
                )
            )
            fig_maj_uns.add_trace(
                go.Scatter(
                    x=maj_uns["Day"],
                    y=maj_uns["Intentional"],
                    mode="lines",
                    name="Intentional",
                    stackgroup="one",
                )
            )
            fig_maj_uns.update_layout(
                title=dict(
                    text="Users per Day by Majority Category (Unsupervised)",
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title="Day",
                yaxis_title="Users (7-day rolling mean)",
                template="plotly_dark",
                font=dict(size=14),
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
                annotations=[
                    dict(
                        text="Each user/day assigned to a single category (tie-break: Intentional > Unintentional > Normal). Sums equal total active users per day.",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.18,
                        showarrow=False,
                        align="center",
                        font=dict(size=12),
                    )
                ],
            )
            fig_maj_uns.write_html(
                str(config.PLOTS / "dataset_inclusion_user_majority_unsupervised.html")
            )
            fig_maj_uns.write_image(
                str(config.PLOTS / "dataset_inclusion_user_majority_unsupervised.png"),
                width=1600,
                height=1000,
                scale=1,
            )

        # --- New Plot 4b: Users per Day by Majority Category (Supervised vs Unsupervised in one figure) ---
        # Show both pipelines simultaneously (six traces). No cross-pipeline stacking to avoid misleading totals.
        if not supervised_majority.empty or not unsupervised_majority.empty:
            # Safe copies with rolling mean on counts
            maj_sup = (
                supervised_majority.copy()
                if not supervised_majority.empty
                else pd.DataFrame(
                    columns=["Day", "Normal", "Intentional", "Unintentional"]
                )
            )
            maj_uns = (
                unsupervised_majority.copy()
                if not unsupervised_majority.empty
                else pd.DataFrame(
                    columns=["Day", "Normal", "Intentional", "Unintentional"]
                )
            )
            for df_ in (maj_sup, maj_uns):
                if not df_.empty:
                    for c in ["Normal", "Intentional", "Unintentional"]:
                        df_[c] = df_[c].rolling(window=7).mean()

            fig_maj_both = go.Figure()
            # Consistent six-color mapping across plots
            color6 = {
                "Normal (Supervised)": "#2ca02c",
                "Intentional (Supervised)": "#d62728",
                "Unintentional (Supervised)": "#9467bd",
                "Normal (Unsupervised)": "#66c2a5",
                "Intentional (Unsupervised)": "#fc8d62",
                "Unintentional (Unsupervised)": "#8da0cb",
            }

            # Supervised traces
            if not maj_sup.empty:
                fig_maj_both.add_trace(
                    go.Scatter(
                        x=maj_sup["Day"],
                        y=maj_sup["Normal"],
                        mode="lines",
                        name="Normal — Supervised",
                        line=dict(color=color6["Normal (Supervised)"]),
                    )
                )
                fig_maj_both.add_trace(
                    go.Scatter(
                        x=maj_sup["Day"],
                        y=maj_sup["Unintentional"],
                        mode="lines",
                        name="Unintentional — Supervised",
                        line=dict(color=color6["Unintentional (Supervised)"]),
                    )
                )
                fig_maj_both.add_trace(
                    go.Scatter(
                        x=maj_sup["Day"],
                        y=maj_sup["Intentional"],
                        mode="lines",
                        name="Intentional — Supervised",
                        line=dict(color=color6["Intentional (Supervised)"]),
                    )
                )

            # Unsupervised traces (dashed to distinguish pipelines)
            if not maj_uns.empty:
                fig_maj_both.add_trace(
                    go.Scatter(
                        x=maj_uns["Day"],
                        y=maj_uns["Normal"],
                        mode="lines",
                        name="Normal — Unsupervised",
                        line=dict(color=color6["Normal (Unsupervised)"], dash="dot"),
                    )
                )
                fig_maj_both.add_trace(
                    go.Scatter(
                        x=maj_uns["Day"],
                        y=maj_uns["Unintentional"],
                        mode="lines",
                        name="Unintentional — Unsupervised",
                        line=dict(
                            color=color6["Unintentional (Unsupervised)"], dash="dot"
                        ),
                    )
                )
                fig_maj_both.add_trace(
                    go.Scatter(
                        x=maj_uns["Day"],
                        y=maj_uns["Intentional"],
                        mode="lines",
                        name="Intentional — Unsupervised",
                        line=dict(
                            color=color6["Intentional (Unsupervised)"], dash="dot"
                        ),
                    )
                )

            fig_maj_both.update_layout(
                title=dict(
                    text="Users per Day by Majority Category (Supervised & Unsupervised — Six Traces)",
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title="Day",
                yaxis_title="Users (7-day rolling mean; majority per user/day)",
                template="plotly_dark",
                font=dict(size=14),
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
            )

            # --- improve line visibility & legend grouping ---
            fig_maj_both.update_traces(line=dict(width=2.5))
            fig_maj_both.update_layout(legend=dict(tracegroupgap=12))

            # Tighter y ticks and a slight opacity tweak for dashed (unsupervised) traces
            fig_maj_both.update_yaxes(dtick=10)
            for tr in fig_maj_both.data:
                try:
                    if (
                        getattr(tr, "line", None)
                        and getattr(tr.line, "dash", None) == "dot"
                    ):
                        tr.opacity = 0.9
                except Exception:
                    pass

            fig_maj_both.write_html(
                str(config.PLOTS / "dataset_inclusion_user_majority_both.html")
            )
            fig_maj_both.write_image(
                str(config.PLOTS / "dataset_inclusion_user_majority_both.png"),
                width=1600,
                height=1000,
                scale=1,
            )
            logger.info(
                "Supervised & Unsupervised majority users per day (six traces) visualization saved."
            )

        # --- New Plot 5: Submission composition per day (percent, sums to 100%) ---
        # Supervised
        supervised_share = _compute_submission_share(split_dfs, supervised_labels)
        if not supervised_share.empty:
            share_sup = supervised_share.copy()
            for c in ["Normal", "Intentional", "Unintentional"]:
                share_sup[c] = share_sup[c].rolling(window=7).mean()
            fig_share_sup = go.Figure()
            fig_share_sup.add_trace(
                go.Scatter(
                    x=share_sup["Day"],
                    y=share_sup["Normal"],
                    mode="lines",
                    name="Normal",
                    stackgroup="one",
                )
            )
            fig_share_sup.add_trace(
                go.Scatter(
                    x=share_sup["Day"],
                    y=share_sup["Unintentional"],
                    mode="lines",
                    name="Unintentional",
                    stackgroup="one",
                )
            )
            fig_share_sup.add_trace(
                go.Scatter(
                    x=share_sup["Day"],
                    y=share_sup["Intentional"],
                    mode="lines",
                    name="Intentional",
                    stackgroup="one",
                )
            )
            fig_share_sup.update_layout(
                title=dict(
                    text="Submission Composition per Day (Supervised)",
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title="Day",
                yaxis_title="Share of Submissions (%) — 7-day rolling mean",
                template="plotly_dark",
                font=dict(size=14),
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
                yaxis=dict(range=[0, 100]),
                annotations=[
                    dict(
                        text="Shares computed over submissions; sums to 100% per day.",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.18,
                        showarrow=False,
                        align="center",
                        font=dict(size=12),
                    )
                ],
            )
            fig_share_sup.write_html(
                str(config.PLOTS / "dataset_inclusion_submission_share_supervised.html")
            )
            fig_share_sup.write_image(
                str(config.PLOTS / "dataset_inclusion_submission_share_supervised.png"),
                width=1600,
                height=1000,
                scale=1,
            )

        # Unsupervised
        unsupervised_share = _compute_submission_share(split_dfs, unsupervised_labels)
        if not unsupervised_share.empty:
            share_uns = unsupervised_share.copy()
            for c in ["Normal", "Intentional", "Unintentional"]:
                share_uns[c] = share_uns[c].rolling(window=7).mean()
            fig_share_uns = go.Figure()
            fig_share_uns.add_trace(
                go.Scatter(
                    x=share_uns["Day"],
                    y=share_uns["Normal"],
                    mode="lines",
                    name="Normal",
                    stackgroup="one",
                )
            )
            fig_share_uns.add_trace(
                go.Scatter(
                    x=share_uns["Day"],
                    y=share_uns["Unintentional"],
                    mode="lines",
                    name="Unintentional",
                    stackgroup="one",
                )
            )
            fig_share_uns.add_trace(
                go.Scatter(
                    x=share_uns["Day"],
                    y=share_uns["Intentional"],
                    mode="lines",
                    name="Intentional",
                    stackgroup="one",
                )
            )
            fig_share_uns.update_layout(
                title=dict(
                    text="Submission Composition per Day (Unsupervised)",
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title="Day",
                yaxis_title="Share of Submissions (%) — 7-day rolling mean",
                template="plotly_dark",
                font=dict(size=14),
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
                yaxis=dict(range=[0, 100]),
                annotations=[
                    dict(
                        text="Shares computed over submissions; sums to 100% per day.",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.18,
                        showarrow=False,
                        align="center",
                        font=dict(size=12),
                    )
                ],
            )
            fig_share_uns.write_html(
                str(
                    config.PLOTS
                    / "dataset_inclusion_submission_share_unsupervised.html"
                )
            )
            fig_share_uns.write_image(
                str(
                    config.PLOTS / "dataset_inclusion_submission_share_unsupervised.png"
                ),
                width=1600,
                height=1000,
                scale=1,
            )
            logger.info(
                "Unsupervised-only submission composition per day visualization saved."
            )

        # --- New Plot 5b: Submission composition per day (Supervised & Unsupervised — Six Traces) ---
        # Show both pipelines simultaneously as percentages (no stacking across pipelines).
        if (
            "share_sup" in locals()
            and isinstance(share_sup, pd.DataFrame)
            and not share_sup.empty
        ) or (
            "share_uns" in locals()
            and isinstance(share_uns, pd.DataFrame)
            and not share_uns.empty
        ):
            fig_share_both = go.Figure()

            # Reuse the six-color mapping
            color6 = {
                "Normal (Supervised)": "#2ca02c",
                "Intentional (Supervised)": "#d62728",
                "Unintentional (Supervised)": "#9467bd",
                "Normal (Unsupervised)": "#66c2a5",
                "Intentional (Unsupervised)": "#fc8d62",
                "Unintentional (Unsupervised)": "#8da0cb",
            }

            # Ensure 7-day smoothing (idempotent)
            def _rolled(df_):
                if df_ is None or df_.empty:
                    return df_
                df2 = df_.copy()
                for c in ["Normal", "Intentional", "Unintentional"]:
                    df2[c] = df2[c].rolling(window=7).mean()
                return df2

            share_sup2 = _rolled(share_sup) if "share_sup" in locals() else None
            share_uns2 = _rolled(share_uns) if "share_uns" in locals() else None

            if share_sup2 is not None and not share_sup2.empty:
                fig_share_both.add_trace(
                    go.Scatter(
                        x=share_sup2["Day"],
                        y=share_sup2["Normal"],
                        mode="lines",
                        name="Normal — Supervised",
                        line=dict(color=color6["Normal (Supervised)"]),
                    )
                )
                fig_share_both.add_trace(
                    go.Scatter(
                        x=share_sup2["Day"],
                        y=share_sup2["Unintentional"],
                        mode="lines",
                        name="Unintentional — Supervised",
                        line=dict(color=color6["Unintentional (Supervised)"]),
                    )
                )
                fig_share_both.add_trace(
                    go.Scatter(
                        x=share_sup2["Day"],
                        y=share_sup2["Intentional"],
                        mode="lines",
                        name="Intentional — Supervised",
                        line=dict(color=color6["Intentional (Supervised)"]),
                    )
                )
            if share_uns2 is not None and not share_uns2.empty:
                fig_share_both.add_trace(
                    go.Scatter(
                        x=share_uns2["Day"],
                        y=share_uns2["Normal"],
                        mode="lines",
                        name="Normal — Unsupervised",
                        line=dict(color=color6["Normal (Unsupervised)"], dash="dot"),
                    )
                )
                fig_share_both.add_trace(
                    go.Scatter(
                        x=share_uns2["Day"],
                        y=share_uns2["Unintentional"],
                        mode="lines",
                        name="Unintentional — Unsupervised",
                        line=dict(
                            color=color6["Unintentional (Unsupervised)"], dash="dot"
                        ),
                    )
                )
                fig_share_both.add_trace(
                    go.Scatter(
                        x=share_uns2["Day"],
                        y=share_uns2["Intentional"],
                        mode="lines",
                        name="Intentional — Unsupervised",
                        line=dict(
                            color=color6["Intentional (Unsupervised)"], dash="dot"
                        ),
                    )
                )

            fig_share_both.update_layout(
                title=dict(
                    text="Submission Composition per Day (Supervised & Unsupervised — Six Traces)",
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title="Day",
                yaxis_title="Share of Submissions (%) — 7-day rolling mean",
                template="plotly_dark",
                font=dict(size=14),
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
                yaxis=dict(range=[0, 100]),
            )
            fig_share_both.write_html(
                str(config.PLOTS / "dataset_inclusion_submission_share_both.html")
            )
            fig_share_both.write_image(
                str(config.PLOTS / "dataset_inclusion_submission_share_both.png"),
                width=1600,
                height=1000,
                scale=1,
            )
            logger.info(
                "Submission composition per day (Supervised & Unsupervised — six traces) saved."
            )

    except Exception as e:  # Catch exceptions and handle them gracefully
        logger.error(f"Failed to plot dataset inclusion: {str(e)}")
        raise


#  ========== Plot Tpot Model Comparison ==========
@system.timer_decorator
def plot_model_comparison():
    """
    Function which generates a bar plot to compare model scores for both supervised and unsupervised processed results, for TPOT and FLAML.

    Features
    ---------------------------------
        - Loads user reputation model results from 'model_results.csv'.
        - Separately loads and identifies processed supervised and unsupervised results for both TPOT and FLAML.
        - Plots two distinct bars for each (TPOT-supervised, TPOT-unsupervised, FLAML-supervised, FLAML-unsupervised).
        - Adds "run_type" (Supervised/Unsupervised) as a hue/color for the bars.
        - Updates the legend and plot title for clarity.
        - Avoids duplicate bars and ensures correct pipeline_file mapping.
        - Logs which run types are being compared.

    Raises Exception
    ---------------------------------
        - If the results file is missing or contains invalid data.
        - If the plot generation fails due to errors in data processing or visualization.
    """
    try:  # Try-except block to catch and handle exceptions
        if (
            not config.EXPORTED_MODELS.exists()
        ):  # Check if the exported models directory exists
            logger.warning(
                "No TPOT model results found to compare."
            )  # Log warning if directory does not exist
            return  # Exit the function if no models are found

        df = utils.load_csv(
            folder_name=config.EXPORTED_MODELS, file_name=config.MODEL_RESULTS
        )  # Load the model results CSV file from the exported models directory
        if not isinstance(df, pd.DataFrame):  # Ensure df is a Pandas DataFrame
            df = pd.DataFrame(df)  # Convert to DataFrame if needed

        expected_cols = {
            "pipeline_file",
            "score",
            "prefix",
        }  # Define the expected columns for the model results
        if not expected_cols.issubset(df.columns):
            logger.warning(
                "Required columns missing in model results; nothing to plot."
            )  # Log warning if required columns are missing
            return  # Exit the function if required columns are missing

        df["pipeline_file"] = (
            df["pipeline_file"].astype(str).fillna("").str.strip()
        )  # Ensure pipeline_file is a string, fill NaN with empty string, and strip whitespace
        models_dir = Path(
            config.EXPORTED_MODELS
        )  # Get the path to the exported models directory

        # Only consider .py files for results, ignoring .pkl or others
        df["pipeline_file"] = df["pipeline_file"].astype(str).fillna("").str.strip()
        df = df[df["pipeline_file"].str.endswith(".py", na=False)]

        df["score"] = pd.to_numeric(
            df["score"], errors="coerce"
        )  # Convert score to numeric, coercing errors to NaN
        df = df.dropna(subset=["score"])  # Drop rows where score is NaN
        df = df[df["pipeline_file"] != ""]  # Remove empty pipeline_file entries
        df = df.drop_duplicates(
            subset=["pipeline_file", "prefix", "score"]
        )  # Drop duplicates based on pipeline_file, prefix, and score

        # Map only the final processed and raw supervised/unsupervised model .py files
        mapping = [
            {
                "prefix": "tpot",
                "run_type": "Supervised",
                "pipeline_file_py": "final_processed_supervised_tpot.py",
                "label": "TPOT",
            },  # TPOT supervised
            {
                "prefix": "tpot",
                "run_type": "Unsupervised",
                "pipeline_file_py": "final_processed_unsupervised_tpot.py",
                "label": "TPOT",
            },  # TPOT unsupervised
            {
                "prefix": "tpot",
                "run_type": "Raw",
                "pipeline_file_py": "final_raw_tpot.py",
                "label": "TPOT",
            },  # TPOT raw
            {
                "prefix": "flaml",
                "run_type": "Raw",
                "pipeline_file_py": "final_raw_flaml.py",
                "label": "FLAML",
            },  # FLAML raw
            {
                "prefix": "flaml",
                "run_type": "Supervised",
                "pipeline_file_py": "final_processed_supervised_flaml.py",
                "label": "FLAML",
            },  # FLAML supervised
            {
                "prefix": "flaml",
                "run_type": "Unsupervised",
                "pipeline_file_py": "final_processed_unsupervised_flaml.py",
                "label": "FLAML",
            },  # FLAML unsupervised
        ]
        plot_rows = []  # List to hold rows for the plot
        for m in mapping:  # Loop through each mapping to filter the DataFrame
            filtered = df[
                (df["prefix"].str.lower() == m["prefix"])
                & (
                    df["pipeline_file"].str.lower().apply(lambda x: Path(x).name)
                    == m["pipeline_file_py"].lower()
                )
            ]  # Filter the DataFrame based on prefix and pipeline_file
            if filtered.empty:  # Check if the filtered DataFrame is empty
                logger.warning(
                    f"No results found for {m['label']} ({m['run_type']}) processed pipeline .py: {m['pipeline_file_py']}"
                )  # Log a warning if no results are found for the current mapping
                continue  # Skip to the next mapping if no results are found
            filtered = filtered.sort_values(
                "score", ascending=False
            )  # Sort the filtered DataFrame by score in descending order
            row = filtered.iloc[0]  # Get the first row of the filtered DataFrame
            plot_rows.append(
                {
                    "pipeline_file": row["pipeline_file"],
                    "score": row["score"],
                    "engine": m["label"],
                    "run_type": m["run_type"],
                }
            )  # Append the row to the plot_rows list with relevant details
        if not plot_rows:  # Check if plot_rows is empty
            logger.warning(
                "No processed supervised/unsupervised results found for TPOT or FLAML."
            )  # Log a warning if no processed results are found
            return  # Exit the function if no processed results are found

        plot_df = pd.DataFrame(plot_rows)  # Convert plot_rows to a DataFrame
        logger.info(
            "Comparing processed supervised and unsupervised results for TPOT and FLAML:\n"
            + plot_df[["engine", "run_type", "pipeline_file", "score"]].to_string(
                index=False
            )
        )  # Log the comparison of processed supervised and unsupervised results for TPOT and FLAML

        plot_df["engine"] = pd.Categorical(
            plot_df["engine"], categories=["TPOT", "FLAML"], ordered=True
        )  # Ensure 'engine' is a categorical variable with TPOT and FLAML as categories
        plot_df["run_type"] = pd.Categorical(
            plot_df["run_type"],
            categories=["Raw", "Supervised", "Unsupervised"],
            ordered=True,
        )  # Ensure 'run_type' is a categorical variable with Raw, Supervised and Unsupervised as categories

        plot_df["engine_run_type"] = (
            plot_df["engine"].astype(str) + " (" + plot_df["run_type"].astype(str) + ")"
        )  # Create a new column combining 'engine' and 'run_type' for better labeling in the plot

        # Extra safety: print and clean plot_df before plotting
        print(plot_df)
        # Ensure no NaN or infinite values
        plot_df = plot_df.replace([float("inf"), float("-inf")], float("nan")).dropna(
            subset=["score", "engine_run_type"]
        )

        # --- build bar chart manually with go.Bar to avoid plotly transform errors ---
        fig = go.Figure()
        colors = {"Raw": "#8da0cb", "Supervised": "#66c2a5", "Unsupervised": "#fc8d62"}

        # enforce consistent ordering of bars
        order = [
            "TPOT (Raw)",
            "TPOT (Supervised)",
            "TPOT (Unsupervised)",
            "FLAML (Raw)",
            "FLAML (Supervised)",
            "FLAML (Unsupervised)",
        ]
        plot_df = plot_df.set_index("engine_run_type").loc[order].reset_index()

        for _, row in plot_df.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[row["engine_run_type"]],
                    y=[row["score"]],
                    name=row["run_type"],
                    marker_color=colors.get(row["run_type"], "#cccccc"),
                    text=f"{row['score']:.3f}",
                    textposition="auto",
                )
            )

        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            font=dict(size=14),
            xaxis_tickangle=30,
            bargap=0.18,
            xaxis_title="Engine (Run Type)",
            yaxis_title="Model Score",
            showlegend=True,
            legend_title="Run Type",
            title=dict(
                text=(
                    "AutoML Model Scores: Raw vs Supervised vs Unsupervised<br>"
                    "<span style='font-size:12px;'>Processed & Raw Results — TPOT vs FLAML</span>"
                ),
                x=0.5,
                xanchor="center",
            ),
        )

        # --- bar value labels formatting ---
        fig.update_traces(texttemplate="%{y:.003f}", textfont_size=12)

        for tr in fig.data:
            tr.textfont.update(size=12)
        fig.update_layout(font=dict(size=16), margin=dict(l=80, r=80, t=80, b=80))
        fig.update_yaxes(tickformat=".003f")

        # Headroom so value labels never touch the top border
        fig.update_yaxes(range=[0, float(plot_df["score"].max()) * 1.15])

        fig.write_html(
            str(config.PLOTS / "model_comparison.html")
        )  # Save the plot as HTML
        fig.write_image(
            str(config.PLOTS / "model_comparison.png"), width=1600, height=1000, scale=1
        )  # Save the plot as PNG

        logger.info(
            "Model comparison visualization (processed, supervised vs unsupervised) saved successfully."
        )  # Log success message

    except Exception as e:  # Catch exceptions and handle them gracefully
        logger.error(
            f"Failed to plot model comparison (supervised vs unsupervised): {str(e)}"
        )  # Log an error message
        raise  # Raise the exception to propagate the error


# ========== Plot Reputation Comparison ==========
@system.timer_decorator
def plot_reputation_comparison():
    """
    Function which generates a scatter plot to compare reputations at all major pipeline stages.

    Features
    ---------------------------------
        - Loads and merges the following CSVs from config.EXPORTED_REPUTATIONS:
            - 'original_reputations.csv' (Original_Reputation)
            - 'predicted_reputations_raw_tpot.csv' (Predicted_TPOT_Raw)
            - 'predicted_reputations_raw_flaml.csv' (Predicted_FLAML_Raw)
            - 'feedback_loop_reputations_supervised.csv' (Feedback_Loop_Supervised)
            - 'predicted_reputations_processed_tpot_supervised.csv' (Processed_TPOT_Supervised)
            - 'predicted_reputations_processed_flaml_supervised.csv' (Processed_FLAML_Supervised)
            - 'feedback_loop_reputations_unsupervised.csv' (Feedback_Loop_Unsupervised)
            - 'predicted_reputations_processed_tpot_unsupervised.csv' (Processed_TPOT_Unsupervised)
            - 'predicted_reputations_processed_flaml_unsupervised.csv' (Processed_FLAML_Unsupervised)
        - Merges all on User_Id (as int).
        - Generates a scatter plot with each reputation column as a separate trace.
        - Saves the merged CSV and the plot (.html and .png).
        - Updates plot legend and axis titles for clarity.

    Returns
    ---------------------------------
        - None: The function saves the plot and merged CSV to the specified directories.

    Raises Exception
    ---------------------------------
        - If any of the required CSV files are missing or contain invalid data.
        - If the plot generation fails due to errors in data processing or visualization.
    """
    try:  # Try-except block to catch and handle exceptions
        file_map = {
            "original": (
                "original_reputations",
                "Original_Reputation",
            ),  # Original reputations
            "tpot_raw": (
                "predicted_reputations_raw_tpot",
                "Predicted_TPOT_Raw",
            ),  # TPOT raw predictions
            "flaml_raw": (
                "predicted_reputations_raw_flaml",
                "Predicted_FLAML_Raw",
            ),  # FLAML raw predictions
            "supervised_feedback": (
                "feedback_loop_reputations_supervised",
                "Feedback_Loop_Supervised",
            ),  # Supervised feedback loop reputations
            "supervised_tpot": (
                "predicted_reputations_processed_tpot_supervised",
                "Processed_TPOT_Supervised",
            ),  # Processed TPOT supervised reputations
            "supervised_flaml": (
                "predicted_reputations_processed_flaml_supervised",
                "Processed_FLAML_Supervised",
            ),  # Processed FLAML supervised reputations
            "unsupervised_feedback": (
                "feedback_loop_reputations_unsupervised",
                "Feedback_Loop_Unsupervised",
            ),  # Unsupervised feedback loop reputations
            "unsupervised_tpot": (
                "predicted_reputations_processed_tpot_unsupervised",
                "Processed_TPOT_Unsupervised",
            ),  # Processed TPOT unsupervised reputations
            "unsupervised_flaml": (
                "predicted_reputations_processed_flaml_unsupervised",
                "Processed_FLAML_Unsupervised",
            ),  # Processed FLAML unsupervised reputations
        }  # Map of file names to their respective column names
        dfs = {}  # Dictionary to hold DataFrames for each file
        for key, (
            fname,
            colname,
        ) in file_map.items():  # Loop through each file in the map
            try:  # Try to load the CSV file
                df = utils.load_csv(
                    config.EXPORTED_REPUTATIONS, fname
                )  # Load the CSV file
            except Exception as e:  # Catch any exceptions during file loading
                logger.error(
                    f"Failed to load file {fname}.csv: {e}"
                )  # Log an error if loading fails
                return  # Exit the function if loading fails

            if not isinstance(df, pd.DataFrame):  # Ensure df is a Pandas DataFrame
                df = pd.DataFrame(df)  # Convert to DataFrame if needed
            df.columns = df.columns.str.strip()  # Strip whitespace from column names
            if "User_Id" not in df.columns:  # Check if 'User_Id' column exists
                logger.error(
                    f"Missing 'User_Id' column in {fname}.csv."
                )  # Log an error if 'User_Id' is missing
                return  # Exit the function if 'User_Id' is missing

            df["User_Id"] = (
                df["User_Id"]
                .astype(str)
                .str.replace("User_", "", regex=False)
                .astype(int)
            )  # Clean 'User_Id' to be int, removing 'User_' prefix if present
            rep_cols = [
                c for c in df.columns if c != "User_Id"
            ]  # Get all columns except 'User_Id'
            if len(rep_cols) != 1:  # Check if there is exactly one reputation column
                logger.error(
                    f"Expected exactly one reputation column in {fname}.csv, found: {rep_cols}"
                )  # Log an error if not exactly one reputation column
                return  # Exit the function if not exactly one reputation column
            rep_col = rep_cols[0]  # Get the reputation column name
            df = df[["User_Id", rep_col]].rename(
                columns={rep_col: colname}
            )  # Rename the reputation column to the specified column name
            dfs[key] = df  # Store the DataFrame in the dictionary with the key

        merge_order = [
            "original",
            "tpot_raw",
            "flaml_raw",
            "supervised_feedback",
            "supervised_tpot",
            "supervised_flaml",
            "unsupervised_feedback",
            "unsupervised_tpot",
            "unsupervised_flaml",
        ]  # Define the order of merging DataFrames
        merged = dfs[
            merge_order[0]
        ]  # Start with the first DataFrame in the merge order
        for key in merge_order[
            1:
        ]:  # Loop through the remaining DataFrames in the merge order
            merged = pd.merge(
                merged, dfs[key], on="User_Id", how="inner"
            )  # Merge the current DataFrame with the merged DataFrame on 'User_Id'

        expected_cols = [
            "Original_Reputation",
            "Predicted_TPOT_Raw",
            "Predicted_FLAML_Raw",
            "Feedback_Loop_Supervised",
            "Processed_TPOT_Supervised",
            "Processed_FLAML_Supervised",
            "Feedback_Loop_Unsupervised",
            "Processed_TPOT_Unsupervised",
            "Processed_FLAML_Unsupervised",
        ]  # Define the expected columns in the merged DataFrame

        missing_cols = [
            c for c in expected_cols if c not in merged.columns
        ]  # Check for missing columns in the merged DataFrame
        if missing_cols:  # If there are missing columns, log an error and exit
            logger.error(
                f"Missing columns in merged dataframe: {missing_cols}"
            )  # Log an error if any expected columns are missing
            return  # Exit the function if any expected columns are missing
        if merged.empty:  # Check if the merged DataFrame is empty
            logger.warning(
                "Merged reputation comparison dataset is empty."
            )  # Log a warning if the merged DataFrame is empty
            return  # Exit the function if the merged DataFrame is empty
        merged = merged.sort_values("User_Id")  # Sort the merged DataFrame by 'User_Id'
        merged_csv_path = (
            config.EXPORTED_REPUTATIONS / "merged_full_reputation_comparison.csv"
        )  # Define the path to save the merged CSV file
        merged.to_csv(
            merged_csv_path, index=False
        )  # Save the merged DataFrame to a CSV file

        color_map = {
            "Original_Reputation": "#66c2a5",
            "Predicted_TPOT_Raw": "#fc8d62",
            "Predicted_FLAML_Raw": "#ffd92f",
            "Feedback_Loop_Supervised": "#8da0cb",
            "Processed_TPOT_Supervised": "#e78ac3",
            "Processed_FLAML_Supervised": "#a6d854",
            "Feedback_Loop_Unsupervised": "#7fc97f",
            "Processed_TPOT_Unsupervised": "#beaed4",
            "Processed_FLAML_Unsupervised": "#fdc086",
        }  # Define a color map for the reputation columns
        symbol_map = {
            "Original_Reputation": "circle",
            "Predicted_TPOT_Raw": "x",
            "Predicted_FLAML_Raw": "cross",
            "Feedback_Loop_Supervised": "square",
            "Processed_TPOT_Supervised": "triangle-up",
            "Processed_FLAML_Supervised": "star",
            "Feedback_Loop_Unsupervised": "diamond",
            "Processed_TPOT_Unsupervised": "triangle-down",
            "Processed_FLAML_Unsupervised": "hexagram",
        }  # Define a symbol map for the reputation columns
        name_map = {
            "Original_Reputation": "Original Reputation",
            "Predicted_TPOT_Raw": "TPOT Prediction (Raw)",
            "Predicted_FLAML_Raw": "FLAML Prediction (Raw)",
            "Feedback_Loop_Supervised": "Feedback Loop (Supervised)",
            "Processed_TPOT_Supervised": "TPOT (Processed, Supervised)",
            "Processed_FLAML_Supervised": "FLAML (Processed, Supervised)",
            "Feedback_Loop_Unsupervised": "Feedback Loop (Unsupervised)",
            "Processed_TPOT_Unsupervised": "TPOT (Processed, Unsupervised)",
            "Processed_FLAML_Unsupervised": "FLAML (Processed, Unsupervised)",
        }  # Define a name map for the reputation columns
        fig = go.Figure()  # Create a new figure for the plot
        for (
            col
        ) in (
            expected_cols
        ):  # Loop through each expected column to add traces to the plot
            fig.add_trace(
                go.Scatter(
                    x=merged["User_Id"],
                    y=merged[col],
                    mode="markers",
                    name=name_map.get(col, col),
                    marker=dict(
                        color=color_map.get(col, None),
                        symbol=symbol_map.get(col, "circle"),
                        size=8,
                        line=dict(width=1, color="black"),
                    ),
                )
            )  # Add a scatter trace for the current column with specified color and symbol
        fig.update_layout(
            title=dict(
                text="Full Pipeline Reputation Comparison",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="User ID",
            yaxis_title="Reputation Score",
            template="plotly_dark",
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", title="Stage"),
            font=dict(size=14),
        )  # Update the layout of the plot with titles and formatting

        # --- legend κάτω, πιο «ήπια» τα Raw, και γραμμή αναφοράς στο 1.0 ---
        fig.update_layout(
            legend=dict(
                orientation="h", y=-0.22, x=0.5, xanchor="center", title_text="Stage"
            )
        )
        for tr in fig.data:
            if "Raw" in tr.name:
                try:
                    tr.marker.size = 6  # λίγο μικρότερα τα raw markers
                except Exception:
                    pass

        # Make raw points slightly less intense to highlight processed stages
        for tr in fig.data:
            if "Raw" in tr.name:
                try:
                    tr.marker.opacity = 0.6
                except Exception:
                    pass

        fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color="#888")

        fig.update_yaxes(range=[0.25, 1.02], dtick=0.1)
        fig.update_xaxes(tickmode="linear", dtick=10)
        fig.update_layout(font=dict(size=16), margin=dict(l=80, r=260, t=70, b=70))

        fig.write_html(
            str(config.PLOTS / "full_reputation_comparison.html")
        )  # Save the plot as HTML
        fig.write_image(
            str(config.PLOTS / "full_reputation_comparison.png"),
            width=1600,
            height=1000,
            scale=1,
        )  # Save the plot as PNG
        logger.info(
            "Full reputation pipeline comparison plot and merged CSV saved successfully."
        )  # Log success message

    except Exception as e:  # Catch exceptions and handle them gracefully
        logger.error(
            f"Failed to plot full reputation comparison: {e}"
        )  # Log an error message
        raise  # Raise the exception to propagate the error


# ========== Plot Final Experience vs Final Reputation ==========
@system.timer_decorator
def plot_final_experience_vs_final_reputation():
    """
    Function which generates a scatter plot to compare user final experience against final reputation,
    for both supervised and unsupervised feedback loop reputations.

    Features
    ---------------------------------
        - Loops over both 'supervised' and 'unsupervised' feedback loop reputation files.
        - Loads each file and merges with the original dataset on User_Id.
        - Plots experience vs final reputation for both run_types on the same figure, using different colors/markers.
        - Adds a legend distinguishing supervised and unsupervised results.
        - Saves the plot as 'experience_vs_final_reputation_comparison.html' and '.png'.
        - Logs the files being loaded and the results of the plot.

    Raises Exception
    ---------------------------------
        - If any of the required CSV files are missing or contain invalid data.
        - If the plot generation fails due to errors in data processing or visualization.
    """
    try:  # Try-except block to catch and handle exceptions
        run_types = [
            {
                "run_type": "Supervised",
                "file": "feedback_loop_reputations_supervised",
                "color": "#1f77b4",
                "marker": "circle",
            },  # Supervised feedback loop
            {
                "run_type": "Unsupervised",
                "file": "feedback_loop_reputations_unsupervised",
                "color": "#ff7f0e",
                "marker": "diamond",
            },  # Unsupervised feedback loop
        ]
        original_data = utils.load_csv(
            config.DATA_FOLDER, "Dataset"
        )  # Load original dataset
        logger.info(
            f"Loaded Dataset.csv from {config.DATA_FOLDER}"
        )  # Log loading message

        if not isinstance(
            original_data, pd.DataFrame
        ):  # Ensure original_data is a DataFrame
            original_data = pd.DataFrame(
                original_data
            )  # Convert to DataFrame if needed
        original_data["User_Id"] = (
            original_data["User_Id"].astype(str).str.replace("User_", "").astype(int)
        )  # Clean User_Id to be int
        latest_experience = (
            original_data.groupby("User_Id")["User_Experience"].max().reset_index()
        )  # Get latest experience per user

        fig = go.Figure()  # Create a new figure for the plot
        plotted_any = False  # Flag to track if any data was plotted
        for run in run_types:  # Loop over each run type
            try:  # Try to load the feedback loop reputation file
                logger.info(
                    f"Loading feedback loop reputations for {run['run_type']}: {run['file']}.csv from {config.EXPORTED_REPUTATIONS}"
                )  # Log loading message
                final_df = utils.load_csv(
                    config.EXPORTED_REPUTATIONS, run["file"]
                )  # Load the CSV file
            except Exception as e:  # Catch any exceptions during file loading
                logger.error(f"Failed to load file {run['file']}.csv: {e}")  # Log error
                continue  # Skip to the next run type if loading fails
            if not isinstance(final_df, pd.DataFrame):  # Ensure final_df is a DataFrame
                final_df = pd.DataFrame(final_df)  # Convert to DataFrame if needed
            if "User_Id" not in final_df.columns:  # Check if User_Id column exists
                logger.warning(
                    f"Missing 'User_Id' in {run['file']}.csv, skipping {run['run_type']}."
                )  # Log warning if User_Id is missing
                continue  # Skip to the next run type if User_Id is missing

            rep_cols = [
                c for c in final_df.columns if c != "User_Id"
            ]  # Get all columns except User_Id
            if len(rep_cols) != 1:  # Check if there is exactly one reputation column
                logger.warning(
                    f"Expected one reputation column in {run['file']}.csv, found: {rep_cols}. Skipping."
                )  # Log warning if not exactly one reputation column
                continue  # Skip to the next run type if not exactly one reputation column
            rep_col = rep_cols[0]  # Get the reputation column name

            final_df["User_Id"] = (
                final_df["User_Id"].astype(str).str.replace("User_", "").astype(int)
            )  # Clean User_Id to be int
            merged = pd.merge(
                final_df[["User_Id", rep_col]],
                latest_experience,
                on="User_Id",
                how="left",
            ).dropna(
                subset=["User_Experience", rep_col]
            )  # Merge with latest experience data and drop rows with NaN values
            if merged.empty:  # Check if merged DataFrame is empty
                logger.warning(
                    f"No data to plot for {run['run_type']} after merge."
                )  # Log warning if no data to plot
                continue  # Skip to the next run type if no data to plot

            fig.add_trace(
                go.Scatter(
                    x=merged["User_Experience"],
                    y=merged[rep_col],
                    mode="markers",
                    name=f"{run['run_type']} Feedback Loop",
                    marker=dict(
                        color=run["color"],
                        symbol=run["marker"],
                        size=10,
                        line=dict(width=1, color="black"),
                    ),
                    hovertemplate=(
                        "User Id: %{customdata[0]}<br>"
                        "Experience: %{x}<br>"
                        "Final Reputation: %{y}<extra></extra>"
                    ),
                    customdata=merged[["User_Id"]],
                )
            )  # Add a scatter trace for the current run type
            logger.info(
                f"Plotted {len(merged)} points for {run['run_type']} (file: {run['file']}.csv)."
            )  # Log the number of points plotted for the current run type
            plotted_any = True  # Set the flag to True if any data was plotted

        if not plotted_any:  # Check if no data was plotted
            logger.warning(
                "No data was plotted for experience vs final reputation comparison."
            )  # Log warning if no data was plotted
            return  # Exit the function if no data was plotted

        fig.update_layout(
            font=dict(size=14),
            title=dict(
                text="Final Experience vs Final Reputation (Supervised vs Unsupervised)<br>"
                "<span style='font-size:13px;'>User ID is available on hover in the interactive HTML plot</span>",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Final Experience",
            yaxis_title="Final Reputation",
            template="plotly_dark",
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", title="Run Type"),
        )  # Update layout of the plot

        fig.write_html(
            str(config.PLOTS / "final_experience_vs_final_reputation_comparison.html")
        )  # Path to save the HTML file
        fig.write_image(
            str(config.PLOTS / "final_experience_vs_final_reputation_comparison.png"),
            width=1600,
            height=1000,
            scale=1,
        )

    except Exception as e:  # Catch exceptions and handle them gracefully
        logger.error(
            f"Failed to plot final experience vs final reputation comparison:\n{e}"  # Log an error message
        )
        raise  # Raise the exception to propagate the error


# ========== Plot All Visualizations ==========
@system.timer_decorator
def plot_all_visualizations():
    """
    Function which generates and displays all visualizations in the specified order.

    Features
    ---------------------------------
        - Calls each visualization function in the specified order.
        - Displays all generated visualizations together in the same session.
        - Logs a success message after generating and displaying all visualizations.

    Raises Exception
    ---------------------------------
        - If any visualization function fails to generate or display the plots.
    """
    try:  # Try-except block to catch and handle exceptions
        # plot_outlier_numbers()  # Plot outlier numbers
        # plot_dataset_inclusion()  # Plot dataset inclusion
        # plot_model_comparison()  # Plot model comparison
        # plot_reputation_comparison()  # Plot reputation comparison
        plot_final_experience_vs_final_reputation()  # Plot final experience vs final reputation

        logger.info(
            "All visualizations have been generated and displayed successfully."
        )  # Log success message

    except Exception as e:  # Catch exceptions and handle them gracefully
        logger.error(
            f"Failed to generate all visualizations: {str(e)}"
        )  # Log an error message
        raise  # Raise the exception to propagate the error
