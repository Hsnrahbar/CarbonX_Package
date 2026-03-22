from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import solve_ivp


K_M = 1.11
D_FM = 2.15
DM0 = 60e-9
DP0 = 5e-9
R_GAS = 8.314

NP0 = K_M * (DM0 / DP0) ** D_FM
A0 = NP0 * np.pi * DP0**2
V_SPHERE = NP0 * (np.pi / 6.0) * DP0**3
D_SPHERE = (6.0 * V_SPHERE / np.pi) ** (1.0 / 3.0)
A_SPHERE = np.pi * D_SPHERE**2


def sintering_rhs(_time: float, y: np.ndarray, temperature_k: float):
    area_ratio = float(y[0])
    tau_mean = (
        9e14
        * (6.0 * V_SPHERE / (A0 * area_ratio)) ** 4
        * temperature_k
        * np.exp(
            (20249.0 / temperature_k)
            * (
                1.0
                - (
                    (
                        ((3.5e-9) / (6.0 * V_SPHERE / (A0 * area_ratio)))
                        * (temperature_k / 3400.0)
                    )
                    ** 3
                )
            )
        )
    )
    dydt = -(area_ratio - A_SPHERE / A0) / tau_mean
    return np.array([dydt], dtype=float)


def run_sintering_loop(
    temperature_k: float,
    step: float = 0.001,
    total_time: float = 7.0,
    ode_dt: float = 0.001,
) -> pd.DataFrame:
    times = np.arange(0.0, total_time + 0.5 * step, step)
    surface_ratio = np.array([1.0], dtype=float)
    rows = []

    for current_time in times:
        solution = solve_ivp(
            fun=lambda t, y: sintering_rhs(t, y, temperature_k),
            t_span=(current_time, current_time + ode_dt),
            y0=surface_ratio,
            method="BDF",
            t_eval=[current_time + ode_dt],
            rtol=1e-6,
            atol=1e-10,
        )
        surface_ratio = solution.y[:, -1]
        area = surface_ratio[0] * A0
        dp = 6.0 * V_SPHERE / area
        dm = dp * (((A_SPHERE / np.pi) ** 1.5) / K_M / dp**3) ** (1.0 / D_FM)
        dm = max(dm, D_SPHERE)
        tau = 1.534e15 * (6.0 * V_SPHERE / (A0 * surface_ratio[0])) ** 4 * temperature_k * np.exp(
            165000.0 / R_GAS / temperature_k
        )
        rows.append((area, current_time, dp, dm, tau))

    return pd.DataFrame(rows, columns=["area_m2", "time_s", "dp_m", "dm_m", "tau_s"])


def normalize_column_name(name: str) -> str:
    normalized = str(name).strip().lower()
    for char in [" ", "[", "]", "(", ")", "-", "/", ".", ","]:
        normalized = normalized.replace(char, "")
    return normalized


def get_column_by_aliases(df: pd.DataFrame, aliases: list[str], required: bool = True):
    normalized_lookup = {normalize_column_name(column): column for column in df.columns}
    for alias in aliases:
        matched_column = normalized_lookup.get(normalize_column_name(alias))
        if matched_column is not None:
            return pd.to_numeric(df[matched_column], errors="coerce").to_numpy(dtype=float)
    if required:
        raise KeyError(f"Could not find any of these columns: {aliases}")
    return None


def load_fig5_data(base_dir: Path) -> dict:
    preferred_files = ["fig_5_data_NEW.xlsx", "fig_5_data.xlsx"]

    for filename in preferred_files:
        file_path = base_dir / filename
        if not file_path.exists():
            continue

        data = pd.read_excel(file_path)
        data = data.dropna(how="all").reset_index(drop=True)
        if data.empty:
            continue

        result = {
            "source": filename,
            "temperature_k": get_column_by_aliases(data, ["Temp", "Temperature", "T", "T [K]"]),
            "dm_avg_nm": get_column_by_aliases(data, ["data_dm_avg", "dm_avg", "dm", "d_m_avg"]),
            "dm_low_nm": get_column_by_aliases(data, ["data_dm_low", "dm_low", "d_m_low"], required=False),
            "dm_high_nm": get_column_by_aliases(data, ["data_dm_high", "dm_high", "d_m_high"], required=False),
            "dp_avg_nm": get_column_by_aliases(data, ["data_dp_avg", "dp_avg", "dp", "d_p_avg"]),
            "dp_low_nm": get_column_by_aliases(data, ["data_dp_low", "dp_low", "d_p_low"]),
            "dp_high_nm": get_column_by_aliases(data, ["data_dp_high", "dp_high", "d_p_high"]),
        }

        valid_rows = np.isfinite(result["temperature_k"])
        for key, values in result.items():
            if key == "source" or values is None:
                continue
            valid_rows &= np.isfinite(values)

        for key, values in result.items():
            if key == "source" or values is None:
                continue
            result[key] = values[valid_rows]

        return result

    raise FileNotFoundError(
        "Could not find `fig_5_data.xlsx` or `fig_5_data_NEW.xlsx` next to the script."
    )


def compute_temperature_sweep(temperatures_k: np.ndarray, step: float = 0.001) -> pd.DataFrame:
    dp_values = []
    dm_values = []
    tau_values = []

    for temperature_k in temperatures_k:
        results = run_sintering_loop(float(temperature_k), step=step)
        dp_values.append(results["dp_m"].max())
        dm_values.append(results["dm_m"].min())
        tau_values.append(results["tau_s"].max())

    output = pd.DataFrame(
        {
            "T [K]": temperatures_k,
            "dp [m]": dp_values,
            "dm [m]": dm_values,
            "dp [nm]": np.array(dp_values) * 1e9,
            "dm [nm]": np.array(dm_values) * 1e9,
            "tau [s]": tau_values,
        }
    )
    return output


def style_axes(ax, include_left: bool, include_right: bool) -> None:
    ax.tick_params(axis="both", which="major", labelsize=34)
    ax.tick_params(axis="x", which="major", direction="out", bottom=True, top=False, width=2, length=8)
    ax.tick_params(axis="x", which="minor", direction="out", bottom=True, top=False, width=2, length=5)
    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        left=include_left,
        right=include_right,
        width=2,
        length=8,
        labelcolor="black",
    )
    ax.tick_params(
        axis="y",
        which="minor",
        direction="out",
        left=include_left,
        right=include_right,
        width=2,
        length=5,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def plot_dual_axis(results_df: pd.DataFrame, fig5_data: dict, save_dir: Path) -> None:
    plt.rcParams["font.family"] = ["Calibri", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.size"] = 24

    fig, ax1 = plt.subplots(figsize=(12, 12 * 6.0 / 8.18))
    ax2 = ax1.twinx()

    dp_symbol_color = "#154357"
    dp_line_color = "#15C5C5"
    dm_symbol_color = "#6F3B79"
    dm_line_color = "orange"

    dm_avg = fig5_data["dm_avg_nm"]
    dm_low = fig5_data["dm_low_nm"]
    dm_high = fig5_data["dm_high_nm"]

    if dm_low is not None and dm_high is not None:
        dm_handle = ax1.errorbar(
            fig5_data["temperature_k"],
            dm_avg,
            yerr=[dm_avg - dm_low, dm_high - dm_avg],
            fmt="s",
            color=dm_symbol_color,
            markerfacecolor="none",
            markeredgecolor=dm_symbol_color,
            markeredgewidth=2,
            markersize=16,
            capsize=5,
            capthick=2,
            linewidth=4,
            label="dm (literature)",
        )
    else:
        dm_handle = ax1.scatter(
            fig5_data["temperature_k"],
            dm_avg,
            s=180,
            marker="s",
            facecolors="none",
            edgecolors=dm_symbol_color,
            linewidths=2,
            label="dm (literature)",
        )

    dm_line, = ax1.plot(
        results_df["T [K]"],
        results_df["dm [nm]"],
        linestyle="-.",
        color=dm_line_color,
        linewidth=4,
        label="dm (this study)",
    )

    dp_handle = ax2.errorbar(
        fig5_data["temperature_k"],
        fig5_data["dp_avg_nm"],
        yerr=[
            fig5_data["dp_avg_nm"] - fig5_data["dp_low_nm"],
            fig5_data["dp_high_nm"] - fig5_data["dp_avg_nm"],
        ],
        fmt="o",
        color=dp_symbol_color,
        markerfacecolor="none",
        markeredgecolor=dp_symbol_color,
        markeredgewidth=2,
        markersize=16,
        capsize=5,
        capthick=2,
        linewidth=4,
        label="dp (literature)",
    )
    dp_line, = ax2.plot(
        results_df["T [K]"],
        results_df["dp [nm]"],
        linestyle="--",
        color=dp_line_color,
        linewidth=4,
        label="dp (this study)",
    )

    ax1.set_xlim(200, 1100)
    combined_ymax = max(
        np.nanmax(results_df["dp [nm]"]),
        np.nanmax(results_df["dm [nm]"]),
        np.nanmax(fig5_data["dp_high_nm"]),
        np.nanmax(dm_high if dm_high is not None else dm_avg),
    )
    upper_limit = max(90.0, np.ceil(combined_ymax / 10.0) * 10.0)
    ax1.set_ylim(0, upper_limit)
    ax2.set_ylim(0, upper_limit)

    ax1.set_xlabel("Temperature (K)", fontsize=34)
    ax1.set_ylabel("dm (nm)", fontsize=34, color="black")
    ax2.set_ylabel("dp (nm)", fontsize=34, color="black")

    plt.grid(False)
    style_axes(ax1, include_left=True, include_right=False)
    style_axes(ax2, include_left=False, include_right=True)

    legend_handles = [dm_handle, dm_line, dp_handle, dp_line]
    legend_labels = ["dm (literature)", "dm (this study)", "dp (literature)", "dp (this study)"]
    ax1.legend(legend_handles, legend_labels, fontsize=22, loc="best")

    fig.tight_layout()

    for suffix, dpi in [("jpg", 300), ("svg", None), ("tiff", 300), ("pdf", None)]:
        output_path = save_dir / f"sintering_dual_axis_plot.{suffix}"
        save_kwargs = {"bbox_inches": "tight", "format": suffix}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)

    plt.show()


def main():
    base_dir = Path.cwd()
    temperatures_k = np.array([293, 473, 523, 573, 623, 673, 723, 773, 873, 973, 1073], dtype=float)

    fig5_data = load_fig5_data(base_dir)
    results_df = compute_temperature_sweep(temperatures_k, step=0.001)

    output_excel = base_dir / "sint_data_python.xlsx"
    results_df.to_excel(output_excel, index=False)

    plot_dual_axis(results_df, fig5_data, base_dir)

    print(f"Loaded literature data from: {fig5_data['source']}")
    print(f"Saved calculated data to: {output_excel}")
    print("Saved figure as JPG, SVG, TIFF, and PDF.")


if __name__ == "__main__":
    main()
