from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import solve_ivp


R_GAS = 8.314

TEMPERATURES_K = np.array([773, 823, 873, 923, 973, 1023, 1073, 1123, 1173, 1223, 1273], dtype=float)
TIME_RES_S = np.array([1.946, 1.782, 1.618, 1.451, 1.282, 1.118, 1.072, 1.026, 0.9808, 0.9301, 0.8821], dtype=float)
TEMP_PROFILES = np.array(
    [
        [317.6, -1788, 3759, -3685, 1798, 290],
        [557.9, -2935, 5752, -5234, 2326, 285.2],
        [1033, -4977, 8930, -7444, 3015, 260.7],
        [1948, -8491, 1.376e4, -1.035e4, 3759, 253.7],
        [4137, -1.611e4, 2.338e4, -1.577e4, 5118, 192.1],
        [8221, -2.835e4, 3.639e4, -2.173e4, 6260, 183.2],
        [1.497e4, -4.792e4, 5.694e4, -3.119e4, 8037, 150.6],
        [1.655e4, -5.076e4, 5.77e4, -3.038e4, 7652, 231.1],
        [2.53e4, -7.308e4, 7.81e4, -3.85e4, 9001, 213.5],
        [3.245e4, -8.898e4, 9.036e4, -4.252e4, 9594, 222.1],
        [4.546e4, -1.165e5, 1.105e5, -4.877e4, 1.041e4, 236.5],
    ],
    dtype=float,
)

K_M_LOOP = 1.11
D_FM_LOOP = 2.20
DM0_LOOP = 60e-9
DP0_LOOP = 4e-9
SIGMA_FE = 2.52

NP0_LOOP = K_M_LOOP * (DM0_LOOP / DP0_LOOP) ** D_FM_LOOP
A0_LOOP = NP0_LOOP * np.pi * DP0_LOOP**2
V_SPHERE_LOOP = NP0_LOOP * (np.pi / 6.0) * DP0_LOOP**3
D_SPHERE_LOOP = (6.0 * V_SPHERE_LOOP / np.pi) ** (1.0 / 3.0)
A_SPHERE_LOOP = np.pi * D_SPHERE_LOOP**2

K_M_ODE = 1.0
D_FM_ODE = 2.20
DM0_ODE = 60e-9
DP0_ODE = 4e-9

NP0_ODE = K_M_ODE * (DM0_ODE / DP0_ODE) ** D_FM_ODE
A0_ODE = NP0_ODE * np.pi * DP0_ODE**2
V_SPHERE_ODE = NP0_ODE * (np.pi / 6.0) * DP0_ODE**3
A_SPHERE_ODE = np.pi * ((6.0 * V_SPHERE_ODE / np.pi) ** (1.0 / 3.0)) ** 2

TAU_MODELS = {
    "tau11": (1.57e4, 1.0, 59150.0),
    "tau12": (5.08e10, 2.0, 83850.0),
    "tau13": (1.1e17, 3.0, 111040.0),
    "tau14": (1.12e20, 4.0, 185610.0),
}


def polynomial_temperature(time_s: float, poly_coeffs: np.ndarray) -> float:
    return float(np.polyval(poly_coeffs, time_s))


def fe_tau_for_ode(dp_m: float, temperature_k: float, tau_model: str) -> float:
    prefactor, power, activation = TAU_MODELS[tau_model]
    return prefactor * dp_m**power * np.exp(activation / (R_GAS * temperature_k))


def fe_tau_saved_in_loop(dp_m: float, temperature_k: float) -> float:
    d_fe_gbd = 11.2e-13 * np.exp(-41500.0 / (R_GAS * temperature_k))
    return ((0.83**6) * 8.314 * temperature_k * dp_m**4) / (
        6.0 * 8.0 * 16.0 * SIGMA_FE * d_fe_gbd * (0.0558 / 7874.0) * 5e-10
    )


def sintering_fe_rhs(
    _time_s: float,
    y: np.ndarray,
    temperature_k: float,
    tau_model: str,
) -> np.ndarray:
    area_ratio = float(y[0])
    dp_m = 6.0 * V_SPHERE_ODE / (A0_ODE * area_ratio)
    tau = fe_tau_for_ode(dp_m, temperature_k, tau_model)
    dydt = -(area_ratio - A_SPHERE_ODE / A0_ODE) / tau
    return np.array([dydt], dtype=float)


def run_fe_sintering_loop(
    temperature_k: float,
    total_time_s: float,
    poly_coeffs: np.ndarray,
    tau_model: str = "tau14",
    step: float = 0.0005,
    ode_window_s: float = 0.0001,
) -> pd.DataFrame:
    times = np.arange(0.0, total_time_s + 0.5 * step, step)
    surface_ratio = np.array([1.0], dtype=float)
    rows = []

    for current_time in times:
        solution = solve_ivp(
            fun=lambda t, y: sintering_fe_rhs(t, y, temperature_k, tau_model),
            t_span=(current_time, current_time + ode_window_s),
            y0=surface_ratio,
            method="BDF",
            t_eval=[current_time + ode_window_s],
            rtol=1e-6,
            atol=1e-10,
        )
        surface_ratio = solution.y[:, -1]

        area_m2 = surface_ratio[0] * A0_LOOP
        dp_m = 6.0 * V_SPHERE_LOOP / area_m2
        dm_m = dp_m * (((A_SPHERE_LOOP / np.pi) ** 1.5) / K_M_LOOP / dp_m**3) ** (1.0 / D_FM_LOOP)
        dm_m = max(dm_m, D_SPHERE_LOOP)

        profile_temperature_k = polynomial_temperature(current_time, poly_coeffs)
        tau_saved_s = fe_tau_saved_in_loop(dp_m, profile_temperature_k)

        rows.append((area_m2, current_time, dp_m, dm_m, tau_saved_s))

    return pd.DataFrame(
        rows,
        columns=["area_m2", "time_s", "dp_m", "dm_m", "tau_s"],
    )


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


def load_fe_reference_data(base_dir: Path) -> dict:
    preferred_files = ["fig_5_data_NEW.xlsx", "data_sint_Fe.xlsx", "fig_5_data.xlsx"]

    for filename in preferred_files:
        file_path = base_dir / filename
        if not file_path.exists():
            continue

        data = pd.read_excel(file_path, sheet_name=0)
        data = data.dropna(how="all").reset_index(drop=True)
        if data.empty:
            continue

        result = {
            "source": filename,
            "temperature_k": get_column_by_aliases(data, ["T", "Temp", "Temperature", "T [K]"]),
            "dp_avg_nm": get_column_by_aliases(data, ["dp", "data_dp_avg", "dp_avg", "d_p_avg"]),
            "dp_low_nm": get_column_by_aliases(data, ["dp_l", "dp_low", "data_dp_low", "d_p_low"]),
            "dp_high_nm": get_column_by_aliases(data, ["dp_u", "dp_high", "data_dp_high", "d_p_high"]),
            "dm_avg_nm": get_column_by_aliases(data, ["dm", "data_dm_avg", "dm_avg", "d_m_avg"]),
            "dm_low_nm": get_column_by_aliases(data, ["dm_l", "dm_low", "data_dm_low", "d_m_low"], required=False),
            "dm_high_nm": get_column_by_aliases(data, ["dm_u", "dm_high", "data_dm_high", "d_m_high"], required=False),
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
        "Could not find `fig_5_data_NEW.xlsx`, `data_sint_Fe.xlsx`, or `fig_5_data.xlsx` in the current working directory."
    )


def compute_fe_temperature(
    temperatures_k: np.ndarray,
    time_res_s: np.ndarray,
    poly_coeff_matrix: np.ndarray,
    tau_model: str = "tau14",
    step: float = 0.0005,
) -> pd.DataFrame:
    dp_values_m = []
    dm_values_m = []
    tau_values_s = []

    for temperature_k, total_time_s, poly_coeffs in zip(temperatures_k, time_res_s, poly_coeff_matrix):
        results = run_fe_sintering_loop(
            temperature_k=float(temperature_k),
            total_time_s=float(total_time_s),
            poly_coeffs=np.asarray(poly_coeffs, dtype=float),
            tau_model=tau_model,
            step=step,
            ode_window_s=0.0001,
        )
        dp_values_m.append(results["dp_m"].max())
        dm_values_m.append(results["dm_m"].min())
        tau_values_s.append(results["tau_s"].min())

    return pd.DataFrame(
        {
            "T [K]": temperatures_k,
            "time_res [s]": time_res_s,
            "dp [m]": dp_values_m,
            "dm [m]": dm_values_m,
            "dp [nm]": np.array(dp_values_m) * 1e9,
            "dm [nm]": np.array(dm_values_m) * 1e9,
            "tau [s]": tau_values_s,
        }
    )


def compute_fe_shaded_bounds(
    temperatures_k: np.ndarray,
    time_res_s: np.ndarray,
    poly_coeff_matrix: np.ndarray,
    step: float = 0.0005,
) -> pd.DataFrame:
    model_runs = [
        compute_fe_temperature(
            temperatures_k=temperatures_k,
            time_res_s=time_res_s,
            poly_coeff_matrix=poly_coeff_matrix,
            tau_model=model_name,
            step=step,
        )
        for model_name in ("tau11", "tau12", "tau13", "tau14")
    ]

    dp_stack_nm = np.vstack([run["dp [nm]"].to_numpy() for run in model_runs])
    dm_stack_nm = np.vstack([run["dm [nm]"].to_numpy() for run in model_runs])
    tau_stack_s = np.vstack([run["tau [s]"].to_numpy() for run in model_runs])

    return pd.DataFrame(
        {
            "T [K]": temperatures_k,
            "time_res [s]": time_res_s,
            "dp_avg [nm]": dp_stack_nm.mean(axis=0),
            "dm_avg [nm]": dm_stack_nm.mean(axis=0),
            "dp_min [nm]": dp_stack_nm.min(axis=0),
            "dp_max [nm]": dp_stack_nm.max(axis=0),
            "dm_min [nm]": dm_stack_nm.min(axis=0),
            "dm_max [nm]": dm_stack_nm.max(axis=0),
            "tau_avg [s]": tau_stack_s.mean(axis=0),
        }
    )


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


def plot_fe_dual_axis(results_df: pd.DataFrame, bounds_df: pd.DataFrame, reference_data: dict, save_dir: Path) -> None:
    plt.rcParams["font.family"] = ["Calibri", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.size"] = 24

    fig, ax1 = plt.subplots(figsize=(13.5, 12 * 6.0 / 8.18))
    ax2 = ax1.twinx()

    dp_symbol_color = "#154357"
    dp_line_color = "#15C5C5"
    dm_symbol_color = "#6F3B79"
    dm_line_color = "orange"

    dm_avg = reference_data["dm_avg_nm"]
    dm_low = reference_data["dm_low_nm"]
    dm_high = reference_data["dm_high_nm"]

    if dm_low is not None and dm_high is not None:
        dm_handle = ax1.errorbar(
            reference_data["temperature_k"],
            dm_avg,
            yerr=[dm_avg - dm_low, dm_high - dm_avg],
            fmt="s",
            color=dm_symbol_color,
            markerfacecolor="none",
            markeredgecolor=dm_symbol_color,
            markeredgewidth=2,
            markersize=22,
            capsize=5,
            capthick=2,
            linewidth=4,
            label="dm (literature)",
        )
    else:
        dm_handle = ax1.scatter(
            reference_data["temperature_k"],
            dm_avg,
            s=180,
            marker="s",
            facecolors="none",
            edgecolors=dm_symbol_color,
            linewidths=2,
            label="dm (literature)",
        )

    dm_band = ax1.fill_between(
        bounds_df["T [K]"],
        bounds_df["dm_min [nm]"],
        bounds_df["dm_max [nm]"],
        color=dm_line_color,
        alpha=0.18,
        linewidth=0,
        label="dm range (tau11-tau14)",
    )
    dm_line, = ax1.plot(
        results_df["T [K]"],
        results_df["dm_avg [nm]"],
        linestyle="-.",
        color=dm_line_color,
        linewidth=4,
        label="dm (average tau11-tau14)",
    )

    dp_handle = ax2.errorbar(
        reference_data["temperature_k"],
        reference_data["dp_avg_nm"],
        yerr=[
            reference_data["dp_avg_nm"] - reference_data["dp_low_nm"],
            reference_data["dp_high_nm"] - reference_data["dp_avg_nm"],
        ],
        fmt="o",
        color=dp_symbol_color,
        markerfacecolor="none",
        markeredgecolor=dp_symbol_color,
        markeredgewidth=2,
        markersize=22,
        capsize=5,
        capthick=2,
        linewidth=4,
        label="dp (literature)",
    )
    dp_band = ax2.fill_between(
        bounds_df["T [K]"],
        bounds_df["dp_min [nm]"],
        bounds_df["dp_max [nm]"],
        color=dp_line_color,
        alpha=0.18,
        linewidth=0,
        label="dp range (tau11-tau14)",
    )
    dp_line, = ax2.plot(
        results_df["T [K]"],
        results_df["dp_avg [nm]"],
        linestyle="-.",
        color=dp_line_color,
        linewidth=4,
        label="dp (average tau11-tau14)",
    )

    ax1.set_xlim(float(np.min(TEMPERATURES_K)) - 20.0, float(np.max(TEMPERATURES_K)) + 20.0)
    ax1.set_ylim(0, 90)
    ax2.set_ylim(0, 90)
    ax1.set_xlabel("Temperature (K)", fontsize=34)
    ax1.set_ylabel("dm (nm)", fontsize=34, color="black")
    ax2.set_ylabel("dp (nm)", fontsize=34, color="black")

    plt.grid(False)
    style_axes(ax1, include_left=True, include_right=False)
    style_axes(ax2, include_left=False, include_right=True)

    legend_handles = [dm_band, dm_handle, dm_line, dp_band, dp_handle, dp_line]
    legend_labels = [
        "dm range (tau11-tau14)",
        "dm (literature)",
        "dm (average tau11-tau14)",
        "dp range (tau11-tau14)",
        "dp (literature)",
        "dp (average tau11-tau14)",
    ]
    ax1.legend(legend_handles, legend_labels, fontsize=22, loc="best")

    fig.tight_layout()

    for suffix, dpi in [("jpg", 300), ("svg", None), ("tiff", 300), ("pdf", None)]:
        output_path = save_dir / f"exp2_Fe_sintering_dual_axis_plot.{suffix}"
        save_kwargs = {"bbox_inches": "tight", "format": suffix}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)

    plt.show()


def main() -> None:
    base_dir = Path.cwd()
    reference_data = load_fe_reference_data(base_dir)
    results_df = compute_fe_shaded_bounds(
        temperatures_k=TEMPERATURES_K,
        time_res_s=TIME_RES_S,
        poly_coeff_matrix=TEMP_PROFILES,
        step=0.0005,
    )

    output_excel = base_dir / "exp2_Fe_sint_data_python.xlsx"
    results_df.to_excel(output_excel, index=False)

    plot_fe_dual_axis(results_df, results_df, reference_data, base_dir)

    print(f"Loaded literature data from: {reference_data['source']}")
    print(f"Saved calculated data to: {output_excel}")
    print("Saved figure as JPG, SVG, TIFF, and PDF.")


if __name__ == "__main__":
    main()
