script_content = '''"""
Pediatric Cardiometabolic Risk and Relative Physical Fitness Analysis
====================================================================
Reproducible statistical pipeline for evaluating the protective associations
of relative physical fitness domains (cardiorespiratory endurance, upper-body
strength, and lower-body explosive strength) against clustered cardiometabolic
risk (Z-Score cMetS) in school-aged children (N = 250).

Author: Study Research Team
Repository: Pediatric-Cardiometabolic-Fitness-Analysis
License: MIT
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def calc_z_age_sex(series: pd.Series, age_col: pd.Series, sex_col: pd.Series) -> pd.Series:
    """Compute internal age- and sex-specific Z-scores (mean = 0, SD = 1)."""
    return series.groupby([age_col, sex_col]).transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )


def prepare_dataset(filepath: str) -> pd.DataFrame:
    """Load raw dataset, derive relative fitness metrics, Z-scores, and tertiles."""
    df = pd.read_csv(filepath)

    # 1. Anthropometric & Blood Pressure Standardization
    df["zBMI"] = calc_z_age_sex(df["BMI"], df["Age"], df["Sex"])
    df["Weight_Status"] = np.where(df["zBMI"] > 1.0, "Overweight/Obese", "Normal-Weight")
    df["WHtR"] = df["Waist_circumference"] / df["height"]
    df["MAP"] = df["Pre_Diastolic"] + (df["Pre_Systolic"] - df["Pre_Diastolic"]) / 3.0

    # 2. Clustered Cardiometabolic Risk Score (Z-Score cMetS)
    df["Z_WC"] = calc_z_age_sex(df["Waist_circumference"], df["Age"], df["Sex"])
    df["Z_WHtR"] = calc_z_age_sex(df["WHtR"], df["Age"], df["Sex"])
    df["Z_Skinfold"] = calc_z_age_sex(df["Sum_of_skinfold"], df["Age"], df["Sex"])
    df["Z_MAP"] = calc_z_age_sex(df["MAP"], df["Age"], df["Sex"])

    z_cmets_raw = (df["Z_WC"] + df["Z_WHtR"] + df["Z_Skinfold"] + df["Z_MAP"]) / 4.0
    df["Z_cMetS"] = (z_cmets_raw - z_cmets_raw.mean()) / z_cmets_raw.std(ddof=1)
    df["High_cMetS_Risk"] = (df["Z_cMetS"] > 1.0).astype(int)

    # 3. Relative Physical Fitness Indices (Body-Size Normalized)
    df["Rel_Endurance"] = df["Distance_covered"] / df["height"]     # 6MWT / Height (m/cm)
    df["Rel_Jump"] = df["Explosive_strength"] / df["height"]        # Jump / Height (cm/cm)
    df["Rel_Grip"] = df["Handgrip_Strength"] / df["weight"]         # Grip / Weight (kg/kg)

    # 4. Age- and Sex-Specific Fitness Z-Scores & Composite Indices
    df["Z_Endurance"] = calc_z_age_sex(df["Rel_Endurance"], df["Age"], df["Sex"])
    df["Z_Jump"] = calc_z_age_sex(df["Rel_Jump"], df["Age"], df["Sex"])
    df["Z_Grip"] = calc_z_age_sex(df["Rel_Grip"], df["Age"], df["Sex"])

    df["Z_Strength"] = (df["Z_Jump"] + df["Z_Grip"]) / 2.0
    df["Z_Fitness"] = (df["Z_Endurance"] + df["Z_Strength"]) / 2.0

    # 5. Fitness Tertile Stratification
    tertile_labels = ["Low", "Moderate", "High"]
    df["Z_Endurance_Tertile"] = pd.qcut(df["Z_Endurance"], q=3, labels=tertile_labels)
    df["Z_Jump_Tertile"] = pd.qcut(df["Z_Jump"], q=3, labels=tertile_labels)
    df["Z_Grip_Tertile"] = pd.qcut(df["Z_Grip"], q=3, labels=tertile_labels)
    df["Z_Strength_Tertile"] = pd.qcut(df["Z_Strength"], q=3, labels=tertile_labels)
    df["Z_Fitness_Tertile"] = pd.qcut(df["Z_Fitness"], q=3, labels=tertile_labels)

    return df


def generate_table1(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 1: Baseline characteristics stratified by WHO weight status."""
    continuous_vars = [
        ("Age (years)", "Age"),
        ("BMI Z-Score (zBMI)", "zBMI"),
        ("Height (cm)", "height"),
        ("Weight (kg)", "weight"),
        ("Waist Circumference (cm)", "Waist_circumference"),
        ("Waist-to-Height Ratio (WHtR)", "WHtR"),
        ("Sum of Skinfolds (mm)", "Sum_of_skinfold"),
        ("Distance / Height (6MWT / Height)", "Rel_Endurance"),
        ("Jump / Height (cm / cm)", "Rel_Jump"),
        ("Grip Strength / Weight (kg / kg)", "Rel_Grip"),
        ("Endurance Score (Z_Endurance)", "Z_Endurance"),
        ("Total Body Strength Score (Z_Strength)", "Z_Strength"),
        ("Total Physical Fitness (Z_Fitness)", "Z_Fitness"),
        ("Clustered Metabolic Risk (Z-Score cMetS)", "Z_cMetS"),
    ]

    nw = df[df["Weight_Status"] == "Normal-Weight"]
    ow = df[df["Weight_Status"] == "Overweight/Obese"]

    rows = []
    # Categorical: Male sex
    male_tot = f"{(df['Sex'] == 1).sum()} ({(df['Sex'] == 1).mean() * 100:.1f}%)"
    male_nw = f"{(nw['Sex'] == 1).sum()} ({(nw['Sex'] == 1).mean() * 100:.1f}%)"
    male_ow = f"{(ow['Sex'] == 1).sum()} ({(ow['Sex'] == 1).mean() * 100:.1f}%)"
    rows.append(["Male sex [n (%)]", male_tot, male_nw, male_ow])

    for label, col in continuous_vars:
        tot_str = f"{df[col].mean():.2f} ± {df[col].std(ddof=1):.2f}"
        nw_str = f"{nw[col].mean():.2f} ± {nw[col].std(ddof=1):.2f}"
        ow_str = f"{ow[col].mean():.2f} ± {ow[col].std(ddof=1):.2f}"
        rows.append([label, tot_str, nw_str, ow_str])

    columns = [
        "Characteristic",
        f"Total (N={len(df)})",
        f"Normal-Weight (n={len(nw)})",
        f"Overweight/Obese (n={len(ow)})",
    ]
    return pd.DataFrame(rows, columns=columns)


def partial_corr(df: pd.DataFrame, x: str, y: str, covariates: list) -> tuple:
    """Compute partial Pearson correlation (r) and two-tailed p-value."""
    sub = df[[x, y] + covariates].dropna()
    X_cov = sm.add_constant(sub[covariates])
    res_x = sm.OLS(sub[x], X_cov).fit().resid
    res_y = sm.OLS(sub[y], X_cov).fit().resid
    return stats.pearsonr(res_x, res_y)


def generate_table2(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 2: Partial correlations adjusted for Age, Sex, and zBMI."""
    covariates = ["Age", "Sex", "zBMI"]
    outcomes = [
        ("Waist Circumference (cm)", "Waist_circumference"),
        ("WHtR", "WHtR"),
        ("Sum of Skinfolds (mm)", "Sum_of_skinfold"),
        ("Blood Pressure (MAP)", "MAP"),
        ("Clustered Metabolic Risk (Z-Score cMetS)", "Z_cMetS"),
    ]
    predictors = [
        ("Distance / Height (Endurance)", "Rel_Endurance"),
        ("Jump / Height (Lower-Body)", "Rel_Jump"),
        ("Grip / Weight (Upper-Body)", "Rel_Grip"),
        ("Total Physical Fitness", "Z_Fitness"),
    ]

    rows = []
    for out_label, out_col in outcomes:
        row = [out_label]
        for _, pred_col in predictors:
            r, p = partial_corr(df, pred_col, out_col, covariates)
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            row.append(f"{r:+.2f} ({p_str})")
        rows.append(row)

    return pd.DataFrame(rows, columns=["Cardiometabolic Marker"] + [p[0] for p in predictors])


def generate_table3(df: pd.DataFrame) -> dict:
    """Generate Table 3: ANCOVA-adjusted Z-Score cMetS across Endurance x Strength tertiles."""
    def compute_matrix(strength_col: str) -> pd.DataFrame:
        model = smf.ols(
            f"Z_cMetS ~ C(Z_Endurance_Tertile) * C({strength_col}) + Age + C(Sex)",
            data=df,
        ).fit()
        mean_age = df["Age"].mean()
        sex_vals = df["Sex"].unique()
        matrix = {}

        for s_level in ["Low", "Moderate", "High"]:
            matrix[s_level] = {}
            for e_level in ["Low", "Moderate", "High"]:
                pred_df = pd.DataFrame({
                    "Z_Endurance_Tertile": [e_level] * len(sex_vals),
                    strength_col: [s_level] * len(sex_vals),
                    "Age": [mean_age] * len(sex_vals),
                    "Sex": sex_vals,
                })
                pred = model.get_prediction(pred_df).summary_frame(alpha=0.05)
                mean_val = pred["mean"].mean()
                se_val = np.sqrt(np.mean(pred["mean_se"] ** 2))
                ci_low = mean_val - 1.96 * se_val
                ci_high = mean_val + 1.96 * se_val
                matrix[s_level][e_level] = f"{mean_val:+.2f} ({ci_low:+.2f}, {ci_high:+.2f})"

        out_df = pd.DataFrame(matrix).T
        out_df.columns = ["Low Endurance", "Moderate Endurance", "High Endurance"]
        return out_df

    return {
        "Upper_Body_Strength": compute_matrix("Z_Grip_Tertile"),
        "Lower_Body_Strength": compute_matrix("Z_Jump_Tertile"),
        "Total_Muscular_Strength": compute_matrix("Z_Strength_Tertile"),
    }


def plot_figure1_forest(df: pd.DataFrame, save_path: str = "Figure1_Forest_Plot.png") -> None:
    """Generate Figure 1: Multivariable-adjusted Odds Ratios (95% CI) for High cMetS Risk."""
    domains = [
        ("A. Endurance", "Z_Endurance_Tertile"),
        ("B. Total Body Strength", "Z_Strength_Tertile"),
        ("C. Total Physical Fitness", "Z_Fitness_Tertile"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    y_positions = [2, 1, 0]
    y_labels = ["High\n(Reference)", "Moderate", "Low\n(Least Fit)"]

    for ax, (title, col) in zip(axes, domains):
        formula = f"High_cMetS_Risk ~ C({col}, Treatment(reference='High')) + Age + C(Sex) + zBMI"
        logit_mod = smf.logit(formula, data=df).fit(disp=False)
        params = logit_mod.params
        conf = logit_mod.conf_int()

        ors = [1.00]
        ci_lows = [1.00]
        ci_highs = [1.00]
        labels = ["1.00 (Ref)"]

        for lvl in ["Moderate", "Low"]:
            key = f"C({col}, Treatment(reference='High'))[T.{lvl}]"
            or_val = np.exp(params[key])
            low_val = np.exp(conf.loc[key, 0])
            high_val = np.exp(conf.loc[key, 1])
            ors.append(or_val)
            ci_lows.append(low_val)
            ci_highs.append(high_val)
            labels.append(f"{or_val:.2f} ({low_val:.2f}–{high_val:.2f})")

        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        for y, or_v, l_v, h_v, txt in zip(y_positions, ors, ci_lows, ci_highs, labels):
            ax.errorbar(
                or_v, y, xerr=[[or_v - l_v], [h_v - or_v]],
                fmt="s", color="black", ecolor="black", capsize=4, markersize=6
            )
            ax.text(or_v, y + 0.18, txt, ha="center", va="bottom", fontsize=9)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Adjusted Odds Ratio (95% CI)", fontsize=10)
        ax.set_ylim(-0.5, 2.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_figure2_moderation(df: pd.DataFrame, save_path: str = "Figure2_Moderation_Plot.png") -> None:
    """Generate Figure 2: Weight-status moderation of adjusted Z-Score cMetS across tertiles."""
    domains = [
        ("A. Endurance ($Z_{Endurance}$)", "Z_Endurance_Tertile"),
        ("B. Total Body Strength ($Z_{Strength}$)", "Z_Strength_Tertile"),
        ("C. Total Physical Fitness ($Z_{Fitness}$)", "Z_Fitness_Tertile"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    x_pos = np.array([0, 1, 2])
    levels = ["Low", "Moderate", "High"]

    for ax, (title, col) in zip(axes, domains):
        model = smf.ols(
            f"Z_cMetS ~ C({col}) * C(Weight_Status) + Age + C(Sex)", data=df
        ).fit()
        anova_res = sm.stats.anova_lm(model, typ=2)
        p_int = anova_res.loc[f"C({col}):C(Weight_Status)", "PR(>F)"]

        mean_age = df["Age"].mean()
        sex_vals = df["Sex"].unique()

        for status, marker, ls, color, offset in [
            ("Normal-Weight", "o", "-", "black", -0.04),
            ("Overweight/Obese", "s", "--", "#444444", 0.04),
        ]:
            means, errs = [], []
            for lvl in levels:
                pred_df = pd.DataFrame({
                    col: [lvl] * len(sex_vals),
                    "Weight_Status": [status] * len(sex_vals),
                    "Age": [mean_age] * len(sex_vals),
                    "Sex": sex_vals,
                })
                pred = model.get_prediction(pred_df).summary_frame(alpha=0.05)
                m = pred["mean"].mean()
                se = np.sqrt(np.mean(pred["mean_se"] ** 2))
                means.append(m)
                errs.append(1.96 * se)

            mfc = "black" if status == "Normal-Weight" else "white"
            ax.errorbar(
                x_pos + offset, means, yerr=errs, marker=marker, linestyle=ls,
                color=color, markerfacecolor=mfc, markersize=7, capsize=4, label=status
            )

        ax.axhline(0, color="lightgray", linestyle=":", linewidth=1)
        ax.text(0.95, 0.92, f"$p_{{interaction}} = {p_int:.2f}$", transform=ax.transAxes,
                ha="right", fontsize=9.5, style="italic")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Low\n(Least Fit)", "Moderate", "High\n(Most Fit)"], fontsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Adjusted Clustered Metabolic Risk (Z-Score cMetS)", fontweight="bold", fontsize=10)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    DATA_FILE = "Rajon.sss.csv"
    dataset = prepare_dataset(DATA_FILE)

    print("=== TABLE 1 ===")
    print(generate_table1(dataset).to_markdown(index=False))

    print("\\n=== TABLE 2 ===")
    print(generate_table2(dataset).to_markdown(index=False))

    print("\\n=== TABLE 3 ===")
    t3_matrices = generate_table3(dataset)
    for domain_name, mat in t3_matrices.items():
        print(f"\\n--- {domain_name} ---")
        print(mat.to_markdown())

    plot_figure1_forest(dataset)
    plot_figure2_moderation(dataset)
    print("\\nFigures saved successfully at 300 DPI.")
'''

output_filename = "pediatric_cardiometabolic_fitness_analysis.py"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(script_content)

print(f"Saved {output_filename} successfully.")