import numpy as np
import pandas as pd
from scipy.stats import levene, shapiro, t as t_dist, ttest_1samp, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _decision(p, alpha):
    return "Reject H0" if p < alpha else "Fail to Reject H0"


def _normality_p(x):
    # Shapiro is only valid up to 5000 points; a fixed-seed sample keeps results stable
    x = x.dropna()
    if len(x) < 3:
        return np.nan
    return shapiro(x.sample(min(len(x), 5000), random_state=0)).pvalue


# ---------- ONE SAMPLE ----------
def run_one_sample_ttest(df, col, mu, alpha=0.05):
    data = df[col].dropna()
    if len(data) < 2:
        raise ValueError("Need at least 2 observations.")
    stat, p = ttest_1samp(data, mu)
    n, mean, sd = len(data), data.mean(), data.std(ddof=1)
    half = t_dist.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n)

    return {
        "n": n,
        "mean": mean,
        "t_stat": stat,
        "p_value": p,
        "ci": (mean - half, mean + half),
        "cohens_d": (mean - mu) / sd,
        "normality_p": _normality_p(data),
        "decision": _decision(p, alpha),
    }


# ---------- TWO SAMPLE ----------
def run_ttest(df, value_col, group_col, g1, g2, alpha=0.05):
    if g1 == g2:
        raise ValueError("Pick two different groups.")
    d1 = df[df[group_col] == g1][value_col].dropna()
    d2 = df[df[group_col] == g2][value_col].dropna()
    if len(d1) < 2 or len(d2) < 2:
        raise ValueError("Each group needs at least 2 observations.")

    stat, p = ttest_ind(d1, d2, equal_var=False)
    pooled_sd = np.sqrt(((len(d1) - 1) * d1.var() + (len(d2) - 1) * d2.var()) / (len(d1) + len(d2) - 2))

    return {
        "n1": len(d1),
        "n2": len(d2),
        "mean1": d1.mean(),
        "mean2": d2.mean(),
        "t_stat": stat,
        "p_value": p,
        "cohens_d": (d1.mean() - d2.mean()) / pooled_sd,
        "levene_p": levene(d1, d2).pvalue,
        "decision": _decision(p, alpha),
    }


# ---------- ANOVA ----------
def _term(var):
    return f"C(Q('{var}'))"


def run_anova(df, dep, cat_vars, interactions=False):
    if len(cat_vars) == 0:
        raise ValueError("Select at least one categorical variable.")

    joiner = " * " if interactions else " + "
    formula = f"Q('{dep}') ~ " + joiner.join(_term(v) for v in cat_vars)

    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table["eta_sq"] = table["sum_sq"] / table["sum_sq"].sum()
    table.loc["Residual", "eta_sq"] = np.nan
    table.index = [i.replace("C(Q('", "").replace("'))", "") for i in table.index]
    return table


def run_tukey(df, dep, cat, alpha=0.05):
    data = df[[dep, cat]].dropna()
    res = pairwise_tukeyhsd(data[dep], data[cat].astype(str), alpha=alpha)
    out = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return out.sort_values("p-adj")


# ---------- MANOVA ----------
def run_manova(df, dep_vars, cat):
    if len(dep_vars) < 2:
        raise ValueError("Select at least 2 dependent variables.")

    formula = " + ".join(f"Q('{v}')" for v in dep_vars) + f" ~ {_term(cat)}"
    result = MANOVA.from_formula(formula, data=df).mv_test()
    return result.results[_term(cat)]["stat"]
