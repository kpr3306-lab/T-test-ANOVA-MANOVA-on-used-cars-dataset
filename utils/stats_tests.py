from scipy.stats import ttest_1samp, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA

# ---------- ONE SAMPLE ----------
def run_one_sample_ttest(df, col, mu):
    data = df[col].dropna()
    stat, p = ttest_1samp(data, mu)

    return {
        "t_stat": stat,
        "p_value": p,
        "decision": "Reject H0" if p < 0.05 else "Fail to Reject H0"
    }

# ---------- TWO SAMPLE ----------
def run_ttest(df, value_col, group_col, g1, g2):
    d1 = df[df[group_col] == g1][value_col]
    d2 = df[df[group_col] == g2][value_col]

    stat, p = ttest_ind(d1, d2, equal_var=False)

    return {
        "t_stat": stat,
        "p_value": p,
        "decision": "Reject H0" if p < 0.05 else "Fail to Reject H0"
    }

# ---------- ANOVA ----------
def run_anova(df, dep, cat_vars):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    if len(cat_vars) == 0:
        return "Select at least one categorical variable"

    # Build formula dynamically
    formula = dep + " ~ " + " + ".join([f"C({var})" for var in cat_vars])

    model = ols(formula, data=df).fit()
    return sm.stats.anova_lm(model, typ=2)

# ---------- MANOVA ----------
def run_manova(df, dep_vars, cat):
    if len(dep_vars) < 2:
        return "Select at least 2 dependent variables"

    formula = " + ".join(dep_vars) + f" ~ {cat}"
    manova = MANOVA.from_formula(formula, data=df)
    return manova.mv_test()