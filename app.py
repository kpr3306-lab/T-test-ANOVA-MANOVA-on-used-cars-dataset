import streamlit as st
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.stats_tests import (
    run_ttest,
    run_anova,
    run_manova,
    run_one_sample_ttest
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Used Cars Dashboard", layout="wide")

st.title("🚗 Used Cars Statistical Dashboard")

# =========================
# LOAD DATA
# =========================
df_raw, df = load_data()

# =========================
# COLUMN FILTERING (FIXED)
# =========================
exclude_cols = ["S.No.", "Name"]

numeric_cols = [
    col for col in df.select_dtypes(include="number").columns
    if col not in exclude_cols
]

categorical_cols = list(df.select_dtypes(include="object").columns)

# ✅ Force important categorical columns
for col in ["Seats", "Brand"]:
    if col not in categorical_cols:
        categorical_cols.append(col)

# Remove duplicates
categorical_cols = list(set(categorical_cols))

# Clean sorting
numeric_cols = sorted(numeric_cols)
categorical_cols = sorted(categorical_cols)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset", "Visualization", "T-Test", "ANOVA", "MANOVA"]
)

# =========================
# DATASET
# =========================
if page == "Dataset":
    st.subheader("Raw Data")
    st.dataframe(df_raw.head())

    st.subheader("Cleaned Data")
    st.dataframe(df.head())

# =========================
# VISUALIZATION
# =========================
elif page == "Visualization":
    st.subheader("Price vs Log Price")

    fig, ax = plt.subplots(1, 2, figsize=(12,4))

    ax[0].hist(df["Price"], bins=50)
    ax[0].set_title("Price")

    ax[1].hist(df["Log_Price"], bins=50)
    ax[1].set_title("Log Price")

    st.pyplot(fig)

# =========================
# T-TEST
# =========================
elif page == "T-Test":
    st.subheader("T-Test Analysis")

    test_type = st.radio("Select Test Type", ["One Sample", "Two Sample"])

    # -------- ONE SAMPLE --------
    if test_type == "One Sample":
        col = st.selectbox("Select Variable", numeric_cols)
        mu = st.number_input("Enter Hypothesized Mean (μ₀)", value=10.0)

        if st.button("Run One-Sample T-Test"):
            result = run_one_sample_ttest(df, col, mu)
            st.write(result)

    # -------- TWO SAMPLE --------
    else:
        val_col = st.selectbox("Numeric Variable", numeric_cols)
        grp_col = st.selectbox("Group Variable", categorical_cols)

        groups = df[grp_col].dropna().unique()

        g1 = st.selectbox("Group 1", groups)
        g2 = st.selectbox("Group 2", groups)

        if st.button("Run Two-Sample T-Test"):
            result = run_ttest(df, val_col, grp_col, g1, g2)
            st.write(result)

# =========================
# ANOVA
# =========================
elif page == "ANOVA":
    st.subheader("ANOVA Analysis")

    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in ["S.No."]
    ]

    categorical_cols = list(df.select_dtypes(include="object").columns)

    # Ensure important categorical columns
    for col in ["Seats", "Brand"]:
        if col not in categorical_cols:
            categorical_cols.append(col)

    categorical_cols = sorted(list(set(categorical_cols)))

    # ✅ SELECT TYPE
    anova_type = st.selectbox("Select ANOVA Type", ["1-Way", "2-Way", "3-Way"])

    dep = st.selectbox("Dependent Variable", numeric_cols)

    if anova_type == "1-Way":
        cat_vars = st.multiselect("Select 1 Categorical Variable", categorical_cols, max_selections=1)

    elif anova_type == "2-Way":
        cat_vars = st.multiselect("Select 2 Categorical Variables", categorical_cols, max_selections=2)

    else:
        cat_vars = st.multiselect("Select 3 Categorical Variables", categorical_cols, max_selections=3)

    if st.button("Run ANOVA"):

        if len(cat_vars) == 0:
            st.warning("Please select required categorical variables")
        else:
            table = run_anova(df, dep, cat_vars)
            st.write(table)

# =========================
# MANOVA
# =========================
elif page == "MANOVA":
    st.subheader("MANOVA")

    dep_vars = st.multiselect("Dependent Variables", numeric_cols)
    cat = st.selectbox("Categorical Variable", categorical_cols)

    if st.button("Run MANOVA"):
        result = run_manova(df, dep_vars, cat)
        st.text(result)