import html as _html
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.formula.api import ols

from utils.data_loader import load_data
from utils.graphics import BODY_BLURBS, BODY_COLORS, car_svg, logo_svg
from utils.stats_tests import run_anova, run_manova, run_one_sample_ttest, run_ttest, run_tukey

# =========================
# PAGE CONFIG & THEME
# =========================
st.set_page_config(page_title="MotorIQ · Used Car Intelligence", page_icon="🚗", layout="wide")

ACCENT, MUTED, TEXT, LINE = "#E87535", "#AAB4C0", "#F4F5F6", "#323A43"
PRICE_TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100]
PAIR = ["#E0682A", "#4A8CD0"]  # two-group comparisons reuse the first two validated slots
BODIES = list(BODY_COLORS)
BODY_SCALE = alt.Scale(domain=BODIES, range=list(BODY_COLORS.values()))

NUMERIC_COLS = ["Price", "Log_Price", "Kilometers_Driven", "Mileage", "Engine", "Power", "Car_Age", "Year"]
CATEGORICAL_COLS = ["Body_Type", "Fuel_Type", "Transmission", "Owner_Type", "Seats", "Location", "Brand"]
LABELS = {
    "Price": "Price (₹ lakh)", "Log_Price": "Log price", "Kilometers_Driven": "Kilometres driven",
    "Mileage": "Mileage (km/l)", "Engine": "Engine (cc)", "Power": "Power (bhp)", "Car_Age": "Car age (yrs)",
    "Year": "Model year", "Body_Type": "Body type", "Fuel_Type": "Fuel type", "Transmission": "Transmission",
    "Owner_Type": "Owner type", "Seats": "Seats", "Location": "City", "Brand": "Brand",
}


@alt.theme.register("motoriq", enable=True)
def _motoriq_theme():
    axis = {
        "labelColor": MUTED, "titleColor": MUTED, "gridColor": "#2B3239", "domainColor": LINE,
        "tickColor": LINE, "labelFont": "Inter", "titleFont": "Inter", "titleFontWeight": 500,
        "labelFontSize": 11, "titleFontSize": 11, "titlePadding": 10,
    }
    return alt.theme.ThemeConfig({
        "config": {
            "background": "transparent",
            "view": {"stroke": None},
            "axis": axis,
            "axisX": {"grid": False},
            "legend": {"labelColor": TEXT, "titleColor": MUTED, "labelFont": "Inter", "titleFont": "Inter",
                       "orient": "top", "symbolType": "square"},
            "header": {"labelColor": TEXT, "titleColor": MUTED},
        }
    })


def html(s):
    # collapse newlines so Markdown never turns indented HTML into a code block
    st.markdown(re.sub(r"\s*\n\s*", " ", s), unsafe_allow_html=True)


def esc(x):
    return _html.escape(str(x))


def lakh(x):
    return f"₹{x / 100:.2f} Cr" if x >= 100 else f"₹{x:.2f} L"


def pfmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def peq(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def chart(c, height=300):
    st.altair_chart(c.properties(height=height), theme=None, width="stretch")


def kpis(items):
    cells = "".join(
        f'<div class="card kpi"><div class="label">{esc(l)}</div><div class="value">{esc(v)}</div>'
        f'<div class="note">{esc(n)}</div></div>'
        for l, v, n in items
    )
    html(f'<div class="kpis">{cells}</div>')


def page_head(eyebrow, title, sub):
    html(f'<div class="page-head"><div class="eyebrow">{esc(eyebrow)}</div><h2>{esc(title)}</h2>'
         f'<div class="muted">{esc(sub)}</div></div>')


def verdict(reject, title, body):
    cls, icon, tag = ("good", "✔", "Significant") if reject else ("neutral", "○", "Not significant")
    html(f'<div class="card verdict"><div class="icon pill {cls}">{icon} {tag}</div>'
         f'<div><h4>{esc(title)}</h4><div class="muted">{esc(body)}</div></div></div>')


def size_word(d):
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


# =========================
# LOAD DATA
# =========================
@st.cache_data
def get_data():
    df_raw, df = load_data()
    # Hedonic price model: what a car "should" cost given its specs, used to flag deals.
    # shortcut: in-sample fit on the whole dataset; hold out a test split if this ever drives real decisions.
    model = ols(
        "np.log(Price) ~ Car_Age + np.log(Kilometers_Driven) + Power + Engine + C(Brand) + C(Fuel_Type)"
        " + C(Transmission) + C(Body_Type) + C(Owner_Type) + C(Location)",
        data=df,
    ).fit()
    df["Fair_Price"] = np.exp(model.fittedvalues)
    ratio = df["Price"] / df["Fair_Price"]
    # Under 40% of fair value is more likely a data-entry error than a bargain
    df["Deal"] = np.select([ratio < 0.4, ratio < 0.85, ratio > 1.15],
                           ["Check listing", "Great deal", "Above market"], "Fair price")
    df["Deal_Score"] = ratio
    return df_raw, df, model.rsquared


df_raw, df_all, fair_r2 = get_data()
st.markdown(f"<style>{Path(__file__).with_name('assets').joinpath('style.css').read_text()}</style>",
            unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    html(f'<div class="brand">{logo_svg(42)}<div><div class="brand-name">Motor<span>IQ</span></div>'
         f'<div class="brand-sub">MARKET INTELLIGENCE</div></div></div>')
    page = st.radio(
        "Navigation",
        [":material/space_dashboard: Overview", ":material/directions_car: Market Explorer",
         ":material/table_chart: Dataset", ":material/functions: T-Test",
         ":material/stacked_bar_chart: ANOVA", ":material/hub: MANOVA"],
        label_visibility="collapsed",
    ).split(": ", 1)[1]

    html('<div class="side-label">Filters · apply to every page</div>')
    bodies = st.pills("Body type", BODIES, selection_mode="multi", default=BODIES)
    fuels_all = sorted(df_all["Fuel_Type"].unique())
    fuels = st.multiselect("Fuel type", fuels_all, default=fuels_all)
    trans = st.pills("Transmission", ["Manual", "Automatic"], selection_mode="multi", default=["Manual", "Automatic"])
    yr = st.slider("Model year", int(df_all.Year.min()), int(df_all.Year.max()),
                   (int(df_all.Year.min()), int(df_all.Year.max())))
    pmax = float(np.ceil(df_all.Price.max()))
    price = st.slider("Price (₹ lakh)", 0.0, pmax, (0.0, pmax), step=0.5)

df = df_all[
    df_all.Body_Type.isin(bodies) & df_all.Fuel_Type.isin(fuels) & df_all.Transmission.isin(trans)
    & df_all.Year.between(*yr) & df_all.Price.between(*price)
]
with st.sidebar:
    st.caption(f"Showing **{len(df):,}** of {len(df_all):,} cars")

if df.empty:
    st.warning("No cars match these filters. Widen the filters in the sidebar.")
    st.stop()

# =========================
# OVERVIEW
# =========================
if page == "Overview":
    hero_cars = "".join(f'<div class="{c}">{car_svg(b)}</div>' for c, b in [("c1", "Hatchback"), ("c2", "SUV"), ("c3", "Sedan")])
    html(f"""
    <div class="hero">
      <div>
        <div class="eyebrow">MotorIQ · Market intelligence</div>
        <h1>Find your next <em>drive.</em></h1>
        <p>Explore India's used car market through data: prices, depreciation, deals and the
           statistics behind them.</p>
        <div class="chips">
          <span class="chip"><b>{len(df):,}</b> listings</span>
          <span class="chip"><b>{df.Brand.nunique()}</b> brands</span>
          <span class="chip"><b>{df.Location.nunique()}</b> cities</span>
          <span class="chip"><b>{df.Year.min()}–{df.Year.max()}</b> model years</span>
        </div>
      </div>
      <div class="hero-cars">{hero_cars}<div class="road"></div></div>
    </div>""")

    auto_share = (df.Transmission == "Automatic").mean()
    kpis([
        ("Average market price", lakh(df.Price.mean()), f"median {lakh(df.Price.median())}"),
        ("Cars analysed", f"{len(df):,}", f"{(df.Deal == 'Great deal').sum():,} priced as great deals"),
        ("Median car age", f"{df.Car_Age.median():.0f} yrs", f"as of {2026}"),
        ("Median distance", f"{df.Kilometers_Driven.median() / 1000:.0f}k km", "kilometres driven"),
        ("Automatic share", f"{auto_share:.0%}", "of listings"),
    ])

    tiles = ""
    for b in BODIES:
        sub = df[df.Body_Type == b]
        n = len(sub)
        med = lakh(sub.Price.median()) if n else "—"
        tiles += f"""
        <div class="card body-tile">
          {car_svg(b)}
          <div class="name"><span class="dot" style="background:{BODY_COLORS[b]}"></span>{b}</div>
          <div class="muted" style="font-size:.82rem">{BODY_BLURBS[b]} silhouette</div>
          <div class="stats"><div><span class="muted">Listings</span><b>{n:,}</b></div>
            <div><span class="muted">Median</span><b>{med}</b></div>
            <div><span class="muted">Share</span><b>{n / len(df):.0%}</b></div></div>
        </div>"""
    html(f'<div class="bodies">{tiles}</div>')

    # ---- computed insights ----
    insights = []
    if df.Car_Age.nunique() > 1:
        slope = np.polyfit(df.Car_Age, df.Log_Price, 1)[0]
        insights.append((f"{(1 - np.exp(slope)):.0%}", "value lost per year of age",
                         "Fitted on log price, so it compounds year over year."))
    med_t = df.groupby("Transmission").Price.median()
    if {"Manual", "Automatic"} <= set(med_t.index):
        insights.append((f"{med_t['Automatic'] / med_t['Manual']:.1f}×", "automatic vs manual price",
                         f"Median {lakh(med_t['Automatic'])} vs {lakh(med_t['Manual'])}."))
    med_f = df.groupby("Fuel_Type").Price.median()
    if {"Diesel", "Petrol"} <= set(med_f.index):
        insights.append((f"{med_f['Diesel'] / med_f['Petrol']:.1f}×", "diesel vs petrol price",
                         f"Median {lakh(med_f['Diesel'])} vs {lakh(med_f['Petrol'])}."))
    city = df.groupby("Location").Price.median().sort_values()
    insights.append((esc(city.index[-1]), "priciest city (median)",
                     f"{lakh(city.iloc[-1])} vs {lakh(city.iloc[0])} in {city.index[0]}."))
    html('<div class="insights">' + "".join(
        f'<div class="card insight"><div class="big">{v}</div><div>{esc(t)}</div>'
        f'<div class="muted" style="font-size:.8rem;margin-top:4px">{esc(n)}</div></div>'
        for v, t, n in insights) + "</div>")

    # ---- charts ----
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        html('<div class="chart-title">Price distribution</div>'
             '<div class="chart-sub">Prices are right-skewed, so bins are on a log scale (axis shows ₹ lakh)</div>')
        chart(alt.Chart(df).mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, binSpacing=2).encode(
            x=alt.X("Log_Price:Q", bin=alt.Bin(maxbins=40), title="Price (₹ lakh)",
                    axis=alt.Axis(values=[float(np.log(v)) for v in PRICE_TICKS], labelExpr="format(exp(datum.value), '.3~r')")),
            y=alt.Y("count():Q", title="Listings"),
            tooltip=[alt.Tooltip("count():Q", title="Listings")],
        ))
    with c2, st.container(border=True):
        html('<div class="chart-title">Median price by brand</div>'
             '<div class="chart-sub">Top 12 brands by number of listings</div>')
        top = (df.groupby("Brand").agg(median=("Price", "median"), n=("Price", "size"))
               .nlargest(12, "n").reset_index())
        chart(alt.Chart(top).mark_bar(color=ACCENT, cornerRadiusEnd=4, height={"band": 0.7}).encode(
            y=alt.Y("Brand:N", sort="-x", title=None, axis=alt.Axis(labelLimit=160)),
            x=alt.X("median:Q", title="Median price (₹ lakh)"),
            tooltip=["Brand", alt.Tooltip("median:Q", title="Median ₹ lakh", format=".2f"),
                     alt.Tooltip("n:Q", title="Listings")],
        ))

    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(border=True):
        html('<div class="chart-title">Depreciation curve</div>'
             '<div class="chart-sub">Median price by car age; points with fewer than 10 cars hidden</div>')
        dep = (df.groupby(["Body_Type", "Car_Age"]).Price.agg(["median", "size"]).reset_index()
               .query("size >= 10"))
        hover = alt.selection_point(fields=["Car_Age"], nearest=True, on="pointerover", empty=False)
        base = alt.Chart(dep).encode(
            x=alt.X("Car_Age:Q", title="Car age (years)"),
            y=alt.Y("median:Q", title="Median price (₹ lakh)"),
            color=alt.Color("Body_Type:N", scale=BODY_SCALE, title=None),
        )
        chart(alt.layer(
            base.mark_line(strokeWidth=2),
            base.mark_point(filled=True, size=70, stroke="#252B32", strokeWidth=2).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                tooltip=[alt.Tooltip("Body_Type:N", title="Body"), alt.Tooltip("Car_Age:Q", title="Age"),
                         alt.Tooltip("median:Q", title="Median ₹ lakh", format=".2f"),
                         alt.Tooltip("size:Q", title="Cars")],
            ).add_params(hover),
            alt.Chart(dep).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(x="Car_Age:Q").transform_filter(hover),
        ))
    with c4, st.container(border=True):
        html('<div class="chart-title">Price by fuel type</div>'
             '<div class="chart-sub">Box = middle 50% of listings, line = median (log scale)</div>')
        chart(alt.Chart(df).mark_boxplot(color=ACCENT, size=36, median={"color": TEXT},
                                         rule={"color": MUTED}, outliers={"color": MUTED, "opacity": 0.35, "size": 8}).encode(
            x=alt.X("Fuel_Type:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Price:Q", scale=alt.Scale(type="log"), title="Price (₹ lakh)", axis=alt.Axis(values=PRICE_TICKS)),
            tooltip=["Fuel_Type"],
        ))

# =========================
# MARKET EXPLORER
# =========================
elif page == "Market Explorer":
    page_head("Market explorer", "Browse listings",
              f"Each car is compared with a fair-value model (R² = {fair_r2:.2f}) built from age, km, power, "
              "engine, brand, fuel, transmission, body, owner and city.")
    a, b = st.columns([2, 1])
    q = a.text_input("Search", placeholder="e.g. Creta, BMW, Pune…")
    sort = b.selectbox("Sort by", ["Best deals first", "Price: low to high", "Price: high to low",
                                   "Newest first", "Lowest km"])
    deal_f = st.pills("Deal rating", ["Great deal", "Fair price", "Above market", "Check listing"],
                      selection_mode="multi")

    view = df
    if q:
        view = view[view.Name.str.contains(q, case=False, regex=False) | view.Location.str.contains(q, case=False, regex=False)]
    if deal_f:
        view = view[view.Deal.isin(deal_f)]
    key, asc = {"Best deals first": ("Deal_Score", True), "Price: low to high": ("Price", True),
                "Price: high to low": ("Price", False), "Newest first": ("Year", False),
                "Lowest km": ("Kilometers_Driven", True)}[sort]
    # suspected data errors sink to the bottom of "best deals"
    order = view[key].where(view.Deal != "Check listing", np.inf) if key == "Deal_Score" else view[key]
    view = view.loc[order.sort_values(ascending=asc).index]

    per_page = 12
    pages = max(1, -(-len(view) // per_page))
    p1, p2 = st.columns([3, 1])
    p1.caption(f"{len(view):,} cars match")
    pg = p2.number_input(f"Page (of {pages})", 1, pages, 1)
    pill_cls = {"Great deal": ("good", "↓"), "Fair price": ("neutral", "●"), "Above market": ("warn", "↑"),
                "Check listing": ("bad", "!")}

    cards = ""
    for _, r in view.iloc[(pg - 1) * per_page: pg * per_page].iterrows():
        cls, icon = pill_cls[r.Deal]
        cards += f"""
        <div class="card listing">
          <div class="art">{car_svg(r.Body_Type)}</div>
          <div class="title" title="{esc(r.Name)}">{esc(r.Name)}</div>
          <div class="meta">{r.Year} · {r.Kilometers_Driven:,} km · {esc(r.Fuel_Type)} · {esc(r.Transmission)}</div>
          <div class="meta">{esc(r.Location)} · {esc(r.Owner_Type)} owner · {esc(r.Body_Type)}</div>
          <div class="row"><div class="price">{lakh(r.Price)}</div><span class="pill {cls}">{icon} {r.Deal}</span></div>
          <div class="meta">Fair value ≈ {lakh(r.Fair_Price)}</div>
        </div>"""
    html(f'<div class="listings">{cards}</div>' if cards else '<div class="card muted">No cars match your search.</div>')
    st.caption(f"Page {pg} of {pages}")

# =========================
# DATASET
# =========================
elif page == "Dataset":
    page_head("Dataset", "Raw vs cleaned data", "What went in, what came out and what changed along the way.")
    kpis([
        ("Raw rows", f"{len(df_raw):,}", f"{df_raw.shape[1]} columns"),
        ("Clean rows", f"{len(df_all):,}", f"{len(df):,} after filters"),
        ("Missing cells (raw)", f"{int(df_raw.isna().sum().sum()):,}", "imputed or dropped"),
        ("Engineered features", "5", "Brand, Model, Body type, Car age, Log price"),
    ])
    t1, t2, t3, t4 = st.tabs(["Cleaned (filtered)", "Raw", "Summary statistics", "Cleaning log"])
    show = ["Name", "Brand", "Body_Type", "Location", "Year", "Kilometers_Driven", "Fuel_Type", "Transmission",
            "Owner_Type", "Mileage", "Engine", "Power", "Seats", "Price", "Fair_Price", "Deal"]
    with t1:
        st.dataframe(df[show], hide_index=True, height=440, column_config={
            "Price": st.column_config.NumberColumn("Price (₹L)", format="%.2f"),
            "Fair_Price": st.column_config.NumberColumn("Fair value (₹L)", format="%.2f"),
            "Kilometers_Driven": st.column_config.NumberColumn("Km", format="%d"),
        })
        st.download_button("Download filtered CSV", df[show].to_csv(index=False), "motoriq_filtered.csv",
                           "text/csv", icon=":material/download:")
    with t2:
        st.dataframe(df_raw, hide_index=True, height=440)
    with t3:
        st.dataframe(df[NUMERIC_COLS].describe().T.round(2), height=330)
    with t4:
        st.markdown("""
- **Units stripped** from Mileage (kmpl / km/kg), Engine (CC) and Power (bhp), then converted to numbers.
- **Zero mileage and zero seats** treated as missing, since no car really has either.
- **Missing values**: Mileage, Engine and Power filled with the median, Seats with the mode.
- **Outlier removed**: one listing with 6.5 million km.
- **New_Price dropped**: missing for about 86% of rows.
- **Brand** taken from the first word of the name, with variants merged (ISUZU → Isuzu, Land → Land Rover).
- **Body type** inferred from the model name (e.g. Creta → SUV, City → Sedan, Innova → MUV), falling back to seat count.
- **Car_Age** = 2026 − Year; **Log_Price** = ln(Price), because price is heavily right-skewed.
""")

# =========================
# T-TEST
# =========================
elif page == "T-Test":
    page_head("Hypothesis testing", "T-Test", "Compare a mean against a benchmark, or two groups against each other.")
    left, right = st.columns([1, 2.2], gap="large")
    with left, st.container(border=True):
        test_type = st.segmented_control("Test", ["One sample", "Two sample"], default="Two sample") or "Two sample"
        alpha = st.segmented_control("Significance level α", [0.01, 0.05, 0.10], default=0.05) or 0.05
        if test_type == "One sample":
            col = st.selectbox("Variable", NUMERIC_COLS, format_func=LABELS.get)
            mu = st.number_input("Hypothesised mean μ₀", value=round(float(df_all[col].median()), 2))
        else:
            val_col = st.selectbox("Numeric variable", NUMERIC_COLS, index=1, format_func=LABELS.get)
            grp_col = st.selectbox("Group by", CATEGORICAL_COLS, index=2, format_func=LABELS.get)
            groups = df[grp_col].value_counts().index.tolist()
            g1 = st.selectbox("Group A", groups, index=0)
            g2 = st.selectbox("Group B", groups, index=min(1, len(groups) - 1))

    with right:
        try:
            if test_type == "One sample":
                r = run_one_sample_ttest(df, col, mu, alpha)
                kpis([
                    ("Sample mean", f"{r['mean']:.2f}", f"n = {r['n']:,}"),
                    ("t statistic", f"{r['t_stat']:.2f}", f"df = {r['n'] - 1:,}"),
                    ("p-value", pfmt(r["p_value"]), f"α = {alpha}"),
                    ("Cohen's d", f"{r['cohens_d']:.2f}", f"{size_word(r['cohens_d'])} effect"),
                ])
                reject = r["p_value"] < alpha
                lo, hi = r["ci"]
                verdict(reject, r["decision"],
                        f"The mean {LABELS[col].lower()} is {r['mean']:.2f} ({(1 - alpha):.0%} CI {lo:.2f}–{hi:.2f}), "
                        + (f"which differs significantly from μ₀ = {mu:g}." if reject
                           else f"which is not significantly different from μ₀ = {mu:g}."))
                st.write("")
                with st.container(border=True):
                    html(f'<div class="chart-title">Distribution of {esc(LABELS[col])}</div>'
                         '<div class="chart-sub">Orange line = sample mean · dashed line = μ₀</div>')
                    rules = pd.DataFrame({"x": [r["mean"], mu], "what": ["Sample mean", "μ₀"]})
                    chart(alt.layer(
                        alt.Chart(df).mark_bar(color="#3A424C", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=50), title=LABELS[col]),
                            y=alt.Y("count():Q", title="Listings"), tooltip=[alt.Tooltip("count():Q", title="Listings")]),
                        alt.Chart(rules).mark_rule(strokeWidth=2).encode(
                            x="x:Q", color=alt.Color("what:N", scale=alt.Scale(domain=["Sample mean", "μ₀"], range=[ACCENT, TEXT]),
                                                     title=None),
                            strokeDash=alt.condition("datum.what == 'μ₀'", alt.value([5, 4]), alt.value([1, 0])),
                            tooltip=["what", alt.Tooltip("x:Q", format=".2f", title="Value")]),
                    ), 260)
                st.caption(f"Normality (Shapiro–Wilk) {peq(r['normality_p'])}. "
                           + ("Data aren't normal, but with large n the t-test is still reliable (central limit theorem)."
                              if r["normality_p"] < 0.05 else "No evidence against normality."))
            else:
                r = run_ttest(df, val_col, grp_col, g1, g2, alpha)
                kpis([
                    (f"Mean · {g1}", f"{r['mean1']:.2f}", f"n = {r['n1']:,}"),
                    (f"Mean · {g2}", f"{r['mean2']:.2f}", f"n = {r['n2']:,}"),
                    ("t statistic", f"{r['t_stat']:.2f}", "Welch's t-test"),
                    ("p-value", pfmt(r["p_value"]), f"α = {alpha}"),
                    ("Cohen's d", f"{r['cohens_d']:.2f}", f"{size_word(r['cohens_d'])} effect"),
                ])
                reject = r["p_value"] < alpha
                higher = g1 if r["mean1"] > r["mean2"] else g2
                verdict(reject, r["decision"],
                        (f"{higher} cars have a significantly higher mean {LABELS[val_col].lower()}." if reject
                         else f"No significant difference in {LABELS[val_col].lower()} between {g1} and {g2}.")
                        + f" The difference is {size_word(r['cohens_d'])} in practical terms (d = {r['cohens_d']:.2f}).")
                st.write("")
                with st.container(border=True):
                    html(f'<div class="chart-title">{esc(LABELS[val_col])}: {esc(g1)} vs {esc(g2)}</div>'
                         '<div class="chart-sub">Box = middle 50%, line = median</div>')
                    sub = df[df[grp_col].isin([g1, g2])].assign(Group=lambda d: d[grp_col].astype(str))
                    chart(alt.Chart(sub).mark_boxplot(size=44, median={"color": TEXT}, rule={"color": MUTED},
                                                      outliers={"opacity": 0.35, "size": 8}).encode(
                        y=alt.Y("Group:N", title=None, sort=[str(g1), str(g2)]),
                        x=alt.X(f"{val_col}:Q", title=LABELS[val_col]),
                        color=alt.Color("Group:N", scale=alt.Scale(domain=[str(g1), str(g2)], range=PAIR), legend=None),
                    ), 200)
                st.caption(f"Levene's test for equal variances {peq(r['levene_p'])}. Welch's t-test is used, "
                           "so unequal variances are already accounted for.")
        except ValueError as e:
            st.info(str(e))

# =========================
# ANOVA
# =========================
elif page == "ANOVA":
    page_head("Variance analysis", "ANOVA", "Do group means differ across one, two or three categorical factors?")
    left, right = st.columns([1, 2.2], gap="large")
    with left, st.container(border=True):
        k = {"1-Way": 1, "2-Way": 2, "3-Way": 3}[st.segmented_control("Design", ["1-Way", "2-Way", "3-Way"], default="1-Way") or "1-Way"]
        dep = st.selectbox("Dependent variable", NUMERIC_COLS, index=1, format_func=LABELS.get)
        cat_vars = st.multiselect(f"Factors (pick {k})", CATEGORICAL_COLS, default=CATEGORICAL_COLS[:k],
                                  max_selections=k, format_func=LABELS.get)
        interactions = st.toggle("Include interaction terms", value=False, disabled=k == 1)
        alpha = st.segmented_control("Significance level α", [0.01, 0.05, 0.10], default=0.05) or 0.05

    with right:
        if len(cat_vars) != k:
            st.info(f"Select {k} factor{'s' if k > 1 else ''} to run a {k}-way ANOVA.")
        else:
            try:
                table = run_anova(df, dep, cat_vars, interactions and k > 1)
            except Exception as e:  # sparse filtered data can leave a factor with one level
                st.error(f"ANOVA could not be fitted on this selection: {e}")
                st.stop()
            terms = table.drop(index="Residual")
            sig = terms[terms["PR(>F)"] < alpha]
            kpis([("Terms tested", str(len(terms)), f"{len(sig)} significant"),
                  ("Largest effect", LABELS.get(terms.eta_sq.idxmax(), terms.eta_sq.idxmax()),
                   f"η² = {terms.eta_sq.max():.3f}"),
                  ("Residual df", f"{int(table.loc['Residual', 'df']):,}", f"n = {len(df):,}")])
            verdict(len(sig) > 0,
                    f"{len(sig)} of {len(terms)} terms significant at α = {alpha}",
                    ("Significant: " + ", ".join(LABELS.get(t, t) for t in sig.index) + "." if len(sig)
                     else "No factor explains a significant share of the variance.")
                    + " η² is the share of total variance each term explains.")
            st.write("")
            with st.container(border=True):
                html('<div class="chart-title">ANOVA table (Type II)</div>')
                fmts = {"sum_sq": "{:,.2f}", "df": "{:,.0f}", "F": "{:,.2f}", "PR(>F)": pfmt, "eta_sq": "{:.3f}"}
                shown = pd.DataFrame({c: table[c].map(lambda v, f=f: "" if pd.isna(v) else
                                                      (f(v) if callable(f) else f.format(v)))
                                      for c, f in fmts.items()})
                shown.index = [LABELS.get(i, i) for i in table.index]
                st.dataframe(shown.rename(columns={"sum_sq": "Sum Sq", "PR(>F)": "p-value", "eta_sq": "η²"}))
                if interactions:
                    st.caption("Sparse factor combinations can make some interaction terms inestimable; read them with care.")

            if k == 1:
                f = cat_vars[0]
                order = df[f].value_counts().index[:15].tolist()
                c1, c2 = st.columns(2)
                with c1, st.container(border=True):
                    html(f'<div class="chart-title">{esc(LABELS[dep])} by {esc(LABELS[f])}</div>'
                         f'<div class="chart-sub">{"Top 15 groups by count · " if df[f].nunique() > 15 else ""}box = middle 50%</div>')
                    sub = df[df[f].isin(order)].assign(G=lambda d: d[f].astype(str))
                    color = (alt.Color("G:N", scale=BODY_SCALE, legend=None) if f == "Body_Type"
                             else alt.value(ACCENT))
                    chart(alt.Chart(sub).mark_boxplot(size=18, median={"color": TEXT}, rule={"color": MUTED},
                                                      outliers={"opacity": 0.3, "size": 6}).encode(
                        y=alt.Y("G:N", title=None, sort=[str(o) for o in order]),
                        x=alt.X(f"{dep}:Q", title=LABELS[dep]), color=color), 60 + 34 * len(order))
                with c2, st.container(border=True):
                    html('<div class="chart-title">Tukey HSD post-hoc</div>'
                         '<div class="chart-sub">Which pairs of groups actually differ</div>')
                    if df[f].nunique() > 15:
                        st.caption("Too many groups for a readable pairwise table; pick a factor with ≤ 15 levels.")
                    else:
                        tk = run_tukey(df, dep, f, alpha)[["group1", "group2", "meandiff", "p-adj", "reject"]]
                        st.dataframe(tk, hide_index=True, height=min(420, 40 + 35 * len(tk)), column_config={
                            "group1": "Group A", "group2": "Group B", "meandiff": "Mean diff",
                            "p-adj": st.column_config.NumberColumn("Adj. p", format="%.4f"),
                            "reject": st.column_config.CheckboxColumn("Differs?")})

# =========================
# MANOVA
# =========================
elif page == "MANOVA":
    page_head("Multivariate analysis", "MANOVA", "Do groups differ on several numeric variables at once?")
    left, right = st.columns([1, 2.2], gap="large")
    with left, st.container(border=True):
        dep_vars = st.multiselect("Dependent variables (2+)", NUMERIC_COLS, default=["Price", "Power", "Mileage"],
                                  format_func=LABELS.get)
        cat = st.selectbox("Grouping factor", CATEGORICAL_COLS, format_func=LABELS.get)
        alpha = st.segmented_control("Significance level α", [0.01, 0.05, 0.10], default=0.05) or 0.05

    with right:
        try:
            res = run_manova(df, dep_vars, cat)
        except ValueError as e:
            st.info(str(e))
            st.stop()
        except Exception as e:  # singular covariance, e.g. Year and Car_Age together
            st.error(f"MANOVA could not be fitted: {e}. Avoid perfectly related variables such as Year with Car age.")
            st.stop()
        pillai = res.loc["Pillai's trace"]
        p = float(pillai["Pr > F"])
        kpis([("Pillai's trace", f"{float(pillai['Value']):.3f}", "most robust statistic"),
              ("F value", f"{float(pillai['F Value']):.2f}", f"df {float(pillai['Num DF']):.0f}, {float(pillai['Den DF']):.0f}"),
              ("p-value", pfmt(p), f"α = {alpha}"),
              ("Groups", str(df[cat].nunique()), LABELS[cat])])
        verdict(p < alpha, "Reject H0" if p < alpha else "Fail to Reject H0",
                f"{LABELS[cat]} groups " + ("differ" if p < alpha else "do not differ significantly")
                + " on the combined profile of " + ", ".join(LABELS[v].lower() for v in dep_vars) + ".")
        st.write("")
        with st.container(border=True):
            html('<div class="chart-title">Multivariate test statistics</div>')
            st.dataframe(res.astype(float).style.format({"Value": "{:.4f}", "Num DF": "{:.0f}", "Den DF": "{:,.1f}",
                                                          "F Value": "{:.2f}", "Pr > F": pfmt}))
        with st.container(border=True):
            html('<div class="chart-title">Group profiles</div>'
                 '<div class="chart-sub">Group mean as a z-score of the overall mean · orange = above average, '
                 'blue = below · hover for actual means</div>')
            top = df[cat].value_counts().index[:15]
            means = df[df[cat].isin(top)].groupby(cat)[dep_vars].mean()
            z = (means - df[dep_vars].mean()) / df[dep_vars].std()
            long = (z.reset_index().melt(id_vars=cat, var_name="Variable", value_name="z")
                    .merge(means.reset_index().melt(id_vars=cat, var_name="Variable", value_name="Mean")))
            long["Variable"] = long["Variable"].map(LABELS)
            long[cat] = long[cat].astype(str)
            m = max(float(long.z.abs().max()), 0.1)
            chart(alt.Chart(long).mark_rect(cornerRadius=4, stroke="#252B32", strokeWidth=2).encode(
                x=alt.X("Variable:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                y=alt.Y(f"{cat}:N", title=None, sort=[str(t) for t in top]),
                color=alt.Color("z:Q", scale=alt.Scale(domain=[-m, 0, m], range=["#4A8CD0", "#3A424C", "#E0682A"],
                                                  interpolate="lab"),
                                title="z-score"),
                tooltip=[alt.Tooltip(f"{cat}:N", title=LABELS[cat]), "Variable",
                         alt.Tooltip("Mean:Q", format=",.2f"), alt.Tooltip("z:Q", format=".2f")],
            ), 50 + 44 * len(top))
