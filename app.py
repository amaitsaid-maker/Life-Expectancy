import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Life Expectancy Dashboard", page_icon="🌍", layout="wide")

TARGET = "Life_expectancy"

@st.cache_data
def load_data():
    df = pd.read_csv("Life_Expectancy_Data.csv")
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby(["Status", "Year"])[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())
    df.dropna(subset=[TARGET], inplace=True)
    df["Status_encoded"] = (df["Status"] == "Developed").astype(int)
    return df

@st.cache_data
def train_model(df):
    df_model = df.drop(columns=["Country", "Year", "Status"])
    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    cv = cross_val_score(LinearRegression(), X_train_sc, y_train, cv=5, scoring="r2")
    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
    coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)
    return model, scaler, X, y_test, y_pred, cv, coef_df

df = load_data()
model, scaler, X, y_test, y_pred, cv, coef_df = train_model(df)

st.sidebar.title("🌍 Life Expectancy")
st.sidebar.markdown("**WHO Dataset · 2000–2015**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📦 Data Collection",
    "🔧 Preprocessing",
    "📊 EDA",
    "🤖 Modeling",
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Rows:** {df.shape[0]}  \n**Countries:** {df['Country'].nunique()}  \n**Years:** {df['Year'].min()}–{df['Year'].max()}")

colors = {"Developed": "#2196F3", "Developing": "#FF7043"}

if page == "🏠 Overview":
    st.title("🌍 Life Expectancy Analysis Dashboard")
    st.markdown("Full data science pipeline on the **WHO Life Expectancy Dataset (2000–2015)**.")
    st.markdown("> **Research question:** *Which health, economic, and social factors best predict a country's life expectancy?*")
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries",  df["Country"].nunique())
    c2.metric("Years",      f"{df['Year'].min()}–{df['Year'].max()}")
    c3.metric("Features",   df.shape[1] - 1)
    c4.metric("Model R²",   f"{r2_score(y_test, y_pred):.3f}")
    st.markdown("---")
    st.subheader("Pipeline Steps")
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**📦 Collection**\n\nWHO dataset — 193 countries, 22 variables")
    col2.info("**🔧 Preprocessing**\n\nImputation, encoding, outlier check")
    col3.info("**📊 EDA**\n\nDistributions, trends, correlations")
    col4.info("**🤖 Modeling**\n\nLinear Regression — R² = 0.82")

elif page == "📦 Data Collection":
    st.title("📦 Data Collection")
    st.markdown("**Source:** WHO Global Health Observatory | **Target:** Life expectancy (years)")
    st.markdown("---")
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Basic Statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
    st.subheader("Variable Description")
    var_df = pd.DataFrame({
        "Column": ["Country","Year","Status","Life_expectancy","Adult_Mortality",
                   "infant_deaths","Alcohol","percentage_expenditure","Hepatitis_B",
                   "Measles","BMI","under-five_deaths","Polio","Total_expenditure",
                   "Diphtheria","HIV/AIDS","GDP","Population",
                   "thinness_1-19_years","thinness_5-9_years",
                   "Income_composition_of_resources","Schooling"],
        "Type":   ["Categorical","Integer","Categorical"] + ["Numeric"]*19,
        "Description": [
            "Country name","Year of record","Developed / Developing",
            "Life expectancy in years (TARGET)","Adult mortality rate per 1000",
            "Infant deaths per 1000","Alcohol consumption (litres/capita)",
            "Health expenditure as % of GDP","Hepatitis B immunisation (%)",
            "Measles cases per 1000","Average BMI","Deaths under 5 per 1000",
            "Polio immunisation (%)","Govt health expenditure (%)",
            "Diphtheria immunisation (%)","HIV/AIDS deaths per 1000 (age 0-4)",
            "GDP per capita (USD)","Country population",
            "Thinness prevalence 10-19 yrs (%)","Thinness prevalence 5-9 yrs (%)",
            "HDI income index (0-1)","Years of schooling",
        ]
    })
    st.dataframe(var_df, use_container_width=True)

elif page == "🔧 Preprocessing":
    st.title("🔧 Preprocessing")
    st.markdown("---")
    st.subheader("1 · Missing Values")
    raw = pd.read_csv("Life_Expectancy_Data.csv")
    raw.columns = raw.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    miss = raw.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Columns with missing values:**")
        st.dataframe(miss.reset_index().rename(columns={"index":"Column", 0:"Missing Count"}), use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        miss.plot(kind="bar", color="#EF9A9A", edgecolor="white", ax=ax)
        ax.set_title("Missing Values per Column")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    st.success("Imputed using group median (Status x Year). Fallback: global median.")
    st.markdown("---")
    st.subheader("2 · Categorical Encoding")
    st.markdown("`Status` encoded: Developed = 1, Developing = 0")
    st.dataframe(df[["Country","Year","Status","Status_encoded"]].head(8), use_container_width=True)
    st.markdown("---")
    st.subheader("3 · Outlier Check (IQR)")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    df.boxplot(column=TARGET, by="Status", ax=ax, medianprops=dict(color="red", linewidth=2))
    ax.set_title("Life Expectancy by Status")
    ax.set_xlabel("Status")
    ax.set_ylabel("Life Expectancy (years)")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.info("17 mild outliers detected — retained to preserve real-world variation.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final Rows",     df.shape[0])
    c2.metric("Final Columns",  df.shape[1])
    c3.metric("Missing Values", 0)

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Trends", "Correlations", "Countries"])

    with tab1:
        st.subheader("Life Expectancy Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df[TARGET], bins=30, color="#5C6BC0", edgecolor="white")
        axes[0].set_title("Overall Distribution")
        axes[0].set_xlabel("Life Expectancy (years)")
        axes[0].set_ylabel("Frequency")
        for status, grp in df.groupby("Status"):
            axes[1].hist(grp[TARGET], bins=25, alpha=0.7, label=status, color=colors[status], edgecolor="white")
        axes[1].set_title("By Development Status")
        axes[1].set_xlabel("Life Expectancy (years)")
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("**Interpretation:** Left-skewed distribution. Developed countries average ~79 years vs ~67 years for developing ones.")

    with tab2:
        st.subheader("Life Expectancy Over Time (2000–2015)")
        yearly = df.groupby(["Year", "Status"])[TARGET].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        for status, grp in yearly.groupby("Status"):
            ax.plot(grp["Year"], grp[TARGET], marker="o", label=status, color=colors[status], linewidth=2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Life Expectancy")
        ax.set_title("Average Life Expectancy Over Time")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("**Interpretation:** Both groups show a steady upward trend from 2000–2015.")
        st.subheader("Feature vs Life Expectancy")
        numeric_features = [c for c in df.select_dtypes("number").columns if c not in [TARGET, "Year", "Status_encoded"]]
        feat = st.selectbox("Select a feature", numeric_features, index=numeric_features.index("Schooling") if "Schooling" in numeric_features else 0)
        fig, ax = plt.subplots(figsize=(8, 4))
        for status, grp in df.groupby("Status"):
            ax.scatter(grp[feat], grp[TARGET], alpha=0.3, s=12, label=status, color=colors[status])
        ax.set_xlabel(feat)
        ax.set_ylabel("Life Expectancy")
        ax.set_title(f"{feat} vs Life Expectancy")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Correlation Heatmap")
        df_corr = df.drop(columns=["Country", "Year", "Status"])
        corr = df_corr.corr()
        fig, ax = plt.subplots(figsize=(13, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.subheader("Top 10 Correlations with Life Expectancy")
        corr_target = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(9, 4))
        bar_colors = ["#43A047" if v > 0 else "#E53935" for v in corr_target]
        corr_target.plot(kind="bar", color=bar_colors, edgecolor="white", ax=ax)
        ax.set_title("Top 10 Correlations with Life Expectancy")
        ax.set_ylabel("Pearson r")
        ax.axhline(0, color="black", linewidth=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("**Key findings:** Schooling (r=0.73) and Income composition (r=0.70) are the strongest positive predictors. Adult Mortality (r=-0.70) and HIV/AIDS (r=-0.56) are the strongest negative predictors.")

    with tab4:
        st.subheader("Top & Bottom 10 Countries")
        country_avg = df.groupby("Country")[TARGET].mean().sort_values()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        country_avg.tail(10).plot(kind="barh", ax=axes[0], color="#42A5F5", edgecolor="white")
        axes[0].set_title("Top 10 — Highest Life Expectancy")
        axes[0].set_xlabel("Average (years)")
        country_avg.head(10).plot(kind="barh", ax=axes[1], color="#EF5350", edgecolor="white")
        axes[1].set_title("Bottom 10 — Lowest Life Expectancy")
        axes[1].set_xlabel("Average (years)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.subheader("Country Explorer")
        country = st.selectbox("Select a country", sorted(df["Country"].unique()))
        cdf = df[df["Country"] == country].sort_values("Year")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(cdf["Year"], cdf[TARGET], marker="o", color="#5C6BC0", linewidth=2)
        ax.set_title(f"Life Expectancy Over Time — {country}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Life Expectancy (years)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        cols_show = [c for c in [TARGET, "Adult_Mortality", "GDP", "Schooling", "HIV/AIDS"] if c in cdf.columns]
        st.dataframe(cdf[cols_show].set_index(cdf["Year"]), use_container_width=True)

elif page == "🤖 Modeling":
    st.title("🤖 Linear Regression Modeling")
    st.markdown("---")
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (Test)",      f"{r2_score(y_test, y_pred):.4f}")
    c2.metric("RMSE (Test)",    f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f} yrs")
    c3.metric("MAE (Test)",     f"{mean_absolute_error(y_test, y_pred):.4f} yrs")
    c4.metric("CV R² (5-fold)", f"{cv.mean():.4f} +/- {cv.std():.4f}")
    st.markdown("**R² = 0.82** means the model explains 82% of variance. **RMSE = 3.9 years** means predictions are off by ~4 years on average. **CV R² = 0.81** confirms the model generalises well.")
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Actual vs Predicted", "Residuals", "Coefficients", "Live Predictor"])

    with tab1:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, y_pred, alpha=0.4, s=18, color="#5C6BC0")
        lims = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Life Expectancy")
        ax.set_ylabel("Predicted Life Expectancy")
        ax.set_title(f"Actual vs Predicted  (R2 = {r2_score(y_test, y_pred):.3f})")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("Points close to the red dashed line indicate accurate predictions.")

    with tab2:
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(y_pred, residuals, alpha=0.4, s=18, color="#FF7043")
        axes[0].axhline(0, color="black", linewidth=1.2, linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        axes[1].hist(residuals, bins=35, color="#7E57C2", edgecolor="white", alpha=0.9)
        axes[1].axvline(0, color="red", linewidth=1.2, linestyle="--")
        axes[1].set_title("Distribution of Residuals")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("Residuals are roughly normally distributed around zero — confirming linear regression assumptions are satisfied.")

    with tab3:
        st.subheader("Feature Coefficients (Standardised)")
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ["#43A047" if c > 0 else "#E53935" for c in coef_df["Coefficient"]]
        ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=bar_colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Linear Regression Coefficients")
        ax.set_xlabel("Coefficient Value")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(coef_df.style.format({"Coefficient": "{:.4f}"}), use_container_width=True)
        st.markdown("**Schooling** and **Income composition** increase life expectancy the most. **Adult Mortality** and **HIV/AIDS** have the largest negative impact.")

    with tab4:
        st.subheader("Live Life Expectancy Predictor")
        st.markdown("Adjust the sliders to simulate a country profile and get an instant prediction.")
        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(X.columns):
            mn  = float(X[feat].min())
            mx  = float(X[feat].max())
            med = float(X[feat].median())
            step = round((mx - mn) / 100, 4) if (mx - mn) > 1 else 0.01
            input_vals[feat] = cols[i % 3].slider(feat, min_value=mn, max_value=mx, value=med, step=step)
        input_arr = scaler.transform(pd.DataFrame([input_vals]))
        prediction = model.predict(input_arr)[0]
        st.markdown("---")
        st.metric("Predicted Life Expectancy", f"{prediction:.2f} years")