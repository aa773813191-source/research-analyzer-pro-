import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from io import BytesIO

st.set_page_config(
    page_title="Research Analyzer Pro",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Research Analyzer Pro")
st.subheader("By Amatalelah")
st.write("Professional Statistical Analysis Platform")

st.divider()

uploaded_file = st.file_uploader(
    "Upload your research data",
    type=["csv", "xlsx"]
)

if uploaded_file:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Data uploaded successfully!")

        st.header("1. Data Preview")
        st.dataframe(df, use_container_width=True)

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        st.header("2. Data Quality")

        quality = pd.DataFrame({
            "Variable": df.columns,
            "Missing": df.isnull().sum().values,
            "Unique": df.nunique().values,
            "Data Type": df.dtypes.astype(str).values
        })

        st.dataframe(quality)

        st.header("3. Descriptive Statistics")

        if numeric_cols:
            descriptive = df[numeric_cols].describe().T
            descriptive["Median"] = df[numeric_cols].median()
            descriptive["Variance"] = df[numeric_cols].var()
            descriptive["Skewness"] = df[numeric_cols].skew()

            st.dataframe(descriptive)

        st.header("4. Data Visualization")

        if numeric_cols:

            variable = st.selectbox(
                "Select Numeric Variable",
                numeric_cols
            )

            chart = st.selectbox(
                "Choose Chart",
                ["Histogram", "Boxplot"]
            )

            fig, ax = plt.subplots()

            if chart == "Histogram":
                ax.hist(df[variable].dropna(), bins=15)
                ax.set_title("Histogram")

            else:
                ax.boxplot(df[variable].dropna())
                ax.set_title("Boxplot")

            st.pyplot(fig)
            plt.close(fig)

        st.header("5. Statistical Analysis")

        if numeric_cols:

            test = st.selectbox(
                "Select Statistical Test",
                [
                    "Independent t-test",
                    "Paired t-test",
                    "One-Way ANOVA",
                    "Mann-Whitney U",
                    "Kruskal-Wallis",
                    "Shapiro-Wilk",
                    "Pearson Correlation",
                    "Spearman Correlation"
                ]
            )

            if test == "Independent t-test":

                value = st.selectbox(
                    "Numeric Variable",
                    numeric_cols
                )

                groups = categorical_cols

                if groups:
                    group = st.selectbox(
                        "Grouping Variable",
                        groups
                    )

                    group_values = df[group].dropna().unique()

                    if len(group_values) == 2:
                        if st.button("Run Analysis"):
                            a = df.loc[
                                df[group] == group_values[0],
                                value
                            ].dropna()

                            b = df.loc[
                                df[group] == group_values[1],
                                value
                            ].dropna()

                            result = stats.ttest_ind(
                                a, b, equal_var=False
                            )

                            st.metric(
                                "Test Statistic",
                                f"{result.statistic:.5f}"
                            )

                            st.metric(
                                "P-value",
                                f"{result.pvalue:.5f}"
                            )

                    else:
                        st.warning(
                            "Select a grouping variable with two groups."
                        )

            elif test == "Paired t-test":

                col1 = st.selectbox(
                    "First Measurement",
                    numeric_cols
                )

                col2 = st.selectbox(
                    "Second Measurement",
                    [c for c in numeric_cols if c != col1]
                )

                if st.button("Run Analysis"):
                    data = df[[col1, col2]].dropna()

                    result = stats.ttest_rel(
                        data[col1], data[col2]
                    )

                    st.metric(
                        "Test Statistic",
                        f"{result.statistic:.5f}"
                    )

                    st.metric(
                        "P-value",
                        f"{result.pvalue:.5f}"
                    )

            elif test == "One-Way ANOVA":

                value = st.selectbox(
                    "Numeric Variable",
                    numeric_cols
                )

                if categorical_cols:
                    group = st.selectbox(
                        "Grouping Variable",
                        categorical_cols
                    )

                    if st.button("Run Analysis"):
                        samples = [
                            x[value].dropna().values
                            for _, x in df.groupby(group)
                        ]

                        samples = [
                            x for x in samples if len(x) > 0
                        ]

                        if len(samples) >= 2:
                            result = stats.f_oneway(*samples)

                            st.metric(
                                "F Statistic",
                                f"{result.statistic:.5f}"
                            )

                            st.metric(
                                "P-value",
                                f"{result.pvalue:.5f}"
                            )

            elif test == "Mann-Whitney U":

                value = st.selectbox(
                    "Numeric Variable",
                    numeric_cols
                )

                if categorical_cols:
                    group = st.selectbox(
                        "Grouping Variable",
                        categorical_cols
                    )

                    group_values = df[group].dropna().unique()

                    if len(group_values) == 2:
                        if st.button("Run Analysis"):
                            a = df.loc[
                                df[group] == group_values[0],
                                value
                            ].dropna()

                            b = df.loc[
                                df[group] == group_values[1],
                                value
                            ].dropna()

                            result = stats.mannwhitneyu(a, b)

                            st.metric(
                                "U Statistic",
                                f"{result.statistic:.5f}"
                            )

                            st.metric(
                                "P-value",
                                f"{result.pvalue:.5f}"
                            )

            elif test == "Kruskal-Wallis":

                value = st.selectbox(
                    "Numeric Variable",
                    numeric_cols
                )

                if categorical_cols:
                    group = st.selectbox(
                        "Grouping Variable",
                        categorical_cols
                    )

                    if st.button("Run Analysis"):
                        samples = [
                            x[value].dropna().values
                            for _, x in df.groupby(group)
                        ]

                        samples = [
                            x for x in samples if len(x) > 0
                        ]

                        if len(samples) >= 2:
                            result = stats.kruskal(*samples)

                            st.metric(
                                "H Statistic",
                                f"{result.statistic:.5f}"
                            )

                            st.metric(
                                "P-value",
                                f"{result.pvalue:.5f}"
                            )

            elif test == "Shapiro-Wilk":

                value = st.selectbox(
                    "Numeric Variable",
                    numeric_cols
                )

                if st.button("Run Analysis"):
                    values = df[value].dropna()

                    if 3 <= len(values) <= 5000:
                        result = stats.shapiro(values)

                        st.metric(
                            "W Statistic",
                            f"{result.statistic:.5f}"
                        )

                        st.metric(
                            "P-value",
                            f"{result.pvalue:.5f}"
                        )
                    else:
                        st.warning(
                            "Sample size must be between 3 and 5000."
                        )

            elif test in [
                "Pearson Correlation",
                "Spearman Correlation"
            ]:

                col1 = st.selectbox(
                    "First Variable",
                    numeric_cols
                )

                col2 = st.selectbox(
                    "Second Variable",
                    [c for c in numeric_cols if c != col1]
                )

                if st.button("Run Analysis"):
                    data = df[[col1, col2]].dropna()

                    if test == "Pearson Correlation":
                        result = stats.pearsonr(
                            data[col1], data[col2]
                        )
                    else:
                        result = stats.spearmanr(
                            data[col1], data[col2]
                        )

                    st.metric(
                        "Correlation Coefficient",
                        f"{result.statistic:.5f}"
                    )

                    st.metric(
                        "P-value",
                        f"{result.pvalue:.5f}"
                    )

        st.header("6. Download Results")

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Download CSV",
            csv_data,
            "research_data.csv",
            "text/csv",
            key="download_csv"
        )

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            df.to_excel(
                writer,
                sheet_name="Data",
                index=False
            )

            quality.to_excel(
                writer,
                sheet_name="Data_Quality",
                index=False
            )

            if numeric_cols:
                df[numeric_cols].describe().to_excel(
                    writer,
                    sheet_name="Descriptive"
                )

        st.download_button(
            "Download Complete Excel Report",
            excel_buffer.getvalue(),
            "Research_Analyzer_Pro.xlsx",
            key="download_excel"
        )

    except Exception as error:
        st.error(f"Error: {error}")

else:
    st.info("Upload a CSV or Excel file to start your analysis.")

st.divider()
st.caption("Research Analyzer Pro | By Amatalelah")
