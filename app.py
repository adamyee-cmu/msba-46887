import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="NBA Stats & Predictor", layout="wide")
st.title("NBA Stats & Interactive Predictor")
st.caption("Upload your NBA CSV file, explore visualizations, and train a regression model.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Requirement 1: Preview Data
    st.subheader("1. Data Preview")
    st.dataframe(df.head(10))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("This file needs at least 2 numeric columns for plotting and modeling.")
        st.stop()

    # Requirement 2: Chart Selector
    st.subheader("2. Data Visualization")
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox("Choose a column to plot:", all_cols)
    with col2:
        chart_type = st.selectbox("Choose a chart type:", ["Line", "Scatter", "Histogram"])

    if chart_type == "Line":
        fig = px.line(df, y=selected_col, title=f"Line Chart of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Scatter":
        with col1:
            x_axis = st.selectbox("Choose X-axis column:", numeric_cols)
        fig = px.scatter(df, x=x_axis, y=selected_col, title=f"Scatter Plot: {selected_col} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=selected_col, title=f"Histogram of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

    # Requirement 3: Linear Regression & Real-time Prediction
    st.subheader("3. Linear Regression Predictor")

    model_col1, model_col2 = st.columns(2)
    with model_col1:
        features = st.multiselect("Choose two input features (X):", numeric_cols, max_selections=2)
        target = st.selectbox("Choose target metric (Y):", [c for c in numeric_cols if c not in features])

    if len(features) == 2 and target:
        X = df[features]
        y = df[target]

        # Train Model
        model = LinearRegression()
        model.fit(X, y)
        r2_score = model.score(X, y)

        st.write(f"### Fitted Regression Plot (R² Score: {r2_score:.2f})")

        # Show 2D scatter plot with fitted regression line (with robust exception tracing)
        try:
            # Ensure clean numeric columns with no NaN values for trendline fitting
            plot_df = df[[features[0], target, features[1]]].dropna()
            
            fig_fit = px.scatter(
                plot_df,
                x=features[0],
                y=target,
                trendline="ols",
                trendline_color_override="red",
                title=f"Fitted Line: {target} vs {features[0]} (controlling for {features[1]})"
            )
            st.plotly_chart(fig_fit, use_container_width=True)
        except Exception as plot_err:
            st.error(f"Plotly failed to generate the regression trendline plot. Error: {plot_err}")
            st.warning("This usually happens if statsmodels is not fully loaded, or if there are invalid inputs in the columns.")

        # Prediction module underneath the plot
        st.write("### Real-time Prediction Inputs")
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            val1 = st.number_input(f"Enter value for {features[0]}:", value=float(df[features[0]].mean()))
        with pred_col2:
            val2 = st.number_input(f"Enter value for {features[1]}:", value=float(df[features[1]].mean()))

        pred_df = pd.DataFrame([[val1, val2]], columns=features)
        prediction = model.predict(pred_df)[0]

        st.metric(label=f"Predicted {target}", value=f"{prediction:.2f}")
