
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models"

DATA_PATH = (
    ROOT / "data" /
    "customer_churn_dataset_80000_records.csv"
)


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #14243a, #245b82);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 5px;
}

.hero p {
    color: #dce8f3;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e7ebf2;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. LOAD SAVED MODELS
# =========================================================

@st.cache_resource
def load_models():

    filenames = {
        "rf_model": "rf_model.pkl",
        "dt_model": "DT_model.pkl",
        "dbscan_model": "dbscan_model.pkl",
        "scaler": "scaler.pkl",
        "minmax_scaler": "MinMax_scaler.pkl",
        "lower_bound": "lower_bound.pkl",
        "upper_bound": "upper_bound.pkl"
    }

    models = {}

    for key, filename in filenames.items():

        file_path = MODEL_PATH / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {filename}"
            )

        models[key] = joblib.load(file_path)

    return models


@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


try:

    models = load_models()
    df = load_dataset()

except Exception as error:

    st.error(f"Application loading error: {error}")

    st.info(
        "Check your data folder and models folder. "
        "Also verify that all required files are present."
    )

    st.stop()


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    "<h1 style='color:#245b82;'>📊 ChurnIQ</h1>",
    unsafe_allow_html=True
)

st.sidebar.caption("CUSTOMER INTELLIGENCE PLATFORM")

st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Churn Prediction",
        "Customer Segmentation",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.success("Models loaded successfully")

st.sidebar.caption(
    "Customer Churn Prediction & Segmentation"
)


# =========================================================
# 6. APPLICATION HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <h1>Customer Churn Intelligence</h1>

    <p>
        Analyze customer behaviour, predict customer churn,
        and discover meaningful customer segments using
        Machine Learning.
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 7. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader("Business Overview")

    dashboard_df = df.copy()

    # Keep one record per CustomerID
    if "CustomerID" in dashboard_df.columns:

        dashboard_df = dashboard_df.drop_duplicates(
            subset=["CustomerID"],
            keep="first"
        )

    if "Churn" not in dashboard_df.columns:

        st.error("Churn column not found in dataset.")
        st.stop()

    total_customers = len(dashboard_df)

    churned = int(
        (dashboard_df["Churn"] == 1).sum()
    )

    retained = int(
        (dashboard_df["Churn"] == 0).sum()
    )

    churn_rate = (
        churned / total_customers * 100
        if total_customers else 0
    )

    # KPI Cards

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    c2.metric(
        "Churned Customers",
        f"{churned:,}"
    )

    c3.metric(
        "Retained Customers",
        f"{retained:,}"
    )

    c4.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.divider()

    # Churn Distribution

    left, right = st.columns(2)

    with left:

        st.markdown("### Churn Distribution")

        churn_data = pd.Series(
            {
                "Retained": retained,
                "Churned": churned
            }
        )

        st.bar_chart(churn_data)

    # Subscription Analysis

    with right:

        st.markdown("### Subscription Analysis")

        if "Subscription Type" in dashboard_df.columns:

            subscription_data = pd.crosstab(
                dashboard_df["Subscription Type"],
                dashboard_df["Churn"].map({
                    0: "Retained",
                    1: "Churned"
                })
            )

            st.bar_chart(subscription_data)

    st.divider()

    left, right = st.columns(2)

    # Contract Analysis

    with left:

        st.markdown("### Contract Analysis")

        if "Contract Length" in dashboard_df.columns:

            contract_data = pd.crosstab(
                dashboard_df["Contract Length"],
                dashboard_df["Churn"].map({
                    0: "Retained",
                    1: "Churned"
                })
            )

            st.bar_chart(contract_data)

    # Gender Analysis

    with right:

        st.markdown("### Gender Distribution")

        if "Gender" in dashboard_df.columns:

            gender_data = dashboard_df[
                "Gender"
            ].value_counts()

            st.bar_chart(gender_data)

    st.divider()

    st.markdown("### Customer Dataset")

    st.dataframe(
        dashboard_df.head(100),
        use_container_width=True
    )

    st.download_button(
        label="Download Customer Dataset",
        data=dashboard_df.to_csv(index=False),
        file_name="customer_churn_data.csv",
        mime="text/csv"
    )


# =========================================================
# 8. CUSTOMER CHURN PREDICTION
# =========================================================

elif page == "Churn Prediction":

    st.subheader("Customer Churn Prediction")

    st.write(
        "Enter the customer information to predict "
        "whether the customer is likely to churn."
    )

    st.info(
        "Prediction is performed using your trained "
        "Random Forest model."
    )

    with st.form("customer_prediction_form"):

        st.markdown("### Customer Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Unknown"]
            )

            tenure = st.number_input(
                "Tenure",
                min_value=0,
                value=12
            )

            usage = st.number_input(
                "Usage Frequency",
                min_value=0,
                value=10
            )

            support_calls = st.number_input(
                "Support Calls",
                min_value=0,
                value=2
            )

        with col2:

            subscription = st.selectbox(
                "Subscription Type",
                ["Basic", "Premium", "Standard", "Unknown"]
            )

            contract = st.selectbox(
                "Contract Length",
                ["Annual", "Monthly", "Quarterly"]
            )

            payment_delay = st.number_input(
                "Payment Delay",
                min_value=0,
                value=5
            )

            total_spend = st.number_input(
                "Total Spend",
                min_value=0.0,
                value=500.0
            )

            last_interaction = st.number_input(
                "Last Interaction",
                min_value=0,
                value=10
            )

        submitted = st.form_submit_button(
            "🔍 Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        try:

            # -----------------------------------------
            # NUMERICAL FEATURES
            # -----------------------------------------

            numerical_data = {
                "Age": age,
                "Tenure": tenure,
                "Usage Frequency": usage,
                "Support Calls": support_calls,
                "Payment Delay": payment_delay,
                "Total Spend": total_spend,
                "Last Interaction": last_interaction
            }

            customer = pd.DataFrame(
                [numerical_data]
            )

            # Apply training IQR bounds

            lower = models["lower_bound"]
            upper = models["upper_bound"]

            for column in customer.columns:

                if column in lower.index and column in upper.index:

                    customer[column] = customer[column].clip(
                        lower=lower[column],
                        upper=upper[column]
                    )

            # -----------------------------------------
            # MANUAL ONE-HOT ENCODING
            # -----------------------------------------

            customer["Gender_Female"] = int(
                gender == "Female"
            )

            customer["Gender_Male"] = int(
                gender == "Male"
            )

            customer["Gender_nan"] = int(
                gender == "Unknown"
            )

            customer["Subscription Type_Basic"] = int(
                subscription == "Basic"
            )

            customer["Subscription Type_Premium"] = int(
                subscription == "Premium"
            )

            customer["Subscription Type_Standard"] = int(
                subscription == "Standard"
            )

            customer["Subscription Type_nan"] = int(
                subscription == "Unknown"
            )

            customer["Contract Length_Annual"] = int(
                contract == "Annual"
            )

            customer["Contract Length_Monthly"] = int(
                contract == "Monthly"
            )

            customer["Contract Length_Quarterly"] = int(
                contract == "Quarterly"
            )

            # -----------------------------------------
            # MATCH MODEL FEATURE ORDER
            # -----------------------------------------

            model = models["rf_model"]

            expected_features = list(
                model.feature_names_in_
            )

            final_data = customer.reindex(
                columns=expected_features,
                fill_value=0
            )

            # -----------------------------------------
            # STANDARD SCALING
            # Scale only the original 15 scaler columns
            # -----------------------------------------

            scaler = models["scaler"]

            scale_columns = list(
                scaler.feature_names_in_
            )

            final_data.loc[:, scale_columns] = (
                scaler.transform(
                    final_data[scale_columns]
                )
            )

            # -----------------------------------------
            # RANDOM FOREST PREDICTION
            # -----------------------------------------

            prediction = model.predict(
                final_data
            )[0]

            probabilities = model.predict_proba(
                final_data
            )[0]

            churn_index = list(
                model.classes_
            ).index(1)

            churn_probability = (
                probabilities[churn_index] * 100
            )

            # -----------------------------------------
            # DISPLAY RESULT
            # -----------------------------------------

            st.divider()

            st.markdown("### Prediction Result")

            result_col, probability_col = st.columns(2)

            with result_col:

                if prediction == 1:

                    st.error(
                        "⚠️ Customer is predicted to CHURN"
                    )

                else:

                    st.success(
                        "✅ Customer is predicted to STAY"
                    )

            with probability_col:

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2f}%"
                )

            st.progress(
                int(round(churn_probability))
            )

            st.caption(
                "This is a model-estimated probability, "
                "not a guarantee of future customer behaviour."
            )

            with st.expander("View processed customer features"):

                st.dataframe(
                    final_data,
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# =========================================================
# 9. CUSTOMER SEGMENTATION USING DBSCAN
# =========================================================

elif page == "Customer Segmentation":

    st.subheader("Customer Segmentation")

    st.write(
        "Explore customer groups identified by "
        "the DBSCAN clustering algorithm."
    )

    st.info(
        "DBSCAN is an unsupervised algorithm. "
        "Noise points are represented by cluster label -1."
    )

    # -----------------------------------------
    # PREPARE DATA AS IN THE DBSCAN NOTEBOOK
    # -----------------------------------------

    segment_source = df.copy()

    # Remove duplicate CustomerIDs only
    if "CustomerID" in segment_source.columns:

        segment_source = segment_source.drop_duplicates(
            subset=["CustomerID"],
            keep="first"
        )

    customer_details = segment_source.reset_index(
        drop=True
    ).copy()

    # Remove ID and target from clustering features

    db_features = customer_details.drop(
        columns=["CustomerID", "Churn"],
        errors="ignore"
    ).copy()

    # -----------------------------------------
    # IMPUTE MISSING VALUES
    # Same approach as the DBSCAN notebook
    # -----------------------------------------

    categorical_columns = db_features.select_dtypes(
        include=["object"]
    ).columns

    numerical_columns = db_features.select_dtypes(
        include=["number"]
    ).columns

    for column in categorical_columns:

        mode_values = db_features[column].mode()

        if not mode_values.empty:

            db_features[column] = (
                db_features[column].fillna(
                    mode_values.iloc[0]
                )
            )

    for column in numerical_columns:

        db_features[column] = (
            db_features[column].fillna(
                db_features[column].median()
            )
        )

    # -----------------------------------------
    # ONE-HOT ENCODING
    # -----------------------------------------

    db_features = pd.get_dummies(
        db_features,
        columns=list(categorical_columns),
        dtype="int"
    )

    # Match original DBSCAN feature order

    db_scaler = models["minmax_scaler"]

    db_columns = list(
        db_scaler.feature_names_in_
    )

    db_features = db_features.reindex(
        columns=db_columns,
        fill_value=0
    )

    # -----------------------------------------
    # MINMAX SCALING
    # -----------------------------------------

    db_scaled = db_scaler.transform(
        db_features
    )

    # -----------------------------------------
    # USE SAVED DBSCAN LABELS
    # -----------------------------------------

    dbscan_model = models["dbscan_model"]

    labels = dbscan_model.labels_

    if len(labels) != len(customer_details):

        st.error(
            "The saved DBSCAN model was fitted on a different "
            "number of customers than the current dataset. "
            "Please regenerate dbscan_model.pkl after updating "
            "your DBSCAN preprocessing."
        )

        st.stop()

    customer_details["Cluster"] = labels

    # -----------------------------------------
    # CLUSTER METRICS
    # -----------------------------------------

    number_of_clusters = len(
        set(labels) - {-1}
    )

    noise_points = int(
        np.sum(labels == -1)
    )

    cluster_col, noise_col, customer_col = st.columns(3)

    cluster_col.metric(
        "Clusters Identified",
        number_of_clusters
    )

    noise_col.metric(
        "Noise Points",
        f"{noise_points:,}"
    )

    customer_col.metric(
        "Total Customers",
        f"{len(labels):,}"
    )

    st.divider()

    # -----------------------------------------
    # CLUSTER DISTRIBUTION
    # -----------------------------------------

    st.markdown("### Cluster Distribution")

    cluster_counts = (
        pd.Series(labels, name="Cluster")
        .value_counts()
        .sort_index()
    )

    st.bar_chart(cluster_counts)

    # -----------------------------------------
    # CUSTOMER PROFILE
    # -----------------------------------------

    st.markdown("### Customer Cluster Profiles")

    if "Churn" in customer_details.columns:

        profile = customer_details.groupby(
            "Cluster"
        ).agg(
            Customer_Count=("Churn", "size"),
            Churn_Rate=("Churn", "mean")
        )

        profile["Churn_Rate"] = (
            profile["Churn_Rate"] * 100
        )

        st.dataframe(
            profile.round(2),
            use_container_width=True
        )

    else:

        profile = customer_details.groupby(
            "Cluster"
        ).size().to_frame("Customer_Count")

        st.dataframe(
            profile,
            use_container_width=True
        )

    # -----------------------------------------
    # PCA VISUALIZATION
    # -----------------------------------------

    st.markdown("### PCA Cluster Visualization")

    with st.spinner("Preparing PCA visualization..."):

        pca = PCA(n_components=2)

        pca_data = pca.fit_transform(
            db_scaled
        )

        pca_df = pd.DataFrame(
            pca_data,
            columns=["PCA_1", "PCA_2"]
        )

        pca_df["Cluster"] = labels

        # Sample points for a responsive chart
        plot_df = pca_df.sample(
            n=min(10000, len(pca_df)),
            random_state=78
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data=plot_df,
            x="PCA_1",
            y="PCA_2",
            hue="Cluster",
            palette="viridis",
            s=15,
            alpha=0.7,
            ax=ax
        )

        ax.set_title(
            "Customer Segmentation Using PCA and DBSCAN"
        )

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

        st.pyplot(fig)

        plt.close(fig)

        st.caption(
            "PCA is used here only to visualize the clusters. "
            "DBSCAN performs the actual clustering."
        )

    # -----------------------------------------
    # FILTER CUSTOMER CLUSTERS
    # -----------------------------------------

    st.markdown("### Explore Customer Segments")

    available_clusters = sorted(
        customer_details["Cluster"].unique()
    )

    selected_cluster = st.selectbox(
        "Choose a cluster",
        available_clusters
    )

    filtered_customers = customer_details[
        customer_details["Cluster"] == selected_cluster
    ]

    st.write(
        f"Customers in selected cluster: "
        f"{len(filtered_customers):,}"
    )

    st.dataframe(
        filtered_customers.head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Customer Segments",
        data=customer_details.to_csv(index=False),
        file_name="customer_segments.csv",
        mime="text/csv"
    )


# =========================================================
# 10. MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.subheader("Machine Learning Model Information")

    st.markdown("### Random Forest Classifier")

    rf = models["rf_model"]

    st.write({
        "Model": "Random Forest Classifier",
        "Number of Trees": rf.n_estimators,
        "Maximum Depth": rf.max_depth,
        "Criterion": rf.criterion,
        "Maximum Features": rf.max_features,
        "Bootstrap": rf.bootstrap,
        "Input Features": rf.n_features_in_
    })

    st.markdown("### Decision Tree Classifier")

    dt = models["dt_model"]

    st.write({
        "Model": "Decision Tree Classifier",
        "Maximum Depth": dt.max_depth,
        "Input Features": dt.n_features_in_
    })

    st.markdown("### DBSCAN Clustering")

    dbscan = models["dbscan_model"]

    st.write({
        "Model": "DBSCAN",
        "Epsilon": dbscan.eps,
        "Minimum Samples": dbscan.min_samples,
        "Clusters": len(
            set(dbscan.labels_) - {-1}
        ),
        "Noise Points": int(
            np.sum(dbscan.labels_ == -1)
        )
    })

    st.divider()

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Matplotlib and Seaborn
    - Streamlit
    - Joblib
    """)


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    <p>
    <b>ChurnIQ</b> |
    Customer Churn Prediction & Segmentation
    </p>

    <p>
    Built using Python, Machine Learning and Streamlit
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

