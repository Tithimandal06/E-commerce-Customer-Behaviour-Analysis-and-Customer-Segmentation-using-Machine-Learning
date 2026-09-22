"""Interactive e-commerce customer behaviour and segmentation dashboard."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="E-commerce Customer Intelligence", page_icon="EC", layout="wide")
REQUIRED = ["CustomerID", "Age", "Gender", "PurchaseAmount", "Frequency", "Recency", "Category"]
NUMERIC = ["Age", "PurchaseAmount", "Frequency", "Recency"]

@st.cache_data
def load_data(source):
    data = pd.read_csv(source)
    missing = [column for column in REQUIRED if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = data[REQUIRED].copy()
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[NUMERIC].isna().any().any():
        raise ValueError("One or more behavioural columns contain non-numeric values.")
    return data.drop_duplicates().drop_duplicates(subset="CustomerID")

def add_rfm(data):
    result = data.copy()
    for column in ["Recency", "Frequency", "PurchaseAmount"]:
        result[column + "Score"] = pd.qcut(result[column].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    result["RecencyScore"] = 6 - result["RecencyScore"]
    result["RFMScore"] = result["RecencyScore"] + result["FrequencyScore"] + result["PurchaseAmountScore"]
    result["RFMSegment"] = np.select(
        [result["RFMScore"] >= 13, (result["FrequencyScore"] >= 4) & (result["PurchaseAmountScore"] >= 4),
         (result["RecencyScore"] >= 4) & (result["FrequencyScore"] >= 3), result["RecencyScore"] <= 2],
        ["High-Value Customers", "Loyal Customers", "Potential Loyal Customers", "At-Risk Customers"],
        default="Low-Engagement Customers")
    return result

def _cluster_names(profiles):
    names = {}
    for cluster, row in profiles.iterrows():
        if row["PurchaseAmount"] >= profiles["PurchaseAmount"].quantile(.75) and row["Frequency"] >= profiles["Frequency"].median():
            names[cluster] = "High-Value Loyal Customers"
        elif row["Recency"] >= profiles["Recency"].quantile(.75):
            names[cluster] = "At-Risk Customers"
        elif row["Frequency"] >= profiles["Frequency"].quantile(.75):
            names[cluster] = "Frequent Customers"
        else:
            names[cluster] = "Regular Customers"
    return pd.Series(names)

def add_clusters(data):
    result = data.copy()
    scaled = StandardScaler().fit_transform(result[NUMERIC])
    evaluations, models = [], {}
    for k in range(2, min(8, len(result) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        evaluations.append((k, silhouette_score(scaled, labels), model.inertia_))
        models[k] = (model, labels)
    evaluation = pd.DataFrame(evaluations, columns=["K", "Silhouette", "Inertia"])
    best_k = int(evaluation.loc[evaluation["Silhouette"].idxmax(), "K"])
    model, labels = models[best_k]
    result["Cluster"] = labels
    profiles = result.groupby("Cluster")[NUMERIC].mean()
    profiles["CustomerCount"] = result["Cluster"].value_counts().sort_index()
    profiles["DominantGender"] = result.groupby("Cluster")["Gender"].agg(lambda values: values.mode().iat[0])
    profiles["DominantCategory"] = result.groupby("Cluster")["Category"].agg(lambda values: values.mode().iat[0])
    profiles["ClusterName"] = _cluster_names(profiles)
    result["ClusterName"] = result["Cluster"].map(profiles["ClusterName"])
    pca = PCA(n_components=2, random_state=42).fit_transform(scaled)
    result["PCA1"], result["PCA2"] = pca[:, 0], pca[:, 1]
    return result, evaluation, profiles, best_k, model

def plot(data, kind, x, y=None, color=None, title=None):
    args = {"data_frame": data, "x": x, "title": title}
    if y: args["y"] = y
    if color: args["color"] = color
    figure = getattr(px, kind)(**args)
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=55, b=20), height=360)
    return figure

st.title("E-commerce Customer Intelligence")
st.caption("E-COMMERCE CUSTOMER BEHAVIOUR ANALYSIS AND CUSTOMER SEGMENTATION | Author: Tithi Mandal")
uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv", help="Upload a file with the seven required columns.")
source = uploaded if uploaded is not None else str(Path(__file__).with_name("customer_data.csv"))
try:
    customers = add_rfm(load_data(source))
    customers, evaluation, profiles, best_k, fitted_model = add_clusters(customers)
except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Executive Dashboard", "Behaviour Analysis", "RFM Analysis", "Customer Segmentation", "Customer Explorer", "Business Insights"])
if page == "Executive Dashboard":
    high_value = int((customers["PurchaseAmount"] >= customers["PurchaseAmount"].quantile(.75)).sum())
    metrics = [("Customers", len(customers)), ("Total Purchase", f"{customers.PurchaseAmount.sum():,.0f}"), ("Average Purchase", f"{customers.PurchaseAmount.mean():,.0f}"), ("Average Frequency", f"{customers.Frequency.mean():.1f}"), ("Average Recency", f"{customers.Recency.mean():.1f}"), ("High-Value", high_value)]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics): col.metric(label, value)
    left, right = st.columns(2)
    with left: st.plotly_chart(plot(customers, "histogram", "Age", title="Customer age distribution"), use_container_width=True)
    with right: st.plotly_chart(plot(customers, "bar", "Category", title="Customers by category"), use_container_width=True)
    st.plotly_chart(plot(customers, "scatter", "Frequency", "PurchaseAmount", "ClusterName", "Purchase amount and frequency by cluster"), use_container_width=True)
elif page == "Behaviour Analysis":
    genders = st.sidebar.multiselect("Gender", sorted(customers.Gender.unique()), default=sorted(customers.Gender.unique()))
    categories = st.sidebar.multiselect("Category", sorted(customers.Category.unique()), default=sorted(customers.Category.unique()))
    age = st.sidebar.slider("Age range", int(customers.Age.min()), int(customers.Age.max()), (int(customers.Age.min()), int(customers.Age.max())))
    filtered = customers[customers.Gender.isin(genders) & customers.Category.isin(categories) & customers.Age.between(*age)]
    cols = st.columns(4)
    values = [len(filtered), filtered.PurchaseAmount.mean(), filtered.Frequency.mean(), filtered.Recency.mean()]
    for col, label, value in zip(cols, ["Customers", "Avg purchase", "Avg frequency", "Avg recency"], values): col.metric(label, f"{value:,.1f}" if isinstance(value, float) else value)
    a, b = st.columns(2)
    with a: st.plotly_chart(plot(filtered, "box", "Gender", "PurchaseAmount", title="Purchase amount by gender"), use_container_width=True)
    with b: st.plotly_chart(plot(filtered, "box", "Category", "Frequency", title="Frequency by category"), use_container_width=True)
    st.dataframe(filtered[REQUIRED], use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", filtered[REQUIRED].to_csv(index=False), "filtered_customers.csv", "text/csv")
elif page == "RFM Analysis":
    st.subheader("RFM analysis")
    st.info("Recency is inverted for scoring: lower recency means more recent activity. Frequency and PurchaseAmount are rewarded when higher.")
    st.plotly_chart(plot(customers, "bar", "RFMSegment", title="RFM segment distribution"), use_container_width=True)
    selected = st.selectbox("Inspect RFM segment", ["All"] + sorted(customers.RFMSegment.unique().tolist()))
    view = customers if selected == "All" else customers[customers.RFMSegment == selected]
    st.dataframe(view[["CustomerID", "Recency", "Frequency", "PurchaseAmount", "RFMScore", "RFMSegment"]], use_container_width=True, hide_index=True)
elif page == "Customer Segmentation":
    left, right = st.columns(2)
    with left: st.metric("Selected K", best_k)
    with right: st.metric("Silhouette score", f"{evaluation.loc[evaluation.K == best_k, 'Silhouette'].iat[0]:.3f}")
    st.plotly_chart(plot(customers, "scatter", "Frequency", "PurchaseAmount", "ClusterName", "Frequency vs purchase amount"), use_container_width=True)
    st.plotly_chart(plot(customers, "scatter", "PCA1", "PCA2", "ClusterName", "PCA projection of customer clusters"), use_container_width=True)
    st.dataframe(profiles.reset_index(), use_container_width=True, hide_index=True)
    st.download_button("Download segmentation results", customers.to_csv(index=False), "segmentation_results.csv", "text/csv")
elif page == "Customer Explorer":
    customer_id = st.selectbox("CustomerID", customers.CustomerID.tolist())
    row = customers.loc[customers.CustomerID == customer_id].iloc[0]
    st.subheader(f"Customer {customer_id}")
    st.json({key: row[key] for key in ["CustomerID", "Age", "Gender", "Category", "PurchaseAmount", "Frequency", "Recency", "RFMSegment", "ClusterName"]})
else:
    top_cluster = profiles["PurchaseAmount"].idxmax()
    frequent_cluster = profiles["Frequency"].idxmax()
    popular_category = customers.Category.mode().iat[0]
    at_risk = int((customers.Recency >= customers.Recency.quantile(.75)).sum())
    st.subheader("Data-driven business insights")
    st.markdown(f"- **Highest spending group:** {profiles.loc[top_cluster, 'ClusterName']} averages {profiles.loc[top_cluster, 'PurchaseAmount']:,.0f} in purchase amount.")
    st.markdown(f"- **Most frequent group:** {profiles.loc[frequent_cluster, 'ClusterName']} averages {profiles.loc[frequent_cluster, 'Frequency']:.1f} purchases.")
    st.markdown(f"- **Category focus:** {popular_category} is the most common category ({(customers.Category == popular_category).sum()} customers).")
    st.markdown(f"- **Re-engagement pool:** {at_risk} customers are at or above the 75th percentile of recency ({customers.Recency.quantile(.75):.0f} days).")
    st.markdown("- These associations describe this dataset; they do not establish causal relationships or predict churn because no churn label or transaction timestamps are available.")