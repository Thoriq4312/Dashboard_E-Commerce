import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("📊 Dashboard Analisis E-Commerce Tahun 2017")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    orders_payments = pd.read_csv(os.path.join(BASE_DIR, "orders_payments.csv"))
    customers = pd.read_csv(os.path.join(BASE_DIR, "customers.csv"))
    url_reviews = "https://drive.google.com/uc?id=1k0X5iJDKfNKJTzErNK4UQpfiCtc9Xpaf"
    reviews_products_items = pd.read_csv(url_reviews)

    orders_payments["order_purchase_timestamp"] = pd.to_datetime(
        orders_payments["order_purchase_timestamp"]
    )

    return orders_payments, customers, reviews_products_items


with st.spinner("Loading data..."):
    orders_payments, customers, reviews_products_items = load_data()

orders_payments = orders_payments[
    orders_payments["order_purchase_timestamp"].dt.year == 2017
]

# =========================
# FILTER TANGGAL
# =========================
st.sidebar.header("Analisis E-Commerce", )
st.sidebar.image("ecommerce.png", width=200)
st.sidebar.markdown("---")

st.sidebar.header("🔎 Filter Data")

min_date = orders_payments["order_purchase_timestamp"].min().date()
max_date = orders_payments["order_purchase_timestamp"].max().date()

start_date, end_date = st.sidebar.date_input(
    label="Pilih Rentang Tanggal",
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date]
)

filtered_orders = orders_payments[
    (orders_payments["order_purchase_timestamp"].dt.date >= start_date) &
    (orders_payments["order_purchase_timestamp"].dt.date <= end_date)
]

if filtered_orders.empty:
    st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
    st.stop()

# =========================
# 1. PERFORMA PENJUALAN
# =========================
st.subheader('Daily Orders')

# resample harian
daily_orders_df = filtered_orders.resample(
    rule='D',
    on='order_purchase_timestamp'
).agg({
    "order_id": "nunique",
    "payment_value": "sum"
}).reset_index()

daily_orders_df.rename(columns={
    "order_id": "order_count",
    "payment_value": "revenue",
    "order_purchase_timestamp": "order_date"
}, inplace=True)

# =========================
# KPI
# =========================
col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df["order_count"].sum()
    st.metric("Total Orders", value=f"{total_orders:,}")

with col2:
    total_revenue = daily_orders_df["revenue"].sum()
    st.metric("Total Revenue", value=f"{total_revenue:,.0f}")

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(
    daily_orders_df["order_date"],  # datetime (PENTING)
    daily_orders_df["order_count"],
    marker='o',
    linewidth=2,
    color="#90CAF9"
)

ax.set_xlabel("Date")
ax.set_ylabel("Order Count")

ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=10)

ax.grid(alpha=0.3)

st.pyplot(fig)

# =========================
# 2. DISTRIBUSI KOTA
# =========================
st.subheader("Distribusi Pelanggan Berdasarkan Kota")

city = filtered_orders.merge(customers, on="customer_id") \
    .groupby("customer_city")["customer_id"].nunique().reset_index()
city.rename(columns={"customer_id": "customer_count"}, inplace=True)

top_city = city.sort_values(by="customer_count", ascending=False).head()
bottom_city = city.sort_values(by="customer_count", ascending=True).head()

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

sns.barplot(y="customer_city", x="customer_count", data=top_city, ax=ax[0])
ax[0].set_title("Top 5 Kota")

sns.barplot(y="customer_city", x="customer_count", data=bottom_city, ax=ax[1])
ax[1].set_title("Bottom 5 Kota")

st.pyplot(fig)

# =========================
# 3. REVENUE KATEGORI PRODUK
# =========================
st.subheader("Kategori Produk dengan Revenue Tertinggi & Terendah")

revenue = reviews_products_items.groupby(
    by="product_category_name"
).agg({
    "total_price": "sum"
}).reset_index()

top_revenue = revenue.sort_values(by="total_price", ascending=False).head(5)
bottom_revenue = revenue.sort_values(by="total_price", ascending=True).head(5)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

sns.barplot(y="product_category_name", x="total_price", data=top_revenue, ax=ax[0])
ax[0].set_title("Top 5 Product Category Revenue")

sns.barplot(y="product_category_name", x="total_price", data=bottom_revenue, ax=ax[1])
ax[1].set_title("Bottom 5 Product Category Revenue")

st.pyplot(fig)

# =========================
# 4. RATING PRODUK
# =========================
st.subheader("Kategori Produk dengan Rating Tertinggi & Terendah")

review = reviews_products_items.groupby(
    by="product_category_name"
).review_score.mean().reset_index()

top_review = review.sort_values(by="review_score", ascending=False).head()
bottom_review = review.sort_values(by="review_score", ascending=True).head()

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

sns.barplot(y="product_category_name", x="review_score", data=top_review, ax=ax[0])
ax[0].set_title("Top 5 Rating")

sns.barplot(y="product_category_name", x="review_score", data=bottom_review, ax=ax[1])
ax[1].set_title("Bottom 5 Rating")

st.pyplot(fig)

# =========================
# 5. RFM ANALYSIS
# =========================
st.subheader("Analisis Pelanggan (RFM)")

rfm_df = orders_payments.merge(customers, on="customer_id")

rfm_df = rfm_df.groupby(by="customer_unique_id", as_index=False).agg({
    "order_purchase_timestamp": "max",
    "order_id": "nunique",
    "payment_value": "sum"
})

rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]

rfm_df["max_order_timestamp"] = pd.to_datetime(rfm_df["max_order_timestamp"]).dt.date
recent_date = orders_payments["order_purchase_timestamp"].dt.date.max()

rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(
    lambda x: (recent_date - x).days
)

rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

df_rec = rfm_df.sort_values(by="recency").head(5)
df_freq = rfm_df.sort_values(by="frequency", ascending=False).head(5)
df_mon = rfm_df.sort_values(by="monetary", ascending=False).head(5)

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.barplot(x="customer_id", y="recency", data=df_rec, ax=ax[0])
ax[0].set_title("Recency")

sns.barplot(x="customer_id", y="frequency", data=df_freq, ax=ax[1])
ax[1].set_title("Frequency")

sns.barplot(x="customer_id", y="monetary", data=df_mon, ax=ax[2])
ax[2].set_title("Monetary")

for a in ax:
    a.tick_params(axis='x', rotation=45)

st.pyplot(fig)
