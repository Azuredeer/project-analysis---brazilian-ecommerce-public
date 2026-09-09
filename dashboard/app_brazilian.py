import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Brazilian E-Commerce Analytics & ML",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E293B;
        margin-top: 0.3rem;
    }
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 600;
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        color: #991B1B;
    }
    .risk-low {
        background-color: #DCFCE7;
        border-left: 5px solid #22C55E;
        padding: 1rem;
        border-radius: 8px;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATABASE CONNECTION & CACHED QUERIES
# ==========================================
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ecommerce.db")

@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_connection()

@st.cache_data(ttl=3600)
def load_kpis():
    q = """
    SELECT 
        COUNT(DISTINCT o.order_id) AS total_orders,
        ROUND(SUM(oi.price), 2) AS total_gmv,
        ROUND(AVG(r.review_score), 2) AS avg_rating,
        ROUND(100.0 * SUM(CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 1 ELSE 0 END) / COUNT(*), 1) AS on_time_rate
    FROM orders_data o
    JOIN order_items_data oi ON o.order_id = oi.order_id
    LEFT JOIN reviews_data r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL;
    """
    return pd.read_sql(q, conn).iloc[0]

@st.cache_data(ttl=3600)
def load_monthly_trends():
    q = """
    SELECT 
        strftime('%Y-%m', o.order_purchase_timestamp) AS month,
        COUNT(DISTINCT o.order_id) AS total_orders,
        ROUND(SUM(oi.price), 2) AS revenue
    FROM orders_data o
    JOIN order_items_data oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
      AND strftime('%Y-%m', o.order_purchase_timestamp) BETWEEN '2017-01' AND '2018-08'
    GROUP BY month
    ORDER BY month;
    """
    return pd.read_sql(q, conn)

@st.cache_data(ttl=3600)
def load_category_data():
    q = """
    SELECT 
        COALESCE(c.product_category_name_english, p.product_category_name, 'Other') AS category,
        COUNT(oi.order_item_id) AS items_sold,
        ROUND(SUM(oi.price), 2) AS total_revenue
    FROM order_items_data oi
    JOIN product_data p ON oi.product_id = p.product_id
    LEFT JOIN pr_category_data c ON p.product_category_name = c.product_category_name
    JOIN orders_data o ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY category;
    """
    return pd.read_sql(q, conn)

@st.cache_data(ttl=3600)
def load_delivery_review():
    q = """
    SELECT 
        CASE 
            WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 'On-Time / Early'
            ELSE 'Delayed'
        END AS status,
        COUNT(DISTINCT o.order_id) AS total_orders,
        ROUND(AVG(r.review_score), 2) AS avg_score,
        ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)), 1) AS avg_days
    FROM orders_data o
    JOIN reviews_data r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered' 
      AND o.order_delivered_customer_date IS NOT NULL 
      AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY status;
    """
    return pd.read_sql(q, conn)

@st.cache_data(ttl=3600)
def load_payment_methods():
    q = """
    SELECT 
        payment_type,
        COUNT(order_id) AS transaction_count,
        ROUND(SUM(payment_value), 2) AS total_amount,
        ROUND(AVG(payment_installments), 1) AS avg_installments
    FROM payments_data
    WHERE payment_type != 'not_defined'
    GROUP BY payment_type
    ORDER BY transaction_count DESC;
    """
    return pd.read_sql(q, conn)

@st.cache_data(ttl=3600)
def load_geo_data():
    q = """
    SELECT customer_state AS state, COUNT(customer_id) AS customer_count
    FROM customer_data
    GROUP BY customer_state
    ORDER BY customer_count DESC
    LIMIT 10;
    """
    return pd.read_sql(q, conn)

# Train ML Delay Classifier Model (Cached)
@st.cache_resource
def train_delay_model():
    q_train = """
    SELECT 
        CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END AS is_delayed,
        ROUND(julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp), 1) AS estimated_delivery_days,
        CAST(strftime('%w', o.order_purchase_timestamp) AS INT) AS purchase_dow,
        CAST(strftime('%m', o.order_purchase_timestamp) AS INT) AS purchase_month,
        CASE WHEN c.customer_state != s.seller_state THEN 1 ELSE 0 END AS is_interstate,
        oi.price,
        oi.freight_value,
        ROUND(oi.freight_value / (oi.price + 0.01), 3) AS freight_ratio,
        COALESCE(p.product_weight_g, 1600.0) AS product_weight_g,
        COALESCE(p.product_length_cm * p.product_height_cm * p.product_width_cm, 10000.0) AS product_volume_cm3
    FROM orders_data o
    JOIN customer_data c ON o.customer_id = c.customer_id
    JOIN order_items_data oi ON o.order_id = oi.order_id
    JOIN seller_data s ON oi.seller_id = s.seller_id
    JOIN product_data p ON oi.product_id = p.product_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
    LIMIT 40000;
    """
    df = pd.read_sql(q_train, conn)
    features = ['estimated_delivery_days', 'purchase_dow', 'purchase_month', 'is_interstate', 
                'price', 'freight_value', 'freight_ratio', 'product_weight_g', 'product_volume_cm3']
    X = df[features]
    y = df['is_delayed']
    model = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, features

ml_model, ml_features = train_delay_model()

# ==========================================
# SIDEBAR CONTROLS & INFO
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=70)
    st.title("E-Commerce Portal")
    st.markdown("**Dataset**: Olist Brazilian E-Commerce")
    st.markdown("---")
    
    st.subheader("⚙️ Navigasi Dashboard")
    st.info("Gunakan tab di halaman utama untuk berpindah antara analisis eksekutif, logistik, segmentasi pelanggan, dan kalkulator AI.")
    
    st.markdown("---")
    st.subheader("💡 Ringkasan Cepat")
    st.write("• **Total Orders**: ~100k transaksi")
    st.write("• **Rentang Waktu**: 2016 - 2018")
    st.write("• **Episentrum**: São Paulo (SP)")
    st.write("• **Tingkat On-Time**: 92.0%")

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================
st.markdown('<div class="main-header">Brazilian E-Commerce Analytics & AI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Platform intelijen bisnis dan prediksi machine learning untuk operasi e-commerce Olist Brasil</div>', unsafe_allow_html=True)

# ----------------- TOP KPI ROW -----------------
kpi_data = load_kpis()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Gross Merchandise Value (GMV)</div>
        <div class="kpi-value">R$ {kpi_data['total_gmv']/1e6:.2f}M</div>
        <div class="kpi-delta" style="color: #10B981;">↑ Total Omzet Terkirim</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Pesanan Selesai (Delivered)</div>
        <div class="kpi-value">{int(kpi_data['total_orders']):,}</div>
        <div class="kpi-delta" style="color: #3B82F6;">97.0% Success Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Rata-rata Rating Ulasan</div>
        <div class="kpi-value">{kpi_data['avg_rating']:.2f} ⭐</div>
        <div class="kpi-delta" style="color: #F59E0B;">Skala 1.0 - 5.0</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Ketepatan Waktu Logistik</div>
        <div class="kpi-value">{kpi_data['on_time_rate']:.1f}%</div>
        <div class="kpi-delta" style="color: #10B981;">Tepat Waktu / Lebih Awal</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- TABS ARCHITECTURE -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Tren Penjualan", 
    "🛍️ Kategori Produk", 
    "🚚 Logistik & Kepuasan", 
    "👥 Segmentasi Pelanggan (RFM)", 
    "🤖 Kalkulator Risiko Keterlambatan (AI)"
])

# ----------------- TAB 1: SALES TRENDS -----------------
with tab1:
    st.subheader("Tren Pertumbuhan Pesanan dan Pendapatan Bulanan (2017 - 2018)")
    df_monthly = load_monthly_trends()
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=df_monthly['month'], 
        y=df_monthly['total_orders'], 
        name="Volume Pesanan", 
        mode="lines+markers", 
        line=dict(color="#1E40AF", width=3)
    ))
    fig_monthly.add_trace(go.Scatter(
        x=df_monthly['month'], 
        y=df_monthly['revenue'], 
        name="Pendapatan (R$)", 
        mode="lines+markers", 
        line=dict(color="#10B981", width=3, dash="dot"), 
        yaxis="y2"
    ))
    
    fig_monthly.update_layout(
        xaxis=dict(title="Bulan Transaksi"),
        yaxis=dict(title=dict(text="Jumlah Pesanan (Orders)", font=dict(color="#1E40AF")), tickfont=dict(color="#1E40AF")),
        yaxis2=dict(title=dict(text="Gross Revenue (BRL R$)", font=dict(color="#10B981")), tickfont=dict(color="#10B981"), overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=420
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Metode Pembayaran Pilihan Konsumen**")
        df_pay = load_payment_methods()
        fig_pay = px.pie(
            df_pay, 
            names='payment_type', 
            values='transaction_count', 
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pay.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
        st.plotly_chart(fig_pay, use_container_width=True)
        
    with col_t2:
        st.markdown("**Konsentrasi Pembeli di 10 Negara Bagian Teratas**")
        df_geo = load_geo_data()
        fig_geo = px.bar(
            df_geo, 
            x='state', 
            y='customer_count', 
            labels={'state': 'State', 'customer_count': 'Total Pelanggan'},
            color='customer_count',
            color_continuous_scale='Blues'
        )
        fig_geo.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_geo, use_container_width=True)

# ----------------- TAB 2: CATEGORY INSIGHTS -----------------
with tab2:
    st.subheader("Top 10 Kategori Produk Berdasarkan Volume Penjualan vs Pendapatan")
    df_cat = load_category_data()
    top_vol = df_cat.sort_values(by='items_sold', ascending=False).head(10)
    top_rev = df_cat.sort_values(by='total_revenue', ascending=False).head(10)
    
    c1, c2 = st.columns(2)
    with c1:
        fig_vol = px.bar(
            top_vol, 
            y='category', 
            x='items_sold', 
            orientation='h',
            title="Top 10 Kategori: Volume Barang Terjual",
            labels={'items_sold': 'Unit Terjual', 'category': 'Kategori'},
            color='items_sold',
            color_continuous_scale='Teal'
        )
        fig_vol.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with c2:
        fig_rev = px.bar(
            top_rev, 
            y='category', 
            x='total_revenue', 
            orientation='h',
            title="Top 10 Kategori: Kontribusi Omzet (R$)",
            labels={'total_revenue': 'Total Omzet (R$)', 'category': 'Kategori'},
            color='total_revenue',
            color_continuous_scale='Greens'
        )
        fig_rev.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_rev, use_container_width=True)

# ----------------- TAB 3: DELIVERY & SATISFACTION -----------------
with tab3:
    st.subheader("Dampak Kinerja Logistik terhadap Kepuasan Pelanggan (Review Score)")
    df_deliv = load_delivery_review()
    
    d1, d2 = st.columns([1, 2])
    with d1:
        st.write("### Perbandingan On-Time vs Delayed")
        for idx, row in df_deliv.iterrows():
            st.metric(
                label=f"Status: {row['status']}", 
                value=f"{row['avg_score']} ⭐", 
                delta=f"{row['avg_days']} hari waktu antar"
            )
        st.info("Pesanan yang terlambat (delayed) rata-rata membutuhkan waktu 31,4 hari untuk sampai, memicu penurunan review score drastis hingga **2.57 / 5.00**.")
        
    with d2:
        fig_deliv_bar = go.Figure(data=[
            go.Bar(name='Rata-rata Rating Ulasan (⭐)', x=df_deliv['status'], y=df_deliv['avg_score'], marker_color=['#10B981', '#EF4444']),
            go.Bar(name='Rata-rata Durasi Pengiriman (Hari)', x=df_deliv['status'], y=df_deliv['avg_days'], marker_color=['#3B82F6', '#F59E0B'])
        ])
        fig_deliv_bar.update_layout(
            barmode='group', 
            title="Rating Ulasan vs Durasi Pengiriman per Status",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_deliv_bar, use_container_width=True)

# ----------------- TAB 4: CUSTOMER SEGMENTATION (RFM) -----------------
with tab4:
    st.subheader("Segmentasi Pelanggan (RFM Analysis & K-Means Clusters)")
    
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        **Pola Pembelian Pelanggan Olist**:
        - **One-time Buyers (97.0%)**: Rata-rata belanja **R$ 137.96**
        - **Repeat Customers (3.0%)**: Rata-rata belanja **R$ 260.05** (Hampir 2x lipat lebih tinggi!)
        
        **Karakteristik Cluster K-Means**:
        1. **Cluster Champions (Repeat Buyers)**: Pelanggan yang belanja berkali-kali dengan frekuensi tinggi.
        2. **Cluster High-Value Buyers**: Pelanggan belanja satu kali dengan nilai transaksi sangat besar.
        3. **Cluster Recent Regulars**: Pelanggan aktif baru dengan nilai belanja normal.
        4. **Cluster Dormant / At-Risk**: Pelanggan yang sudah lama tidak aktif (>1 tahun).
        """)
        
    with r2:
        # Mini simulated chart for RFM proportions
        df_rfm_chart = pd.DataFrame({
            'Tipe': ['One-Time Buyers (97%)', 'Repeat Buyers (3%)'],
            'Belanja_Rata_Rata': [137.96, 260.05]
        })
        fig_rfm = px.bar(
            df_rfm_chart, 
            x='Tipe', 
            y='Belanja_Rata_Rata', 
            color='Tipe', 
            text='Belanja_Rata_Rata',
            title="Perbandingan Rata-rata Total Belanja (R$)",
            color_discrete_sequence=['#94A3B8', '#10B981']
        )
        fig_rfm.update_traces(texttemplate='R$ %{text:.2f}', textposition='outside')
        fig_rfm.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_rfm, use_container_width=True)

# ----------------- TAB 5: AI DELAY CALCULATOR -----------------
with tab5:
    st.subheader("🤖 AI Delivery Delay Risk Calculator")
    st.write("Masukkan parameter pesanan baru untuk memprediksi probabilitas risiko keterlambatan pengiriman secara instan sebelum paket dikirim:")
    
    with st.form("delay_calc_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            est_days = st.slider("Estimasi Hari Pengiriman (SLA Hari)", min_value=3, max_value=60, value=20)
            is_interstate = st.selectbox("Pengiriman Antar Negara Bagian?", ["Ya (Beda State)", "Tidak (Sama State)"])
            dow = st.selectbox("Hari Pembelian", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
            
        with col_in2:
            item_price = st.number_input("Harga Barang (R$)", min_value=5.0, max_value=5000.0, value=120.0)
            freight_val = st.number_input("Ongkos Kirim (R$)", min_value=1.0, max_value=500.0, value=18.5)
            month = st.slider("Bulan Transaksi (1 - 12)", min_value=1, max_value=12, value=11)
            
        with col_in3:
            weight_g = st.number_input("Berat Produk (Gram)", min_value=50.0, max_value=30000.0, value=1500.0)
            length_cm = st.number_input("Panjang (cm)", min_value=5.0, max_value=150.0, value=25.0)
            height_cm = st.number_input("Tinggi (cm)", min_value=2.0, max_value=150.0, value=15.0)
            width_cm = st.number_input("Lebar (cm)", min_value=5.0, max_value=150.0, value=20.0)
            
        submitted = st.form_submit_button("🔍 Prediksi Risiko Keterlambatan")
        
    if submitted:
        dow_map = {"Senin": 1, "Selasa": 2, "Rabu": 3, "Kamis": 4, "Jumat": 5, "Sabtu": 6, "Minggu": 0}
        vol_cm3 = length_cm * height_cm * width_cm
        fr_ratio = freight_val / (item_price + 0.01)
        interstate_val = 1 if "Ya" in is_interstate else 0
        
        input_data = pd.DataFrame([[
            est_days, 
            dow_map[dow], 
            month, 
            interstate_val, 
            item_price, 
            freight_val, 
            fr_ratio, 
            weight_g, 
            vol_cm3
        ]], columns=ml_features)
        
        prob_delay = ml_model.predict_proba(input_data)[0][1] * 100
        
        st.markdown("### 📋 Hasil Analisis Prediktif AI")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Probabilitas Risiko Keterlambatan", f"{prob_delay:.1f}%")
            if prob_delay >= 25.0:
                st.markdown('<div class="risk-high">⚠️ <b>STATUS: RISIKO TINGGI (DELAY RISK)</b><br>Pesanan ini memiliki indikasi potensi keterlambatan pengiriman melebihi ambang batas normal.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">✅ <b>STATUS: RISIKO RENDAH (ON-TIME SAFE)</b><br>Pesanan ini berada dalam batas aman dan diperkirakan akan tiba sesuai atau sebelum tanggal estimasi.</div>', unsafe_allow_html=True)
                
        with res_col2:
            st.write("**Rekomendasi Tindakan Operasional Bisnis:**")
            if prob_delay >= 25.0:
                st.warning("""
                1. **Alihkan ke Mitra Kurir Prioritas**: Gunakan layanan ekspres (*expedited shipping*) untuk rute pengiriman ini.
                2. **Tambah Buffer SLA 2-3 Hari**: Sesuaikan tanggal estimasi yang ditampilkan ke pembeli agar ekspektasi pelanggan terjaga.
                3. **Proactive Customer Notification**: Kirim notifikasi pelacakan real-time otomatis via WhatsApp/Email untuk menjaga kepuasan pembeli.
                """)
            else:
                st.success("""
                1. **Standard Fulfillment**: Paket dapat diproses melalui alur kurir reguler berbiaya efisien.
                2. **Optimasi Pengemasan**: Pastikan perlindungan kemasan terjaga sesuai standar untuk mengamankan review bintang 5.
                """)

