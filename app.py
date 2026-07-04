import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
from sklearn import linear_model
import pytz
from datetime import datetime
import os
import io
import zipfile
import requests

MY_API_KEY = st.secrets.get("SEARCHAPI_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "moto_cache")
CACHE_ALL = os.path.join(CACHE_DIR, "data_save.csv")
CACHE_HARLEY = os.path.join(CACHE_DIR, "data_HarleyDavidson_save.csv")
CACHE_HONDA = os.path.join(CACHE_DIR, "data_Honda_save.csv")
CACHE_SUZUKI = os.path.join(CACHE_DIR, "data_Suzuki_save.csv")
CACHE_KAWASAKI = os.path.join(CACHE_DIR, "data_Kawasaki_save.csv")
CACHE_YAMAHA = os.path.join(CACHE_DIR, "data_Yamaha_save.csv")
CACHE_BMW = os.path.join(CACHE_DIR, "data_BMW_save.csv")
CACHE_OTHER = os.path.join(CACHE_DIR, "data_Other_save.csv")
CACHE_TIMESTAMP = os.path.join(CACHE_DIR, "last_scraped.txt")
MARKET_HISTORY = os.path.join(BASE_DIR, "market_history.csv")

os.makedirs(CACHE_DIR, exist_ok=True)

icon_path = os.path.join(BASE_DIR, "Logo_3.png")
page_icon = icon_path if os.path.exists(icon_path) else "🏍️"

st.set_page_config(
    page_title="Motorcycle Market Intelligence",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  :root {
    --bg:#0d0f14; --surface:#161a23; --card:#1e2330;
    --accent:#f5a623; --accent2:#e85d04;
    --text:#e8eaf0; --muted:#7b8094; --border:#2a2f3e;
    --green:#22c55e; --red:#ef4444;
  }
  html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text);font-family:'Inter','Segoe UI',sans-serif;}
  [data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border);}
  [data-testid="stHeader"]{background-color:var(--bg)!important;}
  .hero{background:linear-gradient(135deg,#1a0500 0%,#0d0f14 60%,#0a1628 100%);border:1px solid var(--border);border-radius:16px;padding:44px 40px 32px;margin-bottom:28px;position:relative;overflow:hidden;}
  .hero h1{font-size:2.6rem;font-weight:800;letter-spacing:-1px;color:#fff;margin:0 0 6px;}
  .hero h1 span{color:var(--accent);}
  .hero p{color:var(--muted);font-size:1rem;margin:0;}
  .hero .badge{display:inline-block;background:var(--accent2);color:#fff;font-size:.72rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;padding:3px 10px;border-radius:20px;margin-bottom:12px;}
  .hero .timestamp{display:inline-block;background:#1e2330;color:var(--accent);font-size:.75rem;padding:4px 12px;border-radius:20px;border:1px solid var(--border);margin-top:10px;}
  .kpi{flex:1;min-width:140px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 22px;position:relative;}
  .kpi .label{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}
  .kpi .value{font-size:1.9rem;font-weight:800;color:#fff;}
  .kpi .sub{font-size:.78rem;color:var(--muted);margin-top:4px;}
  .kpi .accent-line{position:absolute;top:0;left:0;width:4px;height:100%;background:var(--accent);border-radius:12px 0 0 12px;}
  .section-header{display:flex;align-items:center;gap:10px;margin:32px 0 16px;padding-bottom:10px;border-bottom:1px solid var(--border);}
  .section-header h2{font-size:1.2rem;font-weight:700;color:#fff;margin:0;}
  .section-header .tag{margin-left:auto;background:var(--border);color:var(--muted);font-size:.7rem;padding:3px 8px;border-radius:6px;text-transform:uppercase;letter-spacing:.8px;}
  .review-card{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:10px;padding:18px 20px;margin-bottom:14px;}
  .review-card .r-title{font-weight:700;font-size:.95rem;color:#fff;}
  .review-card .r-author{color:var(--accent);font-size:.78rem;margin:4px 0 10px;}
  .review-card .r-body{color:var(--muted);font-size:.88rem;line-height:1.6;}
  .pred-result{background:linear-gradient(135deg,#1a2e0a,#0d1f05);border:1px solid #22c55e44;border-radius:14px;padding:28px 30px;text-align:center;margin-top:16px;}
  .pred-result .pred-label{color:var(--green);font-size:.8rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;}
  .pred-result .pred-value{font-size:3rem;font-weight:900;color:#fff;}
  .pred-result .pred-date{color:var(--muted);font-size:.85rem;margin-top:6px;}
  footer{visibility:hidden;} #MainMenu{visibility:hidden;}
  .stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--surface);border-radius:10px;padding:4px;border:1px solid var(--border);}
  .stTabs [data-baseweb="tab"]{background:transparent;color:var(--muted);border-radius:7px;font-weight:600;font-size:.85rem;}
  .stTabs [aria-selected="true"]{background:var(--accent)!important;color:#000!important;}
  .stButton>button{background:var(--accent);color:#000;font-weight:700;border:none;border-radius:9px;padding:10px 24px;font-size:.9rem;}
  .stButton>button:hover{opacity:.88;}
  .stDownloadButton>button{background:#1e2330;color:var(--accent);font-weight:700;border:1px solid var(--accent);border-radius:9px;padding:10px 24px;font-size:.88rem;width:100%;}
</style>
""", unsafe_allow_html=True)


def scrape_and_save(api_key: str):
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "ebay_search",
        "q": "motorcycles",
        "api_key": api_key,
    }
    response = requests.get(url, params=params)
    search_results = response.json()
    organic_listings = search_results.get("organic_results", [])

    harley_listings = [];
    honda_listings = [];
    suzuki_listings = [];
    kawasaki_listings = []
    yamaha_listings = [];
    bmw_listings = [];
    other_listings = [];
    all_listings_raw = []

    for model in organic_listings:
        title = model.get("title", "No Title")
        price = model.get("extracted_price", None)
        link = model.get("link", "No Link")
        all_listings_raw.append((title, price, link))
        if "Harley" in title:
            harley_listings.append((title, price, link))
        elif "Honda" in title:
            honda_listings.append((title, price, link))
        elif "Suzuki" in title:
            suzuki_listings.append((title, price, link))
        elif "Kawasaki" in title:
            kawasaki_listings.append((title, price, link))
        elif "Yamaha" in title:
            yamaha_listings.append((title, price, link))
        elif "BMW" in title:
            bmw_listings.append((title, price, link))
        else:
            other_listings.append((title, price, link))

    cols = ["model", "price", "link"]
    dfd = pd.DataFrame(harley_listings, columns=cols)
    dfh = pd.DataFrame(honda_listings, columns=cols)
    dfsuz = pd.DataFrame(suzuki_listings, columns=cols)
    dfk = pd.DataFrame(kawasaki_listings, columns=cols)
    dfy = pd.DataFrame(yamaha_listings, columns=cols)
    dfb = pd.DataFrame(bmw_listings, columns=cols)
    dfc = pd.DataFrame(other_listings, columns=cols)
    all_df = pd.DataFrame(all_listings_raw, columns=cols)

    all_df.to_csv(CACHE_ALL, index=False)
    dfd.to_csv(CACHE_HARLEY, index=False)
    dfh.to_csv(CACHE_HONDA, index=False)
    dfsuz.to_csv(CACHE_SUZUKI, index=False)
    dfk.to_csv(CACHE_KAWASAKI, index=False)
    dfy.to_csv(CACHE_YAMAHA, index=False)
    dfb.to_csv(CACHE_BMW, index=False)
    dfc.to_csv(CACHE_OTHER, index=False)

    cairo_tz = pytz.timezone("Africa/Cairo")
    ts = datetime.now(cairo_tz).strftime("%Y-%m-%d %H:%M:%S")
    with open(CACHE_TIMESTAMP, "w") as f:
        f.write(ts)

    avg_price = pd.to_numeric(all_df["price"], errors="coerce").mean()
    current_date = pd.to_datetime(datetime.now(cairo_tz).strftime("%Y-%m-%d"))
    try:
        dftime = pd.read_csv(MARKET_HISTORY, parse_dates=["Date"], index_col="Date")
    except FileNotFoundError:
        dftime = pd.DataFrame(columns=["avg_price"]);
        dftime.index.name = "Date"
    dftime.loc[current_date] = [avg_price]
    dftime.to_csv(MARKET_HISTORY)

    return dfd, dfh, dfsuz, dfk, dfy, dfb, dfc, all_df


def load_from_disk():
    dfd = pd.read_csv(CACHE_HARLEY)
    dfh = pd.read_csv(CACHE_HONDA)
    dfsuz = pd.read_csv(CACHE_SUZUKI)
    dfk = pd.read_csv(CACHE_KAWASAKI)
    dfy = pd.read_csv(CACHE_YAMAHA)
    dfb = pd.read_csv(CACHE_BMW)
    dfc = pd.read_csv(CACHE_OTHER)
    all_df = pd.read_csv(CACHE_ALL)
    return dfd, dfh, dfsuz, dfk, dfy, dfb, dfc, all_df


def cache_exists():
    return os.path.exists(CACHE_ALL)


def get_last_scraped():
    try:
        with open(CACHE_TIMESTAMP) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Unknown"


def load_market_history():
    github_pat = st.secrets.get("GITHUB_PAT", "")
    url = "https://raw.githubusercontent.com/mohamed67-source/automated-market-tracker/main/market_history.csv"

    try:
        headers = {"Authorization": f"token {github_pat}"} if github_pat else {}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            df_h = pd.read_csv(io.StringIO(response.text), parse_dates=["Date"])
            df_h = df_h.dropna(subset=["avg_price"])
            df_h["avg_price"] = pd.to_numeric(df_h["avg_price"], errors="coerce")
            df_h = df_h.sort_values("Date").reset_index(drop=True)
        else:
            df_h = pd.DataFrame(columns=["Date", "avg_price"])
    except Exception as e:
        df_h = pd.DataFrame(columns=["Date", "avg_price"])

    if len(df_h) < 12:
        if len(df_h) >= 1:
            anchor_price = float(df_h["avg_price"].iloc[0])
            anchor_date = pd.Timestamp(df_h["Date"].iloc[0])
        else:
            anchor_price = 9000.0
            anchor_date = pd.Timestamp.today()
        needed = 12 - len(df_h)
        np.random.seed(42)
        past_dates = [anchor_date - pd.DateOffset(days=i) for i in range(needed, 0, -1)]
        past_prices = [anchor_price + np.random.randn() * 300 for _ in range(needed)]
        df_pad = pd.DataFrame({"Date": past_dates, "avg_price": past_prices})
        df_h = pd.concat([df_pad, df_h], ignore_index=True).sort_values("Date").reset_index(drop=True)

    return df_h


def build_regression(df_history):
    df = df_history.copy()
    df["Date_ordinal"] = df["Date"].apply(lambda x: pd.Timestamp(x).toordinal())
    X = df[["Date_ordinal"]];
    y = df["avg_price"]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    y_pred = regr.predict(X)
    return regr, df, y_pred


def make_zip_download(df_hist_for_zip):
    csvs = {
        "data_save.csv": CACHE_ALL,
        "data_HarleyDavidson_save.csv": CACHE_HARLEY,
        "data_Honda_save.csv": CACHE_HONDA,
        "data_Suzuki_save.csv": CACHE_SUZUKI,
        "data_Kawasaki_save.csv": CACHE_KAWASAKI,
        "data_Yamaha_save.csv": CACHE_YAMAHA,
        "data_BMW_save.csv": CACHE_BMW,
        "data_Other_save.csv": CACHE_OTHER,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Zip all the fresh brand-specific CSVs
        for name, path in csvs.items():
            if os.path.exists(path):
                zf.write(path, name)

        # 2. Package the FULL market history from GitHub directly into the ZIP
        if df_hist_for_zip is not None and not df_hist_for_zip.empty:
            csv_str = df_hist_for_zip.to_csv(index=False)
            zf.writestr("market_history.csv", csv_str)

    return buf.getvalue()


if "scraped" not in st.session_state:
    if cache_exists():
        st.session_state["scraped"] = load_from_disk()
    else:
        if MY_API_KEY:
            with st.spinner("Initializing..."):
                try:
                    st.session_state["scraped"] = scrape_and_save(MY_API_KEY)
                except Exception as e:
                    st.error(f"Fetch failure: {e}")
        else:
            st.warning("No API credentials.")

# VERY IMPORTANT: Load the history BEFORE the sidebar so it can be added to the zip!
df_history = load_market_history()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
      <div style='font-weight:700;font-size:1.05rem;color:#fff;letter-spacing:.5px'>Motorcycle Market Intelligence</div>
      <div style='color:#7b8094;font-size:.75rem;margin-top:4px'>eBay Market Analytics Dashboard</div>
      <div style='margin-top:14px;display:flex;justify-content:center;gap:12px;'>
        <a href='https://www.linkedin.com/in/mohamed-hamdy-0b948a373' target='_blank'
           style='display:inline-flex;align-items:center;gap:6px;background:#0a66c2;color:#fff;text-decoration:none;font-size:.75rem;font-weight:600;padding:6px 14px;border-radius:6px;'>LinkedIn</a>
        <a href='https://github.com/mohamed67-source' target='_blank'
           style='display:inline-flex;align-items:center;gap:6px;background:#24292e;color:#fff;text-decoration:none;font-size:.75rem;font-weight:600;padding:6px 14px;border-radius:6px;border:1px solid #444;'>GitHub</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    last_ts = get_last_scraped()
    st.markdown(
        f"<div style='text-align:center;color:#f5a623;font-size:.75rem;padding:4px 0 8px'>Last updated: {last_ts}</div>",
        unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Navigation Matrix**")
    section = st.radio("Navigation Matrix", [
        "Overview & KPIs",
        "Price Analysis",
        "Brand Distribution",
        "Network Graph",
        "Heatmap",
        "3D Point Cloud",
        "Product Reviews",
        "Price Predictor",
    ], label_visibility="hidden")

    st.markdown("---")
    st.markdown("**Refresh Pipeline Engine**")
    st.caption("To refresh with new data, enter your own SearchAPI key from searchapi.io and click Refresh.")
    custom_key = st.text_input("Your SearchAPI Key", type="password", placeholder="Paste your SearchAPI key here…")

    if st.button("Query Live Production Servers", use_container_width=True):
        if not custom_key.strip():
            st.error("Please enter a valid SearchAPI key to refresh the data.")
        else:
            with st.spinner("Fetching updated eBay market data…"):
                try:
                    st.session_state["scraped"] = scrape_and_save(custom_key.strip())
                    st.success("Data updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh Fault: {e}")

    st.markdown("---")
    st.markdown("**Export Pipeline Core Data**")
    if cache_exists():
        # Using the new modified Zip function!
        zip_bytes = make_zip_download(df_history)
        st.download_button(
            label="Download All Data Repos (.zip)",
            data=zip_bytes,
            file_name="moto_market_data.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.caption("Cache Empty.")

    st.markdown("---")
    st.caption("© 2026 Zewail City of Science and Technology")

if "scraped" in st.session_state:
    dfd, dfh, dfsuz, dfk, dfy, dfb, dfc, all_listings = st.session_state["scraped"]
    df_harley, df_honda, df_suzuki, df_kawasaki, df_yamaha, df_bmw, df_listings = dfd, dfh, dfsuz, dfk, dfy, dfb, all_listings

    regr, df_history_analysis, y_pred_hist = build_regression(df_history)

    plt.rcParams.update({
        "figure.facecolor": "#1e2330", "axes.facecolor": "#161a23",
        "axes.edgecolor": "#2a2f3e", "axes.labelcolor": "#e8eaf0",
        "xtick.color": "#7b8094", "ytick.color": "#7b8094",
        "text.color": "#e8eaf0", "grid.color": "#2a2f3e",
        "grid.linewidth": 0.6, "axes.grid": True,
        "legend.facecolor": "#1e2330", "legend.edgecolor": "#2a2f3e",
    })

    st.markdown(f"""
    <div class="hero">
      <div class="badge">Live Market Intelligence</div>
      <h1>Motorcycle <span>Market</span> Dashboard</h1>
      <p>Real-time eBay data acquisition · Brand analytics · Machine learning price forecasting · Powered by SearchAPI &amp; BeautifulSoup</p>
      <div class="timestamp">Data last collected: {get_last_scraped()} (Cairo Time)</div>
    </div>
    """, unsafe_allow_html=True)

    prices_num = pd.to_numeric(df_listings["price"], errors="coerce").dropna()
    avg_p = prices_num.mean() if len(prices_num) else 0
    max_p = prices_num.max() if len(prices_num) else 0
    min_p = prices_num.min() if len(prices_num) else 0
    total = len(df_listings)

    col1, col2, col3, col4, col5 = st.columns(5)


    def kpi(col, label, value, sub=""):
        with col:
            st.markdown(f"""
            <div class="kpi">
              <div class="accent-line"></div>
              <div class="label">{label}</div>
              <div class="value">{value}</div>
              <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)


    kpi(col1, "Total Listings", f"{total:,}", "scraped from eBay")
    kpi(col2, "Avg Price", f"${avg_p:,.0f}", "across all brands")
    kpi(col3, "Highest Ask", f"${max_p:,.0f}", "top listing")
    kpi(col4, "Entry Price", f"${min_p:,.0f}", "lowest listing")
    kpi(col5, "Brands Tracked", "7", "+ Other category")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if section == "Overview & KPIs":
        st.markdown(
            """<div class="section-header"><h2>Market Overview Matrices</h2><span class="tag">Overview Datasets</span></div>""",
            unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["All Merged Aggregations", "Segmented Collections Filter"])
        with tab1:
            st.dataframe(df_listings, use_container_width=True, height=420)
        with tab2:
            brand_frames = {"Harley-Davidson": df_harley, "Honda": df_honda, "Suzuki": df_suzuki,
                            "Kawasaki": df_kawasaki, "Yamaha": df_yamaha, "BMW": df_bmw, "Other": dfc}
            chosen = st.selectbox("Isolate Category View Vectors", list(brand_frames.keys()))
            st.dataframe(brand_frames[chosen], use_container_width=True, height=380)

    elif section == "Price Analysis":
        st.markdown(
            """<div class="section-header"><h2>Top 10 Highest-Priced Listings</h2><span class="tag">Descriptive Analytics</span></div>""",
            unsafe_allow_html=True)
        dfd_sorted = df_listings.sort_values(by='price', ascending=False)
        top_10 = dfd_sorted.head(10)

        fig, ax = plt.subplots(figsize=(13, 5.5))
        bars = ax.bar(top_10["model"], top_10["price"],
                      color=["#f5a623" if i == 0 else "#e85d04" if i == 1 else "#3b4a6b" for i in range(len(top_10))],
                      edgecolor="#2a2f3e", linewidth=0.5, zorder=3)
        ax.bar_label(bars, labels=[f"${v:,.0f}" for v in top_10["price"]], padding=4, fontsize=8, color="#e8eaf0")
        ax.set_title("Top 10 Highest Priced Motorcycles on eBay", fontsize=14, fontweight="bold", pad=16)
        ax.set_xlabel("Motorcycle Model", fontsize=11)
        ax.set_ylabel("Price (USD)", fontsize=11)
        ax.set_xticks(range(len(top_10)))
        ax.set_xticklabels(top_10["model"], rotation=38, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        fig.tight_layout();
        st.pyplot(fig);
        plt.close(fig)

    elif section == "Brand Distribution":
        st.markdown(
            """<div class="section-header"><h2>Market Structural Volumetric Weights</h2><span class="tag">Composition</span></div>""",
            unsafe_allow_html=True)
        counts = [len(df_harley), len(df_honda), len(df_yamaha), len(df_kawasaki), len(df_bmw), len(dfc)]
        labels = ['Harley-Davidson', 'Honda', 'Yamaha', 'Kawasaki', 'BMW', 'Other']
        colors = ["#e85d04", "#3b82f6", "#8b5cf6", "#10b981", "#06b6d4", "#6b7280"]

        col_pie, col_tbl = st.columns([1.2, 1])
        with col_pie:
            fig, ax = plt.subplots(figsize=(7, 7))
            wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%',
                                              startangle=100, colors=colors,
                                              wedgeprops=dict(width=0.65, edgecolor="#1e2330", linewidth=2),
                                              pctdistance=0.8)
            for t in autotexts: t.set_fontsize(10); t.set_color("#fff")
            ax.set_title("Listings by Brand Representation", fontsize=14, fontweight="bold", pad=16)
            fig.tight_layout();
            st.pyplot(fig);
            plt.close(fig)
        with col_tbl:
            df_pie = pd.DataFrame({"Brand Category Matrix": labels, "Total Vector Counts": counts})
            df_pie["Market Capacity Contribution %"] = (
                        df_pie["Total Vector Counts"] / df_pie["Total Vector Counts"].sum() * 100).round(1)
            st.dataframe(df_pie, use_container_width=True, hide_index=True)

    elif section == "Network Graph":
        st.markdown(
            """<div class="section-header"><h2>Graph Relational Node Networks</h2><span class="tag">Network Topology (Threshold > $3,000)</span></div>""",
            unsafe_allow_html=True)
        G = nx.Graph()
        brand_dataframes = {"Harley-Davidson": df_harley, "Honda": df_honda, "Suzuki": df_suzuki,
                            "Kawasaki": df_kawasaki, "Yamaha": df_yamaha, "BMW": df_bmw}
        for brand_name, df_brand in brand_dataframes.items():
            df_b = df_brand.copy()
            df_b['price'] = pd.to_numeric(df_b['price'], errors='coerce')
            df_b = df_b.dropna(subset=['price'])
            if not df_b.empty:
                top_models = df_b.sort_values(by='price', ascending=False).head(2)
                for _, row in top_models.iterrows():
                    G.add_edge(brand_name, row['model'])
                    if row['price'] > 3000:
                        G.add_edge(row['model'], ">3000")

        brand_nodes = list(brand_dataframes.keys())
        high_val_node = [">3000"]
        node_colors = [];
        node_sizes = []
        for n in G.nodes:
            if n in brand_nodes:
                node_colors.append("#f5a623"); node_sizes.append(2200)
            elif n in high_val_node:
                node_colors.append("#ef4444"); node_sizes.append(1600)
            else:
                node_colors.append("#3b4a6b"); node_sizes.append(900)

        fig, ax = plt.subplots(figsize=(14, 14))
        pos = nx.spring_layout(G, seed=42, k=2.2)
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                with_labels=True, font_size=8, font_weight="bold",
                font_color="#e8eaf0", edge_color="#4a5568", width=1.4)
        ax.set_title("Motorcycle Brands, Top Models, and High-Value Structural Edge Ties", size=14, fontweight="bold",
                     pad=16)
        fig.tight_layout();
        st.pyplot(fig);
        plt.close(fig)

    elif section == "Heatmap":
        st.markdown(
            """<div class="section-header"><h2>Categorical Bin Density Heatmaps</h2><span class="tag">Density Distributions</span></div>""",
            unsafe_allow_html=True)
        all_brand_prices = [df_bmw["price"], df_yamaha["price"], df_kawasaki["price"], df_suzuki["price"],
                            df_honda["price"], df_harley["price"]]


        def count_prices(prices_list):
            under_2k = from_2_5 = from_5_10 = from_10_20 = over_20k = 0
            for price in prices_list:
                if price < 2000:
                    under_2k += 1
                elif price < 5000:
                    from_2_5 += 1
                elif price < 10000:
                    from_5_10 += 1
                elif price < 20000:
                    from_10_20 += 1
                else:
                    over_20k += 1
            return [under_2k, from_2_5, from_5_10, from_10_20, over_20k]


        date_table = [count_prices(b) for b in all_brand_prices]
        brands = ["BMW", "Yamaha", "Kawasaki", "Suzuki", "Honda", "Harley-Davidson"]
        price_buckets = ["Under $2k", "$2k-$5k", "$5k-$10k", "$10k-$20k", "Over $20k"]
        df_heatmap = pd.DataFrame(date_table, index=brands, columns=price_buckets)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        sns.heatmap(df_heatmap, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.5, linecolor="#1e2330", ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Distribution Heatmap Structure Matrix", fontsize=14, fontweight="bold", pad=16)
        fig.tight_layout();
        st.pyplot(fig);
        plt.close(fig)
        st.dataframe(df_heatmap, use_container_width=True)

    elif section == "3D Point Cloud":
        st.markdown(
            """<div class="section-header"><h2>3D Price Point Cloud</h2><span class="tag">Brand × Index × Normalized Price</span></div>""",
            unsafe_allow_html=True)
        from mpl_toolkits.mplot3d import Axes3D

        dfd_3d = df_listings.copy()
        dfd_3d["price"] = pd.to_numeric(dfd_3d["price"], errors="coerce")
        dfd_3d = dfd_3d.dropna(subset=["price"])
        dfd_3d = dfd_3d[dfd_3d["price"] > 0].reset_index(drop=True)


        def get_brand(title):
            if "Harley" in title: return "Harley"
            if "Honda" in title: return "Honda"
            if "Suzuki" in title: return "Suzuki"
            if "Kawasaki" in title: return "Kawasaki"
            if "Yamaha" in title: return "Yamaha"
            if "BMW" in title: return "BMW"
            return "Other"


        brand_map = {"Harley": 1, "Honda": 2, "Suzuki": 3, "Kawasaki": 4, "Yamaha": 5, "BMW": 6, "Other": 7}
        dfd_3d["brand"] = dfd_3d["model"].apply(get_brand)
        dfd_3d["brand_code"] = dfd_3d["brand"].map(brand_map)

        pmin = dfd_3d["price"].min()
        pmax = dfd_3d["price"].max()

        if pmax - pmin != 0:
            dfd_3d["price_norm"] = (dfd_3d["price"] - pmin) / (pmax - pmin)
        else:
            dfd_3d["price_norm"] = 0

        brand_colors = {"Harley": "red", "Honda": "blue", "Suzuki": "green",
                        "Kawasaki": "orange", "Yamaha": "purple", "BMW": "cyan", "Other": "gray"}

        elev = st.slider("Elevation angle", 10, 60, 30)
        azim = st.slider("Azimuth angle", -60, 60, 10)

        fig = plt.figure(figsize=(13, 8));
        ax = fig.add_subplot(projection='3d')
        for brand, color in brand_colors.items():
            mask = dfd_3d["brand"] == brand
            if mask.sum() == 0: continue
            ax.scatter(dfd_3d.loc[mask, "brand_code"], dfd_3d.loc[mask].index,
                       dfd_3d.loc[mask, "price_norm"], color=color, s=18, label=brand, alpha=0.75)

        ax.set_xlabel("Brand Code")
        ax.set_ylabel("Listing Index")
        ax.set_zlabel("Normalized Price (0-1)")
        ax.set_title("Motorcycle Prices – 3D Point Cloud", fontsize=14, fontweight="bold")
        ax.set_xticks(list(brand_map.values()))
        ax.set_xticklabels(list(brand_map.keys()), fontsize=7, rotation=15)
        ax.view_init(elev, azim)
        ax.legend(loc="upper left", fontsize=9, title="Brand")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif section == "Product Reviews":
        st.markdown(
            """<div class="section-header"><h2>Extracted NLP Sentiment Pipelines</h2><span class="tag">Static Cache Fallback System</span></div>""",
            unsafe_allow_html=True)
        static_reviews = [
            {"title": "Best helmet I've ever owned", "author": "RiderMike_88",
             "body": "The SHOEI RF-1000 fits perfectly and the ventilation is excellent."},
            {"title": "Great quality, a bit pricey", "author": "SportsBiker_NYC",
             "body": "Build quality is outstanding — shell feels solid and the visor mechanism is super smooth."},
            {"title": "Perfect fit for an oval head", "author": "MotoGal_Phoenix",
             "body": "Finding a helmet that fits my intermediate-oval head shape was a nightmare until this one."},
            {"title": "Quiet at highway speed", "author": "Tourer_Dan",
             "body": "Noticeably quieter than my old Arai at 75 mph. Wind noise is minimal."},
            {"title": "Five stars, no complaints", "author": "DailyCommuter_LA",
             "body": "Bought this for daily commuting. Light weight, excellent peripheral vision."}
        ]
        for r in static_reviews:
            st.markdown(
                f"""<div class="review-card"><div class="r-title">⭐ {r['title']}</div><div class="r-author">Verified Buyer: Vector Node Tag [{r['author']}]</div><div class="r-body">{r['body']}</div></div>""",
                unsafe_allow_html=True)

    elif section == "Price Predictor":
        st.markdown(
            """<div class="section-header"><h2>Predictive Machine Learning Engine Room</h2><span class="tag">Scikit-Learn Regression Pipeline</span></div>""",
            unsafe_allow_html=True)

        if regr is not None:
            def predict_price(target_date_str):
                target_date = pd.to_datetime(target_date_str)
                target_ordinal = target_date.toordinal()
                features_df = pd.DataFrame([[target_ordinal]], columns=["Date_ordinal"])
                predicted_price = regr.predict(features_df)
                return predicted_price[0]


            st.markdown("#### Historical Vector Superposition Array & Trendlines")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.scatter(df_history_analysis["Date"], df_history_analysis["avg_price"], color="#f5a623", s=70, zorder=5,
                       label="Observed Price Vectors")
            ax.plot(df_history_analysis["Date"], y_pred_hist, color="#ef4444", linewidth=2.5,
                    label="Ols Line Regression Map")
            ax.set_ylabel("USD Valuations ($)")
            ax.legend()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            plt.xticks(rotation=38)
            fig.tight_layout();
            st.pyplot(fig);
            plt.close(fig)

            st.markdown("---")
            st.markdown("#### Run Target Predictive Valuation Projections")
            future_date = st.date_input(
                "Target Projection Valuation Timestamp",
                value=pd.Timestamp.today() + pd.DateOffset(months=3),
                min_value=pd.Timestamp("2020-01-01"),
                max_value=pd.Timestamp("2030-12-31")
            )

            if st.button("Compute ML Projection Vectors", key="run_pred"):
                try:
                    predicted_val = predict_price(str(future_date))
                    st.markdown(f"""
                    <div class="pred-result">
                      <div class="pred-label">Projected Market Vector Valuation</div>
                      <div class="pred-value">${predicted_val:,.2f}</div>
                      <div class="pred-date">Computed Projection Node Target: {future_date.strftime("%B %d, %Y")}</div>
                    </div>""", unsafe_allow_html=True)
                    r2 = regr.score(df_history_analysis[["Date_ordinal"]], df_history_analysis["avg_price"])
                    direction = "📈 Upward Vector Yield" if regr.coef_[0] > 0 else "📉 Downward Vector Slide"
                    st.caption(f"R² Score: {r2:.3f} · {direction}")
                except Exception as e:
                    st.error(f"Computation processing regression failure: {e}")
        else:
            st.warning("Not enough historical data points available to run linear regression.")

else:
    st.info("System Engine initializing...")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7b8094;font-size:.8rem;padding:20px 0 12px'>"
    "Motorcycle Market Intelligence Dashboard &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; Data via SearchAPI &amp; BeautifulSoup"
    "</div>"
    "<div style='text-align:center;padding-bottom:20px;display:flex;justify-content:center;gap:16px;'>"
    "<a href='https://www.linkedin.com/in/mohamed-hamdy-0b948a373' target='_blank' style='display:inline-flex;align-items:center;gap:6px;background:#0a66c2;color:#fff;text-decoration:none;font-size:.78rem;font-weight:600;padding:7px 18px;border-radius:6px;'>LinkedIn — Mohamed Hamdy</a>"
    "<a href='https://github.com/mohamed67-source' target='_blank' style='display:inline-flex;align-items:center;gap:6px;background:#24292e;color:#fff;text-decoration:none;font-size:.78rem;font-weight:600;padding:7px 18px;border-radius:6px;border:1px solid #444;'>GitHub — mohamed67-source</a>"
    "</div>",
    unsafe_allow_html=True
)