import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import ast
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Dashboard Sentimen BUMI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background:#1e2130; border:1px solid #2e3250;
    border-radius:12px; padding:16px 20px;
}
[data-testid="metric-container"] label {
    color:#8892b0 !important; font-size:0.78rem !important;
    text-transform:uppercase; letter-spacing:1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#ccd6f6 !important; font-size:1.8rem !important; font-weight:700;
}
.sh { color:#64ffda; font-size:1.05rem; font-weight:600;
      border-left:4px solid #64ffda; padding-left:10px; margin:20px 0 12px 0; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────
BG    = "#1e2130"
TEXT  = "#ccd6f6"
BLUE  = "#5A8FBF"
RED   = "#E07B54"
GREEN = "#6DBF6D"

# ── LOAD DATA ────────────────────────────────
@st.cache_data
def load_data(f):
    df = pd.read_csv(f)
    df["tanggal_dt"] = pd.to_datetime(df["tanggal"], errors="coerce", utc=True)
    # parse kolom list yang tersimpan sebagai string
    for col in ["stems", "tokens", "normalized", "no_stopword", "negation"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

# ── PLOT HELPERS ─────────────────────────────
def new_fig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#2e3250")
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT); ax.title.set_color(TEXT)
    return fig, ax

def fig_hist(series, title, xlabel, color):
    fig, ax = new_fig()
    ax.hist(series.dropna(), bins=30, color=color, alpha=0.85, edgecolor=BG)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Jumlah Komentar")
    plt.tight_layout(); return fig

def fig_bar(keys, vals, title, color, horiz=False):
    keys, vals = list(keys), list(vals)
    fig, ax = new_fig()
    if horiz:
        ax.barh(keys, vals, color=color); ax.invert_yaxis()
    else:
        ax.bar(keys, vals, color=color)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_title(title); plt.tight_layout(); return fig

def fig_trend(tren):
    fig, ax = new_fig(12, 4)
    ax.plot(tren.index, tren.values, color=BLUE, linewidth=1.5, marker="o", markersize=3)
    ax.fill_between(tren.index, tren.values, alpha=0.15, color=BLUE)
    if len(tren) > 0:
        peak = tren.idxmax()
        ax.annotate(f"Puncak: {int(tren.max())}",
                    xy=(peak, tren.max()), xytext=(peak, tren.max()*1.15),
                    ha="center", fontsize=9, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_title("Tren Komentar BUMI per Hari")
    ax.set_xlabel("Tanggal"); ax.set_ylabel("Jumlah Komentar")
    plt.tight_layout(); return fig

def fig_wc(text, title="WordCloud"):
    if not text or not text.strip(): text = "kosong"
    wc = WordCloud(width=800, height=350, background_color=BG,
                   colormap="cool", max_words=150).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title(title, color=TEXT); plt.tight_layout(); return fig

def fig_pie(pos, neg, neu):
    sizes  = [max(neg,0), max(pos,0), max(neu,0)]
    if sum(sizes) == 0: sizes = [1,1,1]
    labels = ["Negatif","Positif","Netral"]
    colors = [RED, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
    for t in texts + autotexts: t.set_color(TEXT)
    ax.set_title("Distribusi Sentimen", color=TEXT)
    plt.tight_layout(); return fig

def fig_cm(cm):
    fig, ax = new_fig(6, 4)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negatif","Positif"],
                yticklabels=["Negatif","Positif"],
                ax=ax, linewidths=0.5, linecolor="#2e3250")
    ax.set_title("Confusion Matrix"); ax.set_ylabel("Aktual"); ax.set_xlabel("Prediksi")
    plt.tight_layout(); return fig

def get_bigram(corpus, n=10):
    try:
        vec = CountVectorizer(ngram_range=(2,2), max_features=1000)
        vec.fit(corpus)
        bag = vec.transform(corpus)
        sw  = bag.sum(axis=0)
        freq = [(w, sw[0, idx]) for w, idx in vec.vocabulary_.items()]
        return sorted(freq, key=lambda x: x[1], reverse=True)[:n]
    except Exception:
        return []

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 BUMI Sentiment Dashboard")
    st.markdown("Analisis sentimen komentar saham **BUMI** dari Stockbit.")
    st.divider()
    uploaded = st.file_uploader(
        "📁 Upload CSV labeled",
        type=["csv"],
        help="Upload file komentar_BUMI_labeled.csv"
    )
    st.divider()
    st.markdown("<small style='color:#8892b0'>Analisis Diskusi Teks Saham<br>Model: Naive Bayes + InSet Lexicon<br>Data: Stockbit Stream</small>",
                unsafe_allow_html=True)

# ── LOAD ─────────────────────────────────────
st.markdown("# 📊 Dashboard Analisis Sentimen Saham BUMI")
st.markdown("Klasifikasi komentar investor Stockbit — **Naive Bayes** & **Leksikon InSet**.")
st.divider()

if uploaded:
    df = load_data(uploaded)
    st.success(f"✅ {len(df):,} komentar berhasil dimuat!")
else:
    st.info("⬆️ Upload file **komentar_BUMI_labeled.csv** di sidebar untuk memulai.")
    st.stop()

# ── HITUNG STATISTIK ─────────────────────────
lc    = df["label"].value_counts()
pos   = int(lc.get(1, 0))
neg   = int(lc.get(0, 0))
neu   = int(lc.get(2, 0))
total = len(df)

# ── TABS ─────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview", "🔍 EDA", "💬 Sentimen", "🤖 Naive Bayes", "📄 Data"
])

# ════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sh">Ringkasan Data</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 Total Komentar", f"{total:,}")
    c2.metric("✅ Positif",  f"{pos:,}",  f"{pos/total*100:.1f}%")
    c3.metric("❌ Negatif",  f"{neg:,}",  f"{neg/total*100:.1f}%")
    c4.metric("➖ Netral",   f"{neu:,}",  f"{neu/total*100:.1f}%")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sh">Distribusi Sentimen</div>', unsafe_allow_html=True)
        st.pyplot(fig_pie(pos, neg, neu))
    with col2:
        st.markdown('<div class="sh">Tren Komentar per Hari</div>', unsafe_allow_html=True)
        tren = df.set_index("tanggal_dt").resample("D")["stream_id"].count().fillna(0)
        st.pyplot(fig_trend(tren))

    st.divider()
    st.markdown('<div class="sh">WordCloud Semua Komentar</div>', unsafe_allow_html=True)
    st.pyplot(fig_wc(" ".join(df["stem_text"].dropna().astype(str)), "WordCloud Komentar BUMI"))

# ════════════════════════════════════════════
#  TAB 2 — EDA
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sh">Distribusi Panjang Dokumen & Token</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.pyplot(fig_hist(df["length"],      "Distribusi Panjang Dokumen", "Panjang Karakter", BLUE))
    with c2: st.pyplot(fig_hist(df["token_count"], "Distribusi Jumlah Token",   "Jumlah Token",     RED))

    st.markdown('<div class="sh">Distribusi Likes & Replies</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3: st.pyplot(fig_hist(df["likes"],   "Distribusi Likes",   "Jumlah Likes",   BLUE))
    with c4: st.pyplot(fig_hist(df["replies"], "Distribusi Replies", "Jumlah Replies", RED))

    st.markdown('<div class="sh">Top 10 User Paling Aktif</div>', unsafe_allow_html=True)
    top_u = df["username"].value_counts().head(10)
    st.pyplot(fig_bar(top_u.index, top_u.values, "Top 10 User Paling Aktif", BLUE, horiz=True))

    st.markdown('<div class="sh">10 Kata Paling Sering Muncul</div>', unsafe_allow_html=True)
    words  = " ".join(df["stem_text"].dropna().astype(str)).lower().split()
    common = dict(Counter(words).most_common(10))
    st.pyplot(fig_bar(common.keys(), common.values(), "10 Kata Paling Sering Muncul", BLUE))

    st.markdown('<div class="sh">Top 10 Bigram</div>', unsafe_allow_html=True)
    bg = get_bigram(df["stem_text"].dropna())
    if bg:
        bg_df = pd.DataFrame(bg, columns=["Bigram","Frekuensi"])
        st.pyplot(fig_bar(bg_df["Bigram"], bg_df["Frekuensi"], "Top 10 Bigram", GREEN, horiz=True))

    st.divider()
    st.markdown("**Statistik Deskriptif**")
    st.dataframe(df[["length","token_count","likes","replies"]].describe().round(2), use_container_width=True)

# ════════════════════════════════════════════
#  TAB 3 — SENTIMEN
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sh">Distribusi Label Sentimen</div>', unsafe_allow_html=True)
    label_map = {0:"Negatif", 1:"Positif", 2:"Netral"}
    lc2 = df["label"].value_counts().sort_index()
    lc2.index = [label_map[i] for i in lc2.index]

    c1, c2 = st.columns([1,2])
    with c1:
        st.dataframe(pd.DataFrame({"Sentimen":lc2.index,"Jumlah":lc2.values}).set_index("Sentimen"),
                     use_container_width=True)
    with c2:
        fig, ax = new_fig(8,4)
        ax.bar(lc2.index, lc2.values, color=[RED, BLUE, GREEN])
        ax.set_title("Distribusi Sentimen (Leksikon InSet)")
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="sh">WordCloud per Sentimen</div>', unsafe_allow_html=True)
    wc1, wc2, wc3 = st.columns(3)
    for col, lbl, name in [(wc1,1,"Positif"),(wc2,0,"Negatif"),(wc3,2,"Netral")]:
        subset = df[df["label"]==lbl]["stem_text"].dropna()
        with col:
            st.markdown(f"**{name}** ({len(subset):,})")
            st.pyplot(fig_wc(" ".join(subset.astype(str)), name))

    st.markdown('<div class="sh">Bigram per Sentimen</div>', unsafe_allow_html=True)
    for lbl, name, color in [(1,"Positif",BLUE),(0,"Negatif",RED),(2,"Netral",GREEN)]:
        corpus = df[df["label"]==lbl]["stem_text"].dropna()
        if len(corpus) < 5: continue
        bg2 = get_bigram(corpus)
        if not bg2: continue
        bg2_df = pd.DataFrame(bg2, columns=["Bigram","Frekuensi"])
        st.markdown(f"**{name}**")
        st.pyplot(fig_bar(bg2_df["Bigram"], bg2_df["Frekuensi"], f"Bigram — {name}", color, horiz=True))

    st.markdown('<div class="sh">Contoh Komentar per Sentimen</div>', unsafe_allow_html=True)
    for lbl, name in [(1,"Positif"),(0,"Negatif"),(2,"Netral")]:
        with st.expander(f"💬 {name}"):
            sample = df[df["label"]==lbl][["komentar","sentiment_score"]].head(5)
            st.dataframe(sample.rename(columns={"komentar":"Komentar","sentiment_score":"Skor"}),
                         use_container_width=True)

# ════════════════════════════════════════════
#  TAB 4 — NAIVE BAYES
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sh">Klasifikasi Naive Bayes (Positif vs Negatif)</div>', unsafe_allow_html=True)
    df_m = df[df["label"] != 2].copy()

    col_s, col_f = st.columns(2)
    test_size = col_s.slider("Ukuran Test Set (%)", 10, 40, 20, 5) / 100
    max_feat  = col_f.slider("Max Features TF-IDF", 100, 2000, 1000, 100)

    if st.button("🚀 Jalankan Model Naive Bayes", use_container_width=True):
        with st.spinner("Training model..."):
            X_tr, X_te, y_tr, y_te = train_test_split(
                df_m["stem_text"], df_m["label"],
                test_size=test_size, random_state=42, stratify=df_m["label"])
            tfidf = TfidfVectorizer(max_features=max_feat)
            Xtr_v = tfidf.fit_transform(X_tr)
            Xte_v = tfidf.transform(X_te)
            nb    = MultinomialNB()
            nb.fit(Xtr_v, y_tr)
            y_pred = nb.predict(Xte_v)
            acc = accuracy_score(y_te, y_pred)
            cm  = confusion_matrix(y_te, y_pred)
            cr  = classification_report(y_te, y_pred,
                    target_names=["Negatif","Positif"], output_dict=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Accuracy",       f"{acc*100:.2f}%")
        m2.metric("📊 Data Train",      f"{len(X_tr):,}")
        m3.metric("🧪 Data Test",       f"{len(X_te):,}")
        m4.metric("📐 TF-IDF Features", f"{Xtr_v.shape[1]:,}")

        st.divider()
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Confusion Matrix**")
            st.pyplot(fig_cm(cm))
        with cb:
            st.markdown("**Classification Report**")
            cr_df = pd.DataFrame(cr).T.drop("accuracy", errors="ignore")
            st.dataframe(cr_df.round(3), use_container_width=True)
    else:
        st.info("👆 Klik tombol di atas untuk menjalankan model.")

# ════════════════════════════════════════════
#  TAB 5 — DATA
# ════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sh">Data Komentar</div>', unsafe_allow_html=True)
    search = st.text_input("🔎 Filter komentar:")
    lf = st.multiselect("Filter Sentimen:", ["Positif","Negatif","Netral"],
                        default=["Positif","Negatif","Netral"])
    linv = {"Positif":1,"Negatif":0,"Netral":2}
    view = df[df["label"].isin([linv[x] for x in lf])]
    if search:
        view = view[view["komentar"].str.contains(search, case=False, na=False)]

    cols = [c for c in ["username","komentar","stem_text","label","sentiment_score","likes","replies","tanggal"] if c in view.columns]
    st.markdown(f"Menampilkan **{len(view):,}** baris")
    st.dataframe(view[cols].rename(columns={
        "username":"User","komentar":"Komentar","stem_text":"Setelah Preprocessing",
        "label":"Label","sentiment_score":"Skor","likes":"Likes","replies":"Replies","tanggal":"Tanggal"
    }), use_container_width=True, height=420)

    st.divider()
    st.download_button("⬇️ Download CSV", data=view[cols].to_csv(index=False, encoding="utf-8-sig"),
                       file_name="sentimen_BUMI.csv", mime="text/csv", use_container_width=True)
