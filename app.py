"""
BIST 100 Dip Bulucu
TKE · Stokastik RSI · MFI · RSI
Tüm kod tek dosyada.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ══════════════════════════════════════════════════════
# BIST 100 HİSSE LİSTESİ
# ══════════════════════════════════════════════════════
BIST100 = [
    "ACSEL","ADEL","AEFES","AGESA","AKBNK","AKFEN","AKSA","AKSEN","ALARK","ALBRK",
    "ALFAS","ALKIM","ALTNY","ANACM","ARCLK","ARDYZ","ASELS","ASTOR","BERA","BFREN",
    "BIMAS","BIOEN","BRISA","BRYAT","BUCIM","CANTE","CCOLA","CEMTS","CIMSA","CLEBI",
    "CVKMD","DOAS","DOHOL","DURDO","DYOBY","ECILC","EGEEN","ENJSA","ENKAI","EREGL",
    "EUPWR","FENER","FROTO","GARAN","GESAN","GLYHO","GOLTS","GOZDE","GUBRF",
    "HALKB","HEKTS","IPEKE","ISCTR","ISFIN","ISGYO","ISYHO","ITTFK","IZOCM","JANTS",
    "KAREL","KARSN","KCAER","KCHOL","KLMSN","KONTR","KONYA","KORDS","KOZAA","KOZAL",
    "KRDMD","LOGO","MAVI","MGROS","MPARK","NETAS","ODAS","OTKAR","OYAKC","PAPIL",
    "PARSN","PEKMT","PETKM","PGSUS","PKENT","POLHO","PRKME","QUAGR","RGYAS","SAHOL",
    "SASA","SELEC","SISE","SKBNK","SMART","SOKM","SUNTK","TABGD","TATGD","TCELL",
    "THYAO","TKFEN","TOASO","TSKB","TTKOM","TTRAK","TUKAS","TUPRS","TURSG","ULKER",
    "VAKBN","VERUS","VESTL","YKBNK","ZOREN"
]

RENKLER = {
    "TKE": "#FF6B35", "StochRSI": "#4ECDC4", "MFI": "#FFE66D", "RSI": "#A8DADC",
    "fiyat": "#F7FFF7", "arkaplan": "#1A1A2E", "panel": "#16213E",
    "metin": "#E0E0E0", "yesil": "#06D6A0", "kirmizi": "#EF476F",
}

# ══════════════════════════════════════════════════════
# VERİ ÇEKME
# ══════════════════════════════════════════════════════
def veri_cek(hisse: str, periyot: str = "1y") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(f"{hisse}.IS").history(period=periyot, interval="1d", auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None

# ══════════════════════════════════════════════════════
# İNDİKATÖRLER
# ══════════════════════════════════════════════════════
def hesapla_tke(df, period=21):
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    hlc3 = (h + l + c) / 3
    momentum = c / c.shift(period) * 100
    cci = (hlc3 - hlc3.rolling(period).mean()) / (0.015 * hlc3.rolling(period).std())
    d = c.diff()
    rsi_v = 100 - 100 / (1 + d.clip(lower=0).rolling(period).mean() /
                         (-d.clip(upper=0)).rolling(period).mean().replace(0, np.nan))
    highest = h.rolling(period).max()
    lowest  = l.rolling(period).min()
    willr   = (highest - c) / (highest - lowest + 1e-10) * -100
    stosk   = (c - lowest) / (highest - lowest + 1e-10) * 100
    diff    = hlc3.diff()
    upper_s = (v * hlc3.where(diff > 0, 0)).rolling(period).sum()
    lower_s = (v * hlc3.where(diff < 0, 0)).rolling(period).sum()
    mfi     = 100 - 100 / (1 + upper_s / lower_s.replace(0, np.nan))
    h_ = pd.concat([h, c.shift(1)], axis=1).max(axis=1)
    l_ = pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    bp, tr_ = c - l_, h_ - l_
    def avg(b, t, n): return b.rolling(n).sum() / t.rolling(n).sum()
    ult = 100 * (4 * avg(bp, tr_, 7) + 2 * avg(bp, tr_, 14) + avg(bp, tr_, 28)) / 7
    return ((ult + mfi + momentum + cci + rsi_v + willr + stosk) / 7).rename("TKE")

def hesapla_stoch_rsi(df, rsi_period=21, stoch_period=21, k=3, d=3):
    c = df["Close"]
    delta = c.diff()
    rsi = 100 - 100 / (1 + delta.clip(lower=0).rolling(rsi_period).mean() /
                       (-delta.clip(upper=0)).rolling(rsi_period).mean().replace(0, np.nan))
    rmin, rmax = rsi.rolling(stoch_period).min(), rsi.rolling(stoch_period).max()
    K = ((rsi - rmin) / (rmax - rmin + 1e-10) * 100).rolling(k).mean()
    D = K.rolling(d).mean()
    return K.rename("K"), D.rename("D")

def hesapla_mfi(df, period=21):
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    hlc3 = (h + l + c) / 3
    diff = hlc3.diff()
    upper = (v * hlc3.where(diff > 0, 0)).rolling(period).sum()
    lower = (v * hlc3.where(diff < 0, 0)).rolling(period).sum()
    return (100 - 100 / (1 + upper / lower.replace(0, np.nan))).rename("MFI")

def hesapla_rsi(df, period=21):
    d = df["Close"].diff()
    return (100 - 100 / (1 + d.clip(lower=0).rolling(period).mean() /
                         (-d.clip(upper=0)).rolling(period).mean().replace(0, np.nan))).rename("RSI")

def mfi_dip_bul(mfi_serisi, varsayilan=35.0):
    dip = mfi_serisi.dropna()
    dip = dip[dip < 50]
    if len(dip) < 10:
        return varsayilan
    try:
        kde = stats.gaussian_kde(dip)
        x = np.linspace(dip.min(), dip.max(), 500)
        return float(np.clip(x[np.argmax(kde(x))], 20.0, 45.0))
    except Exception:
        return varsayilan

# ══════════════════════════════════════════════════════
# PUANLAMA
# ══════════════════════════════════════════════════════
def puan_hesapla(tke_v, sk_v, sd_v, mfi_v, rsi_v, mfi_dip=35.0):
    def _p(val, dip, tavan, maks):
        if pd.isna(val): return 0.0
        if val <= dip:   return float(maks)
        if val >= tavan: return 0.0
        return round(maks * (tavan - val) / (tavan - dip), 2)
    stoch_ort = (sk_v + sd_v) / 2 if not (pd.isna(sk_v) or pd.isna(sd_v)) else np.nan
    p = {
        "TKE":      _p(tke_v,     20,      50, 30),
        "StochRSI": _p(stoch_ort, 20,      50, 30),
        "MFI":      _p(mfi_v,     mfi_dip, 55, 30),
        "RSI":      _p(rsi_v,     40,      60, 10),
    }
    p["TOPLAM"] = round(sum(p.values()), 2)
    return p

# ══════════════════════════════════════════════════════
# ANALİZ
# ══════════════════════════════════════════════════════
def hisse_analiz(hisse, df):
    try:
        tke    = hesapla_tke(df)
        sk, sd = hesapla_stoch_rsi(df)
        mfi    = hesapla_mfi(df)
        rsi    = hesapla_rsi(df)
        mfi_dip = mfi_dip_bul(mfi)
        vals = dict(tke=float(tke.iloc[-1]), sk=float(sk.iloc[-1]),
                    sd=float(sd.iloc[-1]),   mfi=float(mfi.iloc[-1]),
                    rsi=float(rsi.iloc[-1]), fiyat=float(df["Close"].iloc[-1]))
        puan = puan_hesapla(vals["tke"], vals["sk"], vals["sd"],
                            vals["mfi"], vals["rsi"], mfi_dip)
        return {
            "Hisse": hisse, "Fiyat": round(vals["fiyat"], 2),
            "TKE": round(vals["tke"], 2), "StochRSI_K": round(vals["sk"], 2),
            "StochRSI_D": round(vals["sd"], 2), "MFI": round(vals["mfi"], 2),
            "MFI_Dip": round(mfi_dip, 2), "RSI": round(vals["rsi"], 2),
            "Puan_TKE": puan["TKE"], "Puan_StochRSI": puan["StochRSI"],
            "Puan_MFI": puan["MFI"], "Puan_RSI": puan["RSI"],
            "Puan_Toplam": puan["TOPLAM"],
            "_tke": tke, "_sk": sk, "_sd": sd, "_mfi": mfi,
            "_rsi": rsi, "_close": df["Close"],
        }
    except Exception:
        return None

# ══════════════════════════════════════════════════════
# GRAFİKLER – PLOTLY
# ══════════════════════════════════════════════════════
def plotly_detay(s):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        row_heights=[0.30, 0.18, 0.18, 0.18, 0.16],
                        vertical_spacing=0.03,
                        subplot_titles=[f"{s['Hisse']} – Fiyat (TL)",
                                        "TKE (30p)", "Stokastik RSI (30p)",
                                        "MFI (30p)", "RSI (10p)"])
    def ln(ser, name, color, row, dash="solid"):
        fig.add_trace(go.Scatter(x=ser.index, y=ser.values, name=name,
                                 line=dict(color=color, width=2, dash=dash)), row=row, col=1)
    def hl(row, y, color, label):
        fig.add_hline(y=y, row=row, col=1, line=dict(color=color, width=1, dash="dash"),
                      annotation_text=label, annotation_font=dict(size=9, color=color),
                      annotation_position="right")
    ln(s["_close"], "Fiyat",  RENKLER["fiyat"],    1)
    ln(s["_tke"],   "TKE",    RENKLER["TKE"],      2)
    hl(2, 20, RENKLER["yesil"],   "Dip(20)")
    hl(2, 80, RENKLER["kirmizi"], "Tepe(80)")
    ln(s["_sk"], "%K", RENKLER["StochRSI"], 3)
    ln(s["_sd"], "%D", "#FF9F1C",           3, dash="dot")
    hl(3, 20, RENKLER["yesil"], "Dip(20)")
    ln(s["_mfi"], "MFI", RENKLER["MFI"], 4)
    hl(4, s["MFI_Dip"], RENKLER["yesil"],   f"Dip({s['MFI_Dip']:.1f})")
    hl(4, 80,            RENKLER["kirmizi"], "Tepe(80)")
    ln(s["_rsi"], "RSI", RENKLER["RSI"], 5)
    hl(5, 40, RENKLER["yesil"],   "Dip(40)")
    hl(5, 60, RENKLER["kirmizi"], "Tepe(60)")
    fig.update_layout(template="plotly_dark", height=750,
                      paper_bgcolor=RENKLER["arkaplan"], plot_bgcolor=RENKLER["panel"],
                      font=dict(color=RENKLER["metin"], family="monospace"),
                      legend=dict(orientation="h", y=-0.04),
                      margin=dict(l=50, r=60, t=50, b=40),
                      title=dict(text=f"📊 {s['Hisse']} – İndikatör Analizi",
                                 font=dict(size=17, color=RENKLER["fiyat"])))
    return fig

def plotly_radar(s):
    cats  = ["TKE (30)", "StochRSI (30)", "MFI (30)", "RSI (10)"]
    vals  = [s["Puan_TKE"], s["Puan_StochRSI"], s["Puan_MFI"], s["Puan_RSI"]]
    maxs  = [30, 30, 30, 10]
    norm  = [v / m * 100 for v, m in zip(vals, maxs)] + [vals[0] / maxs[0] * 100]
    cats += [cats[0]]
    fig = go.Figure(go.Scatterpolar(r=norm, theta=cats, fill="toself",
                                    fillcolor="rgba(78,205,196,0.25)",
                                    line=dict(color=RENKLER["StochRSI"], width=2)))
    fig.update_layout(
        polar=dict(bgcolor=RENKLER["panel"],
                   radialaxis=dict(visible=True, range=[0, 100],
                                   color=RENKLER["metin"], tickfont=dict(size=9)),
                   angularaxis=dict(color=RENKLER["metin"])),
        paper_bgcolor=RENKLER["arkaplan"], font=dict(color=RENKLER["metin"]),
        title=dict(text=f"🎯 {s['Hisse']} – Puan Radar",
                   font=dict(size=14, color=RENKLER["fiyat"])),
        showlegend=False, height=340, margin=dict(l=40, r=40, t=50, b=30))
    return fig

def plotly_siralama(df, top_n=20):
    top = df.head(top_n).sort_values("Puan_Toplam", ascending=True)
    fig = go.Figure()
    for kolon, renk, etiket in [
        ("Puan_TKE",      RENKLER["TKE"],      "TKE (30p)"),
        ("Puan_StochRSI", RENKLER["StochRSI"], "StochRSI (30p)"),
        ("Puan_MFI",      RENKLER["MFI"],      "MFI (30p)"),
        ("Puan_RSI",      RENKLER["RSI"],      "RSI (10p)"),
    ]:
        fig.add_trace(go.Bar(y=top["Hisse"], x=top[kolon], name=etiket,
                             orientation="h", marker=dict(color=renk, opacity=0.85)))
    fig.update_layout(barmode="stack", template="plotly_dark",
                      paper_bgcolor=RENKLER["arkaplan"], plot_bgcolor=RENKLER["panel"],
                      font=dict(color=RENKLER["metin"], family="monospace"),
                      title=dict(text=f"🏆 En Yüksek Puanlı {top_n} Hisse",
                                 font=dict(size=16, color=RENKLER["fiyat"])),
                      xaxis=dict(title="Toplam Puan (maks 100)", range=[0, 100]),
                      legend=dict(orientation="h", y=-0.12),
                      height=max(400, top_n * 28),
                      margin=dict(l=70, r=30, t=60, b=80))
    return fig

def plotly_heatmap(df, top_n=30):
    top = df.head(top_n)
    kolonlar = ["Puan_TKE","Puan_StochRSI","Puan_MFI","Puan_RSI","Puan_Toplam"]
    etiket   = ["TKE","StochRSI","MFI","RSI","TOPLAM"]
    z = top[kolonlar].values.tolist()
    fig = go.Figure(go.Heatmap(z=z, x=etiket, y=top["Hisse"].tolist(),
                               colorscale="RdYlGn", zmin=0, zmax=100,
                               text=[[f"{v:.1f}" for v in row] for row in z],
                               texttemplate="%{text}", textfont=dict(size=10),
                               colorbar=dict(title="Puan")))
    fig.update_layout(template="plotly_dark", paper_bgcolor=RENKLER["arkaplan"],
                      plot_bgcolor=RENKLER["panel"], font=dict(color=RENKLER["metin"]),
                      title=dict(text=f"🌡️ Puan Isı Haritası – İlk {top_n} Hisse",
                                 font=dict(size=16, color=RENKLER["fiyat"])),
                      height=max(500, top_n * 22), margin=dict(l=80, r=30, t=60, b=40))
    return fig

# ══════════════════════════════════════════════════════
# GRAFİKLER – MATLOTLİB
# ══════════════════════════════════════════════════════
def mpl_detay(s):
    fig = plt.figure(figsize=(14, 10), facecolor=RENKLER["arkaplan"])
    gs  = gridspec.GridSpec(5, 1, figure=fig, hspace=0.08,
                            height_ratios=[2.5, 1.2, 1.2, 1.2, 1.0])
    axler = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axler:
        ax.set_facecolor(RENKLER["panel"])
        ax.tick_params(colors=RENKLER["metin"], labelsize=8)
        ax.spines[:].set_color("#333355")
    x = lambda s_: range(len(s_))
    axler[0].plot(x(s["_close"]), s["_close"].values, color=RENKLER["fiyat"], lw=1.5)
    axler[0].set_ylabel("Fiyat", color=RENKLER["metin"], fontsize=9)
    axler[0].set_title(f"{s['Hisse']} – Toplam Puan: {s['Puan_Toplam']:.1f}/100",
                       color=RENKLER["fiyat"], fontsize=12)
    for ax, ser, renk, lbl, dip, tepe in [
        (axler[1], s["_tke"], RENKLER["TKE"], "TKE", 20, 80),
        (axler[3], s["_mfi"], RENKLER["MFI"], "MFI", s["MFI_Dip"], 80),
        (axler[4], s["_rsi"], RENKLER["RSI"], "RSI", 40, 60),
    ]:
        ax.plot(x(ser), ser.values, color=renk, lw=1.5)
        ax.axhline(dip,  color=RENKLER["yesil"],   lw=0.8, ls="--")
        ax.axhline(tepe, color=RENKLER["kirmizi"], lw=0.8, ls="--")
        ax.set_ylabel(lbl, color=RENKLER["metin"], fontsize=9)
        ax.set_ylim(0, 100)
    axler[2].plot(x(s["_sk"]), s["_sk"].values, color=RENKLER["StochRSI"], lw=1.5, label="%K")
    axler[2].plot(x(s["_sd"]), s["_sd"].values, color="#FF9F1C", lw=1, ls=":", label="%D")
    axler[2].axhline(20, color=RENKLER["yesil"], lw=0.8, ls="--")
    axler[2].set_ylabel("StochRSI", color=RENKLER["metin"], fontsize=9)
    axler[2].set_ylim(0, 100)
    axler[2].legend(fontsize=7, facecolor=RENKLER["panel"],
                    labelcolor=RENKLER["metin"], loc="upper left")
    for ax in axler[:-1]:
        ax.tick_params(labelbottom=False)
    axler[4].set_xlabel("Gün", color=RENKLER["metin"], fontsize=8)
    plt.tight_layout()
    return fig

def mpl_puan_bar(s):
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=RENKLER["arkaplan"])
    ax.set_facecolor(RENKLER["panel"])
    cats    = ["TKE\n(30p)", "StochRSI\n(30p)", "MFI\n(30p)", "RSI\n(10p)"]
    puanlar = [s["Puan_TKE"], s["Puan_StochRSI"], s["Puan_MFI"], s["Puan_RSI"]]
    renkler = [RENKLER["TKE"], RENKLER["StochRSI"], RENKLER["MFI"], RENKLER["RSI"]]
    bars = ax.bar(cats, puanlar, color=renkler, edgecolor="none", width=0.55)
    for bar, p in zip(bars, puanlar):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{p:.1f}", ha="center", va="bottom", color=RENKLER["metin"], fontsize=9)
    ax.set_ylim(0, 35)
    ax.set_title(f"{s['Hisse']} – Puan Dağılımı  (Toplam: {s['Puan_Toplam']:.1f}/100)",
                 color=RENKLER["fiyat"], fontsize=10)
    ax.tick_params(colors=RENKLER["metin"])
    ax.spines[:].set_color("#333355")
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════
# STREAMLIT ARAYÜZ
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="BIST 100 Dip Bulucu", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background:#0f0f1a; color:#e0e0e0; }
.stApp { background-color: #0f0f1a; }
h1 { font-size:2rem !important; color:#FF6B35 !important; letter-spacing:-1px; }
h2 { color:#4ECDC4 !important; } h3 { color:#FFE66D !important; }
.kart { background:#16213E; border:1px solid #1a1a3e; border-radius:12px;
        padding:16px 20px; text-align:center; }
.kart-val { font-size:1.9rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.kart-lbl { font-size:0.72rem; color:#888; margin-top:4px; }
.yesil { color:#06D6A0; } .sari { color:#FFE66D; }
.turuncu { color:#FF9F1C; } .kirmizi { color:#EF476F; }
.stButton > button { background:linear-gradient(135deg,#FF6B35,#FF9F1C); color:white;
    font-weight:700; border:none; border-radius:8px; padding:.5rem 1.5rem; }
</style>""", unsafe_allow_html=True)

st.markdown("# 📉 BIST 100 Dip Bulucu")
st.markdown("**TKE · Stokastik RSI · MFI · RSI** kombinasyonuyla dipte olan hisseleri bulur")
st.divider()

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    periyot = st.selectbox("Veri Periyodu", ["6mo","1y","2y"], index=1,
                           format_func=lambda x: {"6mo":"6 Ay","1y":"1 Yıl","2y":"2 Yıl"}[x])
    min_puan = st.slider("Minimum Puan Filtresi", 0, 100, 40, step=5)
    top_n    = st.slider("Sıralamada Gösterilecek Hisse", 5, 50, 20, step=5)
    st.divider()
    st.caption("TKE: period=21 | StochRSI: 21/21 | MFI: 21 | RSI: 21")
    st.caption("TKE 30p · StochRSI 30p · MFI 30p · RSI 10p = 100p")
    st.divider()
    analiz_baslat = st.button("🔍 Tüm BIST 100'ü Tara", use_container_width=True)
    st.divider()
    st.markdown("**Tek Hisse Analizi**")
    tek_hisse = st.selectbox("Hisse Seç", sorted(BIST100))
    tek_btn   = st.button("📈 Analiz Et", use_container_width=True)

# ── Session state ─────────────────────────────────────
for k in ["df_sonuc", "sonuclar", "tek_sonuc"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Hisse detay görünümü ──────────────────────────────
def goster_detay(s):
    st.markdown(f"## 📈 {s['Hisse']} – Detay Analizi")
    c1,c2,c3,c4,c5 = st.columns(5)
    kart_data = [
        ("TOPLAM", s["Puan_Toplam"], 100),
        ("TKE",    s["Puan_TKE"],    30),
        ("StochRSI", s["Puan_StochRSI"], 30),
        ("MFI",    s["Puan_MFI"],    30),
        ("RSI",    s["Puan_RSI"],    10),
    ]
    for col, (lbl, val, maks) in zip([c1,c2,c3,c4,c5], kart_data):
        r = "yesil" if val >= maks*0.7 else "sari" if val >= maks*0.4 else "kirmizi"
        col.markdown(f'<div class="kart"><div class="kart-val {r}">{val:.1f}</div>'
                     f'<div class="kart-lbl">{lbl} / {maks}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(plotly_detay(s), use_container_width=True)
    with col_b:
        st.plotly_chart(plotly_radar(s), use_container_width=True)
        st.pyplot(mpl_puan_bar(s))
    with st.expander("🖨️ Statik PNG İndir"):
        fig_mpl = mpl_detay(s)
        buf = io.BytesIO()
        fig_mpl.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.image(buf)
        st.download_button("⬇️ PNG İndir", data=buf,
                           file_name=f"{s['Hisse']}_analiz.png", mime="image/png")

# ── Tam tarama ────────────────────────────────────────
if analiz_baslat:
    st.info("📡 Veriler çekiliyor... (~2 dk sürebilir)")
    prog = st.progress(0)
    txt  = st.empty()
    veriler, toplam = {}, len(BIST100)
    for i, hisse in enumerate(BIST100):
        df_h = veri_cek(hisse, periyot)
        if df_h is not None:
            veriler[hisse] = df_h
        prog.progress((i+1)/toplam)
        txt.text(f"⏳ {hisse} ({i+1}/{toplam})")
        time.sleep(0.04)
    txt.text("📊 Puanlama hesaplanıyor...")
    sonuclar = [hisse_analiz(h, d) for h, d in veriler.items()]
    sonuclar = [s for s in sonuclar if s]
    df_sonuc = pd.DataFrame([{k:v for k,v in s.items() if not k.startswith("_")}
                              for s in sonuclar])
    df_sonuc = df_sonuc.sort_values("Puan_Toplam", ascending=False).reset_index(drop=True)
    prog.empty(); txt.empty()
    st.session_state.df_sonuc = df_sonuc
    st.session_state.sonuclar = sonuclar
    st.success(f"✅ {len(df_sonuc)} hisse analiz edildi!")

# ── Tarama sonuçları ──────────────────────────────────
if st.session_state.df_sonuc is not None:
    df  = st.session_state.df_sonuc
    df_f = df[df["Puan_Toplam"] >= min_puan]
    st.markdown("## 🏆 Dip Puanlama Sonuçları")
    c1,c2,c3,c4 = st.columns(4)
    top1 = df.iloc[0]
    for col, val, lbl in [
        (c1, len(df_f),                        "Filtreden Geçen"),
        (c2, top1["Hisse"],                     f"En İyi: {top1['Puan_Toplam']:.1f}p"),
        (c3, f"{df['Puan_Toplam'].mean():.1f}", "Ortalama Puan"),
        (c4, (df["Puan_Toplam"] >= 50).sum(),   "50+ Puan"),
    ]:
        col.markdown(f'<div class="kart"><div class="kart-val sari">{val}</div>'
                     f'<div class="kart-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    tab1, tab2, tab3 = st.tabs(["📊 Bar Sıralama", "🌡️ Isı Haritası", "📋 Tablo"])
    with tab1:
        st.plotly_chart(plotly_siralama(df_f, top_n), use_container_width=True)
    with tab2:
        st.plotly_chart(plotly_heatmap(df_f, top_n), use_container_width=True)
    with tab3:
        kolonlar = ["Hisse","Fiyat","Puan_Toplam","TKE","Puan_TKE",
                    "StochRSI_K","StochRSI_D","Puan_StochRSI",
                    "MFI","MFI_Dip","Puan_MFI","RSI","Puan_RSI"]
        st.dataframe(df_f[kolonlar].style.background_gradient(
            subset=["Puan_Toplam"], cmap="RdYlGn").format(precision=2),
            use_container_width=True, height=500)
        csv = df_f[kolonlar].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV İndir", data=csv,
                           file_name="bist100_dip.csv", mime="text/csv")
    if st.session_state.sonuclar:
        st.divider()
        st.markdown("### 🔎 Listeden Hisse Detayı")
        secim = st.selectbox("Hisse", [s["Hisse"] for s in st.session_state.sonuclar],
                             key="liste_sec")
        s_sec = next((s for s in st.session_state.sonuclar if s["Hisse"] == secim), None)
        if s_sec:
            goster_detay(s_sec)

# ── Tek hisse analizi ─────────────────────────────────
if tek_btn:
    with st.spinner(f"📡 {tek_hisse} çekiliyor..."):
        df_tek = veri_cek(tek_hisse, periyot)
    if df_tek is None:
        st.error(f"❌ {tek_hisse} için veri alınamadı.")
    else:
        s = hisse_analiz(tek_hisse, df_tek)
        st.session_state.tek_sonuc = s if s else None
        if not s:
            st.error("❌ Analiz hesaplanamadı.")

if st.session_state.tek_sonuc and not analiz_baslat:
    goster_detay(st.session_state.tek_sonuc)

# ── Footer ────────────────────────────────────────────
st.divider()
st.markdown("""<div style='text-align:center;color:#444;font-size:.75rem;'>
BIST 100 Dip Bulucu · TKE (Kıvanç Özbilgiç) · Stokastik RSI · MFI · RSI<br>
⚠️ Bu uygulama yatırım tavsiyesi değildir. Veriler Yahoo Finance'den çekilmektedir.
</div>""", unsafe_allow_html=True)
