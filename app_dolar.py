import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA (ESTILO TERMINAL US FINTECH) ---
st.set_page_config(
    page_title="USD/BRL Quant Terminal | Predictive Forex & Arbitrage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS WALL STREET / BLOOMBERG TERMINAL DARK UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #07090e;
        color: #e2e8f0;
    }

    /* Top Market Tape */
    .market-tape {
        background: #0d111a;
        border-bottom: 1px solid #1e293b;
        padding: 0.6rem 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin: -4rem -4rem 1.5rem -4rem;
    }
    .tape-item { display: inline-flex; align-items: center; gap: 0.5rem; }
    .tape-green { color: #10b981; font-weight: 600; }
    .tape-red { color: #f43f5e; font-weight: 600; }

    /* US Fintech Card Style */
    .fin-card {
        background: #0d111c;
        border: 1px solid #1e2638;
        border-radius: 12px;
        padding: 1.25rem 1.1rem;
        transition: border-color 0.2s ease;
    }
    .fin-card:hover {
        border-color: #3b82f6;
    }
    .fin-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .fin-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
    }
    .fin-sub {
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Recommendation Pill */
    .signal-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .pill-buy { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.35); }
    .pill-sell { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.35); }
    .pill-hold { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.35); }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d111c;
        border: 1px solid #1e2638;
        padding: 4px;
        border-radius: 8px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 6px 16px;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CARREGAMENTO DE DADOS COM TRATAMENTO QUANTITATIVO ---
@st.cache_data(ttl=1800)
def load_quant_data():
    try:
        df = pd.read_parquet("dolar_master_analytics.parquet")
    except Exception:
        import etl_dolar
        etl_dolar.run_etl()
        df = pd.read_parquet("dolar_master_analytics.parquet")
    return df

df = load_quant_data()

# --- TOP MARKET TAPE (HEADER DE TERMINAL) ---
ultimo = df.iloc[-1]
penultimo = df.iloc[-2]
cotacao_atual = float(ultimo['dolar_venda'])
var_diaria = ((cotacao_atual - float(penultimo['dolar_venda'])) / float(penultimo['dolar_venda'])) * 100
rsi_val = float(ultimo['rsi'])
z_val = float(ultimo['z_score'])

st.markdown(f"""
<div class="market-tape">
    <div class="tape-item">
        <span>MARKET: <strong>USD / BRL (BACEN PTAX)</strong></span>
        <span style="color:#64748b">|</span>
        <span>LAST: <strong>R$ {cotacao_atual:.4f}</strong></span>
        <span class="{'tape-green' if var_diaria >= 0 else 'tape-red'}">({var_diaria:+.2f}%)</span>
    </div>
    <div class="tape-item">
        <span>STATUS: <strong style="color:#10b981">● LIVE FEED</strong></span>
        <span style="color:#64748b">|</span>
        <span>SESSION: <strong>{ultimo['data'].strftime('%b %d, %Y').upper()}</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR (QUANT CONTROLS) ---
st.sidebar.markdown("### ⚡ QUANT PARAMETERS")
st.sidebar.caption("High-Frequency Analytics & Forecast Model")

lookback_days = st.sidebar.slider("Lookback Window (Trading Days):", 30, 500, 120)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days):", 5, 30, 14)

df_view = df.tail(lookback_days).copy()

# Regra de Decisão Quantitativa
if rsi_val <= 38 or z_val <= -1.4:
    signal_label = "BUY / LONG"
    signal_class = "pill-buy"
    signal_desc = "Asset oversold relative to 20-day mean. High statistical probability of mean reversion upwards."
elif rsi_val >= 64 or z_val >= 1.4:
    signal_label = "SELL / TAKE PROFIT"
    signal_class = "pill-sell"
    signal_desc = "Asset overbought near upper statistical barrier. Strong opportunity to lock in arbitrage profit."
else:
    signal_label = "HOLD / NEUTRAL"
    signal_class = "pill-hold"
    signal_desc = "Price oscillating within 1-sigma standard deviation. Wait for directional breakout."

# --- HERO TITLE ---
st.markdown("## **USD/BRL** Quantitative Terminal & Arbitrage")
st.markdown("<p style='color:#64748b; font-size:0.88rem; margin-top:-0.5rem; margin-bottom:1.5rem;'>Automated central bank data feed & predictive volatility models for retail foreign exchange arbitrage.</p>", unsafe_allow_html=True)

# --- FINTECH KPIS (GRID OF 4) ---
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="fin-card">
        <div class="fin-label">Spot Exchange (PTAX)</div>
        <div class="fin-val">R$ {cotacao_atual:.4f}</div>
        <div class="fin-sub" style="color: {'#10b981' if var_diaria >= 0 else '#f43f5e'};">
            {'▲' if var_diaria >= 0 else '▼'} {abs(var_diaria):.2f}% Daily Change
        </div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="fin-card">
        <div class="fin-label">Relative Strength Index (RSI 14)</div>
        <div class="fin-val">{rsi_val:.1f}</div>
        <div class="fin-sub" style="color: {'#10b981' if rsi_val < 40 else '#f43f5e' if rsi_val > 60 else '#f59e0b'};">
            {'Oversold (< 40)' if rsi_val < 40 else 'Overbought (> 60)' if rsi_val > 60 else 'Neutral Band'}
        </div>
    </div>""", unsafe_allow_html=True)

with k3:
    spread_val = float(ultimo['spread'])
    st.markdown(f"""
    <div class="fin-card">
        <div class="fin-label">Bid / Ask Spread</div>
        <div class="fin-val">R$ {spread_val:.4f}</div>
        <div class="fin-sub" style="color:#64748b;">
            Arbitrage Friction: {float(ultimo['spread_pct']):.2f}%
        </div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="fin-card">
        <div class="fin-label">Quant Recommendation</div>
        <div style="margin-top: 0.5rem;">
            <span class="signal-pill {signal_class}">{signal_label}</span>
        </div>
        <div class="fin-sub" style="color:#64748b;">
            Z-Score: {z_val:+.2f}σ
        </div>
    </div>""", unsafe_allow_html=True)

st.write("")

# --- TABS DO DASHBOARD AMERICANO ---
tab1, tab2, tab3 = st.tabs([
    "📈 Predictive Forecast Model",
    "⚡ Arbitrage & PnL Calculator",
    "📊 Volatility & Bollinger Bands"
])

# --- TAB 1: PREVISÃO QUANTITATIVA WALL STREET STYLE ---
with tab1:
    x_idx = np.arange(len(df_view))
    y_vals = df_view['dolar_venda'].values
    poly = np.polyfit(x_idx, y_vals, 1)

    fut_x = np.arange(len(df_view), len(df_view) + forecast_horizon)
    fut_dates = [df_view['data'].iloc[-1] + timedelta(days=int(i*1.42)) for i in range(1, forecast_horizon + 1)]
    fut_pred = np.poly1d(poly)(fut_x)

    vol = df_view['dolar_venda'].pct_change().std() * cotacao_atual * np.sqrt(np.arange(1, forecast_horizon + 1))
    up_band = fut_pred + (1.96 * vol)
    low_band = fut_pred - (1.96 * vol)

    fig = go.Figure()

    # Preço Histórico
    fig.add_trace(go.Scatter(
        x=df_view['data'], y=df_view['dolar_venda'],
        name="Spot Price (Actual)", line=dict(color="#3b82f6", width=2.2)
    ))

    # Projeção
    fig.add_trace(go.Scatter(
        x=fut_dates, y=fut_pred,
        name="Central Forecast", line=dict(color="#f59e0b", width=2, dash="dash")
    ))

    # Intervalo de Confiança 95%
    fig.add_trace(go.Scatter(
        x=fut_dates, y=up_band,
        name="Upper Bound (95% CI)", line=dict(color="rgba(16, 185, 129, 0.4)", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=fut_dates, y=low_band,
        name="Lower Bound (95% CI)", fill='tonexty', fillcolor='rgba(59, 130, 246, 0.08)',
        line=dict(color="rgba(244, 63, 94, 0.4)", width=1)
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#07090e",
        paper_bgcolor="#07090e",
        height=420,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="#1e2638", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2638", zeroline=False, tickformat="$.4f"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="background:#0d111c; border:1px solid #1e2638; border-radius:8px; padding:0.8rem 1.2rem; font-family:'JetBrains Mono', monospace; font-size:0.82rem; color:#94a3b8;">
        Target Projected Price at <strong>{fut_dates[-1].strftime('%Y-%m-%d')}</strong>: <strong style="color:#f8fafc;">R$ {fut_pred[-1]:.4f}</strong> 
        &nbsp;|&nbsp; Expected Range: <span style="color:#10b981;">R$ {low_band[-1]:.4f}</span> — <span style="color:#3b82f6;">R$ {up_band[-1]:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 2: CALCULADORA DE ARBITRAGEM & PnL TERMINAL ---
with tab2:
    st.markdown("### ⚡ Profit & Loss (PnL) Arbitrage Engine")
    st.caption("Calculate exact net profit, fee friction, and return on invested capital (ROIC) for USD buy-and-resell positions.")

    c_in, c_out = st.columns([1, 1.3])

    with c_in:
        capital_brl = st.number_input("Capital Invested (BRL R$):", min_value=1000.0, max_value=5000000.0, value=25000.0, step=1000.0)
        buy_price = st.number_input("Entry Buy Price (R$ / USD):", value=float(cotacao_atual), step=0.005, format="%.4f")
        target_resell = st.number_input("Target Resell Price (R$ / USD):", value=float(round(cotacao_atual * 1.045, 4)), step=0.005, format="%.4f")
        fee_pct = st.slider("Broker Fee + IOF Spread (%):", 0.0, 4.0, 1.1)

        usd_acquired = capital_brl / buy_price
        gross_return = usd_acquired * target_resell
        fees_paid = gross_return * (fee_pct / 100)
        net_pnl = gross_return - capital_brl - fees_paid
        roic = (net_pnl / capital_brl) * 100

    with c_out:
        st.markdown("#### Position Summary")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("Acquired USD Balance", f"${usd_acquired:,.2f} USD")
            st.metric("Total Return on Exit", f"R$ {(capital_brl + net_pnl):,.2f}")
        with col_p2:
            st.metric("Net Arbitrage Profit (PnL)", f"R$ {net_pnl:,.2f}", f"{roic:+.2f}% ROI", delta_color="normal")
            st.metric("Estimated Fee Friction", f"R$ {fees_paid:,.2f}")

        if net_pnl > 0:
            st.markdown(f"""
            <div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.3); padding:1rem; border-radius:8px; margin-top:1rem;">
                <strong style="color:#10b981;">● POSITIVE SPREAD DETECTED:</strong> This trade yields <strong style="color:#f8fafc;">R$ {net_pnl:,.2f}</strong> clean profit after all transactional and broker spreads are settled.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(244, 63, 94, 0.1); border:1px solid rgba(244, 63, 94, 0.3); padding:1rem; border-radius:8px; margin-top:1rem;">
                <strong style="color:#f43f5e;">● NEGATIVE SPREAD WARNING:</strong> The target exit price does not cover your broker friction and IOF taxes.
            </div>
            """, unsafe_allow_html=True)

# --- TAB 3: BOLLINGER & QUANT VOLATILITY BANDS ---
with tab3:
    st.markdown("### 📊 Mean Reversion & Volatility Envelopes")
    
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=df_view['data'], y=df_view['bollinger_up'], name="Upper Barrier (+2σ)", line=dict(color="#f43f5e", width=1, dash="dot")))
    fig_b.add_trace(go.Scatter(x=df_view['data'], y=df_view['dolar_venda'], name="Spot Price", line=dict(color="#3b82f6", width=2)))
    fig_b.add_trace(go.Scatter(x=df_view['data'], y=df_view['bollinger_low'], name="Lower Barrier (-2σ)", fill='tonexty', fillcolor='rgba(255, 255, 255, 0.02)', line=dict(color="#10b981", width=1, dash="dot")))
    fig_b.add_trace(go.Scatter(x=df_view['data'], y=df_view['mm_50d'], name="50-SMA Mean", line=dict(color="#a855f7", width=1.5)))

    fig_b.update_layout(
        template="plotly_dark",
        plot_bgcolor="#07090e",
        paper_bgcolor="#07090e",
        height=400,
        margin=dict(t=10, b=20, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="#1e2638"),
        yaxis=dict(showgrid=True, gridcolor="#1e2638"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_b, use_container_width=True)

# --- RAW DATA EXPANDER ---
with st.expander("📥 Inspect Real-Time Data Store (Parquet Table)"):
    st.dataframe(df_view.tail(50), use_container_width=True)