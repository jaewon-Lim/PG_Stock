import os
import streamlit as st
import requests
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="📈 주식 매수·매도 도우미", layout="wide")
st.title("📈 주식 매수·매도 도우미 (베타)")

# CSS (가로형 알럿)
st.markdown("""
<style>
.alert-inline{
  display:flex; align-items:center; gap:10px;
  padding:10px 12px; border-radius:8px;
  background:#fff2f2; border:1px solid #ffd9d9;
  color:#b00020; font-size:14px;
}
.alert-inline .icon{font-size:16px; line-height:1;}
</style>
""", unsafe_allow_html=True)

# 세션 초기화
if "symbol" not in st.session_state: st.session_state.symbol = "TSLA"
if "analysis" not in st.session_state: st.session_state.analysis = None
if "insights" not in st.session_state: st.session_state.insights = None
if "last_error" not in st.session_state: st.session_state.last_error = None
if "auto_refresh" not in st.session_state: st.session_state.auto_refresh = False
if "period" not in st.session_state: st.session_state.period = "1y"
if "interval" not in st.session_state: st.session_state.interval = "1d"

#BACKEND = "http://localhost:8000"
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

# API
def fetch_analysis(symbol, period, interval):
    res = requests.post(f"{BACKEND}/stock/analyze", json={"symbol": symbol, "period": period, "interval": interval}, timeout=30)
    if res.status_code == 404:
        raise Exception("검색결과 없음 — 주식종목(티커)을 확인해주세요.")
    res.raise_for_status()
    return res.json()

def fetch_insights(symbol, period):
    res = requests.post(f"{BACKEND}/stock/insights", json={"symbol": symbol, "period": period}, timeout=60)
    if res.status_code == 404:
        raise Exception("검색결과 없음 — 주식종목(티커)을 확인해주세요.")
    res.raise_for_status()
    return res.json()

# 매핑
def tf_to_interval(tf):
    return {"1분":"1m","5분":"5m","15분":"15m","60분":"60m","일봉":"1d","주봉":"1wk","월봉":"1mo"}[tf]

def default_period_for_interval(interval):
    if interval == "1m": return "5d"
    if interval in ["5m","15m","30m"]: return "1mo"
    if interval == "60m": return "1y"
    if interval == "1d": return "1y"
    if interval == "1wk": return "5y"
    if interval == "1mo": return "10y"
    return "1y"

# 콜백
def on_tf_change():
    interval = tf_to_interval(st.session_state.tf_label)
    st.session_state.interval = interval
    st.session_state.period = default_period_for_interval(interval)
    if st.session_state.auto_refresh and st.session_state.symbol:
        try:
            st.session_state.analysis = fetch_analysis(st.session_state.symbol, st.session_state.period, st.session_state.interval)
            # 필요시 AI 코멘트도 갱신하려면 다음 줄 주석 해제
            # st.session_state.insights = fetch_insights(st.session_state.symbol, st.session_state.period)
            st.session_state.last_error = None
        except Exception as e:
            st.session_state.last_error = str(e)

def on_period_change():
    if st.session_state.auto_refresh and st.session_state.symbol:
        try:
            st.session_state.analysis = fetch_analysis(st.session_state.symbol, st.session_state.period, st.session_state.interval)
            # st.session_state.insights = fetch_insights(st.session_state.symbol, st.session_state.period)
            st.session_state.last_error = None
        except Exception as e:
            st.session_state.last_error = str(e)

# 입력/컨트롤
symbol_input = st.text_input("티커 입력 (예: TSLA, AAPL, NVDA)", value=st.session_state.symbol)

bcol_btn, bcol_msg = st.columns([2, 8])
with bcol_btn:
    if st.button("분석하기", type="primary"):
        try:
            st.session_state.symbol = symbol_input.strip().upper()
            st.session_state.analysis = fetch_analysis(st.session_state.symbol, st.session_state.period, st.session_state.interval)
            st.session_state.insights = fetch_insights(st.session_state.symbol, st.session_state.period)
            st.session_state.auto_refresh = True
            st.session_state.last_error = None
        except Exception as e:
            st.session_state.last_error = str(e)
with bcol_msg:
    if st.session_state.last_error:
        st.markdown(f"""<div class="alert-inline"><span class="icon">❌</span>
        <span>{st.session_state['last_error']}</span></div>""", unsafe_allow_html=True)

col_tf, col_pd = st.columns(2)
with col_tf:
    tf_options = ["1분","5분","15분","60분","일봉","주봉","월봉"]
    current_tf = {"1m":"1분","5m":"5분","15m":"15분","60m":"60분","1d":"일봉","1wk":"주봉","1mo":"월봉"}[st.session_state.interval]
    st.radio("타임프레임", tf_options, index=tf_options.index(current_tf), horizontal=True, key="tf_label", on_change=on_tf_change)
with col_pd:
    period_options = ["5d","1mo","3mo","6mo","1y","5y","10y"]
    if st.session_state.period not in period_options:
        period_options = [st.session_state.period] + period_options
    st.selectbox("기간", period_options,index=period_options.index(st.session_state.period),key="period",on_change=on_period_change)

# 분석 결과
if st.session_state.analysis:
    data = st.session_state.analysis

    st.subheader(f"요약: {data['info']['symbol']}")

    # --- KPI 영역: 1줄 6개 지표 (배당 포함) ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("종가", data["info"].get("last_close", "-"))
    c2.metric("RSI(14)", data["info"].get("rsi14", "-"))
    c3.metric("SMA50", data["info"].get("sma50", "-"))
    c4.metric("PER", data["info"].get("pe_ratio", "-"))

    dy = data["info"].get("dividend_yield")
    c5.metric("배당수익률(12M)", f"{dy}%" if dy is not None else "-")

    last_div = data["info"].get("last_dividend")
    last_div_dt = data["info"].get("last_dividend_date")
    c6.metric("최근 배당", f"{last_div} ( {last_div_dt} )" if last_div is not None and last_div_dt else (str(last_div) if last_div is not None else "-"))

    st.caption(f"🔶 최근 골든크로스: {data['info'].get('last_golden_cross') or '없음'} / 🔻 최근 데드크로스: {data['info'].get('last_dead_cross') or '없음'}")

    decision, score = data["headline"]["decision"], data["headline"]["score"]
    if score >= 90: st.success(f"적극매수 ({score})")
    elif score >= 70: st.success(f"매수추천 ({score})")
    elif score >= 50: st.info(f"관망추천 ({score})")
    else: st.warning(f"매수비추 ({score})")

    df = pd.DataFrame(data["prices"])

    with st.expander("표시 옵션", expanded=True):
        show_ma20 = st.checkbox("SMA20", True, key="show_ma20")
        show_ma50 = st.checkbox("SMA50", True, key="show_ma50")
        show_ma200 = st.checkbox("SMA200", True, key="show_ma200")
        show_volume = st.checkbox("거래량", True, key="show_volume")

    # 차트: 캔들 + 점선 이평 + 거래량
    rows = 2 if show_volume else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3] if rows==2 else [1.0], vertical_spacing=0.05)

    fig.add_trace(
        go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
            increasing_line_color="red",   increasing_fillcolor="red",
            decreasing_line_color="blue",  decreasing_fillcolor="blue"
        ),
        row=1, col=1
    )

    def add_ma(col, label):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(
                go.Scatter(x=df["date"], y=df[col], name=label, mode="lines",
                           line=dict(width=1.2, dash="dot"), opacity=0.85),
                row=1, col=1
            )
    if show_ma20:  add_ma("sma20", "SMA20")
    if show_ma50:  add_ma("sma50", "SMA50")
    if show_ma200: add_ma("sma200", "SMA200")

    if show_volume and "volume" in df.columns:
        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.6), row=2, col=1)

    fig.update_layout(
        title=f"{st.session_state.symbol} ({st.session_state.period}, {st.session_state.interval})",
        hovermode="x unified", showlegend=True, xaxis_rangeslider_visible=False,
        margin=dict(t=50, r=20, l=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # AI 코멘트 (필드만)
    if st.session_state.insights:
        ins = st.session_state.insights
        st.header("🧠 AI 코멘트")
        st.subheader("회사 한눈 요약");      st.write(ins.get("company_overview", "—"))
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("R&D 평가");      st.write(ins.get("rnd_assessment", "—"))
        with cc2:
            st.subheader("전망(1~2년)");    st.write(ins.get("outlook", "—"))

        st.subheader("관련 링크 (최대 3개)")
        links = ins.get("top_links", []) or []
        for i, item in enumerate(links[:3], start=1):
            title = item.get("title", "링크"); src = item.get("source", ""); url = item.get("url", "")
            if url:
                st.markdown(f"{i}. [{title}]({url}) — {src}")
            else:
                st.markdown(f"{i}. {title} — {src}")

        st.subheader("유사 기업")
        peers = ins.get("similar_companies", []) or []
        st.write(", ".join(peers) if peers else "—")

        st.subheader("AI 한줄평")
        st.info(ins.get("one_line", "—"))
