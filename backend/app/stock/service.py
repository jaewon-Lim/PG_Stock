import json
import os
from fastapi import HTTPException
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import httpx
import logging
import traceback
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------
# interval/period 보정
# ---------------------------

def _normalize_period_for_interval(period: str, interval: str) -> str:
    if interval == "1m":
        return "5d"
    if interval in ["5m", "15m", "30m"]:
        return "1mo"
    if interval in ["60m"]:
        return "1y"
    return period

# ---------------------------
# 가격 로딩 & 지표
# ---------------------------

def load_price(symbol: str, period: str = "1y", interval: str | None = None) -> pd.DataFrame:
    if interval is None:
        interval = {"5d": "1h", "1mo": "1h", "3mo": "1d", "6mo": "1d",
                    "1y": "1d", "5y": "1wk", "10y": "1mo"}.get(period, "1d")
    else:
        period = _normalize_period_for_interval(period, interval)

    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"데이터 공급자 오류: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="검색결과 없음 — 주식종목(티커)을 확인해주세요.")

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df.dropna()
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_dn = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_dn + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    cross = np.sign(df["sma50"] - df["sma200"]).diff().fillna(0)
    df["golden_cross"] = cross > 0
    df["dead_cross"] = cross < 0
    return df

# ---------------------------
# 재무(매출 추세)
# ---------------------------

def _pick_row_value(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    lowers = [c.strip().lower() for c in candidates]
    for idx in df.index:
        if str(idx).strip().lower() in lowers:
            return df.loc[idx]
    return None

def get_revenue_series(symbol: str) -> pd.Series | None:
    tk = yf.Ticker(symbol)
    for attr in ["financials", "income_stmt", "quarterly_financials", "quarterly_income_stmt"]:
        df = getattr(tk, attr, None)
        row = _pick_row_value(df, ["Total Revenue", "TotalRevenue", "Revenue"])
        if row is not None:
            ser = row.dropna()
            if ser.empty:
                continue
            try:
                ser.index = pd.to_datetime(ser.index)
                ser = ser.sort_index()
            except Exception:
                pass
            return ser
    return None

def revenue_trend_up_5y(symbol: str) -> bool | None:
    ser = get_revenue_series(symbol)
    if ser is None or ser.size < 2:
        return None
    vals = ser.values[-5:] if ser.size >= 5 else ser.values
    return bool(vals[-1] > vals[0])

# ---------------------------
# 배당 정보
# ---------------------------

def get_dividend_info(tk: yf.Ticker, last_close: float) -> dict | None:
    """yfinance dividends 시리즈를 이용해 최근 2년 데이터에서 최근 4회 배당합, TTM 수익률, 최근 배당 정보 계산."""
    try:
        div = tk.dividends
        if div is None or div.empty:
            print(f"[경고] {tk.ticker} 배당 데이터 없음")
            return None
        div = div.dropna()
        # 최근 2년 데이터 중 최근 4회 배당만 사용
        cut = pd.Timestamp.today() - pd.DateOffset(years=2)
        div_recent = div[div.index >= cut]
        if div_recent.empty:
            print(f"[경고] {tk.ticker} 최근 2년 배당 데이터 없음")
            return None
        last_div_date = div_recent.index.max()
        last_div_amount = float(div_recent.loc[last_div_date])
        # 최근 4회 배당 합계
        ttm_sum = float(div_recent.tail(4).sum())
        yld = (ttm_sum / last_close) if last_close and last_close > 0 else None
        return {
            "dividend_ttm": ttm_sum if ttm_sum is not None else None,
            "dividend_yield_ttm": yld if yld is not None else None,
            "last_dividend_date": pd.to_datetime(last_div_date).strftime("%Y-%m-%d") if last_div_date is not None else None,
            "last_dividend": last_div_amount if last_div_amount is not None else None,
        }
    except Exception as e:
        print(f"[오류] 배당 정보 계산 실패: {e}")
        return None



def get_dividend_info(tk: yf.Ticker, last_close: float, symbol: str | None = None) -> dict | None:
    """
    yfinance dividends 시리즈를 이용해 TTM 배당수익률과 최근 배당 정보를 계산.
    - 최근 2년치에서 '가장 최근 4회' 배당만 합산(연배당/분기배당 모두 대응)
    - 데이터가 비어있으면 None
    """
    try:
        div = tk.dividends
        if div is None or div.empty:
            print(f"[배당] 데이터 없음: {symbol or ''}")
            return None
        div = div.dropna()
        # 최근 2년 기준으로 필터
        cut = pd.Timestamp.today() - pd.DateOffset(years=2)
        div_recent = div[div.index >= cut]
        if div_recent.empty:
            # 그래도 비면 전체에서 최근 4회 사용
            div_recent = div

        # 최근 지급일/금액
        last_div_date = div_recent.index.max()
        last_div_amount = float(div_recent.loc[last_div_date]) if last_div_date is not None else None

        # 가장 최근 4회만 합산(분기배당은 대략 최근 1년, 연배당은 최근 1~2년도 커버)
        ttm_sum = float(div_recent.tail(4).sum())

        yld = (ttm_sum / last_close) if last_close and last_close > 0 else None
        return {
            "dividend_ttm": ttm_sum if ttm_sum is not None else None,
            "dividend_yield_ttm": yld if yld is not None else None,
            "last_dividend_date": pd.to_datetime(last_div_date).strftime("%Y-%m-%d") if last_div_date is not None else None,
            "last_dividend": last_div_amount if last_div_amount is not None else None,
        }
    except Exception as e:
        print(f"[배당] 처리 오류: {symbol or ''} - {e}")
        return None


# ---------------------------
# 크로스 날짜
# ---------------------------

def last_cross_dates(df: pd.DataFrame):
    last_golden = df.index[df["golden_cross"]].max() if df["golden_cross"].any() else None
    last_dead = df.index[df["dead_cross"]].max() if df["dead_cross"].any() else None
    fmt = "%Y-%m-%d"
    lg = pd.to_datetime(last_golden).strftime(fmt) if last_golden is not None else None
    ld = pd.to_datetime(last_dead).strftime(fmt) if last_dead is not None else None
    return lg, ld

# ---------------------------
# 점수 계산
# ---------------------------

def calc_score(df: pd.DataFrame, revenue_up: bool | None) -> dict:
    last = df.iloc[-1]
    score = 50

    # 1) 고점 대비 낙폭
    max_price = df["close"].max()
    drop_ratio = (max_price - last["close"]) / max_price if max_price > 0 else 0.0
    if drop_ratio > 0.5: score += 20
    elif drop_ratio > 0.3: score += 15
    elif drop_ratio > 0.15: score += 8
    else: score -= 5

    # 2) 골든크로스/근접
    recent_window = min(20, len(df))
    recent_golden = df.iloc[-recent_window:]["golden_cross"].any() if recent_window > 0 else False
    near_golden = False
    if not np.isnan(last.get("sma200", np.nan)) and last.get("sma200", 0) != 0:
        near_golden = abs(last.get("sma50", 0) - last.get("sma200", 0)) / abs(last.get("sma200", 1)) < 0.01
    if recent_golden or near_golden: score += 15

    # 3) 최근 1년 고점과의 위치
    lookback = min(len(df), 252)
    recent_year_high = df["close"].tail(lookback).max()
    if last["close"] >= recent_year_high * 0.999: score -= 10
    elif last["close"] >= recent_year_high * 0.98: score -= 5
    elif last["close"] > recent_year_high * 1.02: score += 10

    # 4) 매출 추세
    if revenue_up is True: score += 10
    elif revenue_up is False: score -= 5

    # 5) 거래량 이례치
    avg_vol_20 = df["volume"].tail(20).mean() if len(df) >= 20 else df["volume"].mean()
    if avg_vol_20 and avg_vol_20 > 0:
        if last["volume"] > avg_vol_20 * 1.5: score += 10
        elif last["volume"] < avg_vol_20 * 0.7: score -= 5

    # 6) RSI(14) 추가 규칙
    rsi = last.get("rsi14", np.nan)
    if not np.isnan(rsi):
        if rsi <= 30:
            score += 8   # 과매도 구간: 반등 여지 가점
        elif rsi >= 70:
            score -= 8   # 과매수 구간: 과열 리스크 감점

    # 최종 보정 & 의사결정
    score = max(0, min(100, int(score)))
    decision = "적극매수" if score >= 90 else ("매수추천" if score >= 70 else ("관망추천" if score >= 50 else "매수비추"))
    return {"score": score, "decision": decision}

# ---------------------------
# 메인 분석
# ---------------------------

def analyze_symbol(symbol: str, period: str = "1y", interval: str | None = None):
    df = load_price(symbol, period=period, interval=interval)
    df = add_indicators(df)

    rev_up = revenue_trend_up_5y(symbol)
    headline = calc_score(df, rev_up)

    tk = yf.Ticker(symbol)
    fast = getattr(tk, "fast_info", {}) or {}
    pe = getattr(fast, "pe_ratio", None)

    lg, ld = last_cross_dates(df)
    last = df.iloc[-1]

    # 배당 정보 계산
    div_info = get_dividend_info(tk, float(last["close"]), symbol)

    info = {
        "symbol": symbol.upper(),
        "last_close": float(round(float(last["close"]), 2)),
        "sma20": float(round(float(last["sma20"]), 2)) if not np.isnan(last["sma20"]) else None,
        "sma50": float(round(float(last["sma50"]), 2)) if not np.isnan(last["sma50"]) else None,
        "sma200": float(round(float(last["sma200"]), 2)) if not np.isnan(last["sma200"]) else None,
        "rsi14": float(round(float(last["rsi14"]), 1)) if not np.isnan(last["rsi14"]) else None,
        "pe_ratio": float(pe) if pe is not None else None,
        "golden_cross_today": bool(last["golden_cross"]),
        "dead_cross_today": bool(last["dead_cross"]),
        "revenue_trend_up": (bool(rev_up) if rev_up is not None else None),
        "last_golden_cross": lg,
        "last_dead_cross": ld,
        # 배당
        "dividend_yield_ttm": (float(div_info["dividend_yield_ttm"]) if div_info and div_info.get("dividend_yield_ttm") is not None else None),
        "last_dividend": (float(div_info["last_dividend"]) if div_info and div_info.get("last_dividend") is not None else None),
        "last_dividend_date": (div_info.get("last_dividend_date") if div_info else None),
        "dividend_ttm": (float(div_info["dividend_ttm"]) if div_info and div_info.get("dividend_ttm") is not None else None),
    }

    out = df.reset_index().rename(columns={"Date": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    prices = json.loads(
        out[["date", "open", "high", "low", "close", "volume", "sma20", "sma50", "sma200"]]
        .to_json(orient="records")
    )

    return {"info": info, "headline": headline, "prices": prices}

# ---------------------------
# AI 코멘트 (프롬프트 숨기고 필요한 필드만 반환)
# ---------------------------

def _pick_row_value_df(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    return _pick_row_value(df, candidates)

def _fetch_company_context(symbol: str) -> dict:
    tk = yf.Ticker(symbol)
    try:
        info = tk.get_info()
    except Exception:
        info = {}

    name = info.get("longName") or info.get("shortName") or symbol.upper()
    sector = info.get("sector")
    industry = info.get("industry")
    summary = info.get("longBusinessSummary")

    news_list = []
    try:
        news = tk.news or []
        for n in news[:6]:
            news_list.append({"title": n.get("title"), "link": n.get("link"), "publisher": n.get("publisher")})
    except Exception:
        pass

    rnd_val = None
    try:
        inc = tk.income_stmt
        row = _pick_row_value_df(inc, ["Research Development", "Research And Development", "ResearchAndDevelopment"])
        if row is None:
            incq = tk.quarterly_income_stmt
            row = _pick_row_value_df(incq, ["Research Development", "Research And Development", "ResearchAndDevelopment"])
        if row is not None:
            vals = row.dropna().values
            if len(vals):
                rnd_val = float(vals[0])
    except Exception:
        pass

    return {"name": name, "sector": sector, "industry": industry, "summary": summary, "news": news_list, "rnd_recent": rnd_val}

def ai_insights(symbol: str, period: str = "1y"):
    # gemini
    #import google.genai as genai
    #from google.genai.types import Content, Part
    #client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    ctx = _fetch_company_context(symbol)
    rnd_text = f"{ctx['rnd_recent']:,}" if ctx["rnd_recent"] is not None else "불명"

    prompt = f"""
너는 초등학생도 이해할 수 있게 설명하는 금융 도우미다.
다음 회사에 대해 한국어로 간단명료하게 대답하고, 반드시 JSON으로만 출력해.
아래 예시 형식을 절대 변경하지 말고, 인터넷에서 회사에관련된 최신(1달이내) 이슈기사 최대 3개를 골라 'top_links'에 그대로 옮겨 적어라.

예시:
"top_links": [
    {{"title": "뉴스제목", "source": "뉴스출처", "url": "https://example.com"}},
    ...
]

[회사기본]
- 심볼: {symbol.upper()}
- 이름: {ctx['name']}
- 섹터/산업: {ctx.get('sector')}/{ctx.get('industry')}
- 개요: {ctx.get('summary') or '설명 없음'}

[R&D 최근 지출]: {rnd_text}

[요청]
- "company_overview": 2~3문장
- "rnd_assessment": 1문장(+근거 1줄)
- "outlook": 2~3문장
- "top_links": [{{"title":"...","source":"...","url":"..."}}, ...] 
- "similar_companies": 3~5개 회사
- "one_line": 한줄평(매수추천/관망/주의 등)
"""
    # gemini
    #response = client.models.generate_content(
    #    model="gemini-2.5-flash",  # 또는 gemini-2.5-pro
    #    contents=Content(parts=[Part.from_text(prompt)])
    #)
    #raw = response.text.strip()
    try:
        # Responses API 대신 Chat Completions 사용 (JSON 강제)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content or "{}"

        try:
            parsed = json.loads(raw)
        except Exception:
            text2 = raw.strip("` \n")
            text2 = text2[text2.find("{"): text2.rfind("}") + 1] if "{" in text2 and "}" in text2 else "{}"
            parsed = json.loads(text2)

        # AI가 top_links를 비웠으면 yfinance 뉴스로 보완
        parsed_links = parsed.get("top_links", [])
        if not parsed_links and ctx["news"]:
            parsed_links = [
                {
                    "title": n.get("title"),
                    "source": n.get("publisher"),
                    "url": n.get("link")
                }
                for n in ctx["news"][:3]
                if n.get("title") and n.get("link")
            ]

        return {
            "company_overview": parsed.get("company_overview"),
            "rnd_assessment": parsed.get("rnd_assessment"),
            "outlook": parsed.get("outlook"),
            "top_links": parsed_links,
            "similar_companies": parsed.get("similar_companies", []),
            "one_line": parsed.get("one_line"),
        }

    except Exception as e:
        # 개발 중에는 트레이스백도 남겨두자
        traceback.print_exc()
        logger.exception("ai_insights 실패")
        raise HTTPException(status_code=500, detail=str(e))
