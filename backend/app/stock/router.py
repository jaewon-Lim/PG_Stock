from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .service import analyze_symbol, ai_insights

router = APIRouter(prefix="/stock", tags=["stock"])

class AnalyzeReq(BaseModel):
    symbol: str
    period: str | None = None      # "5d","1mo","3mo","6mo","1y","5y","10y"
    interval: str | None = None    # "1m","5m","15m","60m","1d","1wk","1mo"

@router.post("/analyze")
def analyze(req: AnalyzeReq):
    try:
        return analyze_symbol(req.symbol, period=req.period or "1y", interval=req.interval)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class InsightsReq(BaseModel):
    symbol: str
    period: str | None = None

@router.post("/insights")
def insights(req: InsightsReq):
    try:
        return ai_insights(req.symbol, period=req.period or "1y")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
