from fastapi import FastAPI
from .stock.router import router as stock_router

app = FastAPI(title="Stock Helper API")
app.include_router(stock_router)  # 신규

#from app.chat.router import router as chat_router
#app.include_router(chat_router)   # 기존
