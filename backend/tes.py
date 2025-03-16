import yfinance as yf
data = yf.download("GOOG", start="2024-01-01")
print(data)