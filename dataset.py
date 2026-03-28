import yfinance as yf

data = yf.download("TSLA", period="10y", interval="1d")
data.to_csv("tsla_data.csv")