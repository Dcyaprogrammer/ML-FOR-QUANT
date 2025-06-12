from quantlearn.indicators import RSI
from quantlearn.models import MovingAverageCrossover
import yfinance as yf

# 获取数据
data = yf.download('AAPL', start='2020-01-01')

# 使用RSI指标
rsi = RSI(window=14)
rsi_values = rsi.fit_transform(data)
rsi.plot()

# 使用移动平均模型
mac = MovingAverageCrossover(short_window=50, long_window=200)
mac.fit(data)
signals = mac.predict(data)
mac.plot()
