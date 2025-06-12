class BaseModel(BaseEstimator):
    """预测模型基类"""
    def predict(self, X):
        """生成预测"""
        self._check_is_fitted()
        return self._predict(X)
    
    def _predict(self, X):
        raise NotImplementedError

# 移动平均模型示例（models/moving_average.py）
class MovingAverageCrossover(BaseModel):
    def __init__(self, short_window=50, long_window=200):
        super().__init__(params={
            'short_window': short_window,
            'long_window': long_window
        })
    
    def _fit(self, X, y=None):
        self.short_ma_ = X['close'].rolling(self.params['short_window']).mean()
        self.long_ma_ = X['close'].rolling(self.params['long_window']).mean()
        self.signal_ = (self.short_ma_ > self.long_ma_).astype(int)
    
    def _transform(self, X):
        return pd.DataFrame({
            'short_ma': self.short_ma_,
            'long_ma': self.long_ma_
        })
    
    def _predict(self, X):
        """预测交易信号"""
        return self.signal_
    
    def _plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        X['close'].plot(ax=ax, label='Price')
        self.short_ma_.plot(ax=ax, label=f'{self.params["short_window"]}MA')
        self.long_ma_.plot(ax=ax, label=f'{self.params["long_window"]}MA')
        
        # 标记交叉点
        cross_points = self.signal_.diff()
        buy_signals = cross_points == 1
        sell_signals = cross_points == -1
        
        ax.plot(self.short_ma_[buy_signals].index, 
                self.short_ma_[buy_signals], 
                '^', markersize=10, color='g', label='Buy')
        ax.plot(self.short_ma_[sell_signals].index, 
                self.short_ma_[sell_signals], 
                'v', markersize=10, color='r', label='Sell')
        
        ax.set_title('Moving Average Crossover')
        ax.legend()
        return ax




class BaseIndicator(BaseEstimator):
    """技术指标基类"""
    def __init__(self, window=14):
        super().__init__(params={'window': window})
        self.values_ = None
    
    def _transform(self, X):
        return self.values_
    
    def signal(self):
        """生成交易信号（如超买/超卖）"""
        self._check_is_fitted()
        return self._signal()
    
    def _signal(self):
        """子类实现具体信号逻辑"""
        raise NotImplementedError

# RSI指标实现示例（indicators/rsi.py）
class RSI(BaseIndicator):
    def __init__(self, window=14):
        super().__init__(window=window)
    
    def _fit(self, X, y=None):
        delta = X['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(self.params['window']).mean()
        avg_loss = loss.rolling(self.params['window']).mean()
        
        rs = avg_gain / avg_loss
        self.values_ = 100 - (100 / (1 + rs))
    
    def _plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        self.values_.plot(ax=ax, label='RSI')
        ax.axhline(30, color='r', linestyle='--')
        ax.axhline(70, color='r', linestyle='--')
        ax.set_title(f'RSI ({self.params["window"]} days)')
        ax.legend()
        return ax
    
    def _signal(self):
        """生成RSI交易信号"""
        return pd.DataFrame({
            'overbought': self.values_ > 70,
            'oversold': self.values_ < 30
        })



class BaseEstimator:
    """所有估计器的基类（类似sklearn的BaseEstimator）"""
    
    def __init__(self, params):
        self.params = params
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """学习/计算必要参数"""
        self._check_data(X)
        self._fit(X, y)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """返回计算结果"""
        self._check_is_fitted()
        return self._transform(X)
    
    def fit_transform(self, X, y=None):
        """组合fit+transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def plot(self, **kwargs):
        """可视化结果"""
        self._check_is_fitted()
        return self._plot(**kwargs)
    
    def _fit(self, X, y):
        """子类需实现的具体拟合逻辑"""
        raise NotImplementedError
    
    def _transform(self, X):
        """子类需实现的具体转换逻辑"""
        raise NotImplementedError
    
    def _plot(self, **kwargs):
        """子类需实现的可视化逻辑"""
        raise NotImplementedError
    
    def _check_data(self, X):
        """数据验证（可扩展）"""
        pass
    
    def _check_is_fitted(self):
        """检查是否已拟合"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call 'fit' first")
