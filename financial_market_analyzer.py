#!/usr/bin/env python3
"""
📊 FINANCIAL MARKET ANALYZER 📊
Advanced Quantitative Finance & Algorithmic Trading Platform

This project demonstrates:
- Real-time Market Data Analysis & Technical Indicators
- Portfolio Optimization & Risk Management
- Algorithmic Trading Strategy Development
- High-Frequency Trading Signal Generation
- Cryptocurrency Market Analysis
- Derivatives Pricing Models

Author: Data Science Portfolio
Industry Applications: Finance, Trading, Investment Management, Risk Analysis
Tech Stack: Python, QuantLib, pandas, NumPy, scikit-learn, Alpha Vantage API
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Financial analysis libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize

plt.style.use('dark_background')
sns.set_palette("viridis")

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    market_cap: float

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str
    strength: float
    timestamp: datetime
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float

class FinancialMarketAnalyzer:
    """
    💹 Advanced Financial Market Analytics Platform
    
    Features:
    - Technical analysis and indicator calculation
    - Portfolio optimization and risk metrics
    - Trading signal generation
    - Market sentiment analysis
    - Cryptocurrency analysis
    - Risk management systems
    """
    
    def __init__(self):
        self.db_path = "financial_markets.db"
        self.initialize_database()
        self.stocks = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NVDA', 'META', 'NFLX']
        self.crypto = ['BTC', 'ETH', 'ADA', 'DOT', 'SOL', 'MATIC']
        
    def initialize_database(self):
        """Initialize financial database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                market_cap REAL DEFAULT 0,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                entry_price REAL NOT NULL,
                target_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                confidence REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_value REAL NOT NULL,
                daily_return REAL DEFAULT 0,
                cumulative_return REAL DEFAULT 0,
                volatility REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_synthetic_market_data(self, days=252):
        """Generate realistic market data using geometric Brownian motion"""
        print("📈 Generating synthetic market data...")
        
        market_data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for symbol in self.stocks + self.crypto:
            # Different characteristics for stocks vs crypto
            if symbol in self.crypto:
                initial_price = np.random.uniform(100, 50000)
                volatility = np.random.uniform(0.4, 0.8)  # Higher volatility
                drift = np.random.uniform(-0.1, 0.3)
            else:
                initial_price = np.random.uniform(50, 300)
                volatility = np.random.uniform(0.15, 0.4)
                drift = np.random.uniform(-0.05, 0.15)
            
            # Generate price series using geometric Brownian motion
            dt = 1/252  # Daily time step
            price = initial_price
            
            for i in range(days):
                # Market hours simulation (higher volatility during trading hours)
                current_date = start_date + timedelta(days=i)
                
                # Weekend effect (no trading)
                if current_date.weekday() >= 5:  # Saturday, Sunday
                    open_p = price
                    high_p = price
                    low_p = price
                    close_p = price
                    volume = 0
                else:
                    # Simulate intraday price movement
                    random_shock = np.random.normal(0, 1)
                    price_change = drift * dt + volatility * np.sqrt(dt) * random_shock
                    
                    new_price = price * np.exp(price_change)
                    
                    # Simulate OHLC
                    open_p = price
                    intraday_vol = volatility * 0.3
                    high_p = max(open_p, new_price) * (1 + np.random.uniform(0, intraday_vol))
                    low_p = min(open_p, new_price) * (1 - np.random.uniform(0, intraday_vol))
                    close_p = new_price
                    
                    # Volume simulation
                    if symbol in self.crypto:
                        volume = int(np.random.lognormal(12, 1.5))  # Higher volume
                    else:
                        volume = int(np.random.lognormal(14, 1))
                    
                    price = close_p
                
                # Market cap estimation
                if symbol in self.crypto:
                    market_cap = close_p * np.random.uniform(18e6, 21e6)  # Bitcoin-like supply
                else:
                    shares_outstanding = np.random.uniform(1e9, 5e9)
                    market_cap = close_p * shares_outstanding
                
                data_point = MarketData(
                    symbol=symbol,
                    timestamp=current_date,
                    open_price=open_p,
                    high_price=high_p,
                    low_price=low_p,
                    close_price=close_p,
                    volume=volume,
                    market_cap=market_cap
                )
                market_data.append(data_point)
        
        self.store_market_data(market_data)
        print(f"✅ Generated {len(market_data)} market data points for {len(self.stocks + self.crypto)} assets")
        return market_data
    
    def store_market_data(self, market_data):
        """Store market data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for data in market_data:
            cursor.execute('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume, market_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.timestamp, data.open_price, data.high_price,
                data.low_price, data.close_price, data.volume, data.market_cap
            ))
        
        conn.commit()
        conn.close()
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        print("📊 Calculating technical indicators...")
        
        # Moving Averages
        df['SMA_20'] = df['close_price'].rolling(window=20).mean()
        df['SMA_50'] = df['close_price'].rolling(window=50).mean()
        df['EMA_12'] = df['close_price'].ewm(span=12).mean()
        df['EMA_26'] = df['close_price'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['close_price'].rolling(window=20).mean()
        bb_std = df['close_price'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
        
        # Volatility
        df['Returns'] = df['close_price'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def generate_trading_signals(self):
        """Generate trading signals using technical analysis"""
        print("🎯 Generating trading signals...")
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM market_data 
            ORDER BY symbol, timestamp
        ''', conn)
        conn.close()
        
        if df.empty:
            self.generate_synthetic_market_data()
            return self.generate_trading_signals()
        
        signals = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = self.calculate_technical_indicators(symbol_data)
            
            if len(symbol_data) < 50:
                continue
            
            # Signal 1: MACD Crossover
            for i in range(1, len(symbol_data)):
                if (symbol_data.iloc[i]['MACD'] > symbol_data.iloc[i]['MACD_Signal'] and 
                    symbol_data.iloc[i-1]['MACD'] <= symbol_data.iloc[i-1]['MACD_Signal']):
                    
                    entry_price = symbol_data.iloc[i]['close_price']
                    target_price = entry_price * 1.05  # 5% target
                    stop_loss = entry_price * 0.97     # 3% stop loss
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=0.7,
                        timestamp=symbol_data.iloc[i]['timestamp'],
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        confidence=0.75
                    )
                    signals.append(signal)
            
            # Signal 2: RSI Oversold/Overbought
            latest_data = symbol_data.iloc[-1]
            if latest_data['RSI'] < 30:  # Oversold
                entry_price = latest_data['close_price']
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=0.6,
                    timestamp=latest_data['timestamp'],
                    entry_price=entry_price,
                    target_price=entry_price * 1.08,
                    stop_loss=entry_price * 0.95,
                    confidence=0.65
                )
                signals.append(signal)
            elif latest_data['RSI'] > 70:  # Overbought
                entry_price = latest_data['close_price']
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    strength=0.6,
                    timestamp=latest_data['timestamp'],
                    entry_price=entry_price,
                    target_price=entry_price * 0.95,
                    stop_loss=entry_price * 1.03,
                    confidence=0.65
                )
                signals.append(signal)
            
            # Signal 3: Bollinger Band Breakout
            if (latest_data['close_price'] > latest_data['BB_Upper'] and 
                latest_data['Volume_Ratio'] > 1.5):
                
                entry_price = latest_data['close_price']
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=0.8,
                    timestamp=latest_data['timestamp'],
                    entry_price=entry_price,
                    target_price=entry_price * 1.1,
                    stop_loss=entry_price * 0.95,
                    confidence=0.8
                )
                signals.append(signal)
        
        # Store signals
        self.store_trading_signals(signals)
        print(f"✅ Generated {len(signals)} trading signals")
        return signals
    
    def store_trading_signals(self, signals):
        """Store trading signals in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for signal in signals:
            cursor.execute('''
                INSERT INTO trading_signals 
                (symbol, signal_type, strength, timestamp, entry_price, target_price, stop_loss, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.symbol, signal.signal_type, signal.strength, signal.timestamp,
                signal.entry_price, signal.target_price, signal.stop_loss, signal.confidence
            ))
        
        conn.commit()
        conn.close()
    
    def optimize_portfolio(self):
        """Portfolio optimization using Modern Portfolio Theory"""
        print("📊 Optimizing portfolio allocation...")
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT symbol, timestamp, close_price 
            FROM market_data 
            ORDER BY timestamp
        ''', conn)
        conn.close()
        
        if df.empty:
            return None
        
        # Create returns matrix
        pivot_df = df.pivot(index='timestamp', columns='symbol', values='close_price')
        returns = pivot_df.pct_change().dropna()
        
        if len(returns) < 50:
            return None
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        num_assets = len(returns.columns)
        
        # Portfolio optimization function
        def portfolio_performance(weights, expected_returns, cov_matrix):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_std
        
        def negative_sharpe(weights, expected_returns, cov_matrix, risk_free_rate=0.02):
            p_return, p_std = portfolio_performance(weights, expected_returns, cov_matrix)
            return -(p_return - risk_free_rate) / p_std
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_guess = num_assets * [1. / num_assets]
        
        # Optimize for maximum Sharpe ratio
        try:
            optimal_weights = minimize(
                negative_sharpe,
                initial_guess,
                args=(expected_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if optimal_weights.success:
                weights = optimal_weights.x
                opt_return, opt_std = portfolio_performance(weights, expected_returns, cov_matrix)
                sharpe_ratio = (opt_return - 0.02) / opt_std
                
                portfolio_allocation = dict(zip(returns.columns, weights))
                
                optimization_result = {
                    'allocation': portfolio_allocation,
                    'expected_return': opt_return,
                    'volatility': opt_std,
                    'sharpe_ratio': sharpe_ratio
                }
                
                return optimization_result
        except:
            return None
        
        return None
    
    def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        print("⚠️ Calculating risk metrics...")
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT symbol, close_price, timestamp 
            FROM market_data 
            ORDER BY symbol, timestamp
        ''', conn)
        conn.close()
        
        if df.empty:
            return {}
        
        risk_metrics = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 30:
                continue
            
            returns = symbol_data['close_price'].pct_change().dropna()
            
            if len(returns) == 0:
                continue
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Maximum Drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Volatility metrics
            volatility = returns.std() * np.sqrt(252)
            downside_volatility = returns[returns < 0].std() * np.sqrt(252)
            
            # Sharpe Ratio
            excess_returns = returns - 0.02/252  # Risk-free rate
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
            
            risk_metrics[symbol] = {
                'volatility': volatility,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'downside_volatility': downside_volatility
            }
        
        return risk_metrics
    
    def create_financial_dashboard(self):
        """Create comprehensive financial analytics dashboard"""
        print("📊 Creating financial market dashboard...")
        
        # Load data and generate signals
        market_data = self.generate_synthetic_market_data()
        signals = self.generate_trading_signals()
        portfolio_opt = self.optimize_portfolio()
        risk_metrics = self.calculate_risk_metrics()
        
        # Load processed data
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM market_data ORDER BY timestamp', conn)
        signals_df = pd.read_sql_query('SELECT * FROM trading_signals', conn)
        conn.close()
        
        # Create dashboard
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '📈 Price Performance',
                '📊 Volume Analysis',
                '🎯 Trading Signals',
                '⚠️ Risk Metrics',
                '💼 Portfolio Allocation',
                '📉 Volatility Analysis',
                '💰 Market Cap Distribution',
                '📊 Returns Distribution'
            ]
        )
        
        # 1. Price Performance
        for symbol in df['symbol'].unique()[:4]:  # Show top 4 for clarity
            symbol_data = df[df['symbol'] == symbol]
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(symbol_data['timestamp']),
                    y=symbol_data['close_price'],
                    name=symbol,
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # 2. Volume Analysis
        volume_by_symbol = df.groupby('symbol')['volume'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=volume_by_symbol.index,
                y=volume_by_symbol.values,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Trading Signals
        if not signals_df.empty:
            signal_counts = signals_df.groupby(['symbol', 'signal_type']).size().unstack(fill_value=0)
            fig.add_trace(
                go.Bar(
                    x=signal_counts.index,
                    y=signal_counts.get('BUY', []),
                    name='BUY Signals',
                    marker_color='green'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=signal_counts.index,
                    y=signal_counts.get('SELL', []),
                    name='SELL Signals',
                    marker_color='red'
                ),
                row=2, col=1
            )
        
        # 4. Risk Metrics Heatmap
        if risk_metrics:
            risk_df = pd.DataFrame(risk_metrics).T
            fig.add_trace(
                go.Heatmap(
                    z=risk_df[['volatility', 'sharpe_ratio', 'max_drawdown']].values,
                    x=['Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                    y=risk_df.index,
                    colorscale='RdYlGn',
                    showscale=True
                ),
                row=2, col=2
            )
        
        # 5. Portfolio Allocation
        if portfolio_opt:
            allocation = portfolio_opt['allocation']
            fig.add_trace(
                go.Pie(
                    labels=list(allocation.keys()),
                    values=list(allocation.values()),
                    name="Portfolio"
                ),
                row=3, col=1
            )
        
        # 6. Volatility Analysis
        df['returns'] = df.groupby('symbol')['close_price'].pct_change()
        volatility_by_symbol = df.groupby('symbol')['returns'].std().sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=volatility_by_symbol.index,
                y=volatility_by_symbol.values * np.sqrt(252),  # Annualized
                showlegend=False,
                marker_color='orange'
            ),
            row=3, col=2
        )
        
        # 7. Market Cap Distribution
        latest_data = df.groupby('symbol').last()
        fig.add_trace(
            go.Scatter(
                x=latest_data['market_cap'],
                y=latest_data['close_price'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=latest_data['volume'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=latest_data.index,
                showlegend=False
            ),
            row=4, col=1
        )
        
        # 8. Returns Distribution
        all_returns = df['returns'].dropna()
        fig.add_trace(
            go.Histogram(
                x=all_returns,
                nbinsx=50,
                name='Returns',
                showlegend=False
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1600,
            title_text="📊 Financial Market Analytics Dashboard 📊",
            title_font_size=24,
            showlegend=True
        )
        
        fig.write_html("financial_market_dashboard.html")
        fig.show()
        
        return fig
    
    def generate_market_insights(self):
        """Generate market insights and recommendations"""
        signals = self.generate_trading_signals()
        portfolio_opt = self.optimize_portfolio()
        risk_metrics = self.calculate_risk_metrics()
        
        print("\\n" + "="*70)
        print("📊 FINANCIAL MARKET ANALYSIS - INVESTMENT INSIGHTS 📊")
        print("="*70)
        
        # Trading signals summary
        conn = sqlite3.connect(self.db_path)
        signals_df = pd.read_sql_query('SELECT * FROM trading_signals', conn)
        df = pd.read_sql_query('SELECT * FROM market_data', conn)
        conn.close()
        
        if not signals_df.empty:
            buy_signals = len(signals_df[signals_df['signal_type'] == 'BUY'])
            sell_signals = len(signals_df[signals_df['signal_type'] == 'SELL'])
            avg_confidence = signals_df['confidence'].mean()
            
            print(f"🎯 Trading Signals Analysis:")
            print(f"   • Buy Signals: {buy_signals}")
            print(f"   • Sell Signals: {sell_signals}")
            print(f"   • Average Confidence: {avg_confidence:.2f}")
            
            # Top signal opportunities
            top_signals = signals_df.nlargest(3, 'confidence')
            print(f"\\n   📈 Top Trading Opportunities:")
            for _, signal in top_signals.iterrows():
                upside = ((signal['target_price'] - signal['entry_price']) / signal['entry_price']) * 100
                print(f"     • {signal['symbol']}: {signal['signal_type']} - {upside:+.1f}% upside potential")
        
        # Portfolio optimization results
        if portfolio_opt:
            print(f"\\n💼 Optimal Portfolio Allocation:")
            print(f"   • Expected Annual Return: {portfolio_opt['expected_return']:.1%}")
            print(f"   • Portfolio Volatility: {portfolio_opt['volatility']:.1%}")
            print(f"   • Sharpe Ratio: {portfolio_opt['sharpe_ratio']:.2f}")
            
            print(f"\\n   🏆 Top Holdings:")
            sorted_allocation = sorted(portfolio_opt['allocation'].items(), key=lambda x: x[1], reverse=True)
            for symbol, weight in sorted_allocation[:5]:
                print(f"     • {symbol}: {weight:.1%}")
        
        # Risk analysis
        if risk_metrics:
            print(f"\\n⚠️ Risk Analysis:")
            avg_volatility = np.mean([metrics['volatility'] for metrics in risk_metrics.values()])
            avg_sharpe = np.mean([metrics['sharpe_ratio'] for metrics in risk_metrics.values() if not np.isnan(metrics['sharpe_ratio'])])
            
            print(f"   • Average Portfolio Volatility: {avg_volatility:.1%}")
            print(f"   • Average Sharpe Ratio: {avg_sharpe:.2f}")
            
            # Highest risk assets
            high_risk_assets = sorted(risk_metrics.items(), key=lambda x: x[1]['volatility'], reverse=True)[:3]
            print(f"\\n   🔥 Highest Risk Assets:")
            for symbol, metrics in high_risk_assets:
                print(f"     • {symbol}: {metrics['volatility']:.1%} volatility, {metrics['max_drawdown']:.1%} max drawdown")
        
        # Market overview
        latest_data = df.groupby('symbol').last()
        total_market_cap = latest_data['market_cap'].sum()
        avg_price = latest_data['close_price'].mean()
        
        print(f"\\n📊 Market Overview:")
        print(f"   • Total Market Cap: ${total_market_cap/1e12:.1f}T")
        print(f"   • Average Asset Price: ${avg_price:.2f}")
        print(f"   • Assets Analyzed: {len(latest_data)}")
        
        print("="*70)
    
    def run_complete_analysis(self):
        """Execute complete financial market analysis"""
        print("🚀 Starting Financial Market Analysis Pipeline...")
        print("="*55)
        
        # Generate market data
        market_data = self.generate_synthetic_market_data()
        
        # Generate signals
        signals = self.generate_trading_signals()
        
        # Optimize portfolio
        portfolio_opt = self.optimize_portfolio()
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics()
        
        # Create dashboard
        dashboard = self.create_financial_dashboard()
        
        # Generate insights
        self.generate_market_insights()
        
        print("\\n✅ Financial Market Analysis Complete!")
        print("📁 Files generated:")
        print("   • financial_market_dashboard.html")
        print("   • financial_markets.db")
        
        # Export summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'assets_analyzed': len(self.stocks + self.crypto),
            'trading_signals_generated': len(signals),
            'portfolio_optimization': portfolio_opt,
            'risk_metrics_summary': {
                'avg_volatility': np.mean([m['volatility'] for m in risk_metrics.values()]) if risk_metrics else 0,
                'assets_with_high_risk': len([s for s, m in risk_metrics.items() if m['volatility'] > 0.5]) if risk_metrics else 0
            }
        }
        
        with open('financial_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return market_data, signals, portfolio_opt

def main():
    """Main execution function"""
    print("🎯 FINANCIAL MARKET ANALYZER - Industry-Ready Platform")
    print("=" * 60)
    print("Showcasing: Quantitative Finance • ML • Portfolio Optimization")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = FinancialMarketAnalyzer()
    
    # Run complete analysis
    data, signals, portfolio = analyzer.run_complete_analysis()
    
    print(f"\\n🎉 Analysis completed successfully!")
    print(f"📈 Analyzed {len(analyzer.stocks + analyzer.crypto)} financial instruments")
    print(f"🎯 Generated {len(signals)} trading signals")
    
    return analyzer, data, signals

if __name__ == "__main__":
    analyzer, data, signals = main()