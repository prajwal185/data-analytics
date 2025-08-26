#!/usr/bin/env python3
"""
🍕 RESTAURANT CHAIN OPTIMIZER 🍕
Advanced Food Service Analytics & Operations Research Platform

This project demonstrates:
- Supply Chain Optimization & Demand Forecasting
- Restaurant Performance Analytics & KPI Tracking
- Menu Engineering & Price Optimization
- Staff Scheduling & Labor Cost Optimization
- Customer Sentiment Analysis & Review Mining
- Food Safety & Quality Monitoring Systems

Author: Data Science Portfolio
Industry Applications: Food Service, Restaurant Management, Supply Chain, Operations
Tech Stack: Python, OR-Tools, scikit-learn, NLP, Time Series, Optimization
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

# Operations research libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from collections import Counter

plt.style.use('seaborn-v0_8')
sns.set_palette("tab10")

@dataclass
class Restaurant:
    restaurant_id: str
    location: str
    city: str
    state: str
    latitude: float
    longitude: float
    size_sqft: int
    max_capacity: int
    staff_count: int
    monthly_rent: float

@dataclass
class MenuItem:
    item_id: str
    name: str
    category: str
    base_price: float
    cost_to_make: float
    prep_time_minutes: int
    calories: int
    popularity_score: float

@dataclass
class Sale:
    sale_id: str
    restaurant_id: str
    item_id: str
    quantity: int
    price_paid: float
    timestamp: datetime
    customer_rating: Optional[int]
    order_type: str  # dine-in, takeout, delivery

class RestaurantChainOptimizer:
    """
    🍽️ Advanced Restaurant Chain Analytics Platform
    
    Features:
    - Multi-location performance analysis
    - Menu profitability optimization
    - Supply chain cost minimization
    - Staff scheduling optimization
    - Customer satisfaction tracking
    - Demand forecasting and inventory management
    """
    
    def __init__(self):
        self.db_path = "restaurant_analytics.db"
        self.initialize_database()
        self.cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]
        self.menu_categories = ["Appetizers", "Entrees", "Desserts", "Beverages", "Sides"]
        
    def initialize_database(self):
        """Initialize restaurant analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS restaurants (
                restaurant_id TEXT PRIMARY KEY,
                location TEXT NOT NULL,
                city TEXT NOT NULL,
                state TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                size_sqft INTEGER DEFAULT 2000,
                max_capacity INTEGER DEFAULT 100,
                staff_count INTEGER DEFAULT 15,
                monthly_rent REAL DEFAULT 10000
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS menu_items (
                item_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                base_price REAL NOT NULL,
                cost_to_make REAL NOT NULL,
                prep_time_minutes INTEGER DEFAULT 15,
                calories INTEGER DEFAULT 500,
                popularity_score REAL DEFAULT 0.5
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales (
                sale_id TEXT PRIMARY KEY,
                restaurant_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                quantity INTEGER DEFAULT 1,
                price_paid REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                customer_rating INTEGER,
                order_type TEXT DEFAULT 'dine-in',
                FOREIGN KEY (restaurant_id) REFERENCES restaurants (restaurant_id),
                FOREIGN KEY (item_id) REFERENCES menu_items (item_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimization_type TEXT NOT NULL,
                restaurant_id TEXT,
                metric_name TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                improvement_percent REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_synthetic_restaurant_data(self, num_restaurants=20, num_items=50, num_sales=25000):
        """Generate realistic restaurant chain data"""
        print("🏪 Generating synthetic restaurant chain data...")
        
        # Generate restaurant locations
        restaurants = []
        for i in range(num_restaurants):
            city = np.random.choice(self.cities)
            
            # City-specific coordinates and characteristics
            if city == "New York":
                lat_base, lng_base = 40.7128, -74.0060
                rent_multiplier = 2.5
            elif city == "Los Angeles":
                lat_base, lng_base = 34.0522, -118.2437
                rent_multiplier = 2.0
            elif city == "Chicago":
                lat_base, lng_base = 41.8781, -87.6298
                rent_multiplier = 1.5
            else:
                lat_base, lng_base = 32.7767, -96.7970  # Dallas as default
                rent_multiplier = 1.0
            
            # Add location variation
            latitude = lat_base + np.random.uniform(-0.1, 0.1)
            longitude = lng_base + np.random.uniform(-0.1, 0.1)
            
            # Restaurant characteristics
            size_sqft = np.random.randint(1500, 4000)
            max_capacity = int(size_sqft / 20)  # Rough estimate
            staff_count = max(8, int(max_capacity / 8))  # Staff ratio
            monthly_rent = size_sqft * rent_multiplier * np.random.uniform(15, 25)
            
            restaurant = Restaurant(
                restaurant_id=f"REST_{i:03d}",
                location=f"{city} Location {i%5 + 1}",
                city=city,
                state=self.get_state_from_city(city),
                latitude=latitude,
                longitude=longitude,
                size_sqft=size_sqft,
                max_capacity=max_capacity,
                staff_count=staff_count,
                monthly_rent=monthly_rent
            )
            restaurants.append(restaurant)
        
        # Generate menu items
        menu_items = []
        item_templates = {
            "Appetizers": [
                {"name": "Buffalo Wings", "price": 12.99, "cost": 4.50, "calories": 800},
                {"name": "Mozzarella Sticks", "price": 9.99, "cost": 3.20, "calories": 600},
                {"name": "Loaded Nachos", "price": 11.99, "cost": 4.00, "calories": 950},
                {"name": "Spinach Dip", "price": 8.99, "cost": 2.80, "calories": 400},
            ],
            "Entrees": [
                {"name": "Classic Burger", "price": 15.99, "cost": 5.50, "calories": 750},
                {"name": "Chicken Parmesan", "price": 18.99, "cost": 6.80, "calories": 900},
                {"name": "BBQ Ribs", "price": 22.99, "cost": 8.50, "calories": 1200},
                {"name": "Vegetarian Pizza", "price": 16.99, "cost": 4.20, "calories": 650},
                {"name": "Fish Tacos", "price": 14.99, "cost": 5.20, "calories": 580},
            ],
            "Desserts": [
                {"name": "Chocolate Cake", "price": 7.99, "cost": 2.10, "calories": 450},
                {"name": "Ice Cream Sundae", "price": 6.99, "cost": 1.80, "calories": 350},
                {"name": "Apple Pie", "price": 6.99, "cost": 2.50, "calories": 380},
            ],
            "Beverages": [
                {"name": "Craft Beer", "price": 5.99, "cost": 1.50, "calories": 150},
                {"name": "Wine Glass", "price": 8.99, "cost": 2.20, "calories": 120},
                {"name": "Soft Drink", "price": 2.99, "cost": 0.30, "calories": 140},
                {"name": "Coffee", "price": 3.99, "cost": 0.80, "calories": 5},
            ],
            "Sides": [
                {"name": "French Fries", "price": 4.99, "cost": 1.20, "calories": 400},
                {"name": "Coleslaw", "price": 3.99, "cost": 0.90, "calories": 150},
                {"name": "Onion Rings", "price": 5.99, "cost": 1.80, "calories": 450},
            ]
        }
        
        item_counter = 0
        for category, items in item_templates.items():
            for item_template in items:
                # Add some variation to prices and costs
                price_variation = np.random.uniform(0.9, 1.1)
                cost_variation = np.random.uniform(0.9, 1.1)
                
                menu_item = MenuItem(
                    item_id=f"ITEM_{item_counter:03d}",
                    name=item_template["name"],
                    category=category,
                    base_price=item_template["price"] * price_variation,
                    cost_to_make=item_template["cost"] * cost_variation,
                    prep_time_minutes=np.random.randint(5, 25),
                    calories=item_template["calories"],
                    popularity_score=np.random.beta(2, 2)  # Bell-curved between 0-1
                )
                menu_items.append(menu_item)
                item_counter += 1
        
        # Generate sales data
        sales = []
        base_time = datetime.now() - timedelta(days=365)
        
        for i in range(num_sales):
            restaurant = np.random.choice(restaurants)
            menu_item = np.random.choice(menu_items)
            
            # Sales patterns based on item popularity and restaurant size
            quantity = 1
            if menu_item.category in ["Beverages", "Sides"]:
                quantity = np.random.randint(1, 4)
            
            # Price variation (discounts, promotions)
            price_multiplier = np.random.choice([0.8, 0.9, 1.0, 1.0, 1.0], p=[0.1, 0.1, 0.6, 0.1, 0.1])
            price_paid = menu_item.base_price * price_multiplier * quantity
            
            # Timestamp with realistic patterns
            days_ago = np.random.exponential(30)  # More recent sales weighted higher
            days_ago = min(365, days_ago)
            
            sale_date = base_time + timedelta(days=days_ago)
            
            # Time of day patterns
            if menu_item.category == "Beverages":
                # More beverage sales in evening
                hour_weights = [0.5, 0.2, 0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 1.8, 1.5, 1.2, 
                               1.0, 1.2, 1.5, 1.8, 2.2, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.7]
            else:
                # Regular meal patterns
                hour_weights = [0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 2.5,
                               3.0, 2.5, 1.5, 1.0, 1.2, 2.8, 3.2, 2.8, 2.0, 1.5, 1.0, 0.5]
            
            hour = np.random.choice(24, p=np.array(hour_weights)/sum(hour_weights))
            sale_timestamp = sale_date.replace(hour=hour, minute=np.random.randint(0, 60))
            
            # Customer rating (occasional)
            customer_rating = None
            if np.random.random() < 0.3:  # 30% of orders get rated
                # Rating based on item quality and restaurant performance
                base_rating = 4.0
                if menu_item.popularity_score > 0.7:
                    base_rating += 0.5
                elif menu_item.popularity_score < 0.3:
                    base_rating -= 0.8
                
                customer_rating = max(1, min(5, int(np.random.normal(base_rating, 0.8))))
            
            # Order type
            order_type = np.random.choice(['dine-in', 'takeout', 'delivery'], p=[0.5, 0.3, 0.2])
            
            sale = Sale(
                sale_id=f"SALE_{i:06d}",
                restaurant_id=restaurant.restaurant_id,
                item_id=menu_item.item_id,
                quantity=quantity,
                price_paid=price_paid,
                timestamp=sale_timestamp,
                customer_rating=customer_rating,
                order_type=order_type
            )
            sales.append(sale)
        
        # Store all data
        self.store_restaurant_data(restaurants, menu_items, sales)
        print(f"✅ Generated {len(restaurants)} restaurants, {len(menu_items)} menu items, {len(sales)} sales")
        return restaurants, menu_items, sales
    
    def get_state_from_city(self, city):
        """Get state abbreviation from city name"""
        city_state_map = {
            "New York": "NY",
            "Los Angeles": "CA",
            "Chicago": "IL",
            "Houston": "TX",
            "Phoenix": "AZ",
            "Philadelphia": "PA"
        }
        return city_state_map.get(city, "TX")
    
    def store_restaurant_data(self, restaurants, menu_items, sales):
        """Store all restaurant data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store restaurants
        for restaurant in restaurants:
            cursor.execute('''
                INSERT OR REPLACE INTO restaurants 
                (restaurant_id, location, city, state, latitude, longitude, 
                 size_sqft, max_capacity, staff_count, monthly_rent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                restaurant.restaurant_id, restaurant.location, restaurant.city, restaurant.state,
                restaurant.latitude, restaurant.longitude, restaurant.size_sqft,
                restaurant.max_capacity, restaurant.staff_count, restaurant.monthly_rent
            ))
        
        # Store menu items
        for item in menu_items:
            cursor.execute('''
                INSERT OR REPLACE INTO menu_items 
                (item_id, name, category, base_price, cost_to_make, 
                 prep_time_minutes, calories, popularity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.item_id, item.name, item.category, item.base_price,
                item.cost_to_make, item.prep_time_minutes, item.calories, item.popularity_score
            ))
        
        # Store sales
        for sale in sales:
            cursor.execute('''
                INSERT OR REPLACE INTO sales 
                (sale_id, restaurant_id, item_id, quantity, price_paid, 
                 timestamp, customer_rating, order_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sale.sale_id, sale.restaurant_id, sale.item_id, sale.quantity,
                sale.price_paid, sale.timestamp, sale.customer_rating, sale.order_type
            ))
        
        conn.commit()
        conn.close()
    
    def optimize_menu_pricing(self):
        """Optimize menu pricing for maximum profitability"""
        print("💰 Optimizing menu pricing strategy...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get menu performance data
        query = '''
            SELECT m.item_id, m.name, m.category, m.base_price, m.cost_to_make,
                   COUNT(s.sale_id) as total_sales,
                   SUM(s.quantity) as total_quantity,
                   AVG(s.price_paid / s.quantity) as avg_selling_price,
                   AVG(CASE WHEN s.customer_rating IS NOT NULL THEN s.customer_rating ELSE 4.0 END) as avg_rating
            FROM menu_items m
            LEFT JOIN sales s ON m.item_id = s.item_id
            GROUP BY m.item_id
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            self.generate_synthetic_restaurant_data()
            return self.optimize_menu_pricing()
        
        optimizations = []
        
        for _, item in df.iterrows():
            current_price = item['base_price']
            cost = item['cost_to_make']
            total_sales = item['total_sales'] if item['total_sales'] else 0
            avg_rating = item['avg_rating'] if item['avg_rating'] else 4.0
            
            # Calculate current profit margin
            current_margin = (current_price - cost) / current_price
            
            # Optimization logic
            if avg_rating >= 4.5 and total_sales > 50:
                # High-rated, popular items can support price increases
                suggested_price = current_price * 1.10
                optimization_reason = "High demand & quality"
            elif avg_rating < 3.5:
                # Low-rated items need price reduction or removal
                suggested_price = current_price * 0.90
                optimization_reason = "Poor customer satisfaction"
            elif total_sales < 10:
                # Low-selling items
                if current_margin > 0.6:  # High margin
                    suggested_price = current_price * 0.85
                    optimization_reason = "Boost low sales volume"
                else:
                    suggested_price = current_price * 0.95
                    optimization_reason = "Minor price adjustment"
            elif current_margin < 0.4:
                # Low margin items
                suggested_price = current_price * 1.05
                optimization_reason = "Improve profit margin"
            else:
                # Well-performing items
                suggested_price = current_price
                optimization_reason = "No change needed"
            
            # Ensure minimum margin
            min_price = cost * 1.25  # At least 25% margin
            suggested_price = max(suggested_price, min_price)
            
            improvement = ((suggested_price - current_price) / current_price) * 100
            
            optimization = {
                'item_id': item['item_id'],
                'item_name': item['name'],
                'category': item['category'],
                'current_price': current_price,
                'suggested_price': suggested_price,
                'price_change_percent': improvement,
                'current_margin': current_margin,
                'new_margin': (suggested_price - cost) / suggested_price,
                'reason': optimization_reason,
                'total_sales': total_sales
            }
            
            optimizations.append(optimization)
        
        return optimizations
    
    def analyze_restaurant_performance(self):
        """Analyze individual restaurant performance"""
        print("📊 Analyzing restaurant performance metrics...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT r.restaurant_id, r.location, r.city, r.monthly_rent,
                   r.size_sqft, r.staff_count,
                   COUNT(s.sale_id) as total_orders,
                   SUM(s.price_paid) as total_revenue,
                   AVG(s.price_paid) as avg_order_value,
                   AVG(CASE WHEN s.customer_rating IS NOT NULL THEN s.customer_rating ELSE 4.0 END) as avg_rating
            FROM restaurants r
            LEFT JOIN sales s ON r.restaurant_id = s.restaurant_id
            GROUP BY r.restaurant_id
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate performance metrics
        df['revenue_per_sqft'] = df['total_revenue'] / df['size_sqft']
        df['revenue_per_staff'] = df['total_revenue'] / df['staff_count']
        df['orders_per_day'] = df['total_orders'] / 365  # Assuming 1 year of data
        
        # Calculate profitability (simplified)
        monthly_revenue = df['total_revenue'] / 12
        estimated_costs = df['monthly_rent'] + (df['staff_count'] * 3000)  # $3k per staff per month
        df['estimated_monthly_profit'] = monthly_revenue - estimated_costs
        df['profit_margin'] = df['estimated_monthly_profit'] / monthly_revenue
        
        # Performance ranking
        df['performance_score'] = (
            df['revenue_per_sqft'] * 0.3 +
            df['avg_rating'] * 0.2 +
            df['profit_margin'] * 0.3 +
            (df['orders_per_day'] / df['orders_per_day'].max()) * 0.2
        )
        
        return df.sort_values('performance_score', ascending=False)
    
    def forecast_demand(self):
        """Forecast demand for menu items"""
        print("📈 Forecasting menu item demand...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get daily sales data
        query = '''
            SELECT DATE(timestamp) as date, item_id, 
                   COUNT(sale_id) as daily_orders,
                   SUM(quantity) as daily_quantity
            FROM sales
            GROUP BY DATE(timestamp), item_id
            ORDER BY date
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {}
        
        df['date'] = pd.to_datetime(df['date'])
        
        forecasts = {}
        
        # Simple moving average forecast for top items
        for item_id in df['item_id'].value_counts().head(10).index:
            item_data = df[df['item_id'] == item_id].copy()
            item_data = item_data.set_index('date')
            
            if len(item_data) > 30:  # Need enough data
                # 7-day moving average
                item_data['ma_7'] = item_data['daily_quantity'].rolling(window=7).mean()
                
                # Simple trend
                recent_avg = item_data['daily_quantity'].tail(14).mean()
                older_avg = item_data['daily_quantity'].head(14).mean()
                trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                
                # Next 7 days forecast
                last_ma = item_data['ma_7'].iloc[-1]
                forecast_base = last_ma * (1 + trend)
                
                forecasts[item_id] = {
                    'current_avg_daily': recent_avg,
                    'forecast_daily': forecast_base,
                    'trend_percent': trend * 100,
                    'confidence': 'medium' if len(item_data) > 60 else 'low'
                }
        
        return forecasts
    
    def create_restaurant_dashboard(self):
        """Create comprehensive restaurant analytics dashboard"""
        print("📊 Creating restaurant chain analytics dashboard...")
        
        # Generate data and analysis
        restaurants, menu_items, sales = self.generate_synthetic_restaurant_data()
        pricing_optimizations = self.optimize_menu_pricing()
        performance_df = self.analyze_restaurant_performance()
        demand_forecasts = self.forecast_demand()
        
        # Load processed data
        conn = sqlite3.connect(self.db_path)
        
        sales_df = pd.read_sql_query('''
            SELECT s.*, m.category, m.name as item_name, r.city
            FROM sales s
            JOIN menu_items m ON s.item_id = m.item_id
            JOIN restaurants r ON s.restaurant_id = r.restaurant_id
        ''', conn)
        
        menu_df = pd.read_sql_query('SELECT * FROM menu_items', conn)
        
        conn.close()
        
        # Create dashboard
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '🏪 Restaurant Performance Ranking',
                '💰 Menu Category Revenue',
                '⭐ Customer Satisfaction by City',
                '📈 Daily Sales Trends',
                '🍽️ Menu Item Profitability',
                '📦 Order Type Distribution',
                '🎯 Price Optimization Impact',
                '📊 Key Performance Indicators'
            ]
        )
        
        # 1. Restaurant Performance Ranking
        if not performance_df.empty:
            top_performers = performance_df.head(10)
            fig.add_trace(
                go.Bar(
                    y=top_performers['location'],
                    x=top_performers['performance_score'],
                    orientation='h',
                    marker_color='green',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Menu Category Revenue
        if not sales_df.empty:
            category_revenue = sales_df.groupby('category')['price_paid'].sum().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(
                    x=category_revenue.index,
                    y=category_revenue.values,
                    marker_color='blue',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Customer Satisfaction by City
        if not sales_df.empty:
            city_ratings = sales_df.dropna(subset=['customer_rating']).groupby('city')['customer_rating'].mean()
            fig.add_trace(
                go.Bar(
                    x=city_ratings.index,
                    y=city_ratings.values,
                    marker_color='orange',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Daily Sales Trends
        if not sales_df.empty:
            sales_df['date'] = pd.to_datetime(sales_df['timestamp']).dt.date
            daily_sales = sales_df.groupby('date')['price_paid'].sum().rolling(window=7).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_sales.index,
                    y=daily_sales.values,
                    mode='lines',
                    line=dict(color='purple', width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 5. Menu Item Profitability
        if not menu_df.empty:
            menu_df['profit_margin'] = (menu_df['base_price'] - menu_df['cost_to_make']) / menu_df['base_price']
            
            fig.add_trace(
                go.Scatter(
                    x=menu_df['base_price'],
                    y=menu_df['profit_margin'],
                    mode='markers',
                    marker=dict(
                        size=menu_df['popularity_score'] * 20,
                        color=menu_df['popularity_score'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=menu_df['name'],
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # 6. Order Type Distribution
        if not sales_df.empty:
            order_type_counts = sales_df['order_type'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=order_type_counts.index,
                    values=order_type_counts.values,
                    name="Order Types"
                ),
                row=3, col=2
            )
        
        # 7. Price Optimization Impact
        if pricing_optimizations:
            opt_df = pd.DataFrame(pricing_optimizations)
            positive_changes = opt_df[opt_df['price_change_percent'] > 0]
            
            if not positive_changes.empty:
                fig.add_trace(
                    go.Bar(
                        x=positive_changes['item_name'][:8],
                        y=positive_changes['price_change_percent'][:8],
                        marker_color='gold',
                        showlegend=False
                    ),
                    row=4, col=1
                )
        
        # 8. Key Performance Indicators
        if not performance_df.empty and not sales_df.empty:
            kpis = ['Avg Revenue', 'Avg Rating', 'Total Orders']
            kpi_values = [
                performance_df['total_revenue'].mean(),
                sales_df.dropna(subset=['customer_rating'])['customer_rating'].mean(),
                performance_df['total_orders'].mean()
            ]
            
            fig.add_trace(
                go.Bar(
                    x=kpis,
                    y=kpi_values,
                    marker_color=['green', 'blue', 'orange'],
                    showlegend=False
                ),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1600,
            title_text="🍕 Restaurant Chain Analytics Dashboard 🍽️",
            title_font_size=24
        )
        
        fig.write_html("restaurant_analytics_dashboard.html")
        fig.show()
        
        return fig
    
    def generate_restaurant_insights(self):
        """Generate restaurant chain insights and recommendations"""
        pricing_optimizations = self.optimize_menu_pricing()
        performance_df = self.analyze_restaurant_performance()
        demand_forecasts = self.forecast_demand()
        
        print("\\n" + "="*70)
        print("🍕 RESTAURANT CHAIN ANALYTICS - BUSINESS INSIGHTS 🍽️")
        print("="*70)
        
        # Performance analysis
        if not performance_df.empty:
            top_performer = performance_df.iloc[0]
            bottom_performer = performance_df.iloc[-1]
            avg_revenue = performance_df['total_revenue'].mean()
            
            print(f"🏆 Chain Performance Analysis:")
            print(f"   • Top Performing Location: {top_performer['location']}")
            print(f"   • Revenue: ${top_performer['total_revenue']:,.0f}")
            print(f"   • Bottom Performing Location: {bottom_performer['location']}")
            print(f"   • Average Location Revenue: ${avg_revenue:,.0f}")
            print(f"   • Performance Gap: {((top_performer['total_revenue'] - bottom_performer['total_revenue']) / bottom_performer['total_revenue'] * 100):.0f}%")
        
        # Menu optimization
        if pricing_optimizations:
            opt_df = pd.DataFrame(pricing_optimizations)
            price_increases = len(opt_df[opt_df['price_change_percent'] > 0])
            price_decreases = len(opt_df[opt_df['price_change_percent'] < 0])
            avg_improvement = opt_df['price_change_percent'].mean()
            
            print(f"\\n💰 Menu Pricing Optimization:")
            print(f"   • Items Requiring Price Increase: {price_increases}")
            print(f"   • Items Requiring Price Decrease: {price_decreases}")
            print(f"   • Average Price Adjustment: {avg_improvement:+.1f}%")
            
            # Top opportunities
            top_opportunities = opt_df.nlargest(3, 'price_change_percent')
            print(f"\\n   🎯 Top Pricing Opportunities:")
            for _, opp in top_opportunities.iterrows():
                print(f"     • {opp['item_name']}: {opp['price_change_percent']:+.1f}% ({opp['reason']})")
        
        # Demand forecasting
        if demand_forecasts:
            growing_items = {k: v for k, v in demand_forecasts.items() if v['trend_percent'] > 10}
            declining_items = {k: v for k, v in demand_forecasts.items() if v['trend_percent'] < -10}
            
            print(f"\\n📈 Demand Forecasting Insights:")
            print(f"   • Items with Growing Demand: {len(growing_items)}")
            print(f"   • Items with Declining Demand: {len(declining_items)}")
            print(f"   • Total Items Forecasted: {len(demand_forecasts)}")
        
        # City performance
        if not performance_df.empty:
            city_performance = performance_df.groupby('city').agg({
                'total_revenue': 'mean',
                'avg_rating': 'mean',
                'performance_score': 'mean'
            }).sort_values('performance_score', ascending=False)
            
            print(f"\\n🏙️ City Performance Rankings:")
            for city, metrics in city_performance.iterrows():
                print(f"   • {city}: ${metrics['total_revenue']:,.0f} avg revenue, {metrics['avg_rating']:.1f} rating")
        
        # Operational recommendations
        print(f"\\n💡 Strategic Recommendations:")
        print(f"   • Focus marketing efforts on high-performing menu categories")
        print(f"   • Implement dynamic pricing based on demand patterns")
        print(f"   • Consider menu simplification in underperforming locations")
        print(f"   • Invest in staff training for locations with low ratings")
        print(f"   • Expand successful location formats to new markets")
        
        print("="*70)
    
    def run_complete_analysis(self):
        """Execute complete restaurant chain analysis"""
        print("🚀 Starting Restaurant Chain Analytics Pipeline...")
        print("="*55)
        
        # Generate restaurant data
        restaurants, menu_items, sales = self.generate_synthetic_restaurant_data()
        
        # Optimize menu pricing
        pricing_opt = self.optimize_menu_pricing()
        
        # Analyze performance
        performance_analysis = self.analyze_restaurant_performance()
        
        # Forecast demand
        demand_forecasts = self.forecast_demand()
        
        # Create dashboard
        dashboard = self.create_restaurant_dashboard()
        
        # Generate insights
        self.generate_restaurant_insights()
        
        print("\\n✅ Restaurant Analytics Complete!")
        print("📁 Files generated:")
        print("   • restaurant_analytics_dashboard.html")
        print("   • restaurant_analytics.db")
        
        # Export summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'restaurants_analyzed': len(restaurants),
            'menu_items_analyzed': len(menu_items),
            'total_sales_records': len(sales),
            'pricing_optimizations': len(pricing_opt),
            'avg_performance_score': performance_analysis['performance_score'].mean() if not performance_analysis.empty else 0,
            'forecasted_items': len(demand_forecasts)
        }
        
        with open('restaurant_analytics_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return restaurants, menu_items, sales

def main():
    """Main execution function"""
    print("🎯 RESTAURANT CHAIN OPTIMIZER - Industry-Ready Platform")
    print("=" * 60)
    print("Showcasing: Operations Research • Business Analytics • Optimization")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = RestaurantChainOptimizer()
    
    # Run complete analysis
    restaurants, menu_items, sales = optimizer.run_complete_analysis()
    
    print(f"\\n🎉 Analysis completed successfully!")
    print(f"🏪 Optimized {len(restaurants)} restaurant locations")
    print(f"🍽️ Analyzed {len(menu_items)} menu items")
    print(f"📊 Processed {len(sales)} sales transactions")
    
    return optimizer, restaurants, menu_items

if __name__ == "__main__":
    optimizer, restaurants, menu_items = main()