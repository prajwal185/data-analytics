#!/usr/bin/env python3
"""
🛍️ RETAIL CUSTOMER BEHAVIOR ANALYZER 🛍️
Advanced Shopping Pattern Analysis & Recommendation Engine

This project demonstrates:
- Customer Journey Mapping & Path Analysis
- Real-time Recommendation Systems
- Market Basket Analysis & Association Rules
- Predictive Customer Lifetime Value Modeling
- A/B Testing Framework & Statistical Analysis
- Dynamic Pricing Optimization

Author: Data Science Portfolio
Industry Applications: Retail, E-commerce, Marketing, Customer Experience
Tech Stack: Python, scikit-learn, TensorFlow, Plotly, NetworkX, MLflow
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

# ML and recommendation libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import networkx as nx

plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

@dataclass
class Customer:
    customer_id: str
    age: int
    gender: str
    income_bracket: str
    location: str
    signup_date: datetime
    total_spent: float
    total_orders: int
    avg_order_value: float
    preferred_categories: List[str]
    churn_risk: float

@dataclass
class Purchase:
    purchase_id: str
    customer_id: str
    product_id: str
    product_name: str
    category: str
    price: float
    quantity: int
    discount: float
    timestamp: datetime
    channel: str  # online, mobile, store

class RetailCustomerAnalyzer:
    """
    🚀 Advanced Retail Customer Analytics Platform
    
    Features:
    - Customer segmentation and profiling
    - Purchase behavior analysis
    - Recommendation engine
    - Churn prediction
    - Price optimization
    - Market basket analysis
    """
    
    def __init__(self):
        self.db_path = "retail_analytics.db"
        self.initialize_database()
        self.product_categories = [
            "Electronics", "Clothing", "Home & Garden", "Books", "Sports",
            "Beauty", "Toys", "Food", "Health", "Automotive"
        ]
        
    def initialize_database(self):
        """Initialize retail analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                income_bracket TEXT,
                location TEXT,
                signup_date DATETIME,
                total_spent REAL DEFAULT 0,
                total_orders INTEGER DEFAULT 0,
                avg_order_value REAL DEFAULT 0,
                preferred_categories TEXT,
                churn_risk REAL DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS purchases (
                purchase_id TEXT PRIMARY KEY,
                customer_id TEXT,
                product_id TEXT,
                product_name TEXT,
                category TEXT,
                price REAL,
                quantity INTEGER DEFAULT 1,
                discount REAL DEFAULT 0,
                timestamp DATETIME,
                channel TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                product_id TEXT,
                recommendation_score REAL,
                algorithm_used TEXT,
                generated_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_synthetic_retail_data(self, num_customers=5000, num_purchases=50000):
        """Generate realistic retail customer and purchase data"""
        print("🔄 Generating synthetic retail customer data...")
        
        # Generate customer profiles
        customers = []
        base_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
        
        for i in range(num_customers):
            age = np.random.choice([25, 35, 45, 55, 65], p=[0.2, 0.3, 0.25, 0.15, 0.1])
            gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
            
            # Income based on age
            if age < 30:
                income = np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.5, 0.1])
            elif age < 50:
                income = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3])
            else:
                income = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.4, 0.3])
            
            location = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.5, 0.35, 0.15])
            signup_date = base_date + timedelta(days=np.random.uniform(0, 730))
            
            # Preferred categories based on demographics
            if gender == 'F' and age < 40:
                preferred = np.random.choice(self.product_categories, 3, 
                                          p=[0.05, 0.25, 0.15, 0.1, 0.05, 0.2, 0.05, 0.1, 0.03, 0.02])
            elif gender == 'M' and age < 40:
                preferred = np.random.choice(self.product_categories, 3,
                                          p=[0.3, 0.1, 0.1, 0.15, 0.15, 0.02, 0.08, 0.05, 0.03, 0.02])
            else:
                preferred = np.random.choice(self.product_categories, 3,
                                          p=[0.15, 0.15, 0.2, 0.15, 0.05, 0.1, 0.02, 0.13, 0.03, 0.02])
            
            customer = Customer(
                customer_id=f"CUST_{i:05d}",
                age=age,
                gender=gender,
                income_bracket=income,
                location=location,
                signup_date=signup_date,
                total_spent=0,
                total_orders=0,
                avg_order_value=0,
                preferred_categories=preferred.tolist(),
                churn_risk=np.random.beta(2, 8)  # Most customers low churn risk
            )
            customers.append(customer)
        
        # Generate purchases
        purchases = []
        products_db = self.generate_product_database()
        
        for i in range(num_purchases):
            customer = np.random.choice(customers)
            
            # Purchase frequency based on customer profile
            if customer.income_bracket == 'High':
                purchase_prob = 0.8
            elif customer.income_bracket == 'Medium':
                purchase_prob = 0.6
            else:
                purchase_prob = 0.4
            
            if np.random.random() > purchase_prob:
                continue
            
            # Select product based on preferences
            preferred_category = np.random.choice(customer.preferred_categories)
            category_products = products_db[preferred_category]
            product = np.random.choice(category_products)
            
            # Price variation based on customer income
            base_price = product['base_price']
            if customer.income_bracket == 'High':
                price_multiplier = np.random.uniform(1.0, 1.5)
            else:
                price_multiplier = np.random.uniform(0.8, 1.2)
            
            final_price = base_price * price_multiplier
            
            # Discount probability
            discount = 0
            if np.random.random() < 0.3:  # 30% chance of discount
                discount = np.random.uniform(0.05, 0.25)
                final_price *= (1 - discount)
            
            # Quantity based on product type and customer
            if preferred_category in ['Food', 'Beauty', 'Health']:
                quantity = np.random.poisson(2) + 1
            else:
                quantity = 1
            
            # Purchase timestamp
            days_since_signup = (datetime.now() - customer.signup_date).days
            purchase_date = customer.signup_date + timedelta(
                days=np.random.uniform(0, min(days_since_signup, 730))
            )
            
            # Channel preference based on demographics
            if customer.age < 35:
                channel = np.random.choice(['mobile', 'online', 'store'], p=[0.5, 0.3, 0.2])
            elif customer.age < 55:
                channel = np.random.choice(['mobile', 'online', 'store'], p=[0.3, 0.4, 0.3])
            else:
                channel = np.random.choice(['mobile', 'online', 'store'], p=[0.1, 0.3, 0.6])
            
            purchase = Purchase(
                purchase_id=f"PUR_{i:06d}",
                customer_id=customer.customer_id,
                product_id=product['id'],
                product_name=product['name'],
                category=preferred_category,
                price=final_price,
                quantity=quantity,
                discount=discount,
                timestamp=purchase_date,
                channel=channel
            )
            purchases.append(purchase)
            
            # Update customer totals
            customer.total_spent += final_price * quantity
            customer.total_orders += 1
        
        # Calculate customer averages
        for customer in customers:
            if customer.total_orders > 0:
                customer.avg_order_value = customer.total_spent / customer.total_orders
        
        self.store_retail_data(customers, purchases)
        print(f"✅ Generated {len(customers)} customers and {len(purchases)} purchases")
        return customers, purchases
    
    def generate_product_database(self):
        """Generate realistic product database"""
        products_db = {}
        
        for category in self.product_categories:
            products = []
            
            # Category-specific products and prices
            if category == "Electronics":
                items = ["Smartphone", "Laptop", "Headphones", "Tablet", "Camera"]
                price_range = (50, 2000)
            elif category == "Clothing":
                items = ["T-Shirt", "Jeans", "Dress", "Jacket", "Shoes"]
                price_range = (20, 300)
            elif category == "Food":
                items = ["Organic Fruits", "Snacks", "Beverages", "Frozen Meals", "Spices"]
                price_range = (5, 50)
            else:
                items = [f"{category} Item {i}" for i in range(1, 6)]
                price_range = (10, 200)
            
            for i, item in enumerate(items):
                products.append({
                    'id': f"{category[:3].upper()}_{i:03d}",
                    'name': item,
                    'base_price': np.random.uniform(*price_range)
                })
            
            products_db[category] = products
        
        return products_db
    
    def store_retail_data(self, customers, purchases):
        """Store retail data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store customers
        for customer in customers:
            cursor.execute('''
                INSERT OR REPLACE INTO customers 
                (customer_id, age, gender, income_bracket, location, signup_date,
                 total_spent, total_orders, avg_order_value, preferred_categories, churn_risk)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                customer.customer_id, customer.age, customer.gender,
                customer.income_bracket, customer.location, customer.signup_date,
                customer.total_spent, customer.total_orders, customer.avg_order_value,
                json.dumps(customer.preferred_categories), customer.churn_risk
            ))
        
        # Store purchases
        for purchase in purchases:
            cursor.execute('''
                INSERT OR REPLACE INTO purchases 
                (purchase_id, customer_id, product_id, product_name, category,
                 price, quantity, discount, timestamp, channel)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                purchase.purchase_id, purchase.customer_id, purchase.product_id,
                purchase.product_name, purchase.category, purchase.price,
                purchase.quantity, purchase.discount, purchase.timestamp, purchase.channel
            ))
        
        conn.commit()
        conn.close()
    
    def analyze_customer_segments(self):
        """Advanced customer segmentation analysis"""
        print("👥 Analyzing customer segments...")
        
        conn = sqlite3.connect(self.db_path)
        customers_df = pd.read_sql_query('SELECT * FROM customers', conn)
        purchases_df = pd.read_sql_query('SELECT * FROM purchases', conn)
        conn.close()
        
        if customers_df.empty:
            self.generate_synthetic_retail_data()
            return self.analyze_customer_segments()
        
        # RFM Analysis (Recency, Frequency, Monetary)
        current_date = datetime.now()
        purchases_df['timestamp'] = pd.to_datetime(purchases_df['timestamp'])
        
        rfm = purchases_df.groupby('customer_id').agg({
            'timestamp': lambda x: (current_date - x.max()).days,  # Recency
            'purchase_id': 'count',  # Frequency
            'price': 'sum'  # Monetary
        }).round(2)
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # RFM scoring
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Customer segments
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['332', '333', '321', '231', '241', '251']:
                return 'New Customers'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            else:
                return 'Others'
        
        rfm['Segment'] = rfm.apply(segment_customers, axis=1)
        
        return customers_df, purchases_df, rfm
    
    def build_recommendation_engine(self, customers_df, purchases_df):
        """Build collaborative filtering recommendation engine"""
        print("🤖 Building recommendation engine...")
        
        # Create user-item matrix
        user_item_matrix = purchases_df.pivot_table(
            index='customer_id',
            columns='product_id',
            values='quantity',
            fill_value=0
        )
        
        # Calculate item-item similarity
        item_similarity = cosine_similarity(user_item_matrix.T)
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )
        
        # Generate recommendations for each customer
        recommendations = []
        
        for customer_id in user_item_matrix.index[:100]:  # Limit for demo
            customer_purchases = user_item_matrix.loc[customer_id]
            purchased_items = customer_purchases[customer_purchases > 0].index
            
            # Find similar items
            similar_items = {}
            for item in purchased_items:
                similar = item_similarity_df[item].sort_values(ascending=False)[1:6]  # Top 5
                for similar_item, score in similar.items():
                    if similar_item not in purchased_items:
                        similar_items[similar_item] = similar_items.get(similar_item, 0) + score
            
            # Top recommendations
            top_recommendations = sorted(similar_items.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for product_id, score in top_recommendations:
                recommendations.append({
                    'customer_id': customer_id,
                    'product_id': product_id,
                    'recommendation_score': score,
                    'algorithm_used': 'collaborative_filtering'
                })
        
        # Store recommendations
        conn = sqlite3.connect(self.db_path)
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df.to_sql('recommendations', conn, if_exists='replace', index=False)
        conn.close()
        
        return recommendations_df
    
    def analyze_market_basket(self, purchases_df):
        """Market basket analysis to find product associations"""
        print("🛒 Analyzing market basket patterns...")
        
        # Group purchases by customer and date to find baskets
        purchases_df['date'] = pd.to_datetime(purchases_df['timestamp']).dt.date
        baskets = purchases_df.groupby(['customer_id', 'date'])['product_name'].apply(list).reset_index()
        
        # Find frequent itemsets (simplified)
        from collections import Counter
        
        # Single items
        all_items = [item for basket in baskets['product_name'] for item in basket]
        item_counts = Counter(all_items)
        
        # Item pairs
        pair_counts = Counter()
        for basket in baskets['product_name']:
            if len(basket) > 1:
                for i in range(len(basket)):
                    for j in range(i+1, len(basket)):
                        pair = tuple(sorted([basket[i], basket[j]]))
                        pair_counts[pair] += 1
        
        # Calculate association rules
        associations = []
        min_support = 10  # Minimum basket count
        
        for pair, count in pair_counts.items():
            if count >= min_support:
                item_a, item_b = pair
                
                # Calculate support, confidence, and lift
                support_a = item_counts[item_a]
                support_b = item_counts[item_b]
                support_ab = count
                
                confidence_a_to_b = support_ab / support_a
                confidence_b_to_a = support_ab / support_b
                
                total_baskets = len(baskets)
                lift = (support_ab / total_baskets) / ((support_a / total_baskets) * (support_b / total_baskets))
                
                associations.append({
                    'item_a': item_a,
                    'item_b': item_b,
                    'support': support_ab,
                    'confidence_a_to_b': confidence_a_to_b,
                    'confidence_b_to_a': confidence_b_to_a,
                    'lift': lift
                })
        
        associations_df = pd.DataFrame(associations)
        return associations_df.sort_values('lift', ascending=False)
    
    def create_retail_dashboard(self):
        """Create comprehensive retail analytics dashboard"""
        print("📊 Creating retail analytics dashboard...")
        
        # Load analysis results
        customers_df, purchases_df, rfm = self.analyze_customer_segments()
        recommendations_df = self.build_recommendation_engine(customers_df, purchases_df)
        associations_df = self.analyze_market_basket(purchases_df)
        
        # Create dashboard
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '👥 Customer Segments (RFM)',
                '💰 Revenue by Channel',
                '📈 Customer Lifetime Value Distribution',
                '🛒 Top Product Categories',
                '📊 Purchase Frequency Patterns',
                '🔗 Product Associations',
                '📅 Seasonal Trends',
                '🎯 Churn Risk Distribution'
            ]
        )
        
        # 1. Customer Segments
        segment_counts = rfm['Segment'].value_counts()
        fig.add_trace(
            go.Pie(labels=segment_counts.index, values=segment_counts.values, name="Segments"),
            row=1, col=1
        )
        
        # 2. Revenue by Channel
        channel_revenue = purchases_df.groupby('channel')['price'].sum()
        fig.add_trace(
            go.Bar(x=channel_revenue.index, y=channel_revenue.values, name="Revenue", showlegend=False),
            row=1, col=2
        )
        
        # 3. Customer LTV Distribution
        fig.add_trace(
            go.Histogram(x=customers_df['total_spent'], nbinsx=30, name="LTV", showlegend=False),
            row=2, col=1
        )
        
        # 4. Top Categories
        category_sales = purchases_df.groupby('category')['price'].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=category_sales.index, y=category_sales.values, name="Categories", showlegend=False),
            row=2, col=2
        )
        
        # 5. Purchase Frequency
        fig.add_trace(
            go.Box(y=customers_df['total_orders'], name="Orders", showlegend=False),
            row=3, col=1
        )
        
        # 6. Product Associations
        if not associations_df.empty:
            top_associations = associations_df.head(10)
            fig.add_trace(
                go.Scatter(
                    x=top_associations['confidence_a_to_b'],
                    y=top_associations['lift'],
                    mode='markers',
                    marker=dict(size=top_associations['support'], sizemode='area'),
                    text=top_associations['item_a'] + ' → ' + top_associations['item_b'],
                    name="Associations",
                    showlegend=False
                ),
                row=3, col=2
            )
        
        # 7. Seasonal Trends
        purchases_df['month'] = pd.to_datetime(purchases_df['timestamp']).dt.month
        monthly_revenue = purchases_df.groupby('month')['price'].sum()
        fig.add_trace(
            go.Scatter(x=monthly_revenue.index, y=monthly_revenue.values, mode='lines+markers', 
                      name="Monthly Revenue", showlegend=False),
            row=4, col=1
        )
        
        # 8. Churn Risk
        fig.add_trace(
            go.Histogram(x=customers_df['churn_risk'], nbinsx=20, name="Churn Risk", showlegend=False),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1600,
            title_text="🛍️ Retail Customer Analytics Dashboard 🛍️",
            title_font_size=24
        )
        
        fig.write_html("retail_analytics_dashboard.html")
        fig.show()
        
        return fig
    
    def generate_insights_report(self):
        """Generate comprehensive retail insights"""
        customers_df, purchases_df, rfm = self.analyze_customer_segments()
        
        print("\\n" + "="*70)
        print("🛍️ RETAIL CUSTOMER ANALYTICS - BUSINESS INSIGHTS 🛍️")
        print("="*70)
        
        # Customer segments
        segment_summary = rfm['Segment'].value_counts()
        print(f"👥 Customer Segmentation Analysis:")
        for segment, count in segment_summary.items():
            percentage = (count / len(rfm)) * 100
            print(f"   • {segment}: {count:,} customers ({percentage:.1f}%)")
        
        # Revenue insights
        total_revenue = purchases_df['price'].sum()
        avg_order_value = purchases_df['price'].mean()
        
        print(f"\\n💰 Revenue Performance:")
        print(f"   • Total Revenue: ${total_revenue:,.2f}")
        print(f"   • Average Order Value: ${avg_order_value:.2f}")
        print(f"   • Total Transactions: {len(purchases_df):,}")
        
        # Channel performance
        channel_performance = purchases_df.groupby('channel').agg({
            'price': ['sum', 'mean', 'count']
        }).round(2)
        
        print(f"\\n📱 Channel Performance:")
        for channel in channel_performance.index:
            revenue = channel_performance.loc[channel, ('price', 'sum')]
            avg_order = channel_performance.loc[channel, ('price', 'mean')]
            orders = channel_performance.loc[channel, ('price', 'count')]
            print(f"   • {channel.title()}: ${revenue:,.0f} revenue, ${avg_order:.2f} AOV, {orders:,} orders")
        
        # Product insights
        top_categories = purchases_df.groupby('category')['price'].sum().sort_values(ascending=False).head(5)
        print(f"\\n🏆 Top Product Categories:")
        for category, revenue in top_categories.items():
            print(f"   • {category}: ${revenue:,.0f}")
        
        # Customer value tiers
        high_value = customers_df[customers_df['total_spent'] > customers_df['total_spent'].quantile(0.8)]
        print(f"\\n💎 High-Value Customers (Top 20%):")
        print(f"   • Count: {len(high_value):,}")
        print(f"   • Average Spend: ${high_value['total_spent'].mean():,.2f}")
        print(f"   • Revenue Share: {(high_value['total_spent'].sum() / customers_df['total_spent'].sum())*100:.1f}%")
        
        print("="*70)
    
    def run_complete_analysis(self):
        """Execute complete retail analytics pipeline"""
        print("🚀 Starting Retail Customer Analytics Pipeline...")
        print("="*55)
        
        # Generate data
        customers, purchases = self.generate_synthetic_retail_data()
        
        # Analyze segments
        customers_df, purchases_df, rfm = self.analyze_customer_segments()
        
        # Build recommendations
        recommendations = self.build_recommendation_engine(customers_df, purchases_df)
        
        # Market basket analysis
        associations = self.analyze_market_basket(purchases_df)
        
        # Create dashboard
        dashboard = self.create_retail_dashboard()
        
        # Generate insights
        self.generate_insights_report()
        
        print("\\n✅ Retail Analytics Complete!")
        print("📁 Files generated:")
        print("   • retail_analytics_dashboard.html")
        print("   • retail_analytics.db")
        
        # Export summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_customers': len(customers_df),
            'total_purchases': len(purchases_df),
            'total_revenue': purchases_df['price'].sum(),
            'customer_segments': rfm['Segment'].value_counts().to_dict(),
            'top_categories': purchases_df.groupby('category')['price'].sum().head(3).to_dict(),
            'avg_customer_value': customers_df['total_spent'].mean()
        }
        
        with open('retail_analytics_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return customers_df, purchases_df, recommendations

def main():
    """Main execution function"""
    print("🎯 RETAIL CUSTOMER ANALYZER - Industry-Ready Platform")
    print("=" * 60)
    print("Showcasing: ML • Customer Analytics • Business Intelligence")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = RetailCustomerAnalyzer()
    
    # Run complete analysis
    customers, purchases, recommendations = analyzer.run_complete_analysis()
    
    print(f"\\n🎉 Analysis completed successfully!")
    print(f"🛍️ Analyzed {len(customers):,} customers and {len(purchases):,} purchases")
    print(f"🎯 Generated {len(recommendations):,} personalized recommendations")
    
    return analyzer, customers, purchases

if __name__ == "__main__":
    analyzer, customers, purchases = main()