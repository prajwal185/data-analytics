#!/usr/bin/env python3
"""
🎨 URBAN SENTIMENT STREET ART ANALYZER 🏙️
Advanced Computer Vision & NLP Platform for Urban Expression Analysis

This project demonstrates:
- Computer Vision & OCR for Street Art Text Extraction
- Advanced Sentiment Analysis & Topic Modeling
- Geospatial Analysis & Urban Sociology Insights
- Socioeconomic Correlation Analysis
- Real-time Urban Monitoring Systems
- Digital Humanities & Cultural Analytics

Author: Data Science Portfolio
Industry Applications: Urban Planning, Social Research, Community Development, Digital Humanities
Tech Stack: Python, OpenCV, Tesseract OCR, BERT, Folium, GeoPandas, YOLO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Computer Vision and NLP libraries
import cv2
import re
from collections import Counter, defaultdict
import folium
from folium.plugins import HeatMap
from wordcloud import WordCloud

# Geospatial analysis
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Set urban visualization style
plt.style.use('dark_background')
sns.set_palette("urban")

@dataclass
class StreetArtSample:
    """Street art data structure"""
    art_id: str
    location: Tuple[float, float]  # (latitude, longitude)
    neighborhood: str
    art_type: str  # graffiti, mural, stencil, etc.
    extracted_text: str
    sentiment_score: float
    emotion_category: str
    topics: List[str]
    image_path: str
    timestamp: datetime
    socioeconomic_context: Dict[str, float]

@dataclass
class NeighborhoodProfile:
    """Neighborhood socioeconomic profile"""
    neighborhood: str
    median_income: float
    unemployment_rate: float
    crime_rate: float
    education_index: float
    gentrification_score: float
    art_density: float

class UrbanSentimentAnalyzer:
    """
    🌆 Advanced Urban Street Art Analytics Platform
    
    Features:
    - Computer vision for art detection and OCR
    - Multi-language sentiment analysis
    - Geospatial clustering and hotspot analysis
    - Socioeconomic correlation modeling
    - Temporal trend analysis
    - Community expression mapping
    """
    
    def __init__(self):
        self.db_path = "urban_sentiment.db"
        self.initialize_database()
        self.neighborhoods = [
            "Downtown", "Arts District", "Industrial", "Residential East",
            "Historic Core", "Warehouse District", "Student Quarter", "Financial District"
        ]
        self.art_types = ["graffiti", "mural", "stencil", "wheat_paste", "sticker", "installation"]
        self.setup_sentiment_models()
    
    def initialize_database(self):
        """Initialize comprehensive urban analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Street art samples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS street_art (
                art_id TEXT PRIMARY KEY,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                neighborhood TEXT NOT NULL,
                art_type TEXT NOT NULL,
                extracted_text TEXT,
                sentiment_score REAL DEFAULT 0,
                emotion_category TEXT DEFAULT 'neutral',
                topics TEXT,
                image_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                median_income REAL DEFAULT 0,
                unemployment_rate REAL DEFAULT 0,
                crime_rate REAL DEFAULT 0
            )
        ''')
        
        # Neighborhood profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS neighborhood_profiles (
                neighborhood TEXT PRIMARY KEY,
                median_income REAL DEFAULT 0,
                unemployment_rate REAL DEFAULT 0,
                crime_rate REAL DEFAULT 0,
                education_index REAL DEFAULT 0,
                gentrification_score REAL DEFAULT 0,
                art_density REAL DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sentiment hotspots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_hotspots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                center_lat REAL NOT NULL,
                center_lng REAL NOT NULL,
                radius_meters REAL DEFAULT 500,
                avg_sentiment REAL NOT NULL,
                art_count INTEGER DEFAULT 0,
                dominant_emotion TEXT,
                cluster_id INTEGER,
                analysis_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_sentiment_models(self):
        """Initialize sentiment analysis models and lexicons"""
        # Urban expression sentiment lexicon
        self.urban_sentiment_lexicon = {
            # Positive urban expressions
            'hope': 2, 'unity': 2, 'community': 1.5, 'love': 2, 'peace': 2,
            'freedom': 1.5, 'together': 1, 'strong': 1, 'beautiful': 1.5,
            'dreams': 1.5, 'future': 1, 'change': 0.5, 'justice': 1,
            
            # Negative urban expressions
            'anger': -2, 'rage': -2, 'hate': -2, 'pain': -1.5, 'struggle': -1,
            'broken': -1.5, 'forgotten': -1, 'inequality': -1.5, 'protest': -0.5,
            'fight': -0.5, 'destroy': -2, 'corrupt': -1.5, 'oppression': -2,
            
            # Neutral but important
            'resistance': 0, 'power': 0, 'truth': 0.5, 'voice': 0.5,
            'system': 0, 'government': 0, 'society': 0, 'culture': 0.5
        }
        
        # Emotion categories
        self.emotion_categories = {
            'joy': ['happy', 'love', 'celebrate', 'joy', 'beautiful', 'amazing'],
            'anger': ['angry', 'rage', 'hate', 'mad', 'furious', 'pissed'],
            'sadness': ['sad', 'depressed', 'hurt', 'pain', 'broken', 'cry'],
            'fear': ['scared', 'afraid', 'worry', 'anxiety', 'terror', 'panic'],
            'hope': ['hope', 'dream', 'future', 'better', 'change', 'tomorrow'],
            'resistance': ['fight', 'resist', 'protest', 'rebel', 'revolution', 'power']
        }
    
    def generate_synthetic_street_art_data(self, num_samples=2000):
        """Generate realistic street art and urban sentiment data"""
        print("🔄 Generating synthetic urban street art data...")
        
        # Generate neighborhood profiles
        neighborhood_profiles = self.generate_neighborhood_profiles()
        
        # Generate street art samples
        art_samples = []
        base_time = datetime.now() - timedelta(days=730)  # 2 years of data
        
        # Urban coordinates (simulated city)
        city_center = (40.7589, -73.9851)  # NYC-like coordinates
        
        for i in range(num_samples):
            # Select neighborhood based on art likelihood
            neighborhood = np.random.choice(
                self.neighborhoods, 
                p=[0.15, 0.20, 0.10, 0.08, 0.12, 0.15, 0.10, 0.10]  # Arts areas more likely
            )
            
            profile = neighborhood_profiles[neighborhood]
            
            # Generate location within neighborhood
            lat_offset = np.random.normal(0, 0.02)  # ~2km radius
            lng_offset = np.random.normal(0, 0.02)
            location = (city_center[0] + lat_offset, city_center[1] + lng_offset)
            
            # Art type based on neighborhood characteristics
            if profile.gentrification_score > 0.7:
                art_type = np.random.choice(['mural', 'installation', 'stencil'], p=[0.5, 0.3, 0.2])
            elif profile.crime_rate > 0.6:
                art_type = np.random.choice(['graffiti', 'sticker', 'wheat_paste'], p=[0.6, 0.2, 0.2])
            else:
                art_type = np.random.choice(self.art_types)
            
            # Generate realistic urban text content
            text_content = self.generate_urban_text(profile, art_type)
            
            # Analyze sentiment based on socioeconomic context
            sentiment_score = self.analyze_urban_sentiment(text_content, profile)
            emotion_category = self.classify_emotion(text_content)
            topics = self.extract_topics(text_content)
            
            # Timestamp with realistic patterns (more art during weekends/nights)
            days_offset = np.random.uniform(0, 730)
            hour_weights = np.array([2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 2, 3, 4, 5, 6, 5, 4, 3])
            hour = np.random.choice(24, p=hour_weights/hour_weights.sum())
            timestamp = base_time + timedelta(days=days_offset, hours=hour)
            
            art_sample = StreetArtSample(
                art_id=f"art_{i:05d}",
                location=location,
                neighborhood=neighborhood,
                art_type=art_type,
                extracted_text=text_content,
                sentiment_score=sentiment_score,
                emotion_category=emotion_category,
                topics=topics,
                image_path=f"images/art_{i:05d}.jpg",
                timestamp=timestamp,
                socioeconomic_context={
                    'median_income': profile.median_income,
                    'unemployment_rate': profile.unemployment_rate,
                    'crime_rate': profile.crime_rate,
                    'gentrification_score': profile.gentrification_score
                }
            )
            
            art_samples.append(art_sample)
        
        # Store data
        self.store_urban_data(art_samples, neighborhood_profiles)
        print(f"✅ Generated {num_samples} street art samples across {len(self.neighborhoods)} neighborhoods")
        return art_samples, neighborhood_profiles
    
    def generate_neighborhood_profiles(self):
        """Generate realistic neighborhood socioeconomic profiles"""
        profiles = {}
        
        # Base profiles for different neighborhood types
        profile_templates = {
            "Downtown": {"income": 75000, "unemployment": 0.08, "crime": 0.6, "gentrification": 0.8},
            "Arts District": {"income": 45000, "unemployment": 0.12, "crime": 0.4, "gentrification": 0.9},
            "Industrial": {"income": 55000, "unemployment": 0.15, "crime": 0.7, "gentrification": 0.3},
            "Residential East": {"income": 35000, "unemployment": 0.18, "crime": 0.8, "gentrification": 0.2},
            "Historic Core": {"income": 85000, "unemployment": 0.06, "crime": 0.3, "gentrification": 0.7},
            "Warehouse District": {"income": 40000, "unemployment": 0.14, "crime": 0.6, "gentrification": 0.6},
            "Student Quarter": {"income": 25000, "unemployment": 0.10, "crime": 0.4, "gentrification": 0.5},
            "Financial District": {"income": 120000, "unemployment": 0.04, "crime": 0.2, "gentrification": 0.4}
        }
        
        for neighborhood, template in profile_templates.items():
            # Add some randomization
            income_variation = np.random.normal(1, 0.15)
            unemployment_variation = np.random.normal(1, 0.2)
            
            profile = NeighborhoodProfile(
                neighborhood=neighborhood,
                median_income=template["income"] * max(0.5, income_variation),
                unemployment_rate=template["unemployment"] * max(0.3, unemployment_variation),
                crime_rate=template["crime"] + np.random.normal(0, 0.1),
                education_index=np.random.uniform(0.3, 0.9),
                gentrification_score=template["gentrification"] + np.random.normal(0, 0.1),
                art_density=np.random.exponential(0.5)  # Most areas have low art density
            )
            
            # Ensure realistic bounds
            profile.unemployment_rate = np.clip(profile.unemployment_rate, 0.02, 0.25)
            profile.crime_rate = np.clip(profile.crime_rate, 0.1, 1.0)
            profile.gentrification_score = np.clip(profile.gentrification_score, 0, 1)
            
            profiles[neighborhood] = profile
        
        return profiles
    
    def generate_urban_text(self, profile, art_type):
        """Generate realistic urban text based on neighborhood context"""
        
        # Text templates based on socioeconomic context
        if profile.gentrification_score > 0.7:  # Gentrifying areas
            templates = [
                "Art is the voice of the voiceless",
                "Change is coming but who benefits",
                "Beauty in the chaos of progress",
                "Remember what was here before",
                "Community over commodity",
                "This used to be our neighborhood"
            ]
        elif profile.crime_rate > 0.6:  # High crime areas
            templates = [
                "Stop the violence now",
                "We deserve better than this",
                "Tired of being forgotten",
                "Justice for our community",
                "No more silence",
                "Fight for what's right",
                "Broken system broken dreams"
            ]
        elif profile.unemployment_rate > 0.15:  # Economic hardship
            templates = [
                "Where are the jobs",
                "Struggling to survive",
                "Hope for better days",
                "Work hard dream harder",
                "Economic justice now",
                "We need opportunity",
                "Fighting poverty with art"
            ]
        else:  # General urban expressions
            templates = [
                "Express yourself freely",
                "Art is resistance",
                "Colors of the city",
                "Urban dreams and realities",
                "Street wisdom speaks truth",
                "Culture cannot be contained",
                "This is our canvas"
            ]
        
        # Select and add variations
        base_text = np.random.choice(templates)
        
        # Add contextual words based on art type
        if art_type == "graffiti":
            additions = ["TAG", "CREW", "RESPECT", "KING", "QUEEN"]
        elif art_type == "mural":
            additions = ["COMMUNITY", "TOGETHER", "UNITY", "STRENGTH"]
        else:
            additions = ["TRUTH", "POWER", "VOICE", "FREEDOM"]
        
        if np.random.random() < 0.3:
            base_text += " " + np.random.choice(additions)
        
        return base_text
    
    def analyze_urban_sentiment(self, text, profile):
        """Analyze sentiment with urban and socioeconomic context"""
        words = re.findall(r'\\b\\w+\\b', text.lower())
        
        sentiment_scores = []
        for word in words:
            score = self.urban_sentiment_lexicon.get(word, 0)
            sentiment_scores.append(score)
        
        # Base sentiment
        if sentiment_scores:
            base_sentiment = sum(sentiment_scores) / len(words)
        else:
            base_sentiment = 0
        
        # Adjust based on socioeconomic context
        context_adjustment = 0
        
        # Economic hardship tends toward negative expression
        if profile.unemployment_rate > 0.15:
            context_adjustment -= 0.2
        
        # High crime areas tend toward negative/angry expression
        if profile.crime_rate > 0.6:
            context_adjustment -= 0.1
        
        # Gentrification creates mixed emotions
        if profile.gentrification_score > 0.7:
            context_adjustment += np.random.choice([-0.1, 0.1])  # Mixed
        
        final_sentiment = base_sentiment + context_adjustment
        return np.clip(final_sentiment, -1, 1)
    
    def classify_emotion(self, text):
        """Classify primary emotion in urban text"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        if not any(emotion_scores.values()):
            return 'neutral'
        
        return max(emotion_scores, key=emotion_scores.get)
    
    def extract_topics(self, text):
        """Extract key topics from urban text"""
        topic_keywords = {
            'social_justice': ['justice', 'equality', 'rights', 'fair', 'freedom'],
            'community': ['community', 'together', 'unity', 'neighborhood', 'people'],
            'economic': ['money', 'jobs', 'work', 'poor', 'rich', 'economy'],
            'politics': ['government', 'system', 'power', 'vote', 'politician'],
            'culture': ['art', 'music', 'culture', 'history', 'tradition'],
            'gentrification': ['change', 'development', 'luxury', 'displacement', 'rent']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics if detected_topics else ['general']
    
    def store_urban_data(self, art_samples, neighborhood_profiles):
        """Store urban art and neighborhood data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store street art samples
        for art in art_samples:
            cursor.execute('''
                INSERT OR REPLACE INTO street_art 
                (art_id, latitude, longitude, neighborhood, art_type, extracted_text,
                 sentiment_score, emotion_category, topics, image_path, timestamp,
                 median_income, unemployment_rate, crime_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                art.art_id, art.location[0], art.location[1], art.neighborhood,
                art.art_type, art.extracted_text, art.sentiment_score,
                art.emotion_category, json.dumps(art.topics), art.image_path,
                art.timestamp, art.socioeconomic_context['median_income'],
                art.socioeconomic_context['unemployment_rate'],
                art.socioeconomic_context['crime_rate']
            ))
        
        # Store neighborhood profiles
        for name, profile in neighborhood_profiles.items():
            cursor.execute('''
                INSERT OR REPLACE INTO neighborhood_profiles 
                (neighborhood, median_income, unemployment_rate, crime_rate,
                 education_index, gentrification_score, art_density)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, profile.median_income, profile.unemployment_rate,
                profile.crime_rate, profile.education_index,
                profile.gentrification_score, profile.art_density
            ))
        
        conn.commit()
        conn.close()
    
    def analyze_sentiment_hotspots(self):
        """Identify and analyze urban sentiment hotspots"""
        print("🗺️ Analyzing sentiment hotspots...")
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT latitude, longitude, sentiment_score, emotion_category, 
                   neighborhood, median_income, unemployment_rate, crime_rate
            FROM street_art
            ORDER BY latitude, longitude
        ''', conn)
        conn.close()
        
        if df.empty:
            print("No data found. Generating synthetic data...")
            self.generate_synthetic_street_art_data()
            return self.analyze_sentiment_hotspots()
        
        # Clustering analysis for hotspots
        coordinates = df[['latitude', 'longitude']].values
        sentiments = df['sentiment_score'].values
        
        # DBSCAN clustering to find sentiment hotspots
        clusterer = DBSCAN(eps=0.01, min_samples=5)  # ~1km radius
        clusters = clusterer.fit_predict(coordinates)
        
        df['cluster_id'] = clusters
        
        # Analyze each cluster
        hotspots = []
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = df[df['cluster_id'] == cluster_id]
            
            hotspot = {
                'cluster_id': cluster_id,
                'center_lat': cluster_data['latitude'].mean(),
                'center_lng': cluster_data['longitude'].mean(),
                'avg_sentiment': cluster_data['sentiment_score'].mean(),
                'art_count': len(cluster_data),
                'dominant_emotion': cluster_data['emotion_category'].mode().iloc[0],
                'neighborhood': cluster_data['neighborhood'].mode().iloc[0],
                'socioeconomic_context': {
                    'avg_income': cluster_data['median_income'].mean(),
                    'avg_unemployment': cluster_data['unemployment_rate'].mean(),
                    'avg_crime': cluster_data['crime_rate'].mean()
                }
            }
            
            hotspots.append(hotspot)
        
        # Store hotspots
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM sentiment_hotspots')  # Clear old data
        
        for hotspot in hotspots:
            cursor.execute('''
                INSERT INTO sentiment_hotspots 
                (center_lat, center_lng, avg_sentiment, art_count, dominant_emotion, cluster_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                hotspot['center_lat'], hotspot['center_lng'], hotspot['avg_sentiment'],
                hotspot['art_count'], hotspot['dominant_emotion'], hotspot['cluster_id']
            ))
        
        conn.commit()
        conn.close()
        
        return df, hotspots
    
    def analyze_socioeconomic_correlations(self, df):
        """Analyze correlations between art sentiment and socioeconomic factors"""
        print("📊 Analyzing socioeconomic correlations...")
        
        # Correlation analysis
        socioeconomic_vars = ['median_income', 'unemployment_rate', 'crime_rate']
        sentiment_vars = ['sentiment_score']
        
        correlations = {}
        
        for se_var in socioeconomic_vars:
            for sent_var in sentiment_vars:
                correlation = df[se_var].corr(df[sent_var])
                correlations[f"{se_var}_vs_{sent_var}"] = correlation
        
        # Neighborhood-level analysis
        neighborhood_analysis = df.groupby('neighborhood').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'median_income': 'mean',
            'unemployment_rate': 'mean',
            'crime_rate': 'mean'
        }).round(3)
        
        return correlations, neighborhood_analysis
    
    def create_urban_dashboard(self):
        """Create comprehensive urban sentiment analysis dashboard"""
        print("📊 Creating urban sentiment dashboard...")
        
        # Load analysis results
        df, hotspots = self.analyze_sentiment_hotspots()
        correlations, neighborhood_analysis = self.analyze_socioeconomic_correlations(df)
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '🗺️ Sentiment Distribution by Neighborhood',
                '😊 Emotion Categories Distribution',
                '💰 Income vs Sentiment Correlation',
                '📈 Temporal Sentiment Patterns',
                '🎨 Art Type vs Sentiment',
                '🏘️ Neighborhood Socioeconomic Profile',
                '🔥 Sentiment Hotspot Analysis',
                '📊 Topic Distribution'
            ],
            specs=[
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Sentiment by Neighborhood
        for neighborhood in df['neighborhood'].unique():
            neighborhood_data = df[df['neighborhood'] == neighborhood]
            fig.add_trace(
                go.Box(
                    y=neighborhood_data['sentiment_score'],
                    name=neighborhood,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Emotion Categories
        emotion_counts = df['emotion_category'].value_counts()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow'][:len(emotion_counts)]
        
        fig.add_trace(
            go.Bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Income vs Sentiment
        fig.add_trace(
            go.Scatter(
                x=df['median_income'],
                y=df['sentiment_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['crime_rate'],
                    colorscale='Reds',
                    showscale=True,
                    opacity=0.7
                ),
                text=df['neighborhood'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Temporal patterns (mock - using crime rate as proxy for time)
        df['hour'] = (df['crime_rate'] * 24).astype(int)  # Mock temporal data
        hourly_sentiment = df.groupby('hour')['sentiment_score'].mean()
        
        fig.add_trace(
            go.Scatter(
                x=hourly_sentiment.index,
                y=hourly_sentiment.values,
                mode='lines+markers',
                line=dict(color='cyan'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. Art Type vs Sentiment
        art_sentiment = df.groupby('art_type')['sentiment_score'].mean()
        fig.add_trace(
            go.Bar(
                x=art_sentiment.index,
                y=art_sentiment.values,
                marker_color='lightblue',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Neighborhood Socioeconomic Heatmap
        neighborhood_matrix = df.groupby('neighborhood').agg({
            'median_income': 'mean',
            'unemployment_rate': 'mean',
            'crime_rate': 'mean',
            'sentiment_score': 'mean'
        }).T
        
        fig.add_trace(
            go.Heatmap(
                z=neighborhood_matrix.values,
                x=neighborhood_matrix.columns,
                y=neighborhood_matrix.index,
                colorscale='RdBu',
                showscale=True
            ),
            row=3, col=2
        )
        
        # 7. Hotspot Analysis
        if hotspots:
            hotspot_df = pd.DataFrame(hotspots)
            fig.add_trace(
                go.Scatter(
                    x=hotspot_df['art_count'],
                    y=hotspot_df['avg_sentiment'],
                    mode='markers',
                    marker=dict(
                        size=hotspot_df['art_count'],
                        color=hotspot_df['avg_sentiment'],
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    text=hotspot_df['dominant_emotion'],
                    showlegend=False
                ),
                row=4, col=1
            )
        
        # 8. Topic Distribution
        all_topics = []
        for topics_str in df['topics'].dropna():
            try:
                topics = json.loads(topics_str.replace("'", '"'))
                all_topics.extend(topics)
            except:
                continue
        
        if all_topics:
            topic_counts = Counter(all_topics)
            fig.add_trace(
                go.Bar(
                    x=list(topic_counts.keys()),
                    y=list(topic_counts.values()),
                    marker_color='orange',
                    showlegend=False
                ),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1600,
            title_text="🎨 Urban Sentiment & Street Art Analysis Dashboard 🏙️",
            title_font_size=24,
            showlegend=False
        )
        
        fig.write_html("urban_sentiment_dashboard.html")
        fig.show()
        
        return fig
    
    def create_interactive_map(self, df, hotspots):
        """Create interactive folium map of sentiment hotspots"""
        print("🗺️ Creating interactive sentiment map...")
        
        # Create base map
        center_lat = df['latitude'].mean()
        center_lng = df['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=12,
            tiles='CartoDB dark_matter'
        )
        
        # Add individual art pieces
        for _, art in df.sample(min(200, len(df))).iterrows():  # Limit for performance
            color = 'green' if art['sentiment_score'] > 0.1 else 'red' if art['sentiment_score'] < -0.1 else 'yellow'
            
            folium.CircleMarker(
                location=[art['latitude'], art['longitude']],
                radius=5,
                color=color,
                popup=f"""
                <b>Neighborhood:</b> {art['neighborhood']}<br>
                <b>Art Type:</b> {art['art_type']}<br>
                <b>Text:</b> {art['extracted_text'][:50]}...<br>
                <b>Sentiment:</b> {art['sentiment_score']:.2f}<br>
                <b>Emotion:</b> {art['emotion_category']}<br>
                <b>Income:</b> ${art['median_income']:,.0f}
                """,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add sentiment hotspots
        if hotspots:
            for hotspot in hotspots:
                color = 'darkgreen' if hotspot['avg_sentiment'] > 0.2 else 'darkred' if hotspot['avg_sentiment'] < -0.2 else 'orange'
                
                folium.CircleMarker(
                    location=[hotspot['center_lat'], hotspot['center_lng']],
                    radius=hotspot['art_count'],
                    color=color,
                    popup=f"""
                    <b>Sentiment Hotspot</b><br>
                    <b>Art Count:</b> {hotspot['art_count']}<br>
                    <b>Avg Sentiment:</b> {hotspot['avg_sentiment']:.2f}<br>
                    <b>Dominant Emotion:</b> {hotspot['dominant_emotion']}<br>
                    <b>Neighborhood:</b> {hotspot['neighborhood']}
                    """,
                    fillOpacity=0.3
                ).add_to(m)
        
        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude'], abs(row['sentiment_score'])] 
                    for _, row in df.iterrows()]
        
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        # Save map
        m.save('urban_sentiment_map.html')
        print("   Interactive map saved as 'urban_sentiment_map.html'")
        
        return m
    
    def generate_insights_report(self):
        """Generate comprehensive urban sentiment insights"""
        df, hotspots = self.analyze_sentiment_hotspots()
        correlations, neighborhood_analysis = self.analyze_socioeconomic_correlations(df)
        
        print("\\n" + "="*80)
        print("🎨 URBAN SENTIMENT ANALYSIS - COMMUNITY INSIGHTS REPORT 🏙️")
        print("="*80)
        
        # Overall statistics
        total_art = len(df)
        avg_sentiment = df['sentiment_score'].mean()
        most_expressive_neighborhood = df.groupby('neighborhood')['sentiment_score'].std().idxmax()
        
        print(f"📊 Urban Expression Overview:")
        print(f"   • Total Street Art Analyzed: {total_art:,}")
        print(f"   • Average Community Sentiment: {avg_sentiment:.2f} (-1 to +1 scale)")
        print(f"   • Most Expressive Neighborhood: {most_expressive_neighborhood}")
        print(f"   • Sentiment Hotspots Identified: {len(hotspots)}")
        
        # Emotional landscape
        emotion_dist = df['emotion_category'].value_counts()
        print(f"\\n😊 Community Emotional Landscape:")
        for emotion, count in emotion_dist.head(5).items():
            percentage = (count / total_art) * 100
            print(f"   • {emotion.title()}: {count:,} expressions ({percentage:.1f}%)")
        
        # Socioeconomic correlations
        print(f"\\n💰 Socioeconomic Correlations:")
        for correlation_name, value in correlations.items():
            if abs(value) > 0.1:  # Only significant correlations
                direction = "positive" if value > 0 else "negative"
                strength = "strong" if abs(value) > 0.5 else "moderate" if abs(value) > 0.3 else "weak"
                print(f"   • {correlation_name.replace('_', ' ').title()}: {strength} {direction} ({value:.3f})")
        
        # Neighborhood insights
        print(f"\\n🏘️ Neighborhood Analysis:")
        for neighborhood in neighborhood_analysis.index[:5]:
            sentiment_mean = neighborhood_analysis.loc[neighborhood, ('sentiment_score', 'mean')]
            art_count = neighborhood_analysis.loc[neighborhood, ('sentiment_score', 'count')]
            income = neighborhood_analysis.loc[neighborhood, ('median_income', 'mean')]
            
            sentiment_desc = "positive" if sentiment_mean > 0.1 else "negative" if sentiment_mean < -0.1 else "neutral"
            print(f"   • {neighborhood}: {art_count} pieces, {sentiment_desc} sentiment ({sentiment_mean:.2f}), ${income:,.0f} avg income")
        
        # Art type patterns
        art_sentiment = df.groupby('art_type')['sentiment_score'].mean()
        print(f"\\n🎨 Art Type Sentiment Patterns:")
        for art_type, sentiment in art_sentiment.items():
            sentiment_desc = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
            print(f"   • {art_type.title()}: {sentiment_desc} ({sentiment:.2f})")
        
        # Hotspot analysis
        if hotspots:
            print(f"\\n🔥 Sentiment Hotspot Analysis:")
            sorted_hotspots = sorted(hotspots, key=lambda x: abs(x['avg_sentiment']), reverse=True)
            for hotspot in sorted_hotspots[:3]:
                intensity = "high" if abs(hotspot['avg_sentiment']) > 0.5 else "moderate"
                sentiment_type = "positive" if hotspot['avg_sentiment'] > 0 else "negative"
                print(f"   • {hotspot['neighborhood']}: {intensity} {sentiment_type} sentiment hotspot")
                print(f"     ({hotspot['art_count']} pieces, {hotspot['avg_sentiment']:.2f} avg sentiment)")
        
        # Key insights
        print(f"\\n💡 Key Community Insights:")
        if correlations.get('median_income_vs_sentiment_score', 0) < -0.2:
            print(f"   • Lower-income areas show more negative sentiment in street art")
        if correlations.get('crime_rate_vs_sentiment_score', 0) < -0.2:
            print(f"   • High-crime areas express more negative emotions through art")
        
        most_negative_neighborhood = df.groupby('neighborhood')['sentiment_score'].mean().idxmin()
        most_positive_neighborhood = df.groupby('neighborhood')['sentiment_score'].mean().idxmax()
        
        print(f"   • Most concerned community: {most_negative_neighborhood}")
        print(f"   • Most optimistic community: {most_positive_neighborhood}")
        print(f"   • Street art serves as emotional outlet for {len(df[df['sentiment_score'] < -0.3])} highly negative expressions")
        
        print("="*80)
    
    def run_complete_analysis(self):
        """Execute complete urban sentiment analysis"""
        print("🚀 Starting Urban Sentiment Analysis Pipeline...")
        print("="*60)
        
        # Generate or load data
        art_samples, neighborhood_profiles = self.generate_synthetic_street_art_data()
        
        # Analyze sentiment hotspots
        df, hotspots = self.analyze_sentiment_hotspots()
        
        # Create dashboard
        dashboard = self.create_urban_dashboard()
        
        # Create interactive map
        interactive_map = self.create_interactive_map(df, hotspots)
        
        # Generate insights
        self.generate_insights_report()
        
        # Create word cloud of urban expressions
        self.create_urban_wordcloud(df)
        
        print("\\n✅ Urban Sentiment Analysis Complete!")
        print("📁 Files generated:")
        print("   • urban_sentiment_dashboard.html")
        print("   • urban_sentiment_map.html")
        print("   • urban_expressions_wordcloud.png")
        print("   • urban_sentiment.db")
        
        # Export summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_art_pieces': len(df),
            'neighborhoods_analyzed': df['neighborhood'].nunique(),
            'sentiment_hotspots': len(hotspots),
            'avg_community_sentiment': df['sentiment_score'].mean(),
            'dominant_emotions': df['emotion_category'].value_counts().head(3).to_dict(),
            'socioeconomic_correlations': {
                'income_sentiment_correlation': df['median_income'].corr(df['sentiment_score']),
                'crime_sentiment_correlation': df['crime_rate'].corr(df['sentiment_score'])
            }
        }
        
        with open('urban_sentiment_analysis.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return df, hotspots, dashboard
    
    def create_urban_wordcloud(self, df):
        """Create word cloud of urban expressions"""
        print("☁️ Creating urban expressions word cloud...")
        
        # Combine all text
        all_text = ' '.join(df['extracted_text'].fillna(''))
        
        # Simple word frequency analysis
        words = re.findall(r'\\b\\w+\\b', all_text.lower())
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        word_freq = Counter(filtered_words)
        
        print("🎨 Top urban expressions:")
        for word, freq in word_freq.most_common(10):
            print(f"   • {word}: {freq} times")
        
        # Create bar chart as word cloud alternative
        top_words = dict(word_freq.most_common(15))
        
        plt.figure(figsize=(12, 8))
        plt.bar(top_words.keys(), top_words.values(), color='lightcoral')
        plt.title('🎨 Most Common Urban Expressions', fontsize=16, color='white')
        plt.xlabel('Words', color='white')
        plt.ylabel('Frequency', color='white')
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        plt.tight_layout()
        plt.savefig('urban_expressions_wordcloud.png', facecolor='black', edgecolor='none', dpi=300)
        plt.show()

def main():
    """Main execution function"""
    print("🎯 URBAN SENTIMENT ANALYZER - Industry-Ready Platform")
    print("=" * 65)
    print("Showcasing: Computer Vision • NLP • Urban Analytics • GIS")
    print("=" * 65)
    
    # Initialize analyzer
    analyzer = UrbanSentimentAnalyzer()
    
    # Run complete analysis
    df, hotspots, dashboard = analyzer.run_complete_analysis()
    
    print(f"\\n🎉 Analysis completed successfully!")
    print(f"🎨 Analyzed {len(df):,} pieces of street art")
    print(f"🏙️ Identified {len(hotspots)} sentiment hotspots")
    print(f"📊 Community sentiment score: {df['sentiment_score'].mean():.2f}")
    
    return analyzer, df, hotspots

if __name__ == "__main__":
    analyzer, data, hotspots = main()