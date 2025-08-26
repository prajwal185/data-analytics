# Project 1: The Lifecycle of an Internet Meme - Complete Implementation

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import praw
import tweepy
import time
import sqlite3
from collections import defaultdict
import hashlib
import os

# Create the main project class
class MemeLifecycleTracker:
    def __init__(self):
        """Initialize the Meme Lifecycle Tracker with API configurations"""
        self.reddit_client_id = "YOUR_REDDIT_CLIENT_ID"
        self.reddit_client_secret = "YOUR_REDDIT_CLIENT_SECRET"
        self.reddit_user_agent = "MemeTracker:v1.0 (by /u/yourusername)"
        
        self.twitter_bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
        
        # Initialize database
        self.setup_database()
        
        # Load pre-trained models
        self.setup_models()
        
    def setup_database(self):
        """Set up SQLite database for storing meme data"""
        self.conn = sqlite3.connect('meme_tracker.db')
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                source_platform TEXT,
                first_seen DATETIME,
                title TEXT,
                url TEXT,
                upvotes INTEGER,
                comments INTEGER,
                viral_score REAL,
                mutation_count INTEGER DEFAULT 0
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS meme_variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_meme_id INTEGER,
                image_hash TEXT,
                similarity_score REAL,
                platform TEXT,
                timestamp DATETIME,
                engagement_metrics TEXT,
                FOREIGN KEY (parent_meme_id) REFERENCES memes (id)
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS viral_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meme_id INTEGER,
                timestamp DATETIME,
                platform TEXT,
                engagement_count INTEGER,
                velocity REAL,
                reach_estimate INTEGER,
                FOREIGN KEY (meme_id) REFERENCES memes (id)
            )
        ''')
        
        self.conn.commit()
    
    def setup_models(self):
        """Initialize computer vision models for meme analysis"""
        # Load ResNet50 for feature extraction
        self.feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        # Initialize clustering model for meme categorization
        self.meme_clusterer = KMeans(n_clusters=10, random_state=42)
        
        # Viral prediction model (placeholder - would be trained on historical data)
        self.viral_predictor = LinearRegression()
    
    def extract_image_features(self, image_path):
        """Extract features from meme image using ResNet50"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            
            # Extract features
            features = self.feature_extractor.predict(img)
            return features.flatten()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def calculate_image_hash(self, image_path):
        """Calculate perceptual hash for meme identification"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
                
            # Resize to small fixed size
            img = cv2.resize(img, (8, 8))
            
            # Calculate hash
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            return img_hash
        except:
            return None
    
    def scrape_reddit_memes(self, subreddit_names=['memes', 'dankmemes', 'AdviceAnimals'], limit=100):
        """Scrape memes from Reddit"""
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            
            meme_data = []
            
            for subreddit_name in subreddit_names:
                print(f"Scraping r/{subreddit_name}...")
                subreddit = reddit.subreddit(subreddit_name)
                
                for submission in subreddit.hot(limit=limit):
                    if submission.url.endswith(('.jpg', '.png', '.gif', '.jpeg')):
                        meme_info = {
                            'id': submission.id,
                            'title': submission.title,
                            'url': submission.url,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': datetime.fromtimestamp(submission.created_utc),
                            'subreddit': subreddit_name,
                            'platform': 'reddit'
                        }
                        meme_data.append(meme_info)
                
                time.sleep(1)  # Rate limiting
            
            return meme_data
            
        except Exception as e:
            print(f"Error scraping Reddit: {e}")
            return []
    
    def calculate_viral_score(self, engagement_metrics):
        """Calculate viral score based on engagement metrics"""
        # Weighted combination of various engagement metrics
        score = (
            engagement_metrics.get('upvotes', 0) * 0.4 +
            engagement_metrics.get('comments', 0) * 0.3 +
            engagement_metrics.get('shares', 0) * 0.2 +
            engagement_metrics.get('velocity', 0) * 0.1
        )
        
        # Normalize to 0-100 scale
        return min(100, score / 1000 * 100)
    
    def detect_meme_variants(self, base_features, candidate_features, threshold=0.8):
        """Detect if two memes are variants using feature similarity"""
        if base_features is None or candidate_features is None:
            return False, 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(base_features, candidate_features) / (
            np.linalg.norm(base_features) * np.linalg.norm(candidate_features)
        )
        
        return similarity > threshold, similarity
    
    def track_meme_lifecycle(self, meme_id, days=30):
        """Track a meme's lifecycle over time"""
        query = '''
            SELECT timestamp, engagement_count, velocity 
            FROM viral_metrics 
            WHERE meme_id = ? 
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, self.conn, params=(meme_id,))
        
        if df.empty:
            return None
        
        # Calculate lifecycle phases
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').resample('H').mean().fillna(method='forward')
        
        # Identify peak
        peak_idx = df['engagement_count'].idxmax()
        peak_value = df.loc[peak_idx, 'engagement_count']
        
        # Calculate phases
        birth_phase = df[df.index <= peak_idx].iloc[:len(df)//4]
        growth_phase = df[df.index <= peak_idx].iloc[len(df)//4:]
        decline_phase = df[df.index > peak_idx]
        
        lifecycle_stats = {
            'birth_duration': len(birth_phase),
            'growth_duration': len(growth_phase),
            'peak_engagement': peak_value,
            'peak_time': peak_idx,
            'decline_rate': decline_phase['engagement_count'].pct_change().mean() if len(decline_phase) > 1 else 0,
            'total_lifecycle_days': (df.index[-1] - df.index[0]).days
        }
        
        return lifecycle_stats, df
    
    def predict_virality(self, meme_features):
        """Predict viral potential of a meme"""
        # This would use a trained model in production
        # For demonstration, using a simple heuristic
        
        if len(meme_features) < 5:
            return 0.5  # Default prediction
        
        # Simple scoring based on various factors
        engagement_score = min(1.0, meme_features[0] / 1000)  # Normalize engagement
        timing_score = meme_features[1] if len(meme_features) > 1 else 0.5
        originality_score = meme_features[2] if len(meme_features) > 2 else 0.5
        
        viral_probability = (engagement_score * 0.5 + timing_score * 0.3 + originality_score * 0.2)
        
        return viral_probability
    
    def analyze_mutation_patterns(self):
        """Analyze how memes mutate over time"""
        query = '''
            SELECT m.id, m.title, mv.similarity_score, mv.timestamp
            FROM memes m
            JOIN meme_variants mv ON m.id = mv.parent_meme_id
            ORDER BY m.id, mv.timestamp
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return None
        
        mutation_analysis = {
            'total_memes_with_variants': df['id'].nunique(),
            'avg_variants_per_meme': df.groupby('id').size().mean(),
            'avg_similarity_score': df['similarity_score'].mean(),
            'mutation_rate_over_time': df.groupby(pd.to_datetime(df['timestamp']).dt.date).size()
        }
        
        return mutation_analysis
    
    def generate_lifecycle_report(self, meme_id):
        """Generate comprehensive lifecycle report for a meme"""
        # Get basic meme info
        meme_query = "SELECT * FROM memes WHERE id = ?"
        meme_info = pd.read_sql_query(meme_query, self.conn, params=(meme_id,))
        
        if meme_info.empty:
            return None
        
        # Get lifecycle data
        lifecycle_stats, timeline_df = self.track_meme_lifecycle(meme_id)
        
        # Get variants
        variants_query = "SELECT * FROM meme_variants WHERE parent_meme_id = ?"
        variants_df = pd.read_sql_query(variants_query, self.conn, params=(meme_id,))
        
        report = {
            'meme_info': meme_info.iloc[0].to_dict(),
            'lifecycle_stats': lifecycle_stats,
            'timeline_data': timeline_df,
            'variants_count': len(variants_df),
            'variants_data': variants_df
        }
        
        return report
    
    def visualize_meme_lifecycle(self, meme_id, save_path=None):
        """Create visualization of meme lifecycle"""
        lifecycle_stats, timeline_df = self.track_meme_lifecycle(meme_id)
        
        if timeline_df is None:
            print("No data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Meme Lifecycle Analysis - ID: {meme_id}', fontsize=16)
        
        # Engagement over time
        axes[0, 0].plot(timeline_df.index, timeline_df['engagement_count'])
        axes[0, 0].set_title('Engagement Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Engagement Count')
        
        # Velocity over time
        axes[0, 1].plot(timeline_df.index, timeline_df['velocity'])
        axes[0, 1].set_title('Viral Velocity Over Time')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Velocity')
        
        # Lifecycle phases
        phases = ['Birth', 'Growth', 'Peak', 'Decline']
        durations = [
            lifecycle_stats.get('birth_duration', 0),
            lifecycle_stats.get('growth_duration', 0),
            1,  # Peak is a moment
            lifecycle_stats.get('total_lifecycle_days', 0) - lifecycle_stats.get('growth_duration', 0)
        ]
        
        axes[1, 0].bar(phases, durations)
        axes[1, 0].set_title('Lifecycle Phase Durations')
        axes[1, 0].set_ylabel('Duration (hours)')
        
        # Engagement distribution
        axes[1, 1].hist(timeline_df['engagement_count'], bins=20, alpha=0.7)
        axes[1, 1].set_title('Engagement Distribution')
        axes[1, 1].set_xlabel('Engagement Count')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def run_analysis_pipeline(self):
        """Run complete meme analysis pipeline"""
        print("Starting Meme Lifecycle Analysis Pipeline...")
        
        # 1. Collect meme data
        print("1. Collecting meme data from Reddit...")
        meme_data = self.scrape_reddit_memes()
        print(f"Collected {len(meme_data)} memes")
        
        # 2. Process and store memes
        print("2. Processing and storing memes...")
        for meme in meme_data[:10]:  # Limit for demo
            # Calculate viral score
            viral_score = self.calculate_viral_score({
                'upvotes': meme['score'],
                'comments': meme['num_comments'],
                'velocity': meme['score'] / max(1, (datetime.now() - meme['created_utc']).hours)
            })
            
            # Store in database
            self.cursor.execute('''
                INSERT OR REPLACE INTO memes 
                (image_hash, source_platform, first_seen, title, url, upvotes, comments, viral_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meme['id'],  # Using reddit ID as hash for demo
                meme['platform'],
                meme['created_utc'],
                meme['title'],
                meme['url'],
                meme['score'],
                meme['num_comments'],
                viral_score
            ))
        
        self.conn.commit()
        
        # 3. Analyze patterns
        print("3. Analyzing mutation patterns...")
        mutation_analysis = self.analyze_mutation_patterns()
        if mutation_analysis:
            print(f"Found variants for {mutation_analysis['total_memes_with_variants']} memes")
        
        # 4. Generate summary report
        print("4. Generating summary report...")
        summary_query = '''
            SELECT 
                COUNT(*) as total_memes,
                AVG(viral_score) as avg_viral_score,
                MAX(viral_score) as max_viral_score,
                AVG(upvotes) as avg_upvotes,
                AVG(comments) as avg_comments
            FROM memes
        '''
        
        summary = pd.read_sql_query(summary_query, self.conn)
        print("\nSummary Statistics:")
        print(summary.to_string(index=False))
        
        return summary

# Example usage and testing
def main():
    # Initialize tracker
    tracker = MemeLifecycleTracker()
    
    # Run analysis pipeline
    summary = tracker.run_analysis_pipeline()
    
    # Save summary to CSV
    summary.to_csv('meme_analysis_summary.csv', index=False)
    print("\nAnalysis complete! Summary saved to 'meme_analysis_summary.csv'")
    
    # Demonstrate specific features
    print("\nDemonstrating specific features...")
    
    # Mock data for demonstration
    sample_meme_features = np.array([500, 0.7, 0.8, 100, 50])  # engagement, timing, originality, etc.
    viral_probability = tracker.predict_virality(sample_meme_features)
    print(f"Viral probability prediction: {viral_probability:.2f}")
    
    # Create sample visualization (mock data)
    dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
    sample_engagement = np.random.poisson(100, 24) * np.exp(-np.arange(24) * 0.1)
    sample_velocity = np.gradient(sample_engagement)
    
    timeline_df = pd.DataFrame({
        'engagement_count': sample_engagement,
        'velocity': sample_velocity
    }, index=dates)
    
    # Store mock data for visualization
    tracker.cursor.execute('''
        INSERT OR REPLACE INTO memes 
        (id, image_hash, source_platform, first_seen, title, viral_score)
        VALUES (999, 'mock_hash', 'demo', datetime('now'), 'Demo Meme', 75.5)
    ''')
    
    for i, (timestamp, row) in enumerate(timeline_df.iterrows()):
        tracker.cursor.execute('''
            INSERT INTO viral_metrics 
            (meme_id, timestamp, platform, engagement_count, velocity)
            VALUES (999, ?, 'demo', ?, ?)
        ''', (timestamp, int(row['engagement_count']), row['velocity']))
    
    tracker.conn.commit()
    
    # Generate visualization
    tracker.visualize_meme_lifecycle(999, 'demo_meme_lifecycle.png')
    
    print("Visualization saved as 'demo_meme_lifecycle.png'")

if __name__ == "__main__":
    main()

print("Project 1: Meme Lifecycle Tracker implementation completed!")
print("This implementation includes:")
print("- Reddit API integration for meme collection")
print("- Computer vision for meme variant detection")
print("- Viral score calculation and prediction")
print("- Lifecycle tracking and analysis")
print("- Database storage with SQLite")
print("- Visualization and reporting capabilities")
print("- Mutation pattern analysis")
print("\nTo use this in production:")
print("1. Add your API keys for Reddit and Twitter")
print("2. Install required packages: praw, tweepy, tensorflow, opencv-python, etc.")
print("3. Implement image download and processing")
print("4. Add more sophisticated ML models")
print("5. Scale with cloud infrastructure")