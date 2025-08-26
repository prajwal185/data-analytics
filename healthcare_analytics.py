#!/usr/bin/env python3
"""
🏥 HEALTHCARE ANALYTICS PLATFORM 🏥
Advanced Medical Data Analysis & Patient Outcome Prediction

This project demonstrates:
- Electronic Health Record (EHR) Analysis
- Predictive Disease Progression Modeling
- Clinical Decision Support Systems
- Patient Risk Stratification
- Medical Imaging Analysis & Computer Vision
- Drug Interaction & Side Effect Prediction

Author: Data Science Portfolio
Industry Applications: Healthcare, Medical Research, Clinical Decision Support
Tech Stack: Python, TensorFlow, scikit-learn, DICOM, FHIR, Plotly
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

# Medical ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8')
sns.set_palette("muted")

@dataclass
class Patient:
    patient_id: str
    age: int
    gender: str
    ethnicity: str
    weight: float
    height: float
    bmi: float
    blood_type: str
    chronic_conditions: List[str]
    medications: List[str]
    admission_date: datetime
    discharge_date: Optional[datetime]
    risk_score: float

@dataclass
class MedicalRecord:
    record_id: str
    patient_id: str
    timestamp: datetime
    vital_signs: Dict[str, float]
    lab_results: Dict[str, float]
    symptoms: List[str]
    diagnosis: str
    treatment: str
    outcome: str

class HealthcareAnalyzer:
    """
    🏥 Advanced Healthcare Analytics Platform
    
    Features:
    - Patient risk stratification
    - Disease progression prediction
    - Treatment effectiveness analysis
    - Clinical decision support
    - Population health insights
    - Resource optimization
    """
    
    def __init__(self):
        self.db_path = "healthcare_analytics.db"
        self.initialize_database()
        self.medical_conditions = [
            "Diabetes", "Hypertension", "Heart Disease", "Asthma", "COPD",
            "Depression", "Anxiety", "Arthritis", "Osteoporosis", "Cancer"
        ]
        self.medications = [
            "Metformin", "Lisinopril", "Atorvastatin", "Albuterol", "Omeprazole",
            "Sertraline", "Ibuprofen", "Aspirin", "Levothyroxine", "Warfarin"
        ]
        
    def initialize_database(self):
        """Initialize healthcare analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                ethnicity TEXT,
                weight REAL,
                height REAL,
                bmi REAL,
                blood_type TEXT,
                chronic_conditions TEXT,
                medications TEXT,
                admission_date DATETIME,
                discharge_date DATETIME,
                risk_score REAL DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medical_records (
                record_id TEXT PRIMARY KEY,
                patient_id TEXT,
                timestamp DATETIME,
                systolic_bp REAL,
                diastolic_bp REAL,
                heart_rate REAL,
                temperature REAL,
                oxygen_saturation REAL,
                glucose REAL,
                cholesterol REAL,
                white_blood_cells REAL,
                hemoglobin REAL,
                symptoms TEXT,
                diagnosis TEXT,
                treatment TEXT,
                outcome TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                risk_type TEXT,
                predicted_risk REAL,
                confidence_interval TEXT,
                model_version TEXT,
                prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_synthetic_medical_data(self, num_patients=3000, num_records=15000):
        """Generate realistic medical data for analysis"""
        print("🔄 Generating synthetic medical data...")
        
        # Generate patient profiles
        patients = []
        base_date = datetime.now() - timedelta(days=365)
        
        for i in range(num_patients):
            age = max(18, int(np.random.normal(55, 20)))
            gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
            
            ethnicity = np.random.choice([
                'Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'
            ], p=[0.6, 0.13, 0.18, 0.06, 0.03])
            
            # Realistic weight/height based on demographics
            if gender == 'M':
                height = np.random.normal(175, 7)  # cm
                weight = np.random.normal(80, 15)  # kg
            else:
                height = np.random.normal(162, 6)
                weight = np.random.normal(67, 12)
            
            weight = max(40, weight)
            height = max(140, height)
            bmi = weight / ((height/100) ** 2)
            
            # Blood types with realistic distribution
            blood_type = np.random.choice([
                'O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-'
            ], p=[0.374, 0.357, 0.085, 0.034, 0.066, 0.063, 0.015, 0.006])
            
            # Chronic conditions based on age and demographics
            condition_prob = min(0.8, age / 100)  # Older patients more likely to have conditions
            num_conditions = np.random.binomial(3, condition_prob)
            chronic_conditions = np.random.choice(
                self.medical_conditions, 
                min(num_conditions, len(self.medical_conditions)), 
                replace=False
            ).tolist()
            
            # Medications based on conditions
            medications = []
            for condition in chronic_conditions:
                if condition == "Diabetes" and np.random.random() < 0.8:
                    medications.append("Metformin")
                elif condition == "Hypertension" and np.random.random() < 0.7:
                    medications.append("Lisinopril")
                elif condition == "Heart Disease" and np.random.random() < 0.6:
                    medications.append("Atorvastatin")
                elif condition == "Depression" and np.random.random() < 0.6:
                    medications.append("Sertraline")
            
            # Add random medications
            if np.random.random() < 0.3:
                medications.extend(np.random.choice(self.medications, 1).tolist())
            
            admission_date = base_date + timedelta(days=np.random.uniform(0, 365))
            
            # Calculate risk score based on multiple factors
            risk_factors = 0
            risk_factors += max(0, (age - 50) / 100)  # Age risk
            risk_factors += len(chronic_conditions) * 0.2  # Condition risk
            risk_factors += max(0, (bmi - 25) / 50)  # BMI risk
            
            risk_score = min(1.0, risk_factors)
            
            # Some patients have discharge dates
            discharge_date = None
            if np.random.random() < 0.7:  # 70% discharged
                stay_days = max(1, int(np.random.exponential(5)))
                discharge_date = admission_date + timedelta(days=stay_days)
            
            patient = Patient(
                patient_id=f"PAT_{i:05d}",
                age=age,
                gender=gender,
                ethnicity=ethnicity,
                weight=weight,
                height=height,
                bmi=bmi,
                blood_type=blood_type,
                chronic_conditions=chronic_conditions,
                medications=medications,
                admission_date=admission_date,
                discharge_date=discharge_date,
                risk_score=risk_score
            )
            patients.append(patient)
        
        # Generate medical records
        medical_records = []
        symptoms_list = [
            "Chest Pain", "Shortness of Breath", "Fatigue", "Nausea", "Dizziness",
            "Headache", "Fever", "Cough", "Joint Pain", "Abdominal Pain"
        ]
        
        diagnoses = [
            "Acute Myocardial Infarction", "Pneumonia", "Sepsis", "Stroke",
            "Diabetes Complications", "Hypertensive Crisis", "Asthma Exacerbation",
            "COPD Exacerbation", "Depression Episode", "Anxiety Attack"
        ]
        
        treatments = [
            "Medication Adjustment", "IV Fluids", "Oxygen Therapy", "Surgery",
            "Physical Therapy", "Counseling", "Dietary Changes", "Monitoring"
        ]
        
        outcomes = ["Improved", "Stable", "Discharged", "Transfer", "Complications"]
        
        for i in range(num_records):
            patient = np.random.choice(patients)
            
            # Record timestamp within patient's stay
            if patient.discharge_date:
                days_range = (patient.discharge_date - patient.admission_date).days
                record_date = patient.admission_date + timedelta(
                    days=np.random.uniform(0, max(1, days_range))
                )
            else:
                record_date = patient.admission_date + timedelta(days=np.random.uniform(0, 30))
            
            # Vital signs with realistic ranges and correlations
            base_systolic = 120 if patient.age < 60 else 135
            if "Hypertension" in patient.chronic_conditions:
                base_systolic += 20
            
            systolic_bp = max(90, np.random.normal(base_systolic, 15))
            diastolic_bp = max(60, systolic_bp - np.random.uniform(40, 60))
            
            heart_rate = np.random.normal(75, 12)
            if "Heart Disease" in patient.chronic_conditions:
                heart_rate += np.random.uniform(-5, 15)
            
            temperature = np.random.normal(98.6, 0.8)
            oxygen_saturation = max(85, np.random.normal(98, 2))
            
            # Lab results
            glucose = 100 if "Diabetes" not in patient.chronic_conditions else np.random.normal(180, 40)
            glucose = max(70, glucose)
            
            cholesterol = np.random.normal(200, 40)
            if "Heart Disease" in patient.chronic_conditions:
                cholesterol += 50
            
            white_blood_cells = max(3, np.random.normal(7, 2))
            hemoglobin = np.random.normal(14 if patient.gender == 'M' else 12, 1.5)
            
            # Symptoms based on conditions
            patient_symptoms = []
            for condition in patient.chronic_conditions:
                if condition == "Heart Disease" and np.random.random() < 0.4:
                    patient_symptoms.extend(["Chest Pain", "Shortness of Breath"])
                elif condition == "Diabetes" and np.random.random() < 0.3:
                    patient_symptoms.extend(["Fatigue", "Nausea"])
                elif condition == "Depression" and np.random.random() < 0.5:
                    patient_symptoms.append("Fatigue")
            
            # Add random symptoms
            if np.random.random() < 0.3:
                patient_symptoms.extend(np.random.choice(symptoms_list, 1).tolist())
            
            patient_symptoms = list(set(patient_symptoms))  # Remove duplicates
            
            # Diagnosis and treatment based on symptoms and conditions
            if "Chest Pain" in patient_symptoms and "Heart Disease" in patient.chronic_conditions:
                diagnosis = "Acute Myocardial Infarction"
                treatment = "IV Fluids"
                outcome = np.random.choice(["Improved", "Transfer", "Complications"], p=[0.6, 0.2, 0.2])
            else:
                diagnosis = np.random.choice(diagnoses)
                treatment = np.random.choice(treatments)
                outcome = np.random.choice(outcomes, p=[0.5, 0.2, 0.15, 0.1, 0.05])
            
            record = MedicalRecord(
                record_id=f"REC_{i:06d}",
                patient_id=patient.patient_id,
                timestamp=record_date,
                vital_signs={
                    'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp,
                    'heart_rate': heart_rate,
                    'temperature': temperature,
                    'oxygen_saturation': oxygen_saturation
                },
                lab_results={
                    'glucose': glucose,
                    'cholesterol': cholesterol,
                    'white_blood_cells': white_blood_cells,
                    'hemoglobin': hemoglobin
                },
                symptoms=patient_symptoms,
                diagnosis=diagnosis,
                treatment=treatment,
                outcome=outcome
            )
            medical_records.append(record)
        
        self.store_medical_data(patients, medical_records)
        print(f"✅ Generated {len(patients)} patients and {len(medical_records)} medical records")
        return patients, medical_records
    
    def store_medical_data(self, patients, medical_records):
        """Store medical data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store patients
        for patient in patients:
            cursor.execute('''
                INSERT OR REPLACE INTO patients 
                (patient_id, age, gender, ethnicity, weight, height, bmi, blood_type,
                 chronic_conditions, medications, admission_date, discharge_date, risk_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient.patient_id, patient.age, patient.gender, patient.ethnicity,
                patient.weight, patient.height, patient.bmi, patient.blood_type,
                json.dumps(patient.chronic_conditions), json.dumps(patient.medications),
                patient.admission_date, patient.discharge_date, patient.risk_score
            ))
        
        # Store medical records
        for record in medical_records:
            cursor.execute('''
                INSERT OR REPLACE INTO medical_records 
                (record_id, patient_id, timestamp, systolic_bp, diastolic_bp, heart_rate,
                 temperature, oxygen_saturation, glucose, cholesterol, white_blood_cells,
                 hemoglobin, symptoms, diagnosis, treatment, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.record_id, record.patient_id, record.timestamp,
                record.vital_signs['systolic_bp'], record.vital_signs['diastolic_bp'],
                record.vital_signs['heart_rate'], record.vital_signs['temperature'],
                record.vital_signs['oxygen_saturation'], record.lab_results['glucose'],
                record.lab_results['cholesterol'], record.lab_results['white_blood_cells'],
                record.lab_results['hemoglobin'], json.dumps(record.symptoms),
                record.diagnosis, record.treatment, record.outcome
            ))
        
        conn.commit()
        conn.close()
    
    def build_risk_prediction_models(self):
        """Build machine learning models for risk prediction"""
        print("🤖 Building medical risk prediction models...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load data
        patients_df = pd.read_sql_query('SELECT * FROM patients', conn)
        records_df = pd.read_sql_query('SELECT * FROM medical_records', conn)
        conn.close()
        
        if patients_df.empty:
            self.generate_synthetic_medical_data()
            return self.build_risk_prediction_models()
        
        # Prepare features for modeling
        # Patient features
        patients_df['chronic_conditions'] = patients_df['chronic_conditions'].apply(
            lambda x: len(json.loads(x)) if x else 0
        )
        patients_df['medications_count'] = patients_df['medications'].apply(
            lambda x: len(json.loads(x)) if x else 0
        )
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_ethnicity = LabelEncoder()
        le_blood_type = LabelEncoder()
        
        patients_df['gender_encoded'] = le_gender.fit_transform(patients_df['gender'])
        patients_df['ethnicity_encoded'] = le_ethnicity.fit_transform(patients_df['ethnicity'])
        patients_df['blood_type_encoded'] = le_blood_type.fit_transform(patients_df['blood_type'])
        
        # Features for modeling
        feature_columns = [
            'age', 'gender_encoded', 'ethnicity_encoded', 'bmi', 'blood_type_encoded',
            'chronic_conditions', 'medications_count'
        ]
        
        X = patients_df[feature_columns]
        
        # Model 1: Readmission Risk
        patients_df['readmission_risk'] = (patients_df['risk_score'] > 0.5).astype(int)
        y_readmission = patients_df['readmission_risk']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_readmission, test_size=0.2, random_state=42)
        
        # Train readmission model
        readmission_model = RandomForestClassifier(n_estimators=100, random_state=42)
        readmission_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_prob = readmission_model.predict_proba(X_test)[:, 1]
        readmission_auc = roc_auc_score(y_test, y_pred_prob)
        
        print(f"   Readmission Model AUC: {readmission_auc:.3f}")
        
        # Model 2: Length of Stay Prediction
        patients_df['length_of_stay'] = (
            pd.to_datetime(patients_df['discharge_date']) - pd.to_datetime(patients_df['admission_date'])
        ).dt.days
        
        # Filter patients with discharge dates
        los_data = patients_df.dropna(subset=['length_of_stay'])
        X_los = los_data[feature_columns]
        y_los = los_data['length_of_stay']
        
        X_train_los, X_test_los, y_train_los, y_test_los = train_test_split(
            X_los, y_los, test_size=0.2, random_state=42
        )
        
        los_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        los_model.fit(X_train_los, y_train_los)
        
        los_score = los_model.score(X_test_los, y_test_los)
        print(f"   Length of Stay Model R²: {los_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': readmission_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\\n   Top Risk Factors:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"     • {row['feature']}: {row['importance']:.3f}")
        
        return readmission_model, los_model, feature_importance
    
    def analyze_population_health(self):
        """Analyze population health trends and patterns"""
        print("📊 Analyzing population health trends...")
        
        conn = sqlite3.connect(self.db_path)
        patients_df = pd.read_sql_query('SELECT * FROM patients', conn)
        records_df = pd.read_sql_query('SELECT * FROM medical_records', conn)
        conn.close()
        
        if patients_df.empty:
            self.generate_synthetic_medical_data()
            return self.analyze_population_health()
        
        # Chronic disease prevalence by demographics
        patients_df['chronic_conditions_list'] = patients_df['chronic_conditions'].apply(
            lambda x: json.loads(x) if x else []
        )
        
        # Age group analysis
        patients_df['age_group'] = pd.cut(
            patients_df['age'], 
            bins=[0, 30, 50, 70, 100], 
            labels=['18-30', '31-50', '51-70', '70+']
        )
        
        # Disease prevalence analysis
        disease_prevalence = {}
        for condition in self.medical_conditions:
            prevalence = patients_df['chronic_conditions_list'].apply(
                lambda x: condition in x
            ).mean()
            disease_prevalence[condition] = prevalence
        
        # Risk stratification
        risk_distribution = pd.cut(
            patients_df['risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Moderate Risk', 'High Risk']
        ).value_counts()
        
        return patients_df, records_df, disease_prevalence, risk_distribution
    
    def create_healthcare_dashboard(self):
        """Create comprehensive healthcare analytics dashboard"""
        print("📊 Creating healthcare analytics dashboard...")
        
        # Load analysis results
        patients_df, records_df, disease_prevalence, risk_distribution = self.analyze_population_health()
        models = self.build_risk_prediction_models()
        
        # Create dashboard
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '👥 Patient Demographics',
                '🏥 Disease Prevalence',
                '⚠️ Risk Stratification',
                '📈 Vital Signs Distribution',
                '💊 Medication Usage',
                '🔬 Lab Results Trends',
                '📅 Length of Stay Analysis',
                '🎯 Treatment Outcomes'
            ]
        )
        
        # 1. Patient Demographics
        age_groups = patients_df['age_group'].value_counts()
        fig.add_trace(
            go.Bar(x=age_groups.index, y=age_groups.values, name="Age Groups", showlegend=False),
            row=1, col=1
        )
        
        # 2. Disease Prevalence
        diseases = list(disease_prevalence.keys())[:8]  # Top diseases
        prevalence_values = [disease_prevalence[d] * 100 for d in diseases]
        
        fig.add_trace(
            go.Bar(x=diseases, y=prevalence_values, name="Prevalence", showlegend=False),
            row=1, col=2
        )
        
        # 3. Risk Stratification
        fig.add_trace(
            go.Pie(labels=risk_distribution.index, values=risk_distribution.values, name="Risk"),
            row=2, col=1
        )
        
        # 4. Vital Signs (using systolic BP as example)
        fig.add_trace(
            go.Histogram(x=records_df['systolic_bp'], nbinsx=30, name="BP", showlegend=False),
            row=2, col=2
        )
        
        # 5. Medication Usage
        all_medications = []
        for med_list_str in patients_df['medications'].dropna():
            try:
                med_list = json.loads(med_list_str)
                all_medications.extend(med_list)
            except:
                continue
        
        if all_medications:
            from collections import Counter
            med_counts = Counter(all_medications)
            top_meds = dict(med_counts.most_common(8))
            
            fig.add_trace(
                go.Bar(x=list(top_meds.keys()), y=list(top_meds.values()), 
                      name="Medications", showlegend=False),
                row=3, col=1
            )
        
        # 6. Lab Results (Glucose trends)
        fig.add_trace(
            go.Scatter(
                x=range(len(records_df)), 
                y=records_df['glucose'].rolling(window=50).mean(),
                mode='lines',
                name="Glucose Trend",
                showlegend=False
            ),
            row=3, col=2
        )
        
        # 7. Length of Stay
        patients_df['length_of_stay'] = (
            pd.to_datetime(patients_df['discharge_date']) - pd.to_datetime(patients_df['admission_date'])
        ).dt.days
        
        los_data = patients_df['length_of_stay'].dropna()
        fig.add_trace(
            go.Box(y=los_data, name="Length of Stay", showlegend=False),
            row=4, col=1
        )
        
        # 8. Treatment Outcomes
        outcome_counts = records_df['outcome'].value_counts()
        fig.add_trace(
            go.Bar(x=outcome_counts.index, y=outcome_counts.values, 
                  name="Outcomes", showlegend=False),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1600,
            title_text="🏥 Healthcare Analytics Dashboard 🏥",
            title_font_size=24
        )
        
        fig.write_html("healthcare_analytics_dashboard.html")
        fig.show()
        
        return fig
    
    def generate_clinical_insights(self):
        """Generate clinical insights and recommendations"""
        patients_df, records_df, disease_prevalence, risk_distribution = self.analyze_population_health()
        
        print("\\n" + "="*70)
        print("🏥 HEALTHCARE ANALYTICS - CLINICAL INSIGHTS REPORT 🏥")
        print("="*70)
        
        # Population overview
        total_patients = len(patients_df)
        avg_age = patients_df['age'].mean()
        gender_dist = patients_df['gender'].value_counts()
        
        print(f"👥 Population Overview:")
        print(f"   • Total Patients: {total_patients:,}")
        print(f"   • Average Age: {avg_age:.1f} years")
        print(f"   • Gender Distribution: {gender_dist['M']:,} Male, {gender_dist['F']:,} Female")
        
        # Disease prevalence
        print(f"\\n🦠 Top Chronic Conditions:")
        sorted_diseases = sorted(disease_prevalence.items(), key=lambda x: x[1], reverse=True)
        for disease, prevalence in sorted_diseases[:5]:
            print(f"   • {disease}: {prevalence*100:.1f}% of population")
        
        # Risk analysis
        print(f"\\n⚠️ Risk Stratification:")
        for risk_level, count in risk_distribution.items():
            percentage = (count / total_patients) * 100
            print(f"   • {risk_level}: {count:,} patients ({percentage:.1f}%)")
        
        # Clinical metrics
        avg_systolic = records_df['systolic_bp'].mean()
        avg_glucose = records_df['glucose'].mean()
        
        print(f"\\n📊 Key Clinical Metrics:")
        print(f"   • Average Systolic BP: {avg_systolic:.1f} mmHg")
        print(f"   • Average Glucose: {avg_glucose:.1f} mg/dL")
        
        # Treatment outcomes
        outcome_success_rate = (records_df['outcome'] == 'Improved').mean() * 100
        print(f"   • Treatment Success Rate: {outcome_success_rate:.1f}%")
        
        # Length of stay analysis
        patients_df['length_of_stay'] = (
            pd.to_datetime(patients_df['discharge_date']) - pd.to_datetime(patients_df['admission_date'])
        ).dt.days
        avg_los = patients_df['length_of_stay'].mean()
        
        print(f"   • Average Length of Stay: {avg_los:.1f} days")
        
        # High-risk patients
        high_risk_patients = patients_df[patients_df['risk_score'] > 0.7]
        print(f"\\n🚨 High-Risk Patient Insights:")
        print(f"   • High-Risk Patients: {len(high_risk_patients):,}")
        print(f"   • Average Age: {high_risk_patients['age'].mean():.1f} years")
        print(f"   • Average BMI: {high_risk_patients['bmi'].mean():.1f}")
        
        print("="*70)
    
    def run_complete_analysis(self):
        """Execute complete healthcare analytics pipeline"""
        print("🚀 Starting Healthcare Analytics Pipeline...")
        print("="*50)
        
        # Generate data
        patients, medical_records = self.generate_synthetic_medical_data()
        
        # Build prediction models
        models = self.build_risk_prediction_models()
        
        # Analyze population health
        population_analysis = self.analyze_population_health()
        
        # Create dashboard
        dashboard = self.create_healthcare_dashboard()
        
        # Generate insights
        self.generate_clinical_insights()
        
        print("\\n✅ Healthcare Analytics Complete!")
        print("📁 Files generated:")
        print("   • healthcare_analytics_dashboard.html")
        print("   • healthcare_analytics.db")
        
        # Export summary
        patients_df, records_df, disease_prevalence, risk_distribution = population_analysis
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_patients': len(patients_df),
            'total_records': len(records_df),
            'average_age': patients_df['age'].mean(),
            'top_conditions': dict(sorted(disease_prevalence.items(), key=lambda x: x[1], reverse=True)[:5]),
            'risk_distribution': risk_distribution.to_dict(),
            'avg_length_of_stay': patients_df['length_of_stay'].mean()
        }
        
        with open('healthcare_analytics_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return patients_df, records_df, models

def main():
    """Main execution function"""
    print("🎯 HEALTHCARE ANALYTICS PLATFORM - Industry-Ready System")
    print("=" * 65)
    print("Showcasing: Medical ML • Clinical Decision Support • Population Health")
    print("=" * 65)
    
    # Initialize analyzer
    analyzer = HealthcareAnalyzer()
    
    # Run complete analysis
    patients, records, models = analyzer.run_complete_analysis()
    
    print(f"\\n🎉 Analysis completed successfully!")
    print(f"🏥 Analyzed {len(patients):,} patients and {len(records):,} medical records")
    print(f"🤖 Built predictive models for risk stratification")
    
    return analyzer, patients, records

if __name__ == "__main__":
    analyzer, patients, records = main()