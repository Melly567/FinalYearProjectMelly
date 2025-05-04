import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Create the ml_models directory if it doesn't exist
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')

def train_model():
    """Train a model to predict online shopping intentions."""
    print("Starting model training...")
    
    try:
        # Load data
        print("Loading the dataset...")
        # Try to load from a file in the current directory
        try:
            data = pd.read_csv('online_shoppers_intention.csv')
            print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
        except FileNotFoundError:
            print("Dataset file not found. Trying to load from different location...")
            # Try a few common alternative paths
            alternative_paths = [
                'data/online_shoppers_intention.csv',
                '../data/online_shoppers_intention.csv',
                'static/data/online_shoppers_intention.csv'
            ]
            
            for path in alternative_paths:
                try:
                    data = pd.read_csv(path)
                    print(f"Loaded dataset from {path}")
                    break
                except FileNotFoundError:
                    continue
            else:
                print("Unable to find dataset file. Please ensure it exists.")
                return None
        
        # Display basic information about the data
        print("\nData Overview:")
        print(f"Shape: {data.shape}")
        print("\nFeature Types:")
        print(data.dtypes)
        
        print("\nChecking for missing values...")
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found: {missing_values[missing_values > 0]}")
            # Handle missing values
            data = data.dropna()
            print(f"Data shape after handling missing values: {data.shape}")
        else:
            print("No missing values found.")
        
        # Data preprocessing
        print("\nPreprocessing data...")
        
        # Split features and target
        X = data.drop('Revenue', axis=1)
        y = data['Revenue']
        
        # Identify categorical and numerical features
        categorical_features = ['Month', 'VisitorType', 'Weekend']
        numerical_features = [col for col in X.columns if col not in categorical_features]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, 
                                               min_samples_split=10, random_state=42))
        ])
        
        # Train the model
        print("\nTraining the model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        print("\nEvaluating the model...")
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC ROC: {auc:.4f}")
        
        # Feature importance analysis
        print("\nFeature importance analysis...")
        feature_names = numerical_features + list(
            pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
        
        # Get feature importances
        importances = pipeline.named_steps['classifier'].feature_importances_
        sorted_idx = np.argsort(importances)
        
        try:
            # Create feature importance plot
            plt.figure(figsize=(10, 12))
            plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'))
            print(f"Feature importance plot saved to {os.path.join(MODEL_DIR, 'feature_importance.png')}")
        except Exception as e:
            print(f"Could not create feature importance plot: {e}")
        
        # Save the model
        print(f"\nSaving model to {MODEL_PATH}")
        joblib.dump(pipeline, MODEL_PATH)
        
        # Try to save metrics to database if Django is set up
        try:
            import django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
            django.setup()
            
            from myapp.models import ModelMetrics
            metrics = ModelMetrics(
                name='RandomForest',
                auc_score=auc,
                precision_score=precision,
                recall_score=recall,
                f1_score=f1
            )
            metrics.save()
            print("Metrics saved to database successfully.")
        except Exception as e:
            print(f"Note: Could not save metrics to database: {e}")
            print("This is normal if you're running the script standalone without Django.")
            
            # Save metrics to a csv file as backup
            metrics_df = pd.DataFrame({
                'name': ['RandomForest'],
                'accuracy': [accuracy],
                'auc_score': [auc],
                'precision_score': [precision], 
                'recall_score': [recall],
                'f1_score': [f1],
            })
            metrics_df.to_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'), index=False)
            print(f"Metrics saved to {os.path.join(MODEL_DIR, 'model_metrics.csv')}")
        
        print("\nModel training and evaluation completed successfully!")
        return pipeline
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_data(data):
    """Create visualizations for the dataset."""
    try:
        # Create a directory for visualizations
        viz_dir = os.path.join(MODEL_DIR, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a few visualizations
        
        # 1. Revenue Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Revenue', data=data)
        plt.title('Revenue Distribution')
        plt.savefig(os.path.join(viz_dir, 'revenue_distribution.png'))
        
        # 2. Weekend vs Revenue
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Weekend', hue='Revenue', data=data)
        plt.title('Weekend vs Revenue')
        plt.savefig(os.path.join(viz_dir, 'weekend_revenue.png'))
        
        # 3. Month vs Revenue
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Month', hue='Revenue', data=data)
        plt.title('Month vs Revenue')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(viz_dir, 'month_revenue.png'))
        
        # 4. Visitor Type vs Revenue
        plt.figure(figsize=(10, 5))
        sns.countplot(x='VisitorType', hue='Revenue', data=data)
        plt.title('Visitor Type vs Revenue')
        plt.savefig(os.path.join(viz_dir, 'visitor_type_revenue.png'))
        
        # 5. Correlation Matrix
        plt.figure(figsize=(14, 10))
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'correlation_matrix.png'))
        
        print(f"Visualizations saved to {viz_dir}")
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    try:
        # Try to load the data
        data_path = 'online_shoppers_intention.csv'
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
            
            # Create visualizations
            visualize_data(data)
            
        else:
            print(f"Dataset file not found at {data_path}. Continuing with model training...")
        
        # Train the model
        model = train_model()
        
        if model:
            print("Model training completed successfully!")
        else:
            print("Model training failed. Please check the error messages.")
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        