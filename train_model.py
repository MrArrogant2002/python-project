"""
NYC Taxi Fare Prediction - Enhanced Model Training Pipeline
============================================================
This script trains an enhanced XGBoost model for predicting NYC taxi fares
with advanced feature engineering including rush hour detection, weekend pricing,
and landmark-based features.

Author: Development Team
Date: December 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_FRACTION = 0.1  # Use 10% of data for faster training
RANDOM_STATE = 42
TRAIN_DATA_PATH = 'train.csv'
MODEL_OUTPUT_PATH = 'taxi_fare_enhanced_model.pkl'
FEATURE_IMPORTANCE_PLOT = 'enhanced_feature_importance.png'

# Data types for efficient loading
DTYPES = {
    'fare_amount': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'dropoff_latitude': 'float32',
    'passenger_count': 'uint8'
}

SELECTED_COLS = [
    'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'passenger_count'
]

# ============================================================================
# CUSTOM TRANSFORMER CLASSES
# ============================================================================

class EnhancedDatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract datetime features including rush hour and peak time indicators"""
    
    def __init__(self, datetime_col='pickup_datetime'):
        self.datetime_col = datetime_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Original datetime features
        X[f'{self.datetime_col}_year'] = X[self.datetime_col].dt.year
        X[f'{self.datetime_col}_month'] = X[self.datetime_col].dt.month
        X[f'{self.datetime_col}_day'] = X[self.datetime_col].dt.day
        X[f'{self.datetime_col}_weekday'] = X[self.datetime_col].dt.weekday
        X[f'{self.datetime_col}_hour'] = X[self.datetime_col].dt.hour
        
        # Rush hour indicators
        X['is_morning_rush'] = ((X[f'{self.datetime_col}_hour'] >= 7) & 
                                (X[f'{self.datetime_col}_hour'] <= 9) & 
                                (X[f'{self.datetime_col}_weekday'] < 5)).astype(int)
        
        X['is_evening_rush'] = ((X[f'{self.datetime_col}_hour'] >= 17) & 
                                (X[f'{self.datetime_col}_hour'] <= 19) & 
                                (X[f'{self.datetime_col}_weekday'] < 5)).astype(int)
        
        # Weekend indicator
        X['is_weekend'] = (X[f'{self.datetime_col}_weekday'] >= 5).astype(int)
        
        # Late night indicator (higher rates)
        X['is_late_night'] = ((X[f'{self.datetime_col}_hour'] >= 23) | 
                              (X[f'{self.datetime_col}_hour'] <= 5)).astype(int)
        
        # Business hours (9 AM - 5 PM weekdays)
        X['is_business_hours'] = ((X[f'{self.datetime_col}_hour'] >= 9) & 
                                  (X[f'{self.datetime_col}_hour'] <= 17) & 
                                  (X[f'{self.datetime_col}_weekday'] < 5)).astype(int)
        
        return X


class DistanceCalculator(BaseEstimator, TransformerMixin):
    """Calculate haversine distance between pickup and dropoff locations"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['trip_distance'] = self._haversine_distance(
            X['pickup_longitude'], X['pickup_latitude'],
            X['dropoff_longitude'], X['dropoff_latitude']
        )
        return X
    
    @staticmethod
    def _haversine_distance(lon1, lat1, lon2, lat2):
        """Calculate great circle distance in kilometers"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        
        return km


class LandmarkDistanceCalculator(BaseEstimator, TransformerMixin):
    """Calculate distances from major NYC landmarks"""
    
    def __init__(self):
        # NYC landmark coordinates (longitude, latitude)
        self.landmarks = {
            'jfk': (-73.7781, 40.6413),
            'lga': (-73.8740, 40.7769),
            'ewr': (-74.1745, 40.6895),
            'met': (-73.9632, 40.7794),
            'wtc': (-74.0099, 40.7126)
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for name, (lon, lat) in self.landmarks.items():
            # Distance from pickup location
            X[f'{name}_pickup_distance'] = self._haversine_distance(
                lon, lat, X['pickup_longitude'], X['pickup_latitude']
            )
            # Distance from dropoff location
            X[f'{name}_drop_distance'] = self._haversine_distance(
                lon, lat, X['dropoff_longitude'], X['dropoff_latitude']
            )
        
        return X
    
    @staticmethod
    def _haversine_distance(lon1, lat1, lon2, lat2):
        """Calculate great circle distance in kilometers"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        
        return km


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers and invalid data based on reasonable ranges"""
    
    def __init__(self, 
                 fare_range=(1, 500),
                 lon_range=(-75, -72),
                 lat_range=(40, 42),
                 passenger_range=(1, 6)):
        self.fare_range = fare_range
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.passenger_range = passenger_range
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Only apply fare filter if fare_amount column exists (training data)
        if 'fare_amount' in X.columns:
            mask = (
                (X['fare_amount'] >= self.fare_range[0]) &
                (X['fare_amount'] <= self.fare_range[1]) &
                (X['pickup_longitude'] >= self.lon_range[0]) & 
                (X['pickup_longitude'] <= self.lon_range[1]) & 
                (X['dropoff_longitude'] >= self.lon_range[0]) & 
                (X['dropoff_longitude'] <= self.lon_range[1]) & 
                (X['pickup_latitude'] >= self.lat_range[0]) & 
                (X['pickup_latitude'] <= self.lat_range[1]) & 
                (X['dropoff_latitude'] >= self.lat_range[0]) & 
                (X['dropoff_latitude'] <= self.lat_range[1]) & 
                (X['passenger_count'] >= self.passenger_range[0]) & 
                (X['passenger_count'] <= self.passenger_range[1])
            )
        else:
            # For test data without fare_amount
            mask = (
                (X['pickup_longitude'] >= self.lon_range[0]) & 
                (X['pickup_longitude'] <= self.lon_range[1]) & 
                (X['dropoff_longitude'] >= self.lon_range[0]) & 
                (X['dropoff_longitude'] <= self.lon_range[1]) & 
                (X['pickup_latitude'] >= self.lat_range[0]) & 
                (X['pickup_latitude'] <= self.lat_range[1]) & 
                (X['dropoff_latitude'] >= self.lat_range[0]) & 
                (X['dropoff_latitude'] <= self.lat_range[1]) & 
                (X['passenger_count'] >= self.passenger_range[0]) & 
                (X['passenger_count'] <= self.passenger_range[1])
            )
        
        return X[mask]


class EnhancedFeatureSelector(BaseEstimator, TransformerMixin):
    """Select features including new enhanced time-based features"""
    
    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns
    
    def fit(self, X, y=None):
        if self.feature_columns is None:
            # Define feature columns including new ones
            self.feature_columns = [
                'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
                'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day',
                'pickup_datetime_weekday', 'pickup_datetime_hour', 'trip_distance',
                'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
                'met_drop_distance', 'wtc_drop_distance', 'jfk_pickup_distance',
                'lga_pickup_distance', 'ewr_pickup_distance', 'met_pickup_distance',
                'wtc_pickup_distance',
                # Enhanced features
                'is_morning_rush', 'is_evening_rush', 'is_weekend',
                'is_late_night', 'is_business_hours'
            ]
        return self
    
    def transform(self, X):
        return X[self.feature_columns]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def skip_row(row_idx):
    """Skip rows randomly to sample data"""
    if row_idx == 0:
        return False
    return random.random() > SAMPLE_FRACTION


def load_data(filepath, sample_fraction=SAMPLE_FRACTION):
    """Load and sample training data"""
    print(f"\n{'='*70}")
    print(f"Loading {sample_fraction*100}% sample of training data...")
    print(f"{'='*70}")
    
    random.seed(RANDOM_STATE)
    
    df = pd.read_csv(
        filepath,
        usecols=SELECTED_COLS,
        dtype=DTYPES,
        parse_dates=['pickup_datetime'],
        skiprows=skip_row
    )
    
    print(f"✓ Loaded {len(df):,} rows")
    print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    
    return df


def split_data(df, test_size=0.2):
    """Split data into train and validation sets"""
    print(f"\n{'='*70}")
    print("Splitting data into train and validation sets...")
    print(f"{'='*70}")
    
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
    
    print(f"✓ Training set: {len(train_df):,} rows")
    print(f"✓ Validation set: {len(val_df):,} rows")
    
    # Remove any missing values
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    
    print(f"\nAfter dropping NAs:")
    print(f"✓ Training set: {len(train_df):,} rows")
    print(f"✓ Validation set: {len(val_df):,} rows")
    
    return train_df, val_df


def create_feature_pipeline():
    """Create the enhanced feature engineering pipeline"""
    print(f"\n{'='*70}")
    print("Creating Enhanced Feature Engineering Pipeline...")
    print(f"{'='*70}")
    
    pipeline = Pipeline([
        ('datetime_features', EnhancedDatetimeFeatureExtractor()),
        ('trip_distance', DistanceCalculator()),
        ('landmark_distances', LandmarkDistanceCalculator()),
        ('outlier_removal', OutlierRemover()),
        ('feature_selection', EnhancedFeatureSelector())
    ])
    
    print("Pipeline steps:")
    for step_name, step in pipeline.steps:
        print(f"  ✓ {step_name}: {step.__class__.__name__}")
    
    return pipeline


def apply_feature_engineering(pipeline, train_df, val_df):
    """Apply feature engineering to train and validation data"""
    print(f"\n{'='*70}")
    print("Applying Feature Engineering...")
    print(f"{'='*70}")
    
    # Separate features and target
    y_train = train_df['fare_amount'].values
    y_val = val_df['fare_amount'].values
    
    print("✓ Transforming training data...")
    X_train = pipeline.fit_transform(train_df)
    
    print("✓ Transforming validation data...")
    X_val = pipeline.transform(val_df)
    
    # Update targets after outlier removal
    y_train = train_df.loc[X_train.index, 'fare_amount'].values
    y_val = val_df.loc[X_val.index, 'fare_amount'].values
    
    print(f"\n✓ Training features shape: {X_train.shape}")
    print(f"✓ Validation features shape: {X_val.shape}")
    print(f"✓ Total features: {len(X_train.columns)}")
    
    # Display new enhanced features
    new_features = ['is_morning_rush', 'is_evening_rush', 'is_weekend', 'is_late_night', 'is_business_hours']
    print(f"\nEnhanced features distribution:")
    for feat in new_features:
        count = X_train[feat].sum()
        pct = (count / len(X_train)) * 100
        print(f"  • {feat:20s}: {count:6,} trips ({pct:5.2f}%)")
    
    return X_train, X_val, y_train, y_val


def train_model(X_train, y_train):
    """Train the enhanced XGBoost model"""
    print(f"\n{'='*70}")
    print("Training Enhanced XGBoost Model...")
    print(f"{'='*70}")
    
    model = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        n_estimators=600,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    
    print("\nModel Hyperparameters:")
    for key, value in model.get_params().items():
        if key in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                   'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']:
            print(f"  • {key:20s}: {value}")
    
    print("\n✓ Training in progress...")
    model.fit(X_train, y_train)
    print("✓ Training complete!")
    
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Comprehensive model evaluation"""
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE EVALUATION")
    print(f"{'='*70}")
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    train_mae = mean_absolute_error(y_train, train_preds)
    val_mae = mean_absolute_error(y_val, val_preds)
    
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    print(f"\nRMSE (Root Mean Squared Error):")
    print(f"  Training:   ${train_rmse:.4f}")
    print(f"  Validation: ${val_rmse:.4f}")
    print(f"  Difference: ${abs(train_rmse - val_rmse):.4f}")
    
    print(f"\nMAE (Mean Absolute Error):")
    print(f"  Training:   ${train_mae:.4f}")
    print(f"  Validation: ${val_mae:.4f}")
    
    print(f"\nR² Score:")
    print(f"  Training:   {train_r2:.4f} ({train_r2*100:.2f}%)")
    print(f"  Validation: {val_r2:.4f} ({val_r2*100:.2f}%)")
    
    print(f"\n{'='*70}")
    
    return {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_r2': train_r2,
        'val_r2': val_r2
    }


def analyze_feature_importance(model, feature_names, top_n=20):
    """Analyze and visualize feature importance"""
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}")
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    for idx, row in feature_importance.head(top_n).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.6f}")
    
    # Check importance of enhanced features
    print(f"\n{'='*70}")
    print("IMPORTANCE OF ENHANCED FEATURES:")
    print(f"{'='*70}")
    new_features = ['is_morning_rush', 'is_evening_rush', 'is_weekend', 'is_late_night', 'is_business_hours']
    for feat in new_features:
        feat_data = feature_importance[feature_importance['feature'] == feat]
        if not feat_data.empty:
            importance = feat_data['importance'].values[0]
            rank = feat_data.index[0] + 1
            print(f"  {feat:25s}: {importance:.6f} (Rank: {rank}/{len(feature_names)})")
    
    # Plot feature importance
    print(f"\n✓ Creating feature importance plot...")
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances - Enhanced Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved as '{FEATURE_IMPORTANCE_PLOT}'")
    
    return feature_importance


def save_model(pipeline, model, metrics, feature_columns):
    """Save the complete model pipeline"""
    print(f"\n{'='*70}")
    print("Saving Enhanced Model...")
    print(f"{'='*70}")
    
    model_data = {
        'feature_pipeline': pipeline,
        'model': model,
        'feature_columns': feature_columns,
        'train_rmse': metrics['train_rmse'],
        'val_rmse': metrics['val_rmse'],
        'train_mae': metrics['train_mae'],
        'val_mae': metrics['val_mae'],
        'train_r2': metrics['train_r2'],
        'val_r2': metrics['val_r2'],
        'model_version': '2.0_enhanced',
        'enhancements': [
            'Rush hour indicators (morning/evening)',
            'Weekend vs weekday detection',
            'Late night surcharge indicator',
            'Business hours detection',
            'Enhanced regularization',
            'Optimized hyperparameters'
        ]
    }
    
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    import os
    file_size = os.path.getsize(MODEL_OUTPUT_PATH) / (1024 * 1024)
    
    print(f"✓ Model saved as '{MODEL_OUTPUT_PATH}'")
    print(f"✓ File size: {file_size:.2f} MB")
    
    print("\nModel Enhancements:")
    for enhancement in model_data['enhancements']:
        print(f"  ✓ {enhancement}")


def test_predictions(pipeline, model, val_df, n_samples=10):
    """Test model on sample predictions"""
    print(f"\n{'='*70}")
    print("TESTING SAMPLE PREDICTIONS")
    print(f"{'='*70}")
    
    sample_data = val_df.head(n_samples).copy()
    
    # Make predictions
    sample_features = pipeline.transform(sample_data)
    predictions = model.predict(sample_features)
    
    # Display results
    results = pd.DataFrame({
        'Actual': sample_data['fare_amount'].values[:len(predictions)],
        'Predicted': predictions,
        'Difference': sample_data['fare_amount'].values[:len(predictions)] - predictions,
        'Error %': (abs(sample_data['fare_amount'].values[:len(predictions)] - predictions) / 
                    sample_data['fare_amount'].values[:len(predictions)] * 100),
        'Hour': sample_data['pickup_datetime'].dt.hour.values[:len(predictions)],
        'Day': sample_data['pickup_datetime'].dt.day_name().values[:len(predictions)]
    })
    
    print(f"\nSample Predictions (n={len(results)}):")
    print("-" * 80)
    for idx, row in results.iterrows():
        print(f"  Actual: ${row['Actual']:6.2f} | Predicted: ${row['Predicted']:6.2f} | "
              f"Diff: ${row['Difference']:6.2f} | Error: {row['Error %']:5.1f}% | "
              f"{row['Day'][:3]} {row['Hour']:02d}:00")
    print("-" * 80)
    print(f"\nAverage Error: ${results['Difference'].abs().mean():.2f}")
    print(f"Average Error %: {results['Error %'].mean():.1f}%")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline execution"""
    print("\n" + "="*70)
    print("NYC TAXI FARE PREDICTION - ENHANCED MODEL TRAINING")
    print("="*70)
    print(f"Version: 2.0")
    print(f"Date: December 2025")
    print("="*70)
    
    # Step 1: Load data
    df = load_data(TRAIN_DATA_PATH)
    
    # Step 2: Split data
    train_df, val_df = split_data(df)
    
    # Step 3: Create feature pipeline
    pipeline = create_feature_pipeline()
    
    # Step 4: Apply feature engineering
    X_train, X_val, y_train, y_val = apply_feature_engineering(pipeline, train_df, val_df)
    
    # Step 5: Train model
    model = train_model(X_train, y_train)
    
    # Step 6: Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
    
    # Step 7: Analyze feature importance
    feature_importance = analyze_feature_importance(model, X_train.columns.tolist())
    
    # Step 8: Test predictions
    test_predictions(pipeline, model, val_df)
    
    # Step 9: Save model
    save_model(pipeline, model, metrics, X_train.columns.tolist())
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✓ Model Version: 2.0 Enhanced")
    print(f"✓ Validation RMSE: ${metrics['val_rmse']:.2f}")
    print(f"✓ Validation R²: {metrics['val_r2']:.4f} ({metrics['val_r2']*100:.2f}%)")
    print(f"✓ Total Features: {len(X_train.columns)}")
    print(f"✓ Training Samples: {len(X_train):,}")
    print(f"✓ Validation Samples: {len(X_val):,}")
    
    print(f"\n✓ Model saved: {MODEL_OUTPUT_PATH}")
    print(f"✓ Feature plot: {FEATURE_IMPORTANCE_PLOT}")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Deploy model to Streamlit app")
    print("  2. Test with real-world data")
    print("  3. Monitor performance metrics")
    print("  4. Consider additional enhancements")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
