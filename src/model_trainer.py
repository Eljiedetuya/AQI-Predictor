import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class HybridAQIModel:
    def __init__(self, csv_path):
        try:
            self.data = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
        
        # Validate required columns
        required_cols = ['AQI', 'Month', 'Year', 'city_name', 'components.co', 'components.no', 
                        'components.no2', 'components.o3', 'components.so2', 
                        'components.pm2_5', 'components.pm10', 'components.nh3']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.prepare_features()
        self.train_model()

    def prepare_features(self):
        # Group by city to prevent data leakage between cities
        self.data['lag_1'] = self.data.groupby('city_name')['AQI'].shift(1)
        self.data['lag_3_avg'] = self.data.groupby('city_name')['AQI'].rolling(3).mean().reset_index(drop=True)
        
        # Cyclic encoding for month
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['Month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['Month'] / 12)
        
        # Year-based features (normalize year to 0-1 range for better model performance)
        # Assuming year range 2020-2030 for normalization
        self.data['year_normalized'] = (self.data['Year'] - 2020) / 10.0
        self.data['year_sin'] = np.sin(2 * np.pi * self.data['year_normalized'])
        self.data['year_cos'] = np.cos(2 * np.pi * self.data['year_normalized'])
        
        # Fill NaN values using forward fill then backward fill
        for col in ['lag_1', 'lag_3_avg']:
            self.data[col] = self.data[col].ffill().bfill()
        
        # Drop only if there are still NaN values after filling
        self.data = self.data.dropna(subset=['lag_1', 'lag_3_avg', 'month_sin', 'month_cos', 'AQI'])

        self.features = [
            'components.co','components.no','components.no2',
            'components.o3','components.so2','components.pm2_5',
            'components.pm10','components.nh3',
            'Month','Year','lag_1','lag_3_avg','month_sin','month_cos',
            'year_normalized','year_sin','year_cos'
        ]
        
        # Add polynomial feature interactions
        self.X_base = self.data[self.features].copy()
        self.X_base['pm_interaction'] = self.X_base['components.pm2_5'] * self.X_base['components.pm10']
        self.X_base['nox_interaction'] = self.X_base['components.no'] * self.X_base['components.no2']
        self.X_base['ozone_no2_ratio'] = (self.X_base['components.o3'] + 1e-6) / (self.X_base['components.no2'] + 1e-6)

        self.X = self.X_base
        self.y = self.data['AQI']
        
        # Store all feature names including interactions
        self.features = list(self.X.columns)

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )

        # 1. Train Random Forest with hyperparameter tuning
        print("  • Tuning Random Forest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10]
        }
        rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        self.rf_model = rf_grid.best_estimator_
        print(f"    Best RF params: {rf_grid.best_params_}")

        # 2. Train Gradient Boosting with hyperparameter tuning
        print("  • Tuning Gradient Boosting...")
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, n_jobs=-1)
        gb_grid.fit(X_train, y_train)
        self.gb_model = gb_grid.best_estimator_
        print(f"    Best GB params: {gb_grid.best_params_}")

        # 3. Train XGBoost with hyperparameter tuning
        print("  • Tuning XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        xgb_grid = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), xgb_params, cv=3, n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        self.xgb_model = xgb_grid.best_estimator_
        print(f"    Best XGB params: {xgb_grid.best_params_}")

        # Make predictions and evaluate all models
        self.models = {
            'Random Forest': self.rf_model,
            'Gradient Boosting': self.gb_model,
            'XGBoost': self.xgb_model
        }
        
        self.results = {}
        for name, model in self.models.items():
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            cv_scores = cross_val_score(model, self.X_scaled, self.y, cv=5, scoring='r2')
            
            self.results[name] = {
                'mae': mae, 'rmse': rmse, 'r2': r2,
                'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
            }
        
        # Hybrid ensemble of all three models
        rf_pred = self.rf_model.predict(X_test)
        gb_pred = self.gb_model.predict(X_test)
        xgb_pred = self.xgb_model.predict(X_test)
        self.hybrid_pred = (rf_pred + gb_pred + xgb_pred) / 3
        
        self.hybrid_mae = mean_absolute_error(y_test, self.hybrid_pred)
        self.hybrid_rmse = np.sqrt(mean_squared_error(y_test, self.hybrid_pred))
        self.hybrid_r2 = r2_score(y_test, self.hybrid_pred)
        hybrid_cv = cross_val_score(self.rf_model, self.X_scaled, self.y, cv=5, scoring='r2')
        self.hybrid_cv_mean = hybrid_cv.mean()
        self.hybrid_cv_std = hybrid_cv.std()

    def predict(self, input_df):
        scaled = self.scaler.transform(input_df)
        rf = self.rf_model.predict(scaled)
        gb = self.gb_model.predict(scaled)
        xgb = self.xgb_model.predict(scaled)
        return (rf + gb + xgb) / 3
    
    def save_model(self, model_dir='models'):
        """Save trained models and scaler for later use"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(parent_dir, model_dir)
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.rf_model, os.path.join(model_path, 'rf_model.pkl'))
        joblib.dump(self.gb_model, os.path.join(model_path, 'gb_model.pkl'))
        joblib.dump(self.xgb_model, os.path.join(model_path, 'xgb_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))
        joblib.dump(self.features, os.path.join(model_path, 'features.pkl'))
        print(f"  [OK] Models saved to {model_path}")
    
    @staticmethod
    def load_model(model_dir='models'):
        """Load pre-trained models"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(parent_dir, model_dir)
        
        loaded_model = HybridAQIModel.__new__(HybridAQIModel)
        loaded_model.rf_model = joblib.load(os.path.join(model_path, 'rf_model.pkl'))
        loaded_model.gb_model = joblib.load(os.path.join(model_path, 'gb_model.pkl'))
        loaded_model.xgb_model = joblib.load(os.path.join(model_path, 'xgb_model.pkl'))
        loaded_model.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        loaded_model.features = joblib.load(os.path.join(model_path, 'features.pkl'))
        print(f"  [OK] Models loaded from {model_path}")
        return loaded_model

if __name__ == '__main__':
    print("Initializing Hybrid AQI Model with Advanced Features...")
    print("=" * 70)
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(parent_dir, 'data', '20242Monthlydata_modified.csv')
    model = HybridAQIModel(csv_path)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    # Display individual model results
    for model_name, metrics in model.results.items():
        print(f"\n{model_name}:")
        print(f"  MAE:           {metrics['mae']:.4f}")
        print(f"  RMSE:          {metrics['rmse']:.4f}")
        print(f"  R² Score:      {metrics['r2']:.4f}")
        print(f"  CV R² (5-fold): {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    
    # Display ensemble results
    print(f"\n{'Ensemble (Hybrid 3-Model Average):'}")
    print(f"  MAE:           {model.hybrid_mae:.4f}")
    print(f"  RMSE:          {model.hybrid_rmse:.4f}")
    print(f"  R² Score:      {model.hybrid_r2:.4f}")
    print(f"  CV R² (5-fold): {model.hybrid_cv_mean:.4f} ± {model.hybrid_cv_std:.4f}")
    
    # Save the trained model
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    model.save_model()
    
    # Make predictions on new data
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS ON NEW DATA")
    print("=" * 70)
    
    # Get sample data
    sample_data = model.X.head(5).copy()
    print(f"\nSample input data (first 5 rows):")
    print(sample_data.head())
    
    # Make predictions
    predictions = model.predict(sample_data)
    print(f"\n✓ Predictions made: {predictions[:5]}")
    print(f"  Mean predicted AQI: {predictions.mean():.2f}")
    print(f"  Min predicted AQI:  {predictions.min():.2f}")
    print(f"  Max predicted AQI:  {predictions.max():.2f}")
    
    print("\n" + "=" * 70)
    print("✓ Model training & saving complete!")
    print("=" * 70)
