import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor  # Added import
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import joblib
import traceback

# Configuration
FILES = [
    'ML-HYDPARK_v0.0.5_cleaned.csv',
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODELS = {
    'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
    'SVR': SVR(),
    'Linear Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(n_neighbors=5)  # Added KNN
}
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting functions
def plot_spearman_correlation(data, file_name, output_dir):
    """Plot Spearman correlation between HtoM and Hydrogen_Weight_Percent"""
    try:
        if 'HtoM' not in data.columns or 'Hydrogen_Weight_Percent' not in data.columns:
            print("Required columns not present for Spearman correlation plot")
            return
            
        plt.figure(figsize=(8, 6))
        
        # Calculate Spearman correlation
        corr = data['HtoM'].corr(data['Hydrogen_Weight_Percent'], method='spearman')
        
        # Create scatter plot with regression line
        sns.regplot(
            x='HtoM', 
            y='Hydrogen_Weight_Percent', 
            data=data, 
            scatter_kws={'alpha': 0.6},
            line_kws={'color': 'red'}
        )
        
        plt.title(f"Spearman Correlation: {corr:.2f}\nHtoM vs Hydrogen_Weight_Percent - {file_name}")
        plt.xlabel('HtoM Ratio')
        plt.ylabel('Hydrogen Weight Percent')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"spearman_correlation_{file_name.replace('.csv', '')}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved Spearman correlation plot to {output_path}")
    except Exception as e:
        print(f"Error in plot_spearman_correlation: {str(e)}")
        traceback.print_exc()

def plot_pearson_correlation_matrix(data, file_name, output_dir):
    """Plot Pearson correlation matrix for all numerical features"""
    try:
        # Select only numerical columns
        numerical_data = data.select_dtypes(include=['int64', 'float64'])
        
        if len(numerical_data.columns) < 2:
            print("Not enough numerical features for correlation matrix")
            return
            
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr(method='pearson')
        
        # Create heatmap - REMOVED THE MASK TO SHOW FULL MATRIX
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1,
            square=True
        )
        plt.title(f"Pearson Correlation Matrix - {file_name}")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"pearson_correlation_matrix_{file_name.replace('.csv', '')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Pearson correlation matrix to {output_path}")
        
        # Also save correlation values to CSV
        csv_path = os.path.join(output_dir, f"pearson_correlation_values_{file_name.replace('.csv', '')}.csv")
        corr_matrix.to_csv(csv_path)
        print(f"Saved Pearson correlation values to {csv_path}")
    except Exception as e:
        print(f"Error in plot_pearson_correlation_matrix: {str(e)}")
        traceback.print_exc()

def plot_model_comparison(file_results, output_dir):
    """Plot model comparison for a single file"""
    try:
        metrics = []
        for model_name, model_metrics in file_results['models'].items():
            metrics.append({
                'Model': model_name,
                'MAE': model_metrics['mae'],
                'RMSE': model_metrics['rmse'],
                'R²': model_metrics['r2']
            })
        
        df = pd.DataFrame(metrics)
        plt.figure(figsize=(10, 6))
        df.set_index('Model').plot(kind='bar', rot=45)
        plt.title(f"Model Comparison - {file_results['file']}")
        plt.ylabel('Score')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"model_comparison_{file_results['file'].replace('.csv', '')}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved model comparison plot to {output_path}")
    except Exception as e:
        print(f"Error in plot_model_comparison: {str(e)}")
        traceback.print_exc()

def plot_residuals(file_results, output_dir):
    """Plot residuals for all models in a file"""
    try:
        plt.figure(figsize=(10, 8))
        for model_name, metrics in file_results['models'].items():
            sns.kdeplot(metrics['residuals'], label=model_name)
        
        plt.axvline(x=0, color='k', linestyle='--')
        plt.title(f"Residuals Distribution - {file_results['file']}")
        plt.xlabel('Residuals (Actual - Predicted)')
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"residuals_{file_results['file'].replace('.csv', '')}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved residuals plot to {output_path}")
    except Exception as e:
        print(f"Error in plot_residuals: {str(e)}")
        traceback.print_exc()

def plot_predictions(file_results, output_dir):
    """Plot actual vs predicted for all models in a file"""
    try:
        plt.figure(figsize=(10, 8))
        
        # Calculate overall min/max first
        overall_min = min(min(metrics['y_test'].min(), metrics['y_pred'].min()) 
                        for metrics in file_results['models'].values())
        overall_max = max(max(metrics['y_test'].max(), metrics['y_pred'].max()) 
                        for metrics in file_results['models'].values())
        
        # Plot each model
        for model_name, metrics in file_results['models'].items():
            sns.scatterplot(
                x=metrics['y_test'], 
                y=metrics['y_pred'], 
                alpha=0.6, 
                label=f"{model_name} (R²={metrics['r2']:.2f})"
            )
        
        # Reference line
        plt.plot([overall_min, overall_max], [overall_min, overall_max], 'k--')
        plt.xlabel('Actual Hydrogen Weight Percent')
        plt.ylabel('Predicted Hydrogen Weight Percent')
        plt.title(f"Actual vs Predicted - {file_results['file']}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"predictions_{file_results['file'].replace('.csv', '')}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved predictions plot to {output_path}")
    except Exception as e:
        print(f"Error in plot_predictions: {str(e)}")
        traceback.print_exc()

def plot_feature_importance(file_results, output_dir):
    """Plot feature importance for tree-based models"""
    try:
        for model_name, metrics in file_results['models'].items():
            if model_name in ['Random Forest', 'Gradient Boosting']:
                try:
                    preprocessor = metrics['model'].named_steps['preprocessor']
                    model = metrics['model'].named_steps['regressor']
                    
                    # Get feature names
                    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
                    all_features = np.concatenate([num_features, cat_features])
                    
                    # Get importances
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    else:
                        continue
                    
                    # Create importance DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(20)
                    
                    # Plot
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.title(f"Feature Importance - {model_name} - {file_results['file']}")
                    plt.tight_layout()
                    
                    output_path = os.path.join(
                        output_dir, 
                        f"feature_importance_{model_name.replace(' ', '_')}_{file_results['file'].replace('.csv', '')}.png"
                    )
                    plt.savefig(output_path, dpi=300)
                    plt.close()
                    print(f"Saved feature importance plot to {output_path}")
                    
                except Exception as e:
                    print(f"Error plotting feature importance for {model_name}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error in plot_feature_importance: {str(e)}")
        traceback.print_exc()

def export_feature_importance(file_results, output_dir):
    """Export feature importance values to CSV"""
    for model_name, metrics in file_results['models'].items():
        if model_name in ['Random Forest', 'Gradient Boosting']:
            try:
                preprocessor = metrics['model'].named_steps['preprocessor']
                model = metrics['model'].named_steps['regressor']
                
                num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
                all_features = np.concatenate([num_features, cat_features])
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    csv_path = os.path.join(
                        output_dir, 
                        f"feature_importance_values_{model_name.replace(' ', '_')}_{file_results['file'].replace('.csv', '')}.csv"
                    )
                    importance_df.to_csv(csv_path, index=False)
                    print(f"Saved feature importance values to {csv_path}")
                    
            except Exception as e:
                print(f"Error exporting feature importance for {model_name}: {str(e)}")

# Combined plotting functions
def plot_combined_model_comparison(all_results, output_dir):
    """Plot combined model comparison across all files"""
    try:
        metrics_data = []
        for file_result in all_results:
            for model_name, metrics in file_result['models'].items():
                metrics_data.append({
                    'File': file_result['file'],
                    'Model': model_name,
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'R²': metrics['r2']
                })
        
        df = pd.DataFrame(metrics_data)
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Model', y='R²', data=df)
        plt.title('Model Performance Comparison Across All Files (R²)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'combined_model_comparison_r2.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved combined R² comparison plot to {output_path}")
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Model', y='RMSE', data=df)
        plt.title('Model Performance Comparison Across All Files (RMSE)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'combined_model_comparison_rmse.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved combined RMSE comparison plot to {output_path}")
    except Exception as e:
        print(f"Error in plot_combined_model_comparison: {str(e)}")
        traceback.print_exc()

def save_results_to_csv(all_results, output_dir):
    """Save all results to CSV files"""
    try:
        # Model metrics
        metrics_data = []
        for file_result in all_results:
            for model_name, metrics in file_result['models'].items():
                metrics_data.append({
                    'File': file_result['file'],
                    'Model': model_name,
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'R2': metrics['r2'],
                    'Samples': len(metrics['y_test'])
                })
        
        metrics_path = os.path.join(output_dir, 'all_model_metrics.csv')
        pd.DataFrame(metrics_data).to_csv(metrics_path, index=False)
        print(f"Saved model metrics to {metrics_path}")
        
        # Combined predictions
        pred_data = []
        for file_result in all_results:
            for model_name, metrics in file_result['models'].items():
                for y_true, y_pred in zip(metrics['y_test'], metrics['y_pred']):
                    pred_data.append({
                        'File': file_result['file'],
                        'Model': model_name,
                        'Actual': y_true,
                        'Predicted': y_pred,
                        'Residual': y_true - y_pred
                    })
        
        preds_path = os.path.join(output_dir, 'all_predictions.csv')
        pd.DataFrame(pred_data).to_csv(preds_path, index=False)
        print(f"Saved predictions data to {preds_path}")
    except Exception as e:
        print(f"Error in save_results_to_csv: {str(e)}")
        traceback.print_exc()

# Main processing function
def main():
    # Store results for all files and models
    all_results = []
    combined_predictions = []

    for file_name in tqdm(FILES, desc="Processing files"):
        try:
            print(f"\nProcessing {file_name}...")
            
            # Load data
            if not os.path.exists(file_name):
                print(f"File not found: {file_name}")
                continue
                
            data = pd.read_csv(file_name, index_col=0)
            print(f"Initial rows: {len(data)}")
            
            if data.empty:
                print("Empty dataset")
                continue
                
            # Plot correlation visualizations before any preprocessing
            plot_spearman_correlation(data, file_name, OUTPUT_DIR)
            plot_pearson_correlation_matrix(data, file_name, OUTPUT_DIR)
                
            # Preprocessing
            data = data.dropna(subset=['Hydrogen_Weight_Percent'])
            print(f"Rows after dropping NA targets: {len(data)}")
            
            if len(data) == 0:
                print("No valid samples after preprocessing")
                continue
                
            X = data.drop(columns=['Hydrogen_Weight_Percent'])
            y = data['Hydrogen_Weight_Percent']
            
            # Identify feature types
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            # Preprocessing pipeline
            preprocessor = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_cols)
            ])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            
            # Model training and evaluation
            file_results = {'file': file_name, 'models': {}}
            
            for model_name, model in MODELS.items():
                try:
                    print(f"Training {model_name}...")
                    
                    # Create and train model pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', model)
                    ])
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = pipeline.predict(X_test)
                    residuals = y_test - y_pred
                    
                    # Store metrics
                    metrics = {
                        'mae': mean_absolute_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'r2': r2_score(y_test, y_pred),
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'residuals': residuals,
                        'model': pipeline
                    }
                    file_results['models'][model_name] = metrics
                    
                    # Add to combined predictions
                    combined_predictions.append({
                        'file': file_name,
                        'model': model_name,
                        'y_test': y_test,
                        'y_pred': y_pred
                    })
                    
                    print(f"{model_name} trained successfully")
                    
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            all_results.append(file_results)
            
            # Plotting for this file
            plot_model_comparison(file_results, OUTPUT_DIR)
            plot_residuals(file_results, OUTPUT_DIR)
            plot_predictions(file_results, OUTPUT_DIR)
            plot_feature_importance(file_results, OUTPUT_DIR)
            export_feature_importance(file_results, OUTPUT_DIR)
            
            # Save model
            for model_name, metrics in file_results['models'].items():
                model_path = os.path.join(OUTPUT_DIR, f"{file_name.replace('.csv', '')}_{model_name.replace(' ', '_')}.pkl")
                joblib.dump(metrics['model'], model_path)
                print(f"Saved model to {model_path}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            traceback.print_exc()
            continue

    # Generate combined visualizations
    if all_results:
        # Save all results to CSV
        save_results_to_csv(all_results, OUTPUT_DIR)
        
        # Combined visualizations
        plot_combined_model_comparison(all_results, OUTPUT_DIR)
        
        print("\nProcessing complete. Results saved in 'results' directory.")
    else:
        print("\nNo valid files were processed.")

if __name__ == "__main__":
    main()
