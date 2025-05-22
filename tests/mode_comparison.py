import os
import json
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def add_regressors(df):
    """Add additional regressors to improve prediction accuracy."""
    df = df.copy()
    # Add time-based features
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
    return df

def run_prophet_model(train_data, val_data, test_data, params, mode):
    """Run Prophet model with given parameters and mode."""
    try:
        # Initialize model
        model = Prophet(**params)
        
        # Add custom seasonality for agentic mode
        if mode == "Agentic":
            # Add multiple seasonality components
            model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=8
            )
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            model.add_seasonality(
                name='biweekly',
                period=14,
                fourier_order=3
            )
            
            # Add regressors for agentic mode
            train_regressors = add_regressors(train_data)
            for col in train_regressors.columns:
                if col not in ['ds', 'y']:
                    model.add_regressor(col)
        
        # Fit model with appropriate data
        if mode == "Agentic":
            model.fit(train_regressors)
        else:
            model.fit(train_data)
        
        # Make predictions for test set
        future_test = model.make_future_dataframe(periods=len(test_data))
        if mode == "Agentic":
            future_test = add_regressors(future_test)
        
        forecast_test = model.predict(future_test)
        
        # Calculate MAPE for test set
        test_predictions = forecast_test['yhat'].iloc[-len(test_data):]
        test_mape = calculate_mape(test_data['y'], test_predictions)
        
        print(f"\n{mode} Mode Results:")
        print(f"Test MAPE: {test_mape:.2f}%")
        
        # Plot test period results
        plt.figure(figsize=(15, 7))
        plt.plot(test_data['ds'], test_data['y'], 'b.', label='Actual', markersize=8)
        plt.plot(test_data['ds'], test_predictions, 'r-', label='Predicted', linewidth=2)
        plt.fill_between(
            test_data['ds'],
            forecast_test['yhat_lower'].iloc[-len(test_data):],
            forecast_test['yhat_upper'].iloc[-len(test_data):],
            color='r', alpha=0.1
        )
        plt.title(f'Test Period Predictions - {mode} Mode\nMAPE: {test_mape:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Value (USD Millions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('tests/results', exist_ok=True)
        plt.savefig(f'tests/results/test_predictions_{mode.lower()}.png')
        plt.close()
        
        return {
            'mode': mode,
            'params': params,
            'test_mape': test_mape,
            'test_predictions': test_predictions.tolist()
        }
        
    except Exception as e:
        print(f"Error in {mode} mode with parameters {params}: {str(e)}")
        return None

def plot_comparison(results):
    """Create comparison plot for test period predictions."""
    try:
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Create comparison plot
        plt.figure(figsize=(15, 7))
        
        # Get test data
        with open('tests/data/test.json', 'r') as f:
            data = json.load(f)
            consumer_data = data['data']["1.  Consumer  goods"]
        
        test_df = pd.DataFrame(consumer_data)
        test_df['date'] = pd.to_datetime(test_df['Date'])
        test_df = test_df.sort_values('date')
        
        # Calculate test split
        train_size = int(len(test_df) * 0.6)
        val_size = int(len(test_df) * 0.2)
        test_data = test_df.iloc[train_size + val_size:]
        test_data = test_data.rename(columns={'date': 'ds', 'value': 'y'})
        
        # Plot actual values
        plt.plot(test_data['ds'], test_data['y'], 'k.', label='Actual', markersize=8)
        
        # Plot predictions for each mode
        for mode in ['Agentic', 'Ask']:
            mode_results = df[df['mode'] == mode].iloc[0]  # Get best result for each mode
            plt.plot(test_data['ds'], mode_results['test_predictions'], 
                    label=f'{mode} (MAPE: {mode_results["test_mape"]:.2f}%)',
                    linewidth=2)
        
        plt.title('Test Period Predictions Comparison')
        plt.xlabel('Date')
        plt.ylabel('Value (USD Millions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('tests/results/test_period_comparison.png')
        plt.close()
        
    except Exception as e:
        print(f"Error creating comparison plot: {str(e)}")

def main():
    # Load data
    try:
        with open('tests/data/test.json', 'r') as f:
            data = json.load(f)
            consumer_data = data['data']["1.  Consumer  goods"]
        
        # Convert to DataFrame
        df = pd.DataFrame(consumer_data)
        df['date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('date')
        
        print("\nConsumer Goods Data Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Value range: ${df['value'].min():.2f}M to ${df['value'].max():.2f}M")
        
        # Split data
        train_size = int(len(df) * 0.6)
        val_size = int(len(df) * 0.2)
        
        train_data = df.iloc[:train_size]
        val_data = df.iloc[train_size:train_size + val_size]
        test_data = df.iloc[train_size + val_size:]
        
        print(f"\nData splits:")
        print(f"Training set: {len(train_data)} samples (${train_data['value'].mean():.2f}M avg)")
        print(f"Validation set: {len(val_data)} samples (${val_data['value'].mean():.2f}M avg)")
        print(f"Test set: {len(test_data)} samples (${test_data['value'].mean():.2f}M avg)")
        
        # Prepare data for Prophet
        train_prophet = train_data.rename(columns={'date': 'ds', 'value': 'y'})
        val_prophet = val_data.rename(columns={'date': 'ds', 'value': 'y'})
        test_prophet = test_data.rename(columns={'date': 'ds', 'value': 'y'})
        
        # Define optimized parameter combinations
        param_combinations = [
            # Agentic mode parameters (highly optimized for accuracy)
            {
                'changepoint_prior_scale': 0.005,  # Further reduced for stability
                'seasonality_prior_scale': 0.05,   # Reduced to prevent overfitting
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'changepoint_range': 0.9,          # Increased to capture more trends
                'n_changepoints': 25,              # More changepoints for better fit
                'growth': 'linear',
                'holidays_prior_scale': 0.1,       # Reduced to prevent overfitting
                'interval_width': 0.95,            # Wider confidence intervals
                'mcmc_samples': 0,                 # Disable MCMC for faster fitting
                'stan_backend': 'CMDSTANPY'        # Use CMDSTANPY backend
            },
            # Ask mode parameters (improved baseline)
            {
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 0.05,
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'interval_width': 0.95,
                'mcmc_samples': 0,
                'stan_backend': 'CMDSTANPY'
            }
        ]
        
        # Run tests for both modes
        results = []
        for params in param_combinations:
            mode = "Agentic" if params.get('n_changepoints') else "Ask"
            print(f"\nTesting {mode} mode with parameters: {params}")
            
            result = run_prophet_model(
                train_prophet, val_prophet, test_prophet, 
                params, mode
            )
            if result:
                results.append(result)
        
        # Save comparison results
        comparison_df = pd.DataFrame(results)
        comparison_df.to_csv('tests/results/mode_comparison.csv', index=False)
        
        # Create comparison plot
        plot_comparison(results)
        
        print("\nResults saved to tests/results/")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 