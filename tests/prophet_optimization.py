import os
import json
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def optimize_prophet_parameters(train_data, val_data):
    """Test different parameter combinations and return the best one."""
    # Define parameter combinations to test
    param_combinations = [
        {
            'changepoint_prior_scale': 0.001,
            'seasonality_prior_scale': 0.01,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        },
        {
            'changepoint_prior_scale': 0.01,
            'seasonality_prior_scale': 0.1,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        },
        {
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 1.0,
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
    ]
    
    best_mape = float('inf')
    best_params = None
    results = []
    
    print("\nTesting parameter combinations...")
    for params in param_combinations:
        try:
            # Initialize and fit model
            model = Prophet(**params)
            model.fit(train_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(val_data))
            forecast = model.predict(future)
            
            # Calculate MAPE
            val_predictions = forecast['yhat'].iloc[-len(val_data):]
            mape = calculate_mape(val_data['y'], val_predictions)
            
            results.append({
                'params': params,
                'mape': mape
            })
            
            print(f"Parameters: {params}")
            print(f"MAPE: {mape:.2f}%")
            
            if mape < best_mape:
                best_mape = mape
                best_params = params
                
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue
    
    return best_params, best_mape, results

def plot_optimization_results(results, output_dir="tests/results"):
    """Plot and save optimization results."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot MAPE distribution
        mape_values = [r['mape'] for r in results]
        plt.figure(figsize=(10, 6))
        plt.hist(mape_values, bins=20)
        plt.title('Distribution of MAPE Values')
        plt.xlabel('MAPE (%)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'mape_distribution.png'))
        plt.close()
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'optimization_results.csv'), index=False)
        
    except Exception as e:
        print(f"Error plotting results: {str(e)}")

def main():
    # Load data directly from JSON
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
        
        # Optimize parameters
        best_params, best_mape, results = optimize_prophet_parameters(train_prophet, val_prophet)
        
        if best_params is None:
            print("No valid parameter combination found. Exiting...")
            return
        
        print("\nBest parameters found:")
        print(f"Parameters: {best_params}")
        print(f"Best MAPE: {best_mape:.2f}%")
        
        # Plot and save results
        plot_optimization_results(results)
        
        # Save best parameters
        with open('tests/results/best_parameters.json', 'w') as f:
            json.dump({
                'parameters': best_params,
                'mape': best_mape
            }, f, indent=4)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 