import os
import json
import pandas as pd
import numpy as np
from prophet import Prophet
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from config import PROPHET_PARAM_GRID, GRAPHS_DIR, RESULTS_DIR

class ModelEvaluator:
    def __init__(self):
        self.best_params = {}
        self.results = {}
    
    def optimize_prophet(self, train_data, val_data, product_name):
        """Optimize Prophet model parameters."""
        results = []
        param_combinations = [dict(zip(PROPHET_PARAM_GRID.keys(), v)) 
                            for v in product(*PROPHET_PARAM_GRID.values())]
        
        for params in param_combinations:
            try:
                # Fit model
                model = Prophet(**params)
                model.fit(train_data)
                
                # Make predictions
                future = model.make_future_dataframe(periods=len(val_data))
                forecast = model.predict(future)
                
                # Calculate MAPE for validation period
                val_forecast = forecast[forecast['ds'].isin(val_data['ds'])]
                if len(val_forecast) > 0 and len(val_data) > 0:
                    mape = mean_absolute_percentage_error(val_data['y'], 
                                                        val_forecast['yhat']) * 100
                    results.append({
                        'parameters': params,
                        'mape': mape
                    })
            
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue
        
        # Sort results by MAPE
        results.sort(key=lambda x: x['mape'])
        
        if results:
            self.best_params[product_name] = results[0]['parameters']
            self.results[product_name] = results
        
        return results
    
    def evaluate_model(self, train_data, test_data, params, product_name, mode="agentic"):
        """Evaluate model with given parameters."""
        try:
            # Fit model
            model = Prophet(**params)
            model.fit(train_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Calculate MAPE for test period
            test_forecast = forecast[forecast['ds'].isin(test_data['ds'])]
            if len(test_forecast) > 0 and len(test_data) > 0:
                mape = mean_absolute_percentage_error(test_data['y'], 
                                                    test_forecast['yhat']) * 100
                
                # Plot results
                self._plot_results(test_data, test_forecast, product_name, mode)
                
                return mape
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return None
    
    def _plot_results(self, actual_data, forecast_data, product_name, mode):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        plt.plot(actual_data['ds'], actual_data['y'], label='Actual', color='blue')
        plt.plot(forecast_data['ds'], forecast_data['yhat'], 
                label='Predicted', color='red', linestyle='--')
        plt.fill_between(forecast_data['ds'],
                        forecast_data['yhat_lower'],
                        forecast_data['yhat_upper'],
                        color='red', alpha=0.1)
        
        plt.title(f'{product_name} - {mode.capitalize()} Mode Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(GRAPHS_DIR, f'{product_name}_{mode}_forecast.png')
        plt.savefig(plot_path)
        plt.close()
    
    def save_results(self):
        """Save optimization and evaluation results."""
        # Save best parameters
        with open(os.path.join(RESULTS_DIR, 'best_parameters.json'), 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save detailed results
        for product_name, results in self.results.items():
            results_df = pd.DataFrame([
                {**r['parameters'], 'mape': r['mape']} for r in results
            ])
            results_df.to_csv(
                os.path.join(RESULTS_DIR, f'{product_name}_optimization_results.csv'),
                index=False
            ) 