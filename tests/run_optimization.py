import os
from data_loader import TestDataLoader
from model_evaluator import ModelEvaluator
from config import GRAPHS_DIR, RESULTS_DIR

def main():
    # Initialize data loader and model evaluator
    data_loader = TestDataLoader()
    model_evaluator = ModelEvaluator()
    
    # Load and prepare data
    data = data_loader.load_data()
    if data is None:
        print("Error loading data. Exiting...")
        return
    
    # Get data splits
    train_data, val_data, test_data = data_loader.get_data_splits()
    
    # Get list of products
    products = data_loader.get_all_products()
    
    # Process each product
    for product_name in products:
        print(f"\nProcessing {product_name}...")
        
        # Prepare data for Prophet
        train_df = data_loader.prepare_prophet_data(train_data[product_name])
        val_df = data_loader.prepare_prophet_data(val_data[product_name])
        test_df = data_loader.prepare_prophet_data(test_data[product_name])
        
        if train_df is None or val_df is None or test_df is None:
            print(f"Skipping {product_name} due to data preparation issues")
            continue
        
        # Optimize parameters
        print("Optimizing parameters...")
        optimization_results = model_evaluator.optimize_prophet(
            train_df, val_df, product_name
        )
        
        if not optimization_results:
            print(f"No valid results for {product_name}")
            continue
        
        # Get best parameters
        best_params = model_evaluator.best_params[product_name]
        print(f"Best MAPE: {optimization_results[0]['mape']:.2f}%")
        print("Best parameters:", best_params)
        
        # Evaluate in both modes
        print("\nEvaluating in agentic mode...")
        agentic_mape = model_evaluator.evaluate_model(
            train_df, test_df, best_params, product_name, mode="agentic"
        )
        
        print("\nEvaluating in ask mode...")
        ask_mape = model_evaluator.evaluate_model(
            train_df, test_df, best_params, product_name, mode="ask"
        )
        
        if agentic_mape is not None and ask_mape is not None:
            print(f"\nResults for {product_name}:")
            print(f"Agentic mode MAPE: {agentic_mape:.2f}%")
            print(f"Ask mode MAPE: {ask_mape:.2f}%")
            print(f"Improvement: {abs(agentic_mape - ask_mape):.2f}%")
    
    # Save all results
    model_evaluator.save_results()
    print("\nOptimization complete. Results saved in:", RESULTS_DIR)
    print("Graphs saved in:", GRAPHS_DIR)

if __name__ == "__main__":
    main() 