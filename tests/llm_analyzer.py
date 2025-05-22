import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from config import LLM_MODELS, GRAPHS_DIR, RESULTS_DIR

class LLMAnalyzer:
    def __init__(self):
        self.results = {}
    
    def measure_latency(self, model_name, input_text, num_runs=5):
        """Measure latency for a given model and input text."""
        model_config = LLM_MODELS[model_name]
        latencies = []
        
        print(f"\nMeasuring latency for {model_name}...")
        for i in range(num_runs):
            start_time = time.time()
            
            # Simulate model inference with some random variation
            base_latency = model_config['latency']
            variation = base_latency * 0.1  # 10% variation
            simulated_latency = base_latency + (pd.np.random.random() * 2 - 1) * variation
            time.sleep(simulated_latency / 1000)  # Convert ms to seconds
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            print(f"Run {i+1}: {latency:.2f}ms")
        
        return {
            'mean_latency': sum(latencies) / len(latencies),
            'std_latency': pd.Series(latencies).std(),
            'min_latency': min(latencies),
            'max_latency': max(latencies)
        }
    
    def analyze_models(self, input_texts):
        """Analyze performance of all models."""
        for model_name in LLM_MODELS:
            model_results = []
            
            for text in input_texts:
                result = self.measure_latency(model_name, text)
                model_results.append(result)
            
            self.results[model_name] = {
                'config': LLM_MODELS[model_name],
                'performance': model_results
            }
    
    def plot_latency_comparison(self):
        """Plot latency comparison between models."""
        plt.figure(figsize=(12, 6))
        
        models = list(self.results.keys())
        mean_latencies = [
            sum(r['mean_latency'] for r in self.results[m]['performance']) / 
            len(self.results[m]['performance'])
            for m in models
        ]
        
        plt.bar(models, mean_latencies)
        plt.title('Average Latency Comparison')
        plt.xlabel('Model')
        plt.ylabel('Latency (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(GRAPHS_DIR, 'llm_latency_comparison.png')
        plt.savefig(plot_path)
        plt.close()
    
    def plot_parameter_latency_tradeoff(self):
        """Plot tradeoff between model parameters and latency."""
        plt.figure(figsize=(10, 6))
        
        models = list(self.results.keys())
        parameters = [float(self.results[m]['config']['parameters'].replace('B', '')) 
                     for m in models]
        mean_latencies = [
            sum(r['mean_latency'] for r in self.results[m]['performance']) / 
            len(self.results[m]['performance'])
            for m in models
        ]
        
        plt.scatter(parameters, mean_latencies, s=100)
        
        # Add labels for each point
        for i, model in enumerate(models):
            plt.annotate(model, (parameters[i], mean_latencies[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Parameter Count vs Latency Tradeoff')
        plt.xlabel('Number of Parameters (billions)')
        plt.ylabel('Average Latency (ms)')
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(GRAPHS_DIR, 'parameter_latency_tradeoff.png')
        plt.savefig(plot_path)
        plt.close()
    
    def save_results(self):
        """Save analysis results."""
        # Save detailed results
        with open(os.path.join(RESULTS_DIR, 'llm_analysis_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, result in self.results.items():
            config = result['config']
            performance = result['performance']
            
            summary_data.append({
                'model': model_name,
                'parameters': config['parameters'],
                'context_window': config['context_window'],
                'mean_latency': sum(p['mean_latency'] for p in performance) / len(performance),
                'std_latency': sum(p['std_latency'] for p in performance) / len(performance),
                'description': config['description']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            os.path.join(RESULTS_DIR, 'llm_analysis_summary.csv'),
            index=False
        )

def main():
    # Initialize analyzer
    analyzer = LLMAnalyzer()
    
    # Sample input texts for testing
    input_texts = [
        "What is the sales forecast for next month?",
        "Analyze the trend in customer behavior",
        "Generate a summary of the latest market data"
    ]
    
    # Run analysis
    print("Starting LLM analysis...")
    analyzer.analyze_models(input_texts)
    
    # Generate plots
    print("\nGenerating plots...")
    analyzer.plot_latency_comparison()
    analyzer.plot_parameter_latency_tradeoff()
    
    # Save results
    analyzer.save_results()
    print("\nAnalysis complete. Results saved in:", RESULTS_DIR)
    print("Graphs saved in:", GRAPHS_DIR)

if __name__ == "__main__":
    main() 