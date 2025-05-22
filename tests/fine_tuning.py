import os
import json
import subprocess
from config import FINE_TUNING_PARAMS, RESULTS_DIR

class FineTuningManager:
    def __init__(self):
        self.results = {}
    
    def prepare_training_data(self, data_path):
        """Prepare training data for fine-tuning."""
        # This is a placeholder for data preparation logic
        # In practice, you would need to format your data according to Llama Factory's requirements
        print(f"Preparing training data from {data_path}...")
        return True
    
    def run_fine_tuning(self):
        """Run fine-tuning using Llama Factory."""
        print("\nStarting fine-tuning process...")
        
        # Prepare training data
        if not self.prepare_training_data(FINE_TUNING_PARAMS['training_data_path']):
            print("Error preparing training data. Exiting...")
            return
        
        # Construct Llama Factory command
        cmd = [
            "python", "-m", "llama_factory.train",
            "--model_name_or_path", FINE_TUNING_PARAMS['base_model'],
            "--data_path", FINE_TUNING_PARAMS['training_data_path'],
            "--output_dir", FINE_TUNING_PARAMS['output_dir'],
            "--num_train_epochs", str(FINE_TUNING_PARAMS['num_epochs']),
            "--per_device_train_batch_size", str(FINE_TUNING_PARAMS['batch_size']),
            "--learning_rate", str(FINE_TUNING_PARAMS['learning_rate']),
            "--save_strategy", "epoch",
            "--logging_steps", "100"
        ]
        
        try:
            # Run fine-tuning
            print("Running fine-tuning command:", " ".join(cmd))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Get return code
            return_code = process.poll()
            
            if return_code == 0:
                print("\nFine-tuning completed successfully!")
                self._save_training_results()
            else:
                print("\nFine-tuning failed with return code:", return_code)
                error = process.stderr.read()
                print("Error:", error)
        
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
    
    def _save_training_results(self):
        """Save fine-tuning results and configuration."""
        results = {
            'config': FINE_TUNING_PARAMS,
            'status': 'completed',
            'output_dir': FINE_TUNING_PARAMS['output_dir']
        }
        
        # Save results
        with open(os.path.join(RESULTS_DIR, 'fine_tuning_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {os.path.join(RESULTS_DIR, 'fine_tuning_results.json')}")
    
    def evaluate_fine_tuned_model(self):
        """Evaluate the fine-tuned model."""
        # This is a placeholder for model evaluation logic
        # In practice, you would need to implement proper evaluation metrics
        print("\nEvaluating fine-tuned model...")
        print("Note: Implement proper evaluation metrics based on your requirements")
    
    def generate_insights(self):
        """Generate insights from the fine-tuning process."""
        print("\nGenerating insights from fine-tuning process...")
        
        # Load results if available
        results_path = os.path.join(RESULTS_DIR, 'fine_tuning_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print("\nFine-tuning Insights:")
            print(f"Base Model: {results['config']['base_model']}")
            print(f"Training Epochs: {results['config']['num_epochs']}")
            print(f"Batch Size: {results['config']['batch_size']}")
            print(f"Learning Rate: {results['config']['learning_rate']}")
            print(f"Output Directory: {results['config']['output_dir']}")
            print(f"Status: {results['status']}")
        else:
            print("No fine-tuning results found.")

def main():
    # Initialize fine-tuning manager
    manager = FineTuningManager()
    
    # Run fine-tuning
    manager.run_fine_tuning()
    
    # Evaluate model
    manager.evaluate_fine_tuned_model()
    
    # Generate insights
    manager.generate_insights()

if __name__ == "__main__":
    main() 