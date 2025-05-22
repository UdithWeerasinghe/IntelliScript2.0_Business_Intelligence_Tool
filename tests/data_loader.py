import os
import json
import pandas as pd
from datetime import datetime
from config import DATA_DIR

class TestDataLoader:
    def __init__(self, data_file="test.json"):
        self.data_file = os.path.join(DATA_DIR, data_file)
        self.data = None
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the test data."""
        try:
            with open(self.data_file, 'r') as f:
                raw_data = json.load(f)
                consumer_data = raw_data['data']["1.  Consumer  goods"]
                
                # Convert directly to DataFrame
                self.df = pd.DataFrame(consumer_data)
                self.df['date'] = pd.to_datetime(self.df['Date'])
                self.df = self.df.sort_values('date')
                
                print("\nConsumer Goods Data Summary:")
                print(f"Total samples: {len(self.df)}")
                print(f"Date range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}")
                print(f"Value range: ${self.df['value'].min():.2f}M to ${self.df['value'].max():.2f}M")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.df = None
    
    def get_data_splits(self):
        """Split data into train, validation, and test sets."""
        if self.df is None or len(self.df) == 0:
            return None, None, None
        
        try:
            # Calculate split points
            total_samples = len(self.df)
            train_end = int(total_samples * 0.6)
            val_end = int(total_samples * 0.8)
            
            # Split the data
            train_data = self.df.iloc[:train_end]
            val_data = self.df.iloc[train_end:val_end]
            test_data = self.df.iloc[val_end:]
            
            print(f"\nData splits:")
            print(f"Training set: {len(train_data)} samples (${train_data['value'].mean():.2f}M avg)")
            print(f"Validation set: {len(val_data)} samples (${val_data['value'].mean():.2f}M avg)")
            print(f"Test set: {len(test_data)} samples (${test_data['value'].mean():.2f}M avg)")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            print(f"Error splitting data: {str(e)}")
            return None, None, None
    
    def prepare_prophet_data(self, df):
        """Prepare data for Prophet model."""
        if df is None or len(df) == 0:
            return None
        
        try:
            prophet_df = df.copy()
            prophet_df = prophet_df.rename(columns={'date': 'ds', 'value': 'y'})
            prophet_df = prophet_df[['ds', 'y']]  # Keep only required columns
            prophet_df = prophet_df.dropna()
            prophet_df = prophet_df.sort_values('ds')
            
            return prophet_df
        except Exception as e:
            print(f"Error preparing Prophet data: {str(e)}")
            return None 