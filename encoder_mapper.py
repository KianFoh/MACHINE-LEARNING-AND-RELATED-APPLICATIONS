import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np

class EncoderMapper:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.mappings = {}

    def encode_column(self, df, column_name):
        """
        Encode a given column in the dataframe using LabelEncoder and 
        store the mapping for future use.
        """
        # Initialize the label encoder
        label_encoder = LabelEncoder()
        
        # Fit the encoder and transform the column
        df[column_name + ' Encoded'] = label_encoder.fit_transform(df[column_name])
        
        # Store the mapping for the column
        self.mappings[column_name] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        
        # Drop the original column to avoid redundancy
        df.drop(columns=[column_name], inplace=True)
        
        return df


    def save_mappings(self, file_path):
        """
        Save the mappings dictionary to a JSON file, converting all int64 values to int.
        """
        # Convert int64 to int
        self.mappings = {k: {key: int(value) if isinstance(value, np.int64) else value for key, value in v.items()}
                         for k, v in self.mappings.items()}
        
        # Save the mappings to a JSON file
        with open(file_path, 'w') as f:
            json.dump(self.mappings, f, indent=4)

    def load_mappings(self, file_path):
        """
        Load the mappings dictionary from a JSON file
        """
        with open(file_path, 'r') as f:
            self.mappings = json.load(f)

    def get_mapping(self, column_name):
        """
        Get the mapping for a specific column
        """
        return self.mappings.get(column_name, {})