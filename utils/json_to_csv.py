import pandas as pd
import json

def json_to_csv(json_file, csv_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"Successfully converted {json_file} to {csv_file}")
    except FileNotFoundError:
        print(f"Error: File not found: {json_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
json_to_csv('annotated_dataset.json', 'annotated_dataset.csv')