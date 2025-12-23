import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
import os
import matplotlib.pyplot as plt

# Explicitly set font for Windows
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Define file paths
base_dir = "visualize_data"
files = [
    "88_test_result_with_verifier.csv",
    "91_test_result_retry_logic_without_verifier_prompt_engineering.csv",
    "93_test_result_with_verifier_name_prompt_engineering.csv",
    "95_test_result_with_verifier_name_prompt_engineering_retry_logic.csv"
]

def plot_error_counts(filename):
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        print(f"[Warn] File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {filename}: {len(df)} rows")
        
        target_col = 'error_log'
        if target_col not in df.columns:
            print(f"  [Error] '{target_col}' column missing in {filename}")
            if 'error_msg' in df.columns:
                target_col = 'error_msg'
                print(f"  -> Found 'error_msg' instead. Using it.")
            else:
                return
            
        # fillna to handle empty errors
        df[target_col] = df[target_col].fillna("Success (None)")
        
        # Filter out Success/None
        df_errors = df[df[target_col] != "Success (None)"]
        
        if len(df_errors) == 0:
            print(f"  [Info] No errors found in {filename}. Skipping plot.")
            return

        plt.figure(figsize=(12, 8))
        
        # Count plot (Vertical bars -> x=target_col)
        counts = df_errors[target_col].value_counts()
        
        ax = sns.countplot(x=target_col, data=df_errors, order=counts.index)
        
        # Add values on top of bars
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.1, str(v), color='black', ha='center', va='bottom')
        
        plt.title(f"Error Log Distribution - {filename}", fontsize=15, fontweight='bold')
        plt.xlabel("Error Type")
        plt.ylabel("Count")
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        output_name = filename.replace(".csv", "_plot.png")
        output_path = os.path.join(base_dir, output_name)
        plt.savefig(output_path, dpi=300)
        print(f"  Saved plot to: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"  [Error] Failed to process {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' does not exist.")
    else:
        for f in files:
            plot_error_counts(f)
