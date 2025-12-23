import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
import os

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

def plot_distribution(filename):
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        print(f"[Warn] File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {filename}: {len(df)} rows")
        
        # 1. Turn Distribution (Count Plot)
        plt.figure(figsize=(10, 6))
        
        ax = sns.countplot(x='turn', data=df, palette='Blues_d')
        plt.title("Data Distribution (Turn)", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Turns")
        plt.ylabel("Count")
        
        # Add values
        for i in ax.containers:
            ax.bar_label(i)

        plt.tight_layout()
        output_turn = filename.replace(".csv", "_turn_dist.png")
        plt.savefig(os.path.join(base_dir, output_turn), dpi=300)
        print(f"  Saved Turn dist: {output_turn}")
        plt.close()

        # 2. Item Count Distribution (Count Plot)
        plt.figure(figsize=(10, 6))
        
        ax = sns.countplot(x='item_count', data=df, palette='Greens_d')
        plt.title("Data Distribution (Item Count)", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Items")
        plt.ylabel("Count")
        
        # Add values
        for i in ax.containers:
            ax.bar_label(i)
            
        plt.tight_layout()    
        output_item = filename.replace(".csv", "_item_dist.png")
        plt.savefig(os.path.join(base_dir, output_item), dpi=300)
        print(f"  Saved Item dist: {output_item}")
        plt.close()
        
    except Exception as e:
        print(f"  [Error] Failed to process {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' does not exist.")
    else:
        for f in files:
            plot_distribution(f)
