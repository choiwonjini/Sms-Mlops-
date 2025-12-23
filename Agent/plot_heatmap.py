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

def plot_heatmap(filename):
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        print(f"[Warn] File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {filename}: {len(df)} rows")
        
        # Create Pivot Table: Rows(Turn) x Cols(Item Count) -> Value(Mean Score)
        pivot_table = df.pivot_table(index='turn', columns='item_count', values='correct_score', aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        
        # Heatmap
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, linewidths=.5)
        
        plt.title(f"Complexity Heatmap (Score) - {filename}", fontsize=14, fontweight='bold')
        plt.xlabel("Item Count")
        plt.ylabel("Turn Count")
        
        # Invert Y axis to have smaller turns at bottom? No, standard matrix is fine.
        # But usually graphs have 0 at bottom. Heatmap index 0 is top.
        plt.gca().invert_yaxis() 
        
        plt.tight_layout()
        
        output_name = filename.replace(".csv", "_heatmap.png")
        plt.savefig(os.path.join(base_dir, output_name), dpi=300)
        print(f"  Saved Heatmap: {output_name}")
        plt.close()
        
    except Exception as e:
        print(f"  [Error] Failed to process {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' does not exist.")
    else:
        for f in files:
            plot_heatmap(f)
