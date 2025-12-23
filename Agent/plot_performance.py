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

def plot_performance(filename):
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        print(f"[Warn] File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {filename}: {len(df)} rows")
        
        # 1. Turn vs Accuracy (Line Plot)
        plt.figure(figsize=(10, 6))
        # Group by turn and calculate mean score
        turn_perf = df.groupby('turn')['correct_score'].mean().reset_index()
        
        sns.lineplot(x='turn', y='correct_score', data=turn_perf, marker='o', linewidth=2.5)
        plt.title(f"Accuracy by Turns - {filename}", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Turns")
        plt.ylabel("Mean Accuracy (Score)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.1)  # Score is 0-1
        
        output_turn = filename.replace(".csv", "_turn_plot.png")
        plt.savefig(os.path.join(base_dir, output_turn), dpi=300)
        print(f"  Saved Turn plot: {output_turn}")
        plt.close()

        # 2. Item Count vs Accuracy (Bar Plot)
        plt.figure(figsize=(10, 6))
        # Group by item_count and calculate mean score
        item_perf = df.groupby('item_count')['correct_score'].mean().reset_index()
        
        ax = sns.barplot(x='item_count', y='correct_score', data=item_perf, hue='item_count', palette='viridis', legend=False)
        
        # Add values on bars
        for i in ax.containers:
            ax.bar_label(i, fmt='%.2f')

        plt.title(f"Accuracy by Item Count - {filename}", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Items")
        plt.ylabel("Mean Accuracy (Score)")
        plt.ylim(0, 1.1)
        
        output_item = filename.replace(".csv", "_item_plot.png")
        plt.savefig(os.path.join(base_dir, output_item), dpi=300)
        print(f"  Saved Item plot: {output_item}")
        plt.close()
        
    except Exception as e:
        print(f"  [Error] Failed to process {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' does not exist.")
    else:
        for f in files:
            plot_performance(f)
