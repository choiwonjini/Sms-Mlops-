import pandas as pd
import json

def analyze_failures(csv_path):
    df = pd.read_csv(csv_path)
    # Use correct_score if score column doesn't exist
    score_col = 'correct_score' if 'correct_score' in df.columns else 'score'
    
    low_scores = df[df[score_col] < 1.0]
    print(f"Total Low Score Cases: {len(low_scores)}")
    
    for _, row in low_scores.head(10).iterrows():
        print(f"\n--- Case {row['no']} (Score: {row[score_col]}) ---")
        print(f"Order Script:\n{row['order']}")
        print(f"\nLabel: {row['label']}")
        print(f"\nPredict: {row['predict']}")
        print("-" * 50)

if __name__ == "__main__":
    analyze_failures('test_result/test_result_20251222_score.csv')
