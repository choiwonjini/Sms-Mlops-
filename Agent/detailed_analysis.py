import pandas as pd
import json
import ast

def safe_parse(val):
    if pd.isna(val): return {}
    if isinstance(val, dict): return val
    try:
        return json.loads(val)
    except:
        try:
            return ast.literal_eval(val)
        except:
            return {}

def analyze_mismatches(csv_path):
    df = pd.read_csv(csv_path)
    score_col = 'correct_score' if 'correct_score' in df.columns else 'score'
    
    mismatches = []
    
    for _, row in df[df[score_col] < 1.0].iterrows():
        label = safe_parse(row['label'])
        predict = safe_parse(row['predict'])
        
        reasons = []
        
        # Check Items
        l_items = label.get('items', [])
        p_items = predict.get('items', [])
        
        if len(l_items) != len(p_items):
            reasons.append(f"Item Count Mismatch: {len(l_items)} vs {len(p_items)}")
        else:
            for li, pi in zip(l_items, p_items):
                l_name = li.get('product_name', '').strip()
                p_name = pi.get('product_name', '').strip()
                if l_name != p_name:
                    reasons.append(f"Product Name: Expected '{l_name}' vs Actual '{p_name}'")
        
        # Check Delivery Date
        if label.get('desired_delivery_date') != predict.get('desired_delivery_date'):
            reasons.append(f"Date: '{label.get('desired_delivery_date')}' != '{predict.get('desired_delivery_date')}'")
            
        # Check Customer/Contact/Address
        for field in ['customer_name', 'contact_number', 'delivery_address']:
            if str(label.get(field)).strip() != str(predict.get(field)).strip():
                 reasons.append(f"{field}: '{label.get(field)}' != '{predict.get(field)}'")

        mismatches.append({
            "no": row['no'],
            "score": row[score_col],
            "reasons": "; ".join(reasons)
        })
    
    res_df = pd.DataFrame(mismatches)
    print(res_df.head(20).to_string())
    
    # Summary of common reasons
    print("\nCommon Reason Patterns:")
    all_reasons = []
    for r in res_df['reasons']:
        all_reasons.extend([x.split(":")[0] for x in r.split("; ") if x])
    print(pd.Series(all_reasons).value_counts())

if __name__ == "__main__":
    import glob
    import os
    list_of_files = glob.glob('test_result/test_result_*.csv') 
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Analyzing Latest File: {latest_file}")
    analyze_mismatches(latest_file)
