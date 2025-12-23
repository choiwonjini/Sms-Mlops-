import os
import json
import pandas as pd
from ocr_manager import OCRManager
from config import settings
from datetime import datetime
import argparse

def run_ocr_tests(limit=None):
    """
    Runs OCR tests on a subset of images and exports results to CSV.
    """
    json_path = "test_data/transfer_images_labels_accurate.json"
    image_dir = "test_data/test_image"
    output_dir = "test_result"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading labels from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    # Apply limit if specified
    if limit:
        print(f"Limiting to first {limit} cases.")
        labels_data = labels_data[:limit]

    ocr_manager = Agent ()
    results = []

    print(f"Starting OCR tests for {len(labels_data)} images...")
    
    for i, entry in enumerate(labels_data):
        image_filename = entry.get("image_filename")
        image_path = os.path.join(image_dir, image_filename)
        label_payment_info = entry.get("payment_info")

        print(f"[{i+1}/{len(labels_data)}] Analyzing: {image_filename}...")
        
        if not os.path.exists(image_path):
            print(f"  [Warning] Image not found: {image_path}")
            predict_result = {"error": "File not found"}
        else:
            try:
                predict_result = ocr_manager.analyze_payment_receipt(image_path)
            except Exception as e:
                print(f"  [Error] OCR failed: {e}")
                predict_result = {"error": str(e)}

        results.append({
            "image_path": image_path,
            "predict": json.dumps(predict_result, ensure_ascii=False),
            "label": json.dumps(label_payment_info, ensure_ascii=False)
        })

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"ocr_test_results_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print("-" * 30)
    print(f"OCR Test Completed!")
    print(f"Results saved to: {csv_path}")
    print("-" * 30)
    
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Evaluation Script")
    parser.add_argument("--limit", type=int, default=5, help="Number of test cases to run (default: 5)")
    args = parser.parse_args()

    # You can also change the default MAX_TEST_CASES here
    MAX_TEST_CASES = args.limit
    
    run_ocr_tests(limit=MAX_TEST_CASES)
