import os
import sys
import pickle
import joblib
import pandas as pd
import numpy as np


def main():
    target_file = 'rainfall_pipeline.pkl'
    target_max_size_mb = 80.0

    if not os.path.exists(target_file):
        print(f"Error: Target file '{target_file}' not found.")
        sys.exit(1)

    orig_size_bytes = os.path.getsize(target_file)
    orig_size_mb = orig_size_bytes / (1024 * 1024)
    print(f"Original File Size: {orig_size_mb:.2f} MB")

    # Load original pipeline
    try:
        pipeline = joblib.load(target_file)
    except Exception:
        with open(target_file, 'rb') as f:
            pipeline = pickle.load(f)

    # Sample test payload for prediction parity verification
    test_data = pd.DataFrame([{
        'Humidity9am': 85.0,
        'Pressure9am': 1005.0,
        'MinTemp': 18.0,
        'MaxTemp': 28.0,
        'Rainfall': 12.0
    }])

    orig_pred = pipeline.predict(test_data)[0]
    orig_proba = pipeline.predict_proba(test_data)[0]

    print("\n--- Baseline Model Output ---")
    print(f"Original Prediction: {orig_pred}")
    print(f"Original Probabilities: {orig_proba}")

    # Strategy 1: High-Efficiency Joblib Compression (zlib level 3 / level 9)
    temp_file = 'compressed_pipeline_temp.pkl'
    print("\nApplying Strategy 1: Joblib Compression (zlib, compress=3)...")
    joblib.dump(pipeline, temp_file, compress=('zlib', 3))
    comp_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
    print(f"Compressed Size (compress=3): {comp_size_mb:.2f} MB")

    if comp_size_mb > target_max_size_mb:
        print("Strategy 1 (compress=3) exceeds 80 MB. Trying compress=9...")
        joblib.dump(pipeline, temp_file, compress=('zlib', 9))
        comp_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
        print(f"Compressed Size (compress=9): {comp_size_mb:.2f} MB")

    # Strategy 2: Model Pruning (If compression alone > 80 MB)
    if comp_size_mb > target_max_size_mb:
        print("\nApplying Strategy 2: Model Pruning...")
        rf = None
        if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
            rf = pipeline.named_steps['model']
        elif hasattr(pipeline, 'steps'):
            rf = pipeline.steps[-1][1]

        if rf is not None and hasattr(rf, 'estimators_'):
            print(f"Current n_estimators: {len(rf.estimators_)}")
            # Trim estimators to 80 estimators if unconstrained
            if len(rf.estimators_) > 80:
                rf.estimators_ = rf.estimators_[:80]
                rf.n_estimators = 80
                print("Pruned estimators to 80.")

        # Re-compress after pruning
        joblib.dump(pipeline, temp_file, compress=('zlib', 3))
        comp_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
        print(f"Post-pruning Compressed Size: {comp_size_mb:.2f} MB")

    # Verify Prediction Parity
    print("\n--- Verifying Prediction Parity ---")
    compressed_pipeline = joblib.load(temp_file)
    new_pred = compressed_pipeline.predict(test_data)[0]
    new_proba = compressed_pipeline.predict_proba(test_data)[0]

    print(f"Compressed Model Prediction: {new_pred}")
    print(f"Compressed Model Probabilities: {new_proba}")

    parity = (orig_pred == new_pred) and np.allclose(orig_proba, new_proba, atol=1e-4)
    print(f"Parity Check Passed: {parity}")

    if comp_size_mb <= target_max_size_mb and parity:
        # Overwrite original pipeline file
        if os.path.exists(target_file):
            os.remove(target_file)
        os.rename(temp_file, target_file)
        final_size_mb = os.path.getsize(target_file) / (1024 * 1024)
        print(f"\nSUCCESS: '{target_file}' optimized from {orig_size_mb:.2f} MB to {final_size_mb:.2f} MB!")
    else:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print("\nERROR: Failed to meet size target or prediction parity requirements.")
        sys.exit(1)


if __name__ == "__main__":
    main()
