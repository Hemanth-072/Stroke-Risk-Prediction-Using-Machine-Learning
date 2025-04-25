```python
# src/predict.py
#!/usr/bin/env python

import os
import pandas as pd
from joblib import load

# Model path
model_path = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'stroke_model_pipeline.joblib'
)


def main(input_csv, output_csv=None):
    # 1. Load input data
    df = pd.read_csv(input_csv)
    # 2. Load trained pipeline
    pipeline = load(model_path)
    # 3. Predict probabilities for stroke
    proba = pipeline.predict_proba(df)[:, 1]
    results = pd.DataFrame({'stroke_risk_probability': proba})
    # 4. Output results
    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stroke risk prediction script')
    parser.add_argument('input_csv', help='Path to input CSV with features')
    parser.add_argument('-o', '--output', help='Path to save predictions CSV')
    args = parser.parse_args()
    main(args.input_csv, args.output)
```
