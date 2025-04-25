```python
# src/train.py
#!/usr/bin/env python

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from joblib import dump

# Paths
data_path = os.path.join(os.path.dirname(__file__), '..', 'brain_stroke.csv')
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_path = os.path.join(model_dir, 'stroke_model_pipeline.joblib')

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# 1. Load data
df = pd.read_csv(data_path)

# 2. Winsorize continuous features
cont_cols = ['age', 'avg_glucose_level', 'bmi']
for col in cont_cols:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lo, upper=hi)

# 3. Define features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Preprocessing pipeline
numeric_transformer = Pipeline([('scaler', StandardScaler())])
cat_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status', 'hypertension', 'heart_disease']
categorical_transformer = Pipeline([('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, cont_cols),
    ('cat', categorical_transformer, cat_cols)
])

# 6. Full pipeline with SMOTE and classifier
pipeline = Pipeline([
    ('pre', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ))
])

# 7. Train and save model
pipeline.fit(X_train, y_train)
dump(pipeline, model_path)
print(f"Model trained and saved to {model_path}")
``` 
