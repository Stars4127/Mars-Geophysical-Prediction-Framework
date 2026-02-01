import pandas as pd
f1 = 'backend/models/datasets/core_set_2ppd.csv'
f2 = 'backend/models/datasets/8ppd_core_features_for_mapping.csv'

with open('headers_output.txt', 'w') as f:
    df1 = pd.read_csv(f1, nrows=1)
    f.write(f"--- {f1} ---\n")
    f.write(", ".join(df1.columns.tolist()) + "\n\n")
    
    df2 = pd.read_csv(f2, nrows=1)
    f.write(f"--- {f2} ---\n")
    f.write(", ".join(df2.columns.tolist()) + "\n")
