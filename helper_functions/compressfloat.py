import pandas as pd
import os
from tqdm import tqdm

# =============================
# CONFIG
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "backend", "models", "datasets", "8ppd_features_full.csv")

TEMP_PATH = FILE_PATH + ".tmp"
CHUNK_SIZE = 1_000_000  # adjust based on RAM

def downcast_numeric(df):
    """Downcast numeric columns to reduce size."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def main():
    first_chunk = True
    total_rows = 0

    reader = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)

    for chunk in tqdm(reader, desc="Processing CSV"):
        chunk = downcast_numeric(chunk)

        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(TEMP_PATH, index=False, mode=mode, header=header)

        total_rows += len(chunk)
        first_chunk = False

    # Replace original file
    os.replace(TEMP_PATH, FILE_PATH)
    print(f"\n Done! Total rows processed: {total_rows:,}")
    print(f" Original CSV overwritten with compressed version: {FILE_PATH}")

if __name__ == "__main__":
    main()