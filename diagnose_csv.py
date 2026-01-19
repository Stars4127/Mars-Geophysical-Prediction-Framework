
import csv

FILE_PATH = 'datasets/core_set_2ppd.csv'
LOG_PATH = 'csv_errors.txt'

def scan_for_errors():
    with open(FILE_PATH, 'r', encoding='utf-8', errors='replace') as f:
        # Read header to know expected column count
        line1 = f.readline()
        header = line1.strip().split(',')
        expected_cols = len(header)
        
    with open(LOG_PATH, 'w', encoding='utf-8') as log:
        log.write(f"Header has {expected_cols} columns.\n")
        log.write(f"Columns: {header}\n\n")
        
        bad_rows = 0
        
        # Open again to read rows
        with open(FILE_PATH, 'r', encoding='utf-8', errors='replace') as f:
            # We use manual splitting to see what the 'raw' split looks like vs csv reader
            # But let's verify if lines are simply broken
            next(f) # skip header
            
            for i, line in enumerate(f, start=2):
                line = line.strip()
                if not line: continue
                
                parts = line.split(',')
                # simple split check first
                if len(parts) != expected_cols:
                    bad_rows += 1
                    if bad_rows <= 10:
                        log.write(f"Row {i} error: Found {len(parts)} parts (expected {expected_cols}).\n")
                        log.write(f"Raw Line: {line}\n")
                        log.write("-" * 20 + "\n")

        log.write(f"\nTotal rows with simple comma anomalies: {bad_rows}\n")

if __name__ == "__main__":
    scan_for_errors()
