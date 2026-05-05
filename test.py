import pandas as pd
import os

# Configuration
INPUT_FILE = 'archive (1)/HI-Medium_Trans.csv'
OUTPUT_FILE = 'MMORPG_Medium_Cleaned.csv'
CHUNK_SIZE = 1000000  # Process 1 million rows at a time

# Column mapping (Same as before)
column_mapping = {
    'Timestamp': 'Trade_Time',
    'Account': 'Sender_Player_ID',
    'Account.1': 'Receiver_Player_ID',
    'Amount Paid': 'In_Game_Currency_Value',
    'Payment Format': 'Trade_Type',
    'Is Laundering': 'Is_Fraudulent_Trade'
}

columns_to_keep = list(column_mapping.keys())

print(f"Starting Big Data Clean: {INPUT_FILE}")

# Initialize the CSV file with headers only
first_chunk = True

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, usecols=columns_to_keep):
    # 1. Rename columns
    chunk = chunk.rename(columns=column_mapping)
    
    # 2. Drop rows with missing Player IDs immediately
    chunk = chunk.dropna(subset=['Sender_Player_ID', 'Receiver_Player_ID'])
    
    # 3. Optimization: Convert Player IDs to Strings (if not already) 
    # and trade times to datetime if you plan on doing temporal analysis later
    
    # 4. Save/Append to the new cleaned file
    if first_chunk:
        chunk.to_csv(OUTPUT_FILE, index=False, mode='w')
        first_chunk = False
    else:
        chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)
    
    print(f"Processed {CHUNK_SIZE} rows...")

print(f"\nDone! Cleaned dataset saved as {OUTPUT_FILE}")