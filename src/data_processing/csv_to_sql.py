import pandas as pd
import sqlite3
from pathlib import Path
import time
import logging
from tqdm import tqdm
import os

def setup_logging(log_dir):
    """
    Set up logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'csv_to_sql_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_chunk(chunk):
    """
    Validate chunk and remove rows with missing essential data.
    Returns cleaned chunk with only valid rows.
    """
    # List of required fields that must not be null
    required_fields = ['white', 'black', 'result', 'pgn', 'moves']
    
    # Remove rows where any of the required fields is null/empty
    valid_rows = chunk.dropna(subset=required_fields)
    
    # Also remove rows where any required field is an empty string
    for field in required_fields:
        valid_rows = valid_rows[valid_rows[field].str.strip() != '']
    
    return valid_rows

def create_chess_database(csv_path, db_path, chunk_size=10000, logger=None):
    """
    Convert a large chess games CSV file to SQLite database.
    Only includes complete records with all essential chess game data.
    """
    # Create database connection
    conn = sqlite3.connect(db_path)
    
    # Create the games table with appropriate data types
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS chess_games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT,
        site TEXT,
        date TEXT,
        round TEXT,
        white TEXT NOT NULL,
        black TEXT NOT NULL,
        result TEXT NOT NULL,
        white_elo INTEGER,
        black_elo INTEGER,
        white_title TEXT,
        black_title TEXT,
        eco TEXT,
        opening TEXT,
        time_control TEXT,
        import_date TEXT,
        source TEXT,
        moves TEXT NOT NULL,
        eval_info TEXT,
        clock_info TEXT,
        pgn TEXT NOT NULL
    )
    """
    
    try:
        conn.execute(create_table_sql)
        conn.commit()
        
        # Get total number of rows for progress bar
        total_rows = sum(1 for _ in pd.read_csv(csv_path, chunksize=chunk_size))
        total_rows *= chunk_size  # Approximate total rows
        
        # Process the CSV file in chunks
        chunk_iterator = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            na_values=[''],
            keep_default_na=True,
            dtype={
                'event': 'str',
                'site': 'str',
                'date': 'str',
                'round': 'str',
                'white': 'str',
                'black': 'str',
                'result': 'str',
                'white_elo': 'Int64',
                'black_elo': 'Int64',
                'white_title': 'str',
                'black_title': 'str',
                'eco': 'str',
                'opening': 'str',
                'time_control': 'str',
                'import_date': 'str',
                'source': 'str',
                'moves': 'str',
                'eval_info': 'str',
                'clock_info': 'str',
                'pgn': 'str'
            },
            low_memory=False
        )
        
        total_rows_processed = 0
        total_rows_kept = 0
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=total_rows, unit='rows', desc='Processing chunks')
        
        # Process each chunk
        for chunk_number, chunk in enumerate(chunk_iterator, 1):
            try:
                chunk_size_before = len(chunk)
                
                # Validate and clean the chunk
                valid_chunk = validate_chunk(chunk)
                chunk_size_after = len(valid_chunk)
                
                # Insert only the valid rows into the database
                if not valid_chunk.empty:
                    valid_chunk.to_sql('chess_games', conn, if_exists='append', index=False)
                
                total_rows_processed += chunk_size_before
                total_rows_kept += chunk_size_after
                
                # Update progress bar
                pbar.update(chunk_size_before)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_number}: {str(e)}")
                continue
        
        pbar.close()
        elapsed_time = time.time() - start_time
        
        # Create indexes for common queries
        logger.info("Creating indexes...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_white ON chess_games(white)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_black ON chess_games(black)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON chess_games(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eco ON chess_games(eco)")
        
        conn.commit()
        
        # Log final statistics
        logger.info(f"\nConversion completed!")
        logger.info(f"Total rows processed: {total_rows_processed:,}")
        logger.info(f"Total rows kept: {total_rows_kept:,}")
        logger.info(f"Rows filtered out: {total_rows_processed - total_rows_kept:,}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Average speed: {total_rows_processed / elapsed_time:.2f} rows/second")
        logger.info(f"Database saved to: {db_path}")
        
        # Verify the conversion
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chess_games")
        db_count = cursor.fetchone()[0]
        logger.info(f"\nVerification: Database contains {db_count:,} games")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    # Define paths
    csv_path = "../../data/processed/lumbrasgigabase/lumbrasgigabase.csv"
    db_path = "../../data/processed/chess_games.db"
    log_dir = "../../logs"
    
    # Setup logging
    logger = setup_logging(log_dir)
    
    # Convert the CSV to SQLite database
    logger.info("Starting CSV to SQL conversion...")
    create_chess_database(csv_path, db_path, logger=logger)
    
    # Final database verification
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get distribution of ECO codes
    cursor.execute("""
        SELECT eco, COUNT(*) as count 
        FROM chess_games 
        WHERE eco IS NOT NULL 
        GROUP BY eco 
        ORDER BY count DESC 
        LIMIT 5
    """)
    
    logger.info("\nMost common ECO codes:")
    for eco, count in cursor.fetchall():
        logger.info(f"ECO {eco}: {count:,} games")
    
    conn.close()