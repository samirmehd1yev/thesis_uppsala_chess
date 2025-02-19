"""
Highly optimized PGN processing script with improved performance.
Key optimizations:
- Uses memory-mapped IO with buffered reading
- Implements concurrent processing with process pools
- Optimizes regex patterns and string operations
- Uses batch processing for CSV writing
- Implements caching for repeated operations
- Now saves pure PGN, separate move text, clock_info, and eval_info where available
"""

import os
import csv
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import chess.pgn  # if needed by downstream code
from tqdm import tqdm
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing as mp
from itertools import islice
import mmap
import io
import pickle
import signal
import sys
from functools import lru_cache
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import tempfile
import shutil
import pandas as pd

# Project structure setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
INPUT_DIR = DATA_DIR / "raw_pgn"
BASE_OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_OUTPUT_DIR / "lumbrasgigabase"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
INTERIM_DIR = DATA_DIR / "interim"

# Create necessary directories
for directory in [LOGS_DIR, OUTPUT_DIR, INDIVIDUAL_DIR, INTERIM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Output files
FINAL_OUTPUT = OUTPUT_DIR / "lumbrasgigabase.csv"
if FINAL_OUTPUT.exists():
    FINAL_OUTPUT.unlink()

# Optimized constants
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for better memory efficiency
NUM_PROCESSES = max(1, mp.cpu_count())  # Use all available CPUs
BATCH_SIZE = 5000  # Increased batch size for better I/O performance

# Pre-compile regex patterns for better performance
EVAL_PATTERN = re.compile(rb'\[%eval ([^\]]+)\]')
CLK_PATTERN = re.compile(rb'\[%clk ([^\]]+)\]')
GAME_START_PATTERN = re.compile(rb'\[Event "')
HEADERS_PATTERN = re.compile(rb'\[(.*?) "(.*?)"\]')

# Initialize logging with a single file
LOG_FILE = LOGS_DIR / "pgn_processing.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

@dataclass
class ChessGame:
    """Optimized data structure for chess game information.
       Now includes:
         - pgn: The complete original PGN text.
         - moves: The cleaned moves (only moves).
         - eval_info: JSON-formatted evaluation info.
         - clock_info: JSON-formatted clock info.
    """
    event: str = field(default="?")
    site: str = field(default="")
    date: str = field(default="")
    round: str = field(default="?")
    white: str = field(default="")
    black: str = field(default="")
    result: str = field(default="")
    white_elo: str = field(default="")
    black_elo: str = field(default="")
    white_title: str = field(default="")
    black_title: str = field(default="")
    eco: str = field(default="")
    opening: str = field(default="")
    time_control: str = field(default="")
    import_date: str = field(default="")
    source: str = field(default="LumbrasGigaBase")  # Default source
    moves: str = field(default="")        # Cleaned moves (only moves)
    eval_info: Optional[str] = field(default=None)   # JSON string for evaluation info
    clock_info: Optional[str] = field(default=None)  # JSON string for clock info
    pgn: str = field(default="")          # Pure original PGN text

    def clean_string(self, s: str) -> str:
        """Clean string values."""
        if not s or s.isspace():
            return ""
        return s.strip().replace('\n', ' ').replace('\r', '')

    def clean_moves(self, moves: str) -> str:
        """Clean moves string."""
        # Remove extra whitespace and normalize
        moves = ' '.join(moves.split())
        # Ensure single space after move numbers
        moves = re.sub(r'(\d+)\.\s+', r'\1. ', moves)
        return moves

    def to_dict(self) -> Dict:
        """Fast conversion to dictionary with cleaned data."""
        return {
            "event": self.clean_string(self.event),
            "site": self.clean_string(self.site),
            "date": self.clean_string(self.date),
            "round": self.clean_string(self.round),
            "white": self.clean_string(self.white),
            "black": self.clean_string(self.black),
            "result": self.clean_string(self.result),
            "white_elo": self.clean_string(self.white_elo),
            "black_elo": self.clean_string(self.black_elo),
            "white_title": self.clean_string(self.white_title),
            "black_title": self.clean_string(self.black_title),
            "eco": self.clean_string(self.eco),
            "opening": self.clean_string(self.opening),
            "time_control": self.clean_string(self.time_control),
            "import_date": self.clean_string(self.import_date),
            "source": self.source,
            "moves": self.clean_moves(self.moves),
            "eval_info": self.eval_info,
            "clock_info": self.clock_info,
            "pgn": self.clean_string(self.pgn)
        }

class FastPGNParser:
    """Optimized PGN parser using raw byte operations."""
    def __init__(self):
        self.headers_cache = {}
    
    @lru_cache(maxsize=1024)
    def _decode_bytes(self, byte_str: bytes) -> str:
        """Cached byte decoding."""
        return byte_str.decode('utf-8', errors='replace')
    
    def parse_game(self, raw_game: bytes) -> Optional[ChessGame]:
        """Parse a single game from raw bytes.
           Saves the full PGN text, separates the moves, evaluation info, and clock info.
        """
        try:
            # Decode the entire PGN text and store it as the pure PGN.
            full_pgn = self._decode_bytes(raw_game).strip()
            
            headers = {}
            for match in HEADERS_PATTERN.finditer(raw_game):
                key = self._decode_bytes(match.group(1))
                value = self._decode_bytes(match.group(2))
                value = value.replace('\\"', '"').strip()
                if value == "?":
                    value = ""
                headers[key] = value
            
            # Determine where the moves start. Assumes moves start after a double newline.
            moves_start = raw_game.find(b'\n\n') + 2
            moves_text = self._decode_bytes(raw_game[moves_start:]).strip()
            moves_text = ' '.join(moves_text.split())
            moves_text = re.sub(r'\s+', ' ', moves_text)
            
            # Extract evaluation information into a list
            eval_list = [self._decode_bytes(match.group(1)) 
                         for match in EVAL_PATTERN.finditer(raw_game)]
            
            # Extract clock information into a list
            clk_list = [self._decode_bytes(match.group(1))
                        for match in CLK_PATTERN.finditer(raw_game)]
            
            # Convert the lists into JSON strings if they exist
            eval_json = json.dumps(eval_list, separators=(',', ':')) if eval_list else None
            clk_json = json.dumps(clk_list, separators=(',', ':')) if clk_list else None
            
            game = ChessGame(
                event=headers.get("Event", ""),
                site=headers.get("Site", ""),
                date=headers.get("Date", ""),
                round=headers.get("Round", ""),
                white=headers.get("White", ""),
                black=headers.get("Black", ""),
                result=headers.get("Result", ""),
                white_elo=headers.get("WhiteElo", ""),
                black_elo=headers.get("BlackElo", ""),
                white_title=headers.get("WhiteTitle", ""),
                black_title=headers.get("BlackTitle", ""),
                eco=headers.get("ECO", ""),
                opening=headers.get("Opening", ""),
                time_control=headers.get("TimeControl", ""),
                import_date=headers.get("ImportDate", ""),
                moves=moves_text,
                eval_info=eval_json,
                clock_info=clk_json,
                pgn=full_pgn
            )
            return game
        
        except Exception as e:
            logging.error(f"Error parsing game: {e}")
            return None

class BatchCSVWriter:
    """Optimized CSV writer with batching."""
    def __init__(self, filename: Path, fieldnames: List[str], batch_size: int = BATCH_SIZE):
        self.filename = filename
        self.fieldnames = fieldnames
        self.batch_size = batch_size
        self.buffer = []
        self.lock = threading.Lock()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.batch_files = []
    
    def add_game(self, game: Dict):
        """Add a game to the buffer."""
        self.buffer.append(game)
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Write buffered games to temporary file."""
        with self.lock:
            if not self.buffer:
                return
            
            temp_file = self.temp_dir / f"batch_{len(self.batch_files)}.csv"
            with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.buffer)
            
            self.batch_files.append(temp_file)
            self.buffer.clear()
    
    def finalize(self):
        """Combine all batch files into final output."""
        self.flush()  # Write any remaining games
        
        # Write or append to the final output file
        mode = 'a' if self.filename.exists() else 'w'
        with open(self.filename, mode, newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=self.fieldnames)
            if mode == 'w':
                writer.writeheader()
            
            for batch_file in self.batch_files:
                with open(batch_file, 'r', encoding='utf-8') as infile:
                    next(infile)  # Skip header
                    for line in infile:
                        outfile.write(line)
        
        # Cleanup temporary files
        for batch_file in self.batch_files:
            batch_file.unlink()
        self.temp_dir.rmdir()

class GameProcessor:
    """Processes games in parallel."""
    def __init__(self):
        self.parser = FastPGNParser()
    
    def process_chunk(self, chunk: bytes) -> List[Dict]:
        """Process a chunk of games."""
        games = []
        game_start = 0
        
        while True:
            # Find the start of the next game
            game_end = chunk.find(b'[Event "', game_start + 8)
            if game_end == -1:
                game_end = len(chunk)
            
            if game_start >= game_end:
                break
            
            game_bytes = chunk[game_start:game_end]
            game = self.parser.parse_game(game_bytes)
            if game:
                games.append(game.to_dict())
            
            game_start = game_end
            if game_start >= len(chunk):
                break
        
        return games

def process_file(input_file: Path) -> int:
    """Process a single PGN file."""
    processor = GameProcessor()
    
    # Create writer for individual output CSV
    individual_output = INDIVIDUAL_DIR / f"{input_file.stem.lower().replace(' ', '_')}.csv"
    # Use the keys from ChessGame.to_dict() as fieldnames (includes pgn, clock_info, eval_info, etc.)
    fieldnames = list(ChessGame().__dict__.keys())
    
    individual_writer = BatchCSVWriter(individual_output, fieldnames)
    
    total_games = 0
    
    with open(input_file, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
                futures = []
                chunk_start = 0
                
                while chunk_start < mm.size():
                    chunk_end = min(chunk_start + CHUNK_SIZE, mm.size())
                    chunk = mm.read(CHUNK_SIZE)
                    
                    if not chunk:
                        break
                    
                    futures.append(executor.submit(processor.process_chunk, chunk))
                    chunk_start = chunk_end
                
                with tqdm(total=len(futures), desc=f"Processing {input_file.name}") as pbar:
                    for future in as_completed(futures):
                        try:
                            games = future.result()
                            for game in games:
                                individual_writer.add_game(game)
                                total_games += 1
                        except Exception as e:
                            logging.error(f"Error processing chunk: {e}")
                        finally:
                            pbar.update(1)
    
    # Finalize the writer (combine temporary batch files)
    individual_writer.finalize()
    return total_games

def merge_all_csvs():
    """Merge all individual CSV files into final combined output."""
    logging.info("Merging all individual CSV files into final output...")
    
    # Get all CSV files from the individual directory
    csv_files = list(INDIVIDUAL_DIR.glob('*.csv'))
    if not csv_files:
        logging.warning("No CSV files found to merge")
        return
    
    # Initialize final output with header from the first file
    first_file = csv_files[0]
    with open(first_file, 'r', newline='', encoding='utf-8') as infile:
        header = infile.readline()
    
    with open(FINAL_OUTPUT, 'w', newline='', encoding='utf-8') as outfile:
        outfile.write(header)
        
        # Merge all files
        with tqdm(total=len(csv_files), desc="Merging files") as pbar:
            for file in csv_files:
                with open(file, 'r', newline='', encoding='utf-8') as infile:
                    next(infile)  # Skip header
                    for line in infile:
                        outfile.write(line)
                pbar.update(1)
    
    logging.info(f"Successfully merged all files into {FINAL_OUTPUT}")

def main():
    """Main processing function with optimized file handling."""
    try:
        start_time = datetime.now()
        logging.info(f"Starting processing at {start_time}")
        
        # Reset final output if it exists
        if FINAL_OUTPUT.exists():
            FINAL_OUTPUT.unlink()
        
        total_games = 0
        input_files = sorted(
            INPUT_DIR.glob('*.pgn'),
            key=lambda x: int(re.search(r'\d+', x.stem).group()) if re.search(r'\d+', x.stem) else 0
        )
        
        for input_file in input_files:
            logging.info(f"Processing file: {input_file.name}")
            games_processed = process_file(input_file)
            total_games += games_processed
            logging.info(f"Processed {games_processed} games from {input_file.name}")
        
        # Merge all individual CSV files into the final output
        merge_all_csvs()
        
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        logging.info(f"\nProcessing completed at {end_time}")
        logging.info(f"Total processing time: {processing_time}")
        logging.info(f"Total games processed: {total_games}")
        logging.info(f"Final output saved to: {FINAL_OUTPUT}")
        
        # head of merged CSV
        df = pd.read_csv(FINAL_OUTPUT)
        print(df.head())
        
        
    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
