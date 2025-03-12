#!/usr/bin/env python3
"""
Download ECO Database

This script downloads the ECO database from Hugging Face's Lichess/chess-openings dataset
and saves it as a parquet file for offline use.
"""

import os
import sys
import argparse
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_eco_database(output_dir, force=False):
    """
    Download the ECO database from Hugging Face and save it locally.
    
    Args:
        output_dir: Directory to save the database
        force: Whether to force download even if the file exists
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to output file
    output_file = os.path.join(output_dir, "eco_database.parquet")
    
    # Check if file already exists
    if os.path.exists(output_file) and not force:
        logger.info(f"ECO database already exists at {output_file}, skipping download")
        return output_file
    
    # Try different methods to download the dataset
    try:
        logger.info("Attempting to download ECO database using datasets library...")
        try:
            # Method 1: Using Hugging Face datasets library
            from datasets import load_dataset
            dataset = load_dataset("Lichess/chess-openings", split="train")
            df = dataset.to_pandas()
            logger.info("Successfully downloaded ECO database using datasets library")
        except Exception as e:
            logger.warning(f"Failed to download using datasets library: {e}")
            logger.info("Attempting to download using pandas...")
            
            try:
                # Method 2: Using pandas with Hugging Face filesystem
                df = pd.read_parquet("hf://datasets/Lichess/chess-openings/data/train-00000-of-00001.parquet")
                logger.info("Successfully downloaded ECO database using pandas")
            except Exception as e:
                logger.warning(f"Failed to download using pandas: {e}")
                logger.info("Attempting direct download via HTTPS...")
                
                # Method 3: Using direct HTTPS request
                import requests
                url = "https://huggingface.co/datasets/Lichess/chess-openings/resolve/main/data/train-00000-of-00001.parquet"
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Save the downloaded file
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Successfully downloaded ECO database via HTTPS to {output_file}")
                    
                    # Read the file to verify it
                    df = pd.read_parquet(output_file)
                else:
                    raise Exception(f"Failed to download ECO database: HTTP {response.status_code}")
        
        # Rename column if necessary
        if 'eco-volume' in df.columns:
            df = df.rename(columns={'eco-volume': 'eco_volume'})
        
        # Write to output file if not already written
        if not os.path.exists(output_file) or force:
            df.to_parquet(output_file, index=False)
            logger.info(f"Saved ECO database to {output_file}")
        
        # Log some statistics
        logger.info(f"ECO database contains {len(df)} entries")
        eco_counts = df['eco'].str[0].value_counts().to_dict()
        logger.info(f"ECO code distribution by volume: {eco_counts}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to download ECO database: {e}")
        return None

def create_sample_database(output_dir):
    """
    Create a small sample ECO database for testing purposes.
    
    Args:
        output_dir: Directory to save the sample database
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to output file
    output_file = os.path.join(output_dir, "sample_eco_database.parquet")
    
    # Sample data
    sample_data = [
        {
            'eco_volume': 'A', 
            'eco': 'A00', 
            'name': 'Grob Opening', 
            'pgn': '1. g4', 
            'uci': 'g2g4',
            'epd': 'rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq -'
        },
        {
            'eco_volume': 'B', 
            'eco': 'B20', 
            'name': 'Sicilian Defense', 
            'pgn': '1. e4 c5', 
            'uci': 'e2e4 c7c5',
            'epd': 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -'
        },
        {
            'eco_volume': 'C', 
            'eco': 'C20', 
            'name': "King's Pawn Game", 
            'pgn': '1. e4 e5', 
            'uci': 'e2e4 e7e5',
            'epd': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -'
        },
        {
            'eco_volume': 'D', 
            'eco': 'D00', 
            'name': "Queen's Pawn Game", 
            'pgn': '1. d4 d5', 
            'uci': 'd2d4 d7d5',
            'epd': 'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -'
        },
        {
            'eco_volume': 'E', 
            'eco': 'E00', 
            'name': 'Indian Game', 
            'pgn': '1. d4 Nf6 2. c4', 
            'uci': 'd2d4 g8f6 c2c4',
            'epd': 'rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq -'
        }
    ]
    
    # Create DataFrame and save it
    df = pd.DataFrame(sample_data)
    df.to_parquet(output_file, index=False)
    
    logger.info(f"Created sample ECO database with {len(df)} entries at {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Download ECO database from Hugging Face')
    parser.add_argument('--output-dir', type=str, default='../../data/eco',
                        help='Directory to save the ECO database')
    parser.add_argument('--force', action='store_true',
                        help='Force download even if the file exists')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create a small sample database for testing')
    
    args = parser.parse_args()
    
    # Get absolute path to output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    
    if args.create_sample:
        create_sample_database(output_dir)
    else:
        download_eco_database(output_dir, args.force)

if __name__ == '__main__':
    main() 