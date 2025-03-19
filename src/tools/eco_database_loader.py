#!/usr/bin/env python3
"""
ECO Database Loader

This module provides functionality to download and load the ECO database from
Hugging Face's Lichess/chess-openings dataset.
"""

import os
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECODatabaseLoader:
    """
    A class for loading and accessing the Encyclopaedia of Chess Openings (ECO) database.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ECO database loader.
        
        Args:
            cache_dir: Directory to cache the downloaded dataset
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "../../data/eco")
        self.eco_data = None
        
    def load_database(self, force_download: bool = False) -> List[Dict[str, Any]]:
        """
        Load the ECO database from Hugging Face or local cache.
        
        Args:
            force_download: If True, force a new download even if cached data exists
            
        Returns:
            List of dictionaries containing ECO data
        """
        # Check if we've already loaded the data
        if self.eco_data is not None:
            return self.eco_data
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Path to cached parquet file
        cached_file = os.path.join(self.cache_dir, "eco_database.parquet")
        
        # Try to load from cache first
        if os.path.exists(cached_file) and not force_download:
            try:
                logger.debug(f"Loading ECO database from cache: {cached_file}")
                self.eco_data = pd.read_parquet(cached_file).to_dict(orient='records')
                
                # Log information about the loaded database
                # self._log_database_info()
                
                return self.eco_data
            except Exception as e:
                logger.warning(f"Error loading from cache, will download: {e}")
        
        # Download from Hugging Face
        try:
            logger.debug("Downloading ECO database from Hugging Face...")
            
            try:
                # First try using the datasets library
                from datasets import load_dataset
                dataset = load_dataset("Lichess/chess-openings", split="train")
                df = dataset.to_pandas()
                logger.debug("Successfully loaded using datasets library")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to load using datasets library: {e}")
                logger.debug("Trying with pandas...")
                
                try:
                    # Fall back to pandas
                    df = pd.read_parquet("hf://datasets/Lichess/chess-openings/data/train-00000-of-00001.parquet")
                    logger.debug("Successfully loaded using pandas")
                except Exception as e:
                    logger.warning(f"Failed to load using pandas: {e}")
                    
                    # If all else fails, try a direct HTTPS request
                    logger.debug("Trying direct download...")
                    import requests
                    url = "https://huggingface.co/datasets/Lichess/chess-openings/resolve/main/data/train-00000-of-00001.parquet"
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(cached_file, 'wb') as f:
                            f.write(response.content)
                        df = pd.read_parquet(cached_file)
                        logger.debug("Successfully downloaded via HTTPS")
                    else:
                        raise Exception(f"Failed to download: HTTP {response.status_code}")
            
            # Rename columns to match our expected format
            if 'eco-volume' in df.columns:
                df = df.rename(columns={'eco-volume': 'eco_volume'})
            
            # Save to cache
            df.to_parquet(cached_file, index=False)
            logger.debug(f"Saved ECO database to cache: {cached_file}")
            
            # Convert to list of dictionaries
            self.eco_data = df.to_dict(orient='records')
            
            # Log information about the loaded database
            # self._log_database_info()
            
            return self.eco_data
            
        except Exception as e:
            logger.error(f"Error downloading ECO database: {e}")
            
            # Fall back to a small sample dataset if available
            sample_file = os.path.join(self.cache_dir, "sample_eco_database.parquet")
            if os.path.exists(sample_file):
                logger.debug(f"Using sample ECO database: {sample_file}")
                self.eco_data = pd.read_parquet(sample_file).to_dict(orient='records')
                
                # Log information about the loaded sample database
                # self._log_database_info(is_sample=True)
                
                return self.eco_data
            
            # Return empty list if all else fails
            logger.warning("Returning empty ECO database")
            self.eco_data = []
            return self.eco_data
    
    def _log_database_info(self, is_sample=False):
        """Log information about the loaded database"""
        if not self.eco_data:
            logger.warning("ECO database is empty")
            return
            
        # Count entries by volume
        volumes = {}
        for entry in self.eco_data:
            vol = entry.get('eco', '')[0] if entry.get('eco') else 'unknown'
            volumes[vol] = volumes.get(vol, 0) + 1
            
        # Log general information
        db_type = "sample" if is_sample else "full"
        logger.debug(f"Loaded {db_type} ECO database with {len(self.eco_data)} entries")
        logger.debug(f"Volume distribution: {volumes}")
        
        # Check for specific ECO codes
        specific_codes = ['B97', 'B00', 'E00', 'A00', 'D00', 'C00']
        for code in specific_codes:
            entries = [e for e in self.eco_data if e.get('eco') == code]
            if entries:
                logger.debug(f"Found {len(entries)} entries for ECO code {code}")
                for e in entries[:2]:  # Log first two entries
                    moves_count = len(e.get('uci', '').split())
                    logger.debug(f"  - {e.get('name', 'No name')}: {moves_count} moves")
    
    def get_theoretical_moves(self, eco_code: str) -> List[str]:
        """
        Get theoretical moves for a specific ECO code.
        
        Args:
            eco_code: The ECO code to look up (always uses first 3 characters)
            
        Returns:
            List of UCI notation moves for the opening
        """
        if self.eco_data is None:
            self.load_database()
        
        if not eco_code:
            logger.warning("Empty ECO code provided")
            return []
        
        # Always use the standard 3-character ECO code
        standard_eco = eco_code[:3] if eco_code and len(eco_code) >= 3 else eco_code
        
        # Log info about the lookup
        logger.debug(f"Looking up theoretical moves for ECO code: {standard_eco} (original: {eco_code})")
        
        # Look for exact match using standard ECO code
        matches = []
        for entry in self.eco_data:
            if entry.get('eco') == standard_eco:
                matches.append(entry)
        
        if matches:
            logger.debug(f"Found {len(matches)} entries for ECO code {standard_eco}")
            for i, match in enumerate(matches[:2]):  # Log first 2 matches
                moves = match.get('uci', '').split()
                logger.debug(f"  - {match.get('name')}: {len(moves)} moves")
            
            # Return the moves from the first match
            return matches[0].get('uci', '').split()
        
        # If no exact match, look for entries with the same volume
        volume = standard_eco[0] if standard_eco else ''
        logger.debug(f"No match for {standard_eco}, trying volume: {volume}")
        
        volume_matches = []
        for entry in self.eco_data:
            entry_volume = entry.get('eco_volume') or entry.get('eco-volume') or entry.get('eco', '')[0]
            if entry_volume == volume:
                volume_matches.append(entry)
        
        if volume_matches:
            logger.debug(f"Found {len(volume_matches)} entries for volume {volume}")
            # Return the moves from the first match
            return volume_matches[0].get('uci', '').split()
        
        logger.warning(f"No theoretical moves found for ECO code: {standard_eco}")
        return []
    
    def get_opening_name(self, eco_code: str) -> str:
        """
        Get the opening name for a specific ECO code.
        
        Args:
            eco_code: The ECO code to look up (always uses first 3 characters)
            
        Returns:
            Name of the opening
        """
        if self.eco_data is None:
            self.load_database()
        
        if not eco_code:
            logger.warning("Empty ECO code provided")
            return "Unknown"
        
        # Always use the standard 3-character ECO code
        standard_eco = eco_code[:3] if eco_code and len(eco_code) >= 3 else eco_code
        
        # Log info about the lookup
        logger.debug(f"Looking up opening name for ECO code: {standard_eco} (original: {eco_code})")
        
        # Look for exact match using standard ECO code
        matches = []
        for entry in self.eco_data:
            if entry.get('eco') == standard_eco:
                matches.append(entry)
        
        if matches:
            logger.debug(f"Found {len(matches)} entries for ECO code {standard_eco}")
            return matches[0].get('name', 'Unknown')
        
        # If no exact match, look for entries with the same volume
        volume = standard_eco[0] if standard_eco else ''
        logger.debug(f"No match for {standard_eco}, trying volume: {volume}")
        
        volume_matches = []
        for entry in self.eco_data:
            entry_volume = entry.get('eco_volume') or entry.get('eco-volume') or entry.get('eco', '')[0]
            if entry_volume == volume:
                volume_matches.append(entry)
        
        if volume_matches:
            logger.debug(f"Found {len(volume_matches)} entries for volume {volume}")
            return volume_matches[0].get('name', f"Unknown Opening ({volume}-family)")
        
        return f"Unknown Opening (ECO: {standard_eco})"
    
    def find_closest_opening(self, moves_uci: List[str]) -> Dict[str, Any]:
        """
        Find the closest opening match for a sequence of UCI moves.
        
        Args:
            moves_uci: List of UCI moves
            
        Returns:
            Dictionary with matched opening information including:
            - eco: ECO code
            - name: Opening name
            - uci: UCI moves of the opening
            - matching_moves: Number of moves that matched
            - total_moves: Total moves in the opening
        """
        if self.eco_data is None:
            self.load_database()
        
        best_match = None
        best_match_length = 0
        
        for entry in self.eco_data:
            entry_uci = entry.get('uci', '').split()
            
            # Count matching moves
            match_length = 0
            for i in range(min(len(moves_uci), len(entry_uci))):
                if moves_uci[i] == entry_uci[i]:
                    match_length += 1
                else:
                    break
            
            # Update best match if this one is better
            if match_length > best_match_length:
                best_match_length = match_length
                best_match = {
                    'eco': entry.get('eco', ''),
                    'name': entry.get('name', ''),
                    'uci': entry_uci,
                    'matching_moves': match_length,
                    'total_moves': len(entry_uci)
                }
        
        # If no match found, return default values
        if best_match is None:
            return {
                'eco': '',
                'name': 'Unknown Opening',
                'uci': [],
                'matching_moves': 0,
                'total_moves': 0
            }
        
        return best_match

# Create a singleton instance for easy access
eco_loader = ECODatabaseLoader()

if __name__ == "__main__":
    # Test the loader
    loader = ECODatabaseLoader()
    eco_data = loader.load_database()
    print(f"Loaded {len(eco_data)} ECO entries")
    
    # Example: Get theoretical moves for Sicilian Defense
    sicilian_moves = loader.get_theoretical_moves("B20")
    print(f"Sicilian Defense moves: {sicilian_moves}")
    
    # Example: Find closest opening for a sequence of moves
    test_moves = ["e2e4", "c7c5", "g1f3", "d7d6"]
    opening = loader.find_closest_opening(test_moves)
    print(f"Closest opening for {test_moves}: {opening['name']} (ECO: {opening['eco']})")
    print(f"Matched {opening['matching_moves']}/{opening['total_moves']} moves") 