"""
Chess Feature Extraction Module
This module handles the extraction of chess game features for player style analysis.
"""

import chess
import chess.pgn
import io
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
import sqlite3
from tqdm import tqdm

class ChessFeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor"""
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
    
    def extract_features_from_game(self, pgn_text):
        """Extract features from a single game"""
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if not game:
                return None
            
            features = {
                'white_player': game.headers.get('White', ''),
                'black_player': game.headers.get('Black', ''),
                'result': game.headers.get('Result', ''),
                'total_moves': 0,
                'captures': 0,
                'checks': 0,
                'pawn_moves': 0,
                'piece_moves': 0,
                'center_moves': 0,
                'material_imbalances': [],
                'piece_mobility': [],
                'center_control': []
            }
            
            board = game.board()
            center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
            
            for move in game.mainline_moves():
                features['total_moves'] += 1
                
                if board.piece_at(move.from_square):
                    if board.piece_at(move.from_square).piece_type == chess.PAWN:
                        features['pawn_moves'] += 1
                    else:
                        features['piece_moves'] += 1
                
                if move.to_square in center_squares:
                    features['center_moves'] += 1
                
                if board.is_capture(move):
                    features['captures'] += 1
                
                material = sum(len(board.pieces(piece_type, True)) * value - 
                             len(board.pieces(piece_type, False)) * value 
                             for piece_type, value in self.piece_values.items())
                features['material_imbalances'].append(material)
                
                features['piece_mobility'].append(len(list(board.legal_moves)))
                
                center_control = sum(1 for sq in center_squares if board.is_attacked_by(True, sq)) - \
                               sum(1 for sq in center_squares if board.is_attacked_by(False, sq))
                features['center_control'].append(center_control)
                
                board.push(move)
                
                if board.is_check():
                    features['checks'] += 1
            
            return self._calculate_aggregate_features(features)
            
        except Exception as e:
            logging.error(f"Error processing game: {str(e)}")
            return None
    
    def _calculate_aggregate_features(self, features):
        """Calculate aggregate features from raw game features"""
        return {
            'white_player': features['white_player'],
            'black_player': features['black_player'],
            'result': features['result'],
            'total_moves': features['total_moves'],
            'captures_per_move': features['captures'] / max(features['total_moves'], 1),
            'checks_per_move': features['checks'] / max(features['total_moves'], 1),
            'pawn_move_ratio': features['pawn_moves'] / max(features['total_moves'], 1),
            'center_move_ratio': features['center_moves'] / max(features['total_moves'], 1),
            'avg_material_imbalance': np.mean(features['material_imbalances']),
            'material_volatility': np.std(features['material_imbalances']),
            'avg_mobility': np.mean(features['piece_mobility']),
            'mobility_volatility': np.std(features['piece_mobility']),
            'avg_center_control': np.mean(features['center_control']),
            'center_control_volatility': np.std(features['center_control'])
        }