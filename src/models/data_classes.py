# src/models/data_classes.py
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

@dataclass
class Info:
    """
    Holds evaluation information for a given move/position.
    Attributes:
        ply (int): The ply number (half-move count) in the game
        eval (dict): The evaluation dictionary from Stockfish
        variation (List[str], optional): The list of best moves
    """
    ply: int
    eval: dict  # Stockfish evaluation
    variation: List[str] = None

    @property
    def color(self) -> bool:
        """True if White is to move (even ply), False if Black"""
        return self.ply % 2 == 0

    @property
    def cp(self) -> Optional[int]:
        """Get centipawn evaluation if available"""
        return self.eval["value"] if self.eval["type"] == "cp" else None

    @property
    def mate(self) -> Optional[int]:
        """Get mate evaluation if available"""
        return self.eval["value"] if self.eval["type"] == "mate" else None

    def eval_comment(self) -> Optional[str]:
        """Generate human-readable evaluation comment"""
        if self.mate is not None:
            return f"#{self.mate}"
        elif self.cp is not None:
            return f"{self.cp/100:+.1f}"
        return None

@dataclass
class FeatureVector:
    """Feature vector for clustering"""
    # Game Phase Features
    total_moves: float = 0.0
    opening_length: float = 0.0
    middlegame_length: float = 0.0
    endgame_length: float = 0.0
    
    # Material/Position Features
    material_balance_changes: float = 0.0
    piece_mobility_avg: float = 0.0
    pawn_structure_changes: float = 0.0
    
    # Move Quality Features - White
    white_brilliant_count: float = 0.0  # New
    white_great_count: float = 0.0      # New
    white_good_moves: float = 0.0
    white_inaccuracy_count: float = 0.0
    white_mistake_count: float = 0.0
    white_blunder_count: float = 0.0
    white_avg_eval_change: float = 0.0
    white_eval_volatility: float = 0.0
    white_sacrifice_count: float = 0.0  # New: Count of sacrifices by White
    
    # Move Quality Features - Black
    black_brilliant_count: float = 0.0  # New
    black_great_count: float = 0.0      # New
    black_good_moves: float = 0.0
    black_inaccuracy_count: float = 0.0
    black_mistake_count: float = 0.0
    black_blunder_count: float = 0.0
    black_avg_eval_change: float = 0.0
    black_eval_volatility: float = 0.0
    black_sacrifice_count: float = 0.0  # New: Count of sacrifices by Black
    
    # Statistical Features
    center_control_avg: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array(list(self.__dict__.values()), dtype=np.float32)