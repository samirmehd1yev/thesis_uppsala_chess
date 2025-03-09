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
        variation (List[Dict], optional): The list of best moves with their evaluations
        wdl (Optional[Dict]): Win-Draw-Loss probabilities (if available)
        multipv (Optional[List[Dict]]): Multiple principal variations with scores
    """
    ply: int
    eval: dict  # Stockfish evaluation
    variation: List[str] = None
    wdl: Optional[Dict[str, float]] = None
    multipv: Optional[List[Dict]] = None

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

    def get_best_move(self) -> Optional[str]:
        """Get the best move in UCI format"""
        if self.variation and len(self.variation) > 0:
            return self.variation[0].get("Move")
        return None

    def get_move_eval(self, move_idx: int) -> Optional[int]:
        """Get evaluation for a specific move variation"""
        if self.variation and len(self.variation) > move_idx:
            move_info = self.variation[move_idx]
            if "Centipawn" in move_info:
                return move_info["Centipawn"]
            elif "Mate" in move_info:
                # Convert mate score to high centipawn value
                mate_in = move_info["Mate"]
                return 10000 if mate_in > 0 else -10000
        return None

@dataclass
class FeatureVector:
    """Feature vector for clustering"""
    # Game Phase Features
    total_moves: float = 0.0
    opening_length: float = 0.0
    middlegame_length: float = 0.0
    endgame_length: float = 0.0
    
    # Material/Position Features - White
    white_material_changes: float = 0.0
    white_piece_mobility_avg: float = 0.0
    white_pawn_structure_changes: float = 0.0
    white_center_control_avg: float = 0.0
    
    # Material/Position Features - Black
    black_material_changes: float = 0.0
    black_piece_mobility_avg: float = 0.0
    black_pawn_structure_changes: float = 0.0
    black_center_control_avg: float = 0.0
    
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
    white_accuracy: float = 0.0         # New: Overall accuracy for White
    
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
    black_accuracy: float = 0.0         # New: Overall accuracy for Black
    
    # King Safety Features - New
    white_king_safety: float = 0.0      # Average king safety for White
    black_king_safety: float = 0.0      # Average king safety for Black
    white_king_safety_min: float = 0.0  # Minimum king safety for White
    black_king_safety_min: float = 0.0  # Minimum king safety for Black
    white_vulnerability_spikes: float = 0.0  # Number of sudden safety drops for White
    black_vulnerability_spikes: float = 0.0  # Number of sudden safety drops for Black
    
    def to_array(self) -> np.ndarray:
        return np.array(list(self.__dict__.values()), dtype=np.float32)