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
    """Feature vector for clustering chess games"""
    
    #--------------------------------------------------------------------------
    # GAME STRUCTURE FEATURES
    #--------------------------------------------------------------------------
    # Game length and phase information
    total_moves: float = 0.0
    opening_length: float = 0.0         # Normalized proportion of game in opening
    middlegame_length: float = 0.0      # Normalized proportion of game in middlegame
    endgame_length: float = 0.0         # Normalized proportion of game in endgame
    
    # Development metrics
    white_minor_piece_development: float = 0.0  # Ratio of game completed when all minor pieces are developed (0-1)
    black_minor_piece_development: float = 0.0  # Ratio of game completed when all minor pieces are developed (0-1)
    white_queen_development: float = 0.0        # Ratio of game completed at first queen move (0-1)
    white_queen_lifetime: float = 0.0        # Ratio of game completed when queen is captured (0-1)
    black_queen_lifetime: float = 0.0        # Ratio of game completed when queen is captured (0-1)
    black_queen_development: float = 0.0        # Ratio of game completed at first queen move (0-1)
    white_castle_move: float = 0.0              # Ratio of game completed when castled (0 if never)
    black_castle_move: float = 0.0              # Ratio of game completed when castled (0 if never)

    #--------------------------------------------------------------------------
    # MATERIAL DYNAMICS FEATURES
    #--------------------------------------------------------------------------
    # Material changes
    white_material_changes: float = 0.0         # Overall material changes for White
    black_material_changes: float = 0.0         # Overall material changes for Black
    material_balance_std: float = 0.0           # Standard deviation of material balance


    #--------------------------------------------------------------------------
    # POSITIONAL FEATURES
    #--------------------------------------------------------------------------
    # Mobility metrics
    white_piece_mobility_avg: float = 0.0       # Average mobility for White pieces
    black_piece_mobility_avg: float = 0.0       # Average mobility for Black pieces
    
    # Pawn structure
    white_pawn_structure_changes: float = 0.0   # Changes in White's pawn structure
    black_pawn_structure_changes: float = 0.0   # Changes in Black's pawn structure
    white_pawn_control: float = 0.0             # Average squares controlled by White pawns
    black_pawn_control: float = 0.0             # Average squares controlled by Black pawns
    
    # Board control
    white_center_control_avg: float = 0.0       # Average center control by White
    black_center_control_avg: float = 0.0       # Average center control by Black
    white_space_advantage: float = 0.0          # Average control of Black's half by White
    black_space_advantage: float = 0.0          # Average control of White's half by Black

    #--------------------------------------------------------------------------
    # KING SAFETY FEATURES
    #--------------------------------------------------------------------------
    white_king_safety: float = 0.0              # Average king safety for White
    black_king_safety: float = 0.0              # Average king safety for Black
    white_vulnerability_spikes: float = 0.0     # Number of sudden safety drops for White
    black_vulnerability_spikes: float = 0.0     # Number of sudden safety drops for Black
    white_check_frequency: float = 0.0          # Ratio of checks to total white moves
    black_check_frequency: float = 0.0          # Ratio of checks to total black moves

    #--------------------------------------------------------------------------
    # CRITICAL DECISION FEATURES
    #--------------------------------------------------------------------------
    white_critical_performance: float = 0.0     # White's performance in critical positions
    black_critical_performance: float = 0.0     # Black's performance in critical positions
    white_weighted_alignment: float = 0.0       # White's alignment with engine (weighted by eval diff)
    black_weighted_alignment: float = 0.0       # Black's alignment with engine (weighted by eval diff)

    #--------------------------------------------------------------------------
    # MOVE QUALITY FEATURES
    #--------------------------------------------------------------------------
    # White move quality
    white_accuracy: float = 0.0                 # Overall accuracy for White
    white_opening_accuracy: float = 0.0         # Opening accuracy for White
    white_middlegame_accuracy: float = 0.0      # Middle game accuracy for White
    white_endgame_accuracy: float = 0.0         # Endgame accuracy for White
    white_avg_eval_change: float = 0.0          # Average evaluation change by White moves
    white_sacrifice_count: float = 0.0          # Count of sacrifices by White
    white_prophylactic_frequency: float = 0.0   # Preventative vs reactive moves ratio
    
    # Black move quality
    black_accuracy: float = 0.0                 # Overall accuracy for Black
    black_opening_accuracy: float = 0.0         # Opening accuracy for Black
    black_middlegame_accuracy: float = 0.0      # Middle game accuracy for Black
    black_endgame_accuracy: float = 0.0         # Endgame accuracy for Black
    black_avg_eval_change: float = 0.0          # Average evaluation change by Black moves
    black_sacrifice_count: float = 0.0          # Count of sacrifices by Black
    black_prophylactic_frequency: float = 0.0   # Preventative vs reactive moves ratio
    
    # White move classification counts
    white_brilliant_count: float = 0.0          # Count of brilliant moves by White
    white_great_count: float = 0.0              # Count of great moves by White
    white_good_moves: float = 0.0               # Count of good moves by White
    white_inaccuracy_count: float = 0.0         # Count of inaccuracies by White
    white_mistake_count: float = 0.0            # Count of mistakes by White
    white_blunder_count: float = 0.0            # Count of blunders by White
    
    # Black move classification counts
    black_brilliant_count: float = 0.0          # Count of brilliant moves by Black
    black_great_count: float = 0.0              # Count of great moves by Black
    black_good_moves: float = 0.0               # Count of good moves by Black
    black_inaccuracy_count: float = 0.0         # Count of inaccuracies by Black
    black_mistake_count: float = 0.0            # Count of mistakes by Black
    black_blunder_count: float = 0.0            # Count of blunders by Black

    #--------------------------------------------------------------------------
    # ENGINE ALIGNMENT FEATURES
    #--------------------------------------------------------------------------
    white_top_move_alignment: float = 0.0       # % of moves where White played top engine move
    black_top_move_alignment: float = 0.0       # % of moves where Black played top engine move
    white_top2_3_move_alignment: float = 0.0    # % of moves where White played 2nd/3rd best move
    black_top2_3_move_alignment: float = 0.0    # % of moves where Black played 2nd/3rd best move
    
    #--------------------------------------------------------------------------
    # STRATEGIC ORIENTATION FEATURES
    #--------------------------------------------------------------------------
    opening_novelty_score: float = 0.0          # Ratio of moves matching ECO theory to total opening moves (higher = more "in book")
    opening_name: str = ""                      # Name of the recognized chess opening (e.g., "Sicilian Defense: Najdorf Variation")
    white_sharpness: float = 0.0                # Cumulative sharpness score for white positions
    black_sharpness: float = 0.0                # Cumulative sharpness score for black positions
    
    def to_array(self) -> np.ndarray:
        return np.array(list(self.__dict__.values()), dtype=np.float32)
