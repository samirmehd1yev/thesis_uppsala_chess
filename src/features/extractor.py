# src/features/extractor.py
import chess
import chess.pgn
from typing import Dict, List, Optional, Tuple
import numpy as np
from models.data_classes import FeatureVector, Info
from models.enums import Judgment
from analysis.phase_detector import GamePhaseDetector
from analysis.move_analyzer import MoveAnalyzer

class FeatureExtractor:
    def __init__(self):
        self.phase_detector = GamePhaseDetector()
        
    def extract_features(self, game: chess.pgn.Game, evals: Optional[List[Info]] = None, judgments: Optional[List[Judgment]] = None) -> FeatureVector:
        """
        Extract all features from a game
        
        Args:
            game: Chess game to analyze
            evals: Optional list of position evaluations
            judgments: Optional pre-calculated move judgments
            
        Returns:
            FeatureVector containing extracted features
        """
        if game is None:
            raise ValueError("Game cannot be None")
            
        if evals is not None and not isinstance(evals, list):
            raise TypeError("evals must be a list of Info objects")
            
        features = FeatureVector()
        
        # Get positions and phases
        positions = self._get_positions(game)
        mg_start, eg_start = self.phase_detector.find_phase_transitions(positions)
        total_moves = len(positions) // 2
        features.total_moves = total_moves
        
        opening_length, middlegame_length, endgame_length = self._phase_length(mg_start, eg_start, total_moves)
        features.opening_length = opening_length
        features.middlegame_length = middlegame_length
        features.endgame_length = endgame_length
        
        # Material and position features
        features.material_balance_changes = self._calculate_material_changes(positions)
        features.piece_mobility_avg = self._calculate_mobility(positions)
        features.pawn_structure_changes = self._calculate_pawn_changes(positions)
        features.center_control_avg = self._calculate_center_control(positions)
        
        # Calculate quality metrics if evaluations available
        if evals and len(evals) > 1:
            # Get the actual moves played
            moves = list(game.mainline_moves())
            
            # Use pre-calculated judgments if provided
            if judgments and len(judgments) > 0:
                self._count_judgment_metrics(judgments, features)
                # Still count sacrifices if we have positions and moves
                if positions and moves:
                    self.count_sacrifices(positions, moves, features)
            else:
                # Otherwise calculate move qualities with board information for brilliant/great detection
                self._calculate_quality_metrics(evals, features, positions, moves)
        
        return features
    
    def _phase_length(self, mg_start: int, eg_start: int, total_moves: int) -> Tuple[float, float, float]:
        """
        Calculate length of each game phase
        
        Args:
            mg_start: Move number where middlegame starts
            eg_start: Move number where endgame starts
            total_moves: Total number of moves in the game
            
        Returns:
            Tuple of (opening_length, middlegame_length, endgame_length)
        """
        if mg_start == 0:
            # Game never left opening phase
            opening_length = float(total_moves)
            middlegame_length = 0.0
            endgame_length = 0.0
        elif eg_start == 0:
            # Game only reached middlegame
            opening_length = float(mg_start - 1)
            middlegame_length = float(total_moves - (mg_start - 1))
            endgame_length = 0.0
        else:
            # Game reached all phases
            opening_length = float(mg_start - 1)
            middlegame_length = float(eg_start - mg_start)
            endgame_length = float(total_moves - eg_start + 1)
        
        return opening_length, middlegame_length, endgame_length
        
    def _count_judgment_metrics(self, judgments: List[Judgment], features: FeatureVector) -> None:
        """
        Count judgment metrics from pre-calculated judgments
        
        Args:
            judgments: List of pre-calculated move judgments
            features: FeatureVector object to update
        """
        # Initialize counters
        white_counts = {judgment: 0 for judgment in Judgment}
        black_counts = {judgment: 0 for judgment in Judgment}
        
        # Count the judgments
        for i, judgment in enumerate(judgments):
            is_white = i % 2 == 0  # Even indices are White's moves
            if is_white:
                white_counts[judgment] += 1
            else:
                black_counts[judgment] += 1
        
        # Update feature vector with counts
        features.white_brilliant_count = white_counts[Judgment.BRILLIANT]
        features.white_great_count = white_counts[Judgment.GREAT]
        features.white_good_moves = white_counts[Judgment.GOOD]
        features.white_inaccuracy_count = white_counts[Judgment.INACCURACY]
        features.white_mistake_count = white_counts[Judgment.MISTAKE]
        features.white_blunder_count = white_counts[Judgment.BLUNDER]
        
        features.black_brilliant_count = black_counts[Judgment.BRILLIANT]
        features.black_great_count = black_counts[Judgment.GREAT]
        features.black_good_moves = black_counts[Judgment.GOOD]
        features.black_inaccuracy_count = black_counts[Judgment.INACCURACY]
        features.black_mistake_count = black_counts[Judgment.MISTAKE]
        features.black_blunder_count = black_counts[Judgment.BLUNDER]
    
    def _get_positions(self, game: chess.pgn.Game) -> List[chess.Board]:
        """
        Get list of positions from game
        
        Args:
            game: Chess game to get positions from
            
        Returns:
            List of chess board positions
        """
        positions = []
        board = game.board()
        
        for move in game.mainline_moves():
            positions.append(board.copy())
            board.push(move)
            
        positions.append(board.copy())  # Final position
        return positions
        
    def _calculate_material_changes(self, positions: List[chess.Board]) -> float:
        """
        Calculate rate of material balance changes
        
        Args:
            positions: List of board positions
            
        Returns:
            Rate of material balance changes (0-1)
        """
        if not positions:
            return 0.0
            
        material_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        changes = 0
        for i in range(1, len(positions)):
            prev_mat = self._get_material_value(positions[i-1], material_values)
            curr_mat = self._get_material_value(positions[i], material_values)
            if prev_mat != curr_mat:
                changes += 1
                
        return changes / (len(positions) - 1) if len(positions) > 1 else 0
    
    def _get_material_value(self, board: chess.Board, values: Dict[chess.PieceType, int]) -> int:
        """
        Get total material value on board
        
        Args:
            board: Chess board to evaluate
            values: Dictionary mapping piece types to values
            
        Returns:
            Material value from white's perspective
        """
        total = 0
        for piece_type, value in values.items():
            total += len(board.pieces(piece_type, chess.WHITE)) * value
            total -= len(board.pieces(piece_type, chess.BLACK)) * value
        return total
    
    def _calculate_mobility(self, positions: List[chess.Board]) -> float:
        """
        Calculate average piece mobility
        
        Args:
            positions: List of board positions
            
        Returns:
            Average number of legal moves per position
        """
        if not positions:
            return 0.0
            
        total_mobility = 0
        for board in positions:
            mobility = len(list(board.legal_moves))
            total_mobility += mobility
            
        return total_mobility / len(positions)
    
    def _calculate_pawn_changes(self, positions: List[chess.Board]) -> float:
        """
        Calculate rate of pawn structure changes
        
        Args:
            positions: List of board positions
            
        Returns:
            Rate of pawn structure changes (0-1)
        """
        if not positions or len(positions) <= 1:
            return 0.0
            
        changes = 0
        for i in range(1, len(positions)):
            prev_pawns = self._get_pawn_structure(positions[i-1])
            curr_pawns = self._get_pawn_structure(positions[i])
            if prev_pawns != curr_pawns:
                changes += 1
                
        return changes / (len(positions) - 1)
    
    def _get_pawn_structure(self, board: chess.Board) -> int:
        """
        Get pawn structure hash
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            Hash value representing pawn structure
        """
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        return white_pawns | (black_pawns << 32)
    
    def _calculate_center_control(self, positions: List[chess.Board]) -> float:
        """
        Calculate average center square control
        
        Args:
            positions: List of board positions
            
        Returns:
            Average center control (0-1)
        """
        if not positions:
            return 0.0
            
        center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
        total_control = 0
        
        for board in positions:
            control = 0
            for square in center_squares:
                if board.piece_at(square):
                    control += 1
            total_control += control / len(center_squares)
            
        return total_control / len(positions)
        
    def _calculate_quality_metrics(self, evals: List[Info], features: FeatureVector, 
                                  positions: List[chess.Board] = None, 
                                  moves: List[chess.Move] = None) -> None:
        """
        Calculate move quality related features and statistics for both players
        
        Args:
            evals: List of position evaluations
            features: FeatureVector to update
            positions: List of board positions
            moves: List of moves played
        """
        if not positions or not moves or len(evals) < 2:
            return
            
        white_counts = {judgment: 0 for judgment in Judgment}
        black_counts = {judgment: 0 for judgment in Judgment}
        
        # Initialize sacrifice counters
        white_sacrifices = 0
        black_sacrifices = 0
        
        # Track evaluation changes
        white_eval_changes = []
        black_eval_changes = []
        
        # Skip first evaluation as we need pairs to analyze moves
        for i in range(1, len(evals)):
            if i-1 >= len(moves):
                break
                
            prev, curr = evals[i-1], evals[i]
            is_white = (i - 1) % 2 == 0  # Even indices are White's moves
            
            # Get move quality with additional board information
            if i-1 < len(moves) and i-1 < len(positions) and i < len(positions):
                move = moves[i-1]
                prev_board = positions[i-1]
                curr_board = positions[i]
                
                # Check for sacrifice
                if MoveAnalyzer.is_piece_sacrifice(prev_board, curr_board, move):
                    if is_white:
                        white_sacrifices += 1
                    else:
                        black_sacrifices += 1
                
                judgment, _ = MoveAnalyzer.analyze_move_with_top_moves(
                    prev, curr, 
                    prev_board=prev_board, 
                    curr_board=curr_board, 
                    move=move
                )
            else:
                # Use basic analysis if board/move info not available
                judgment = MoveAnalyzer.analyze_move(prev, curr)
            
            # Increment the appropriate counter
            if is_white:
                white_counts[judgment] += 1
            else:
                black_counts[judgment] += 1
            
            # Track eval changes
            if prev.cp is not None and curr.cp is not None:
                # Get raw eval change from current player's perspective
                eval_change = curr.cp - prev.cp
                # For Black's moves, negate the change to get their perspective
                if not is_white:
                    eval_change = -eval_change
                    black_eval_changes.append(eval_change)
                else:
                    white_eval_changes.append(eval_change)
        
        # Set judgment counts
        features.white_brilliant_count = white_counts[Judgment.BRILLIANT]
        features.white_great_count = white_counts[Judgment.GREAT]
        features.white_good_moves = white_counts[Judgment.GOOD]
        features.white_inaccuracy_count = white_counts[Judgment.INACCURACY]
        features.white_mistake_count = white_counts[Judgment.MISTAKE]
        features.white_blunder_count = white_counts[Judgment.BLUNDER]
        features.white_sacrifice_count = white_sacrifices
        
        features.black_brilliant_count = black_counts[Judgment.BRILLIANT]
        features.black_great_count = black_counts[Judgment.GREAT]
        features.black_good_moves = black_counts[Judgment.GOOD]
        features.black_inaccuracy_count = black_counts[Judgment.INACCURACY]
        features.black_mistake_count = black_counts[Judgment.MISTAKE]
        features.black_blunder_count = black_counts[Judgment.BLUNDER]
        features.black_sacrifice_count = black_sacrifices
        
        # Calculate eval metrics
        if white_eval_changes:
            features.white_avg_eval_change = float(np.mean(np.abs(white_eval_changes)))
            features.white_eval_volatility = float(np.std(white_eval_changes))
        
        if black_eval_changes:
            features.black_avg_eval_change = float(np.mean(np.abs(black_eval_changes)))
            features.black_eval_volatility = float(np.std(black_eval_changes))

    def count_sacrifices(self, positions: List[chess.Board], moves: List[chess.Move], features: FeatureVector) -> None:
        """
        Count sacrifices for both players
        
        Args:
            positions: List of board positions
            moves: List of moves played
            features: FeatureVector object to update
        """
        white_sacrifices = 0
        black_sacrifices = 0
        
        for i in range(len(moves)):
            if i < len(positions) and i+1 < len(positions):
                prev_board = positions[i]
                curr_board = positions[i+1]
                move = moves[i]
                
                if MoveAnalyzer.is_piece_sacrifice(prev_board, curr_board, move):
                    is_white = i % 2 == 0  # Even indices are White's moves
                    if is_white:
                        white_sacrifices += 1
                    else:
                        black_sacrifices += 1
        
        features.white_sacrifice_count = white_sacrifices
        features.black_sacrifice_count = black_sacrifices