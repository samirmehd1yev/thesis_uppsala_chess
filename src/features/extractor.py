import chess
import chess.pgn
from typing import Dict, List, Optional, Tuple
import numpy as np
from models.data_classes import FeatureVector, Info
from models.enums import Judgment
from analysis.phase_detector import GamePhaseDetector
from analysis.move_analyzer import MoveAnalyzer
from analysis.king_safety import KingSafetyEvaluator

class FeatureExtractor:
    def __init__(self):
        self.phase_detector = GamePhaseDetector()
        self.king_safety_evaluator = KingSafetyEvaluator()
        
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
        
        # Material and position features - separate for white and black
        white_material, black_material = self._calculate_material_changes_by_color(positions)
        features.white_material_changes = white_material
        features.black_material_changes = black_material
        
        white_mobility, black_mobility = self._calculate_mobility_by_color(positions)
        features.white_piece_mobility_avg = white_mobility
        features.black_piece_mobility_avg = black_mobility
        
        white_pawn_changes, black_pawn_changes = self._calculate_pawn_changes_by_color(positions)
        features.white_pawn_structure_changes = white_pawn_changes
        features.black_pawn_structure_changes = black_pawn_changes
        
        white_center, black_center = self._calculate_center_control_by_color(positions)
        features.white_center_control_avg = white_center
        features.black_center_control_avg = black_center
        
        # King safety features - new addition
        king_safety_metrics = self._calculate_king_safety(positions)
        features.white_king_safety = king_safety_metrics['white']['avg_safety'] 
        features.black_king_safety = king_safety_metrics['black']['avg_safety']
        features.white_king_safety_min = king_safety_metrics['white']['min_safety']
        features.black_king_safety_min = king_safety_metrics['black']['min_safety']
        features.white_vulnerability_spikes = king_safety_metrics['white']['vulnerability_spikes']
        features.black_vulnerability_spikes = king_safety_metrics['black']['vulnerability_spikes']
        
        # Move statistics
        # Convert mainline_moves to a list before passing to _calculate_move_statistics
        moves_list = list(game.mainline_moves())
        capture_frequency_white, capture_frequency_black, check_frequency_white, check_frequency_black, castle_move_white, castle_move_black = self._calculate_move_statistics(positions, moves_list)
        features.capture_frequency_white = capture_frequency_white
        features.capture_frequency_black = capture_frequency_black
        features.check_frequency_white = check_frequency_white
        features.check_frequency_black = check_frequency_black
        features.castle_move_white = castle_move_white
        features.castle_move_black = castle_move_black
        
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
    
    def _calculate_king_safety(self, positions: List[chess.Board]) -> Dict[str, Dict[str, float]]:
        """
        Calculate king safety metrics for both players throughout the game.
        
        Args:
            positions: List of board positions
            
        Returns:
            Dictionary with king safety metrics for both players
        """
        if not positions:
            return {
                "white": {
                    "avg_safety": 0,
                    "min_safety": 0,
                    "safety_drop": 0,
                    "vulnerability_spikes": 0
                },
                "black": {
                    "avg_safety": 0,
                    "min_safety": 0,
                    "safety_drop": 0,
                    "vulnerability_spikes": 0
                }
            }
        
        # Track safety scores for both colors
        white_scores = []
        black_scores = []
        
        # Skip first few positions to avoid early game anomalies
        start_idx = min(10, len(positions) // 4)
        
        # Evaluate king safety for each position
        for position in positions[start_idx:]:
            # Skip invalid positions
            if position.king(chess.WHITE) is None or position.king(chess.BLACK) is None:
                continue
                
            # Get safety scores for both sides
            white_safety = self.king_safety_evaluator.evaluate_king_safety(position, chess.WHITE)
            black_safety = self.king_safety_evaluator.evaluate_king_safety(position, chess.BLACK)
            
            white_scores.append(white_safety)
            black_scores.append(black_safety)
        
        # Calculate derived metrics
        white_features = self._calculate_safety_metrics(white_scores)
        black_features = self._calculate_safety_metrics(black_scores)
        
        return {
            "white": white_features,
            "black": black_features
        }
    
    def _calculate_safety_metrics(self, safety_scores: List[int]) -> Dict[str, float]:
        """
        Calculate derived metrics from raw safety scores.
        
        Args:
            safety_scores: List of safety scores for a player
            
        Returns:
            Dictionary of derived metrics
        """
        if not safety_scores:
            return {
                "avg_safety": 0,
                "min_safety": 0,
                "safety_drop": 0,
                "vulnerability_spikes": 0
            }
        
        # Average safety score
        avg_safety = sum(safety_scores) / len(safety_scores)
        
        # Minimum safety score
        min_safety = min(safety_scores)
        
        # Calculate maximum safety drop between consecutive positions
        max_drop = 0
        for i in range(1, len(safety_scores)):
            drop = safety_scores[i-1] - safety_scores[i]
            max_drop = max(max_drop, drop)
        
        # Count vulnerability spikes (sudden drops in safety)
        threshold = min(100, abs(avg_safety * 0.3))  # 30% of average as threshold
        vulnerability_spikes = 0
        for i in range(1, len(safety_scores)):
            if safety_scores[i-1] - safety_scores[i] > threshold:
                vulnerability_spikes += 1
        
        return {
            "avg_safety": avg_safety,
            "min_safety": min_safety,
            "safety_drop": max_drop,
            "vulnerability_spikes": vulnerability_spikes
        }
    
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
        
    def _calculate_material_changes_by_color(self, positions: List[chess.Board]) -> tuple:
        """
        Calculate rate of material changes separately for white and black
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of (white_changes_rate, black_changes_rate)
        """
        if not positions:
            return 0.0, 0.0
            
        material_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        white_changes = 0
        black_changes = 0
        
        for i in range(1, len(positions)):
            prev_white_mat = self._get_material_value_for_color(positions[i-1], material_values, chess.WHITE)
            curr_white_mat = self._get_material_value_for_color(positions[i], material_values, chess.WHITE)
            
            prev_black_mat = self._get_material_value_for_color(positions[i-1], material_values, chess.BLACK)
            curr_black_mat = self._get_material_value_for_color(positions[i], material_values, chess.BLACK)
            
            if prev_white_mat != curr_white_mat:
                white_changes += 1
                
            if prev_black_mat != curr_black_mat:
                black_changes += 1
                
        white_change_rate = white_changes / (len(positions) - 1) if len(positions) > 1 else 0
        black_change_rate = black_changes / (len(positions) - 1) if len(positions) > 1 else 0
        
        return white_change_rate, black_change_rate
    
    def _get_material_value_for_color(self, board: chess.Board, values: Dict[chess.PieceType, int], color: chess.Color) -> int:
        """
        Get total material value for a specific color
        
        Args:
            board: Chess board to evaluate
            values: Dictionary mapping piece types to values
            color: Chess color to calculate value for
            
        Returns:
            Material value for the specified color
        """
        total = 0
        for piece_type, value in values.items():
            total += len(board.pieces(piece_type, color)) * value
        return total
    
    def _calculate_mobility_by_color(self, positions: List[chess.Board]) -> tuple:
        """
        Calculate average piece mobility separately for white and black
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of (white_mobility_avg, black_mobility_avg)
        """
        if not positions:
            return 0.0, 0.0
            
        white_mobility_total = 0
        black_mobility_total = 0
        white_positions = 0
        black_positions = 0
        
        for board in positions:
            if board.turn == chess.WHITE:
                white_mobility_total += len(list(board.legal_moves))
                white_positions += 1
            else:
                black_mobility_total += len(list(board.legal_moves))
                black_positions += 1
                
        white_avg = white_mobility_total / white_positions if white_positions > 0 else 0
        black_avg = black_mobility_total / black_positions if black_positions > 0 else 0
        
        return white_avg, black_avg
    
    def _calculate_pawn_changes_by_color(self, positions: List[chess.Board]) -> tuple:
        """
        Calculate rate of pawn structure changes separately for white and black
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of (white_pawn_changes_rate, black_pawn_changes_rate)
        """
        if not positions or len(positions) <= 1:
            return 0.0, 0.0
            
        white_changes = 0
        black_changes = 0
        
        for i in range(1, len(positions)):
            prev_white_pawns = self._get_pawn_structure_for_color(positions[i-1], chess.WHITE)
            curr_white_pawns = self._get_pawn_structure_for_color(positions[i], chess.WHITE)
            
            prev_black_pawns = self._get_pawn_structure_for_color(positions[i-1], chess.BLACK)
            curr_black_pawns = self._get_pawn_structure_for_color(positions[i], chess.BLACK)
            
            if prev_white_pawns != curr_white_pawns:
                white_changes += 1
                
            if prev_black_pawns != curr_black_pawns:
                black_changes += 1
                
        white_change_rate = white_changes / (len(positions) - 1)
        black_change_rate = black_changes / (len(positions) - 1)
        
        return white_change_rate, black_change_rate
    
    def _get_pawn_structure_for_color(self, board: chess.Board, color: chess.Color) -> int:
        """
        Get pawn structure hash for a specific color
        
        Args:
            board: Chess board to evaluate
            color: Chess color to calculate pawn structure for
            
        Returns:
            Hash value representing pawn structure for the specified color
        """
        return board.pieces(chess.PAWN, color)
    
    def _calculate_center_control_by_color(self, positions: List[chess.Board]) -> tuple:
        """
        Calculate average center square control separately for white and black
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of (white_center_control_avg, black_center_control_avg)
        """
        if not positions:
            return 0.0, 0.0
            
        center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
        white_control_total = 0
        black_control_total = 0
        
        for board in positions:
            white_control = 0
            black_control = 0
            
            for square in center_squares:
                piece = board.piece_at(square)
                if piece:
                    if piece.color == chess.WHITE:
                        white_control += 1
                    else:
                        black_control += 1
            
            white_control_total += white_control / len(center_squares)
            black_control_total += black_control / len(center_squares)
            
        white_avg = white_control_total / len(positions)
        black_avg = black_control_total / len(positions)
        
        return white_avg, black_avg
        
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

    def _calculate_move_statistics(self, positions: List[chess.Board], moves: List[chess.Move]) -> Tuple[float, float, float, float, int, int]:
        """
        Calculate move-related statistics including capture frequency, check frequency,
        and castle timing for both players separately.
        
        Args:
            positions: List of board positions (positions[i] is the board before moves[i])
            moves: List of moves played
            
        Returns:
            Tuple of (capture_frequency_white, capture_frequency_black, 
                    check_frequency_white, check_frequency_black, 
                    castle_move_white, castle_move_black)
        """
        if not positions or not moves:
            return 0.0, 0.0, 0.0, 0.0, 0, 0
        
        # Convert moves to a list if it's not already one
        if not isinstance(moves, list):
            moves = list(moves)
                
        total_white_moves = 0
        total_black_moves = 0
        white_capture_count = 0
        black_capture_count = 0
        white_check_count = 0
        black_check_count = 0
        white_castle_move = 0  # Changed to 0 for never castled
        black_castle_move = 0  # Changed to 0 for never castled
        
        for i, move in enumerate(moves):
            if i >= len(positions):
                break
                    
            board = positions[i]
            is_white = board.turn == chess.WHITE
            
            # Count total moves by color
            if is_white:
                total_white_moves += 1
            else:
                total_black_moves += 1
                
            # Check for captures (including en passant)
            is_capture = board.piece_at(move.to_square) is not None
            is_en_passant = False
            
            # Check for en passant if it's a pawn move
            if board.piece_type_at(move.from_square) == chess.PAWN:
                if board.ep_square == move.to_square:
                    is_en_passant = True
            
            if is_capture or is_en_passant:
                if is_white:
                    white_capture_count += 1
                else:
                    black_capture_count += 1
            
            # Check for castling
            if board.piece_type_at(move.from_square) == chess.KING:
                # Check if it's a castling move (king moves two squares)
                file_diff = abs(chess.square_file(move.from_square) - chess.square_file(move.to_square))
                if file_diff > 1:  # King moved more than one file, must be castling
                    move_number = (i // 2) + 1  # Convert to move number (1-based)
                    if is_white:
                        white_castle_move = move_number
                    else:
                        black_castle_move = move_number
            
            # Make a copy of the board and apply the move to check for check
            next_board = board.copy()
            next_board.push(move)
            
            if next_board.is_check():
                if is_white:
                    white_check_count += 1
                else:
                    black_check_count += 1
        
        # Calculate frequencies
        capture_frequency_white = white_capture_count / total_white_moves if total_white_moves > 0 else 0.0
        capture_frequency_black = black_capture_count / total_black_moves if total_black_moves > 0 else 0.0
        check_frequency_white = white_check_count / total_white_moves if total_white_moves > 0 else 0.0
        check_frequency_black = black_check_count / total_black_moves if total_black_moves > 0 else 0.0
        
        # Return actual move numbers for castling (not normalized)
        return (capture_frequency_white, capture_frequency_black, 
                check_frequency_white, check_frequency_black, 
                white_castle_move, black_castle_move)
    
    def count_sacrifices(self, positions: List[chess.Board], moves: List[chess.Move], features: FeatureVector) -> None:
        """
        Count sacrifices for both players
        
        Args:
            positions: List of board positions
            moves: List of moves played
            features: FeatureVector object to update
        """
        # Convert moves to a list if it's not already one
        if not isinstance(moves, list):
            moves = list(moves)
            
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