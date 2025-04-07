import chess
import chess.pgn
from typing import Dict, List, Optional, Tuple
import numpy as np
from models.data_classes import FeatureVector, Info
from models.enums import Judgment
from analysis.move_analyzer import MoveAnalyzer

class PlayerTypeFeatures:
    """
    Features to distinguish between the four chess player types:
    - Activist: Dynamic, tactical, initiative-driven
    - Theorist: Principled, structured, theoretical
    - Reflector: Subtle, positional, harmonious
    - Pragmatist: Calculating, practical, concrete
    """
    
    def __init__(self, feature_extractor):
        """Initialize with reference to the parent feature extractor"""
        self.extractor = feature_extractor
    
    def extract_distinctive_features(self, positions: List[chess.Board], moves: List[chess.Move], 
                                    evals: List[Info], game: chess.pgn.Game, 
                                    mg_start: int, eg_start: int, total_moves: int) -> Dict[str, float]:
        """
        Extract features that distinguish between all four player types
        
        Args:
            positions: List of board positions
            moves: List of chess moves
            evals: List of position evaluations
            game: The chess game
            mg_start: Move number where middlegame starts
            eg_start: Move number where endgame starts
            total_moves: Total moves in the game
            
        Returns:
            Dictionary with features that distinguish player types
        """
        # Initialize feature dictionary
        features = {}
        
        # Calculate phase-specific metrics
        opening_length, middlegame_length, endgame_length = self._phase_length(mg_start, eg_start, total_moves)
        
        # Add initiative features (Activist vs others)
        initiative_metrics = self._calculate_initiative_features(positions, moves, evals)
        features.update(initiative_metrics)
        
        # Add theoretical preparation features (Theorist vs others)
        theoretical_metrics = self._calculate_theoretical_features(game, positions, moves, evals, opening_length, total_moves)
        features.update(theoretical_metrics)
        
        # Add positional harmony features (Reflector vs others)
        harmony_metrics = self._calculate_harmony_features(positions, moves, evals)
        features.update(harmony_metrics)
        
        # Add calculation clarity features (Pragmatist vs others)
        clarity_metrics = self._calculate_clarity_features(positions, moves, evals)
        features.update(clarity_metrics)
        
        return features
    
    def _phase_length(self, mg_start: int, eg_start: int, total_moves: int) -> Tuple[float, float, float]:
        """Calculate phase lengths - reused from your existing code"""
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
    
    #---------------------------------------------------------------------------
    # ACTIVIST DISTINGUISHING FEATURES
    #---------------------------------------------------------------------------
    
    def _calculate_initiative_features(self, positions: List[chess.Board], moves: List[chess.Move], 
                                      evals: List[Info]) -> Dict[str, float]:
        """
        Calculate features related to initiative and dynamic play
        that distinguish Activists from other player types
        """
        initiative_metrics = {}
        
        # Track initiative-related metrics
        white_initiative_moves = 0
        black_initiative_moves = 0
        white_counterplay_moves = 0
        black_counterplay_moves = 0
        white_total_moves = 0
        black_total_moves = 0
        
        # Track forcing move sequence lengths
        white_forcing_sequences = []
        black_forcing_sequences = []
        current_white_sequence = 0
        current_black_sequence = 0
        
        # Analyze each move for initiative characteristics
        for i in range(len(moves)):
            if i >= len(positions) - 1 or i >= len(evals):
                continue
                
            board = positions[i]
            next_board = positions[i+1]
            is_white = board.turn == chess.WHITE
            
            # Count moves by color
            if is_white:
                white_total_moves += 1
            else:
                black_total_moves += 1
            
            # Check for initiative-taking moves
            is_initiative_move = self._is_initiative_move(board, next_board, moves[i])
            is_counterplay_move = self._is_counterplay_move(board, next_board, moves[i], evals[i])
            
            # Record metrics by color
            if is_white:
                if is_initiative_move:
                    white_initiative_moves += 1
                    current_white_sequence += 1
                else:
                    # End of sequence
                    if current_white_sequence > 0:
                        white_forcing_sequences.append(current_white_sequence)
                        current_white_sequence = 0
                        
                if is_counterplay_move:
                    white_counterplay_moves += 1
            else:
                if is_initiative_move:
                    black_initiative_moves += 1
                    current_black_sequence += 1
                else:
                    # End of sequence
                    if current_black_sequence > 0:
                        black_forcing_sequences.append(current_black_sequence)
                        current_black_sequence = 0
                        
                if is_counterplay_move:
                    black_counterplay_moves += 1
        
        # Add final sequences if they exist
        if current_white_sequence > 0:
            white_forcing_sequences.append(current_white_sequence)
        if current_black_sequence > 0:
            black_forcing_sequences.append(current_black_sequence)
        
        # Calculate initiative frequency
        white_initiative_ratio = white_initiative_moves / max(1, white_total_moves)
        black_initiative_ratio = black_initiative_moves / max(1, black_total_moves)
        
        # Calculate counterplay ratio
        white_counterplay_ratio = white_counterplay_moves / max(1, white_total_moves)
        black_counterplay_ratio = black_counterplay_moves / max(1, black_total_moves)
        
        # Calculate average forcing sequence length
        white_avg_sequence = sum(white_forcing_sequences) / max(1, len(white_forcing_sequences))
        black_avg_sequence = sum(black_forcing_sequences) / max(1, len(black_forcing_sequences))
        
        # Store features
        initiative_metrics['white_initiative_ratio'] = white_initiative_ratio
        initiative_metrics['black_initiative_ratio'] = black_initiative_ratio
        initiative_metrics['white_counterplay_ratio'] = white_counterplay_ratio
        initiative_metrics['black_counterplay_ratio'] = black_counterplay_ratio
        initiative_metrics['white_forcing_sequence_length'] = white_avg_sequence
        initiative_metrics['black_forcing_sequence_length'] = black_avg_sequence
        
        return initiative_metrics
    
    def _is_initiative_move(self, board: chess.Board, next_board: chess.Board, move: chess.Move) -> bool:
        """
        Determine if a move is initiative-taking (forcing responses)
        
        Initiative moves: checks, captures, direct threats
        """
        # Check if the move gives check
        if next_board.is_check():
            return True
        
        # Check if it's a capture
        is_capture = board.piece_at(move.to_square) is not None
        if is_capture:
            return True
        
        # Check if it directly threatens a piece (not defended or inadequately defended)
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
            
        # Get attacked squares after the move
        next_board.push(chess.Move.null())  # Switch turn to original player
        attacked_squares = set()
        for sq in chess.SQUARES:
            # Get piece at the square
            target = next_board.piece_at(sq)
            # If square is occupied by an opponent's piece
            if target and target.color != piece.color:
                # And our moved piece attacks it
                if next_board.is_attacked_by(piece.color, sq):
                    # It's a threat if the piece is undefended or higher value than attacker
                    attacked_squares.add(sq)
        next_board.pop()  # Restore board
        
        # If the move creates new attacks, it's an initiative move
        return len(attacked_squares) > 0
    
    def _is_counterplay_move(self, board: chess.Board, next_board: chess.Board, 
                            move: chess.Move, eval_info: Info) -> bool:
        """
        Determine if a move is creating counterplay despite a disadvantage
        """
        # Check if player is at a disadvantage
        if not hasattr(eval_info, 'cp') or eval_info.cp is None:
            return False
            
        cp_value = eval_info.cp
        is_disadvantage = (board.turn == chess.WHITE and cp_value < -100) or \
                          (board.turn == chess.BLACK and cp_value > 100)
        
        if not is_disadvantage:
            return False
            
        # Counterplay is creating active possibilities despite disadvantage
        # Check if move is developing attack despite disadvantage
        return self._is_initiative_move(board, next_board, move)
        
    # Piece values for various calculations
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King value is not used in evaluation
    } 
    
    #---------------------------------------------------------------------------
    # THEORIST DISTINGUISHING FEATURES
    #---------------------------------------------------------------------------
    
    def _calculate_theoretical_features(self, game: chess.pgn.Game, positions: List[chess.Board], 
                                       moves: List[chess.Move], evals: List[Info], 
                                       opening_length: float, total_moves: int) -> Dict[str, float]:
        """
        Calculate features related to theoretical understanding and structured play
        that distinguish Theorists from other player types
        """
        theoretical_metrics = {}
        
        # Track opening theory adherence
        opening_moves = int(opening_length)
        
        # Get ECO code and opening name from game headers
        headers = dict(game.headers) if hasattr(game, 'headers') else {}
        eco_code = headers.get('ECO', '')
        
        # Calculate opening novelty score using your existing method
        from tools.eco_database_loader import eco_loader
        game_moves_uci = [move.uci() for move in moves[:min(opening_moves*2, len(moves))]]
        
        # Only calculate if ECO code and moves available
        if eco_code and game_moves_uci:
            novelty_score, opening_name, matched_eco, matching_plies = self.extractor.calculate_opening_novelty_score(
                eco_code, game_moves_uci, opening_length, total_moves
            )
            theoretical_metrics['opening_theory_adherence'] = novelty_score
        else:
            theoretical_metrics['opening_theory_adherence'] = 0.0
        
        # Track structural consistency (pawn structure maintenance)
        white_structure_changes = 0
        black_structure_changes = 0
        white_moves_count = 0
        black_moves_count = 0
        
        # Initialize structure references
        if positions:
            prev_white_pawns = self._get_pawn_structure(positions[0], chess.WHITE)
            prev_black_pawns = self._get_pawn_structure(positions[0], chess.BLACK)
            
            # Calculate structural changes throughout game
            for i in range(1, len(positions)):
                board = positions[i]
                is_white_to_move = positions[i-1].turn == chess.WHITE
                
                # Count moves by color
                if is_white_to_move:
                    white_moves_count += 1
                else:
                    black_moves_count += 1
                
                # Get current pawn structure
                curr_white_pawns = self._get_pawn_structure(board, chess.WHITE)
                curr_black_pawns = self._get_pawn_structure(board, chess.BLACK)
                
                # Check for changes
                if curr_white_pawns != prev_white_pawns:
                    white_structure_changes += 1
                if curr_black_pawns != prev_black_pawns:
                    black_structure_changes += 1
                
                # Update previous structures
                prev_white_pawns = curr_white_pawns
                prev_black_pawns = curr_black_pawns
                
        # Calculate structural consistency (inverted changes)
        white_structural_consistency = 1.0 - (white_structure_changes / max(1, white_moves_count))
        black_structural_consistency = 1.0 - (black_structure_changes / max(1, black_moves_count))
        
        theoretical_metrics['white_structural_consistency'] = white_structural_consistency
        theoretical_metrics['black_structural_consistency'] = black_structural_consistency
        
        # Track adherence to common strategic patterns
        white_common_patterns = 0
        black_common_patterns = 0
        
        # Check for common theoretical patterns
        for i in range(len(moves)):
            if i >= len(positions) - 1:
                continue
                
            board = positions[i]
            is_white = board.turn == chess.WHITE
            
            # Check if move follows common theoretical patterns
            follows_pattern = self._follows_theoretical_pattern(board, moves[i])
            
            if follows_pattern:
                if is_white:
                    white_common_patterns += 1
                else:
                    black_common_patterns += 1
        
        # Calculate pattern adherence ratio
        white_pattern_ratio = white_common_patterns / max(1, white_moves_count)
        black_pattern_ratio = black_common_patterns / max(1, black_moves_count)
        
        theoretical_metrics['white_pattern_adherence'] = white_pattern_ratio
        theoretical_metrics['black_pattern_adherence'] = black_pattern_ratio
        
        return theoretical_metrics
    
    def _get_pawn_structure(self, board: chess.Board, color: chess.Color) -> chess.SquareSet:
        """Get the pawn structure for a given color"""
        return board.pieces(chess.PAWN, color)
    
    def _follows_theoretical_pattern(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Check if a move follows common theoretical patterns
        (e.g., central control, piece development, etc.)
        """
        # Centre control pattern
        central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        if move.to_square in central_squares:
            return True
            
        # Development pattern (early knight, bishop development)
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
            
        move_count = len(board.move_stack)
        if move_count < 10:  # Early game
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                
                # Knight/Bishop developing from back rank
                if (piece.color == chess.WHITE and from_rank == 0 and to_rank > 0) or \
                   (piece.color == chess.BLACK and from_rank == 7 and to_rank < 7):
                    return True
        
        # Castling pattern
        if piece.piece_type == chess.KING:
            # Check if it's a castling move
            from_file = chess.square_file(move.from_square)
            to_file = chess.square_file(move.to_square)
            if abs(from_file - to_file) > 1:  # King moved more than one file
                return True
                
        return False 
    
    #---------------------------------------------------------------------------
    # REFLECTOR DISTINGUISHING FEATURES
    #---------------------------------------------------------------------------
    
    def _calculate_harmony_features(self, positions: List[chess.Board], moves: List[chess.Move], 
                                   evals: List[Info]) -> Dict[str, float]:
        """
        Calculate features related to piece harmony and positional play
        that distinguish Reflectors from other player types
        """
        harmony_metrics = {}
        
        # Track harmony-related metrics
        white_harmony_scores = []
        black_harmony_scores = []
        white_prophylactic_moves = 0
        black_prophylactic_moves = 0
        white_total_moves = 0
        black_total_moves = 0
        
        # Analyze each position for piece harmony
        for i in range(len(positions)):
            board = positions[i]
            
            # Calculate harmony score for this position
            white_harmony = self._calculate_piece_harmony(board, chess.WHITE)
            black_harmony = self._calculate_piece_harmony(board, chess.BLACK)
            
            white_harmony_scores.append(white_harmony)
            black_harmony_scores.append(black_harmony)
            
            # Check for prophylactic moves
            if i > 0 and i-1 < len(moves):
                is_white = positions[i-1].turn == chess.WHITE
                move = moves[i-1]
                
                # Count moves by color
                if is_white:
                    white_total_moves += 1
                else:
                    black_total_moves += 1
                
                # Check if this was a prophylactic move
                if self._is_prophylactic_move(positions[i-1], positions[i], move):
                    if is_white:
                        white_prophylactic_moves += 1
                    else:
                        black_prophylactic_moves += 1
        
        # Calculate average harmony scores
        white_avg_harmony = sum(white_harmony_scores) / max(1, len(white_harmony_scores))
        black_avg_harmony = sum(black_harmony_scores) / max(1, len(black_harmony_scores))
        
        # Calculate prophylactic move ratio
        white_prophylactic_ratio = white_prophylactic_moves / max(1, white_total_moves)
        black_prophylactic_ratio = black_prophylactic_moves / max(1, black_total_moves)
        
        # Store harmony-related metrics
        harmony_metrics['white_piece_harmony'] = white_avg_harmony
        harmony_metrics['black_piece_harmony'] = black_avg_harmony
        harmony_metrics['white_prophylactic_ratio'] = white_prophylactic_ratio
        harmony_metrics['black_prophylactic_ratio'] = black_prophylactic_ratio
        
        # Calculate exchange sacrifice ratio if positions available
        if len(positions) > 1 and len(moves) > 0:
            white_exchange_sacrifices = 0
            black_exchange_sacrifices = 0
            
            for i in range(len(moves)):
                if i >= len(positions) - 1:
                    continue
                    
                board = positions[i]
                next_board = positions[i+1]
                is_white = board.turn == chess.WHITE
                
                # Check for exchange sacrifice
                if self._is_exchange_sacrifice(board, next_board, moves[i]):
                    if is_white:
                        white_exchange_sacrifices += 1
                    else:
                        black_exchange_sacrifices += 1
            
            # Calculate exchange sacrifice ratios
            white_exchange_ratio = white_exchange_sacrifices / max(1, white_total_moves)
            black_exchange_ratio = black_exchange_sacrifices / max(1, black_total_moves)
            
            harmony_metrics['white_exchange_sacrifice_ratio'] = white_exchange_ratio
            harmony_metrics['black_exchange_sacrifice_ratio'] = black_exchange_ratio
        else:
            harmony_metrics['white_exchange_sacrifice_ratio'] = 0.0
            harmony_metrics['black_exchange_sacrifice_ratio'] = 0.0
        
        return harmony_metrics
    
    def _calculate_piece_harmony(self, board: chess.Board, color: chess.Color) -> float:
        """
        Calculate a piece harmony score for a given position and color
        High scores indicate good piece coordination and harmony
        """
        harmony_score = 0.0
        piece_count = 0
        
        # Metrics for harmony:
        # 1. Piece coordination (pieces defending each other)
        # 2. Space control ratio (control relative to piece count)
        # 3. Development completeness
        # 4. Pawn structure quality
        
        # Check for piece coordination
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece or piece.color != color:
                continue
                
            piece_count += 1
            
            # Check if this piece is defended
            is_defended = board.is_attacked_by(color, square)
            if is_defended:
                harmony_score += 1.0
                
            # Bonus for pieces on good squares
            if self._is_good_square(board, square, piece):
                harmony_score += 0.5
        
        # Normalize by piece count
        avg_harmony = harmony_score / max(1, piece_count)
        
        # Space control factor
        space_control = 0
        for square in chess.SQUARES:
            if board.is_attacked_by(color, square):
                space_control += 1
        
        space_ratio = space_control / max(8, piece_count * 4)  # Expect 4 squares per piece minimum
        
        # Final harmony score combines both factors
        return (avg_harmony + space_ratio) / 2.0
    
    def _is_good_square(self, board: chess.Board, square: chess.Square, piece: chess.Piece) -> bool:
        """Determine if a piece is on a 'good' square for harmony"""
        piece_type = piece.piece_type
        color = piece.color
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        # Knights on central outposts
        if piece_type == chess.KNIGHT:
            # Central knights are good
            if 2 <= file <= 5 and 2 <= rank <= 5:
                return True
                
        # Bishops on long diagonals
        elif piece_type == chess.BISHOP:
            # Long diagonals or fianchetto positions
            if (file + rank) % 2 == 0:  # Bishop on correct color
                # Main diagonals
                if (file == rank) or (file + rank == 7):
                    return True
                # Fianchetto positions
                if (color == chess.WHITE and rank == 1 and (file == 1 or file == 6)) or \
                   (color == chess.BLACK and rank == 6 and (file == 1 or file == 6)):
                    return True
        
        # Rooks on open files
        elif piece_type == chess.ROOK:
            # Check if file is open (no pawns)
            file_open = True
            for r in range(8):
                sq = chess.square(file, r)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN:
                    file_open = False
                    break
            if file_open:
                return True
            
            # Rook on 7th rank is good
            if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                return True
        
        # Queens in center with space
        elif piece_type == chess.QUEEN:
            if 2 <= file <= 5 and 2 <= rank <= 5:
                return True
        
        # Kings on safe squares
        elif piece_type == chess.KING:
            # Early game: castled king is good
            if len(board.piece_map()) > 20:  # Early/middle game
                if (color == chess.WHITE and rank == 0 and (file == 6 or file == 2)) or \
                   (color == chess.BLACK and rank == 7 and (file == 6 or file == 2)):
                    return True
            else:  # Endgame: central king is good
                if 2 <= file <= 5 and 2 <= rank <= 5:
                    return True
                    
        return False
    
    def _is_prophylactic_move(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> bool:
        """
        Determine if a move is prophylactic (preventing future threats)
        A key characteristic of reflector players
        """
        # Check if the move blocks a future attack
        piece = prev_board.piece_at(move.from_square)
        if not piece:
            return False
            
        # Prophylactic moves are not captures or checks
        if prev_board.is_capture(move) or curr_board.is_check():
            return False
            
        # Check if the move prevents an attack on a friendly piece
        for sq in chess.SQUARES:
            target = prev_board.piece_at(sq)
            # If square has a friendly piece
            if target and target.color == piece.color:
                # Check if it was potentially attacked before our move
                attackers_before = prev_board.attackers(not piece.color, sq)
                if attackers_before:
                    # Check if our move reduced potential attackers
                    attackers_after = curr_board.attackers(not piece.color, sq)
                    if len(attackers_after) < len(attackers_before):
                        return True
        
        # Check if the move occupies a square that opponent would want to use
        # This is a common prophylactic idea
        dest_square = move.to_square
        # Check if any opponent piece could use this square
        for sq in chess.SQUARES:
            opponent_piece = prev_board.piece_at(sq)
            if opponent_piece and opponent_piece.color != piece.color:
                # If opponent piece could move to destination
                test_move = chess.Move(sq, dest_square)
                # Skip if move is clearly illegal
                if test_move in prev_board.legal_moves:
                    return True
        
        return False
    
    def _is_exchange_sacrifice(self, prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> bool:
        """
        Check if a move is an exchange sacrifice (giving rook for minor piece)
        A characteristic feature of reflector players like Petrosian
        """
        # If move is not a capture, it can't be an exchange sacrifice
        if not prev_board.is_capture(move):
            return False
            
        piece = prev_board.piece_at(move.from_square)
        captured = prev_board.piece_at(move.to_square)
        
        if not piece or not captured:
            return False
            
        # A genuine exchange sacrifice is giving up a rook for a minor piece
        # This requires checking if the rook can be recaptured by a minor piece
        if piece.piece_type == chess.ROOK and captured.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # After rook takes minor piece, check if the square is attacked by another minor piece
            for sq in chess.SQUARES:
                defender = curr_board.piece_at(sq)
                if defender and defender.color != piece.color and defender.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    recap_move = chess.Move(sq, move.to_square)
                    if recap_move in curr_board.legal_moves:
                        return True
        
        return False
    
    #---------------------------------------------------------------------------
    # PRAGMATIST DISTINGUISHING FEATURES
    #---------------------------------------------------------------------------
    
    def _calculate_clarity_features(self, positions: List[chess.Board], moves: List[chess.Move], 
                                   evals: List[Info]) -> Dict[str, float]:
        """
        Calculate features related to calculation clarity and practical play
        that distinguish Pragmatists from other player types
        """
        clarity_metrics = {}
        
        # Track metrics for calculation clarity
        white_concrete_decisions = []
        black_concrete_decisions = []
        white_defensive_accuracy = []
        black_defensive_accuracy = []
        white_total_moves = 0
        black_total_moves = 0
        white_objective_choices = 0
        black_objective_choices = 0
        
        for i in range(len(moves)):
            if i >= len(positions) - 1 or i >= len(evals):
                continue
                
            board = positions[i]
            next_board = positions[i+1]
            move = moves[i]
            is_white = board.turn == chess.WHITE
            
            # Count moves by color
            if is_white:
                white_total_moves += 1
            else:
                black_total_moves += 1
            
            # Check for concrete calculation quality
            concrete_score = self._calculate_concrete_decision_quality(board, move, evals[i])
            if is_white:
                white_concrete_decisions.append(concrete_score)
            else:
                black_concrete_decisions.append(concrete_score)
            
            # Check for defensive accuracy
            is_defending = self._is_defending(board)
            if is_defending:
                defense_quality = self._calculate_defensive_quality(board, next_board, move, evals[i], evals[i+1])
                if is_white:
                    white_defensive_accuracy.append(defense_quality)
                else:
                    black_defensive_accuracy.append(defense_quality)
            
            # Check for objective decision-making
            if self._is_objective_choice(board, move, evals[i]):
                if is_white:
                    white_objective_choices += 1
                else:
                    black_objective_choices += 1
        
        # Calculate average concrete decision quality
        white_concrete_avg = sum(white_concrete_decisions) / max(1, len(white_concrete_decisions))
        black_concrete_avg = sum(black_concrete_decisions) / max(1, len(black_concrete_decisions))
        
        # Calculate defense quality (key pragmatist trait)
        white_defense_avg = sum(white_defensive_accuracy) / max(1, len(white_defensive_accuracy))
        black_defense_avg = sum(black_defensive_accuracy) / max(1, len(black_defensive_accuracy))
        
        # Calculate objective decision ratio
        white_objective_ratio = white_objective_choices / max(1, white_total_moves)
        black_objective_ratio = black_objective_choices / max(1, black_total_moves)
        
        # Store features
        clarity_metrics['white_concrete_calculation'] = white_concrete_avg
        clarity_metrics['black_concrete_calculation'] = black_concrete_avg
        clarity_metrics['white_defensive_precision'] = white_defense_avg
        clarity_metrics['black_defensive_precision'] = black_defense_avg
        clarity_metrics['white_objective_decision_ratio'] = white_objective_ratio
        clarity_metrics['black_objective_decision_ratio'] = black_objective_ratio
        
        # Track clarity of evaluation changes
        white_eval_clarity = []
        black_eval_clarity = []
        
        for i in range(1, len(evals)):
            if i >= len(positions):
                continue
                
            prev_eval = evals[i-1]
            curr_eval = evals[i]
            
            # Skip if evaluation data is missing
            if not hasattr(prev_eval, 'cp') or not hasattr(curr_eval, 'cp') or \
               prev_eval.cp is None or curr_eval.cp is None:
                continue
                
            # Determine whose move it was
            is_white = positions[i-1].turn == chess.WHITE
            
            # Calculate evaluation clarity (small changes are clear, big jumps are not)
            prev_cp = prev_eval.cp 
            curr_cp = curr_eval.cp 
            eval_diff = abs(curr_cp - prev_cp)
            
            # Convert to clarity score (1.0 is maximally clear)
            clarity_score = 1.0 / (1.0 + eval_diff/100.0)
            
            if is_white:
                white_eval_clarity.append(clarity_score)
            else:
                black_eval_clarity.append(clarity_score)
        
        # Calculate average evaluation clarity
        white_clarity_avg = sum(white_eval_clarity) / max(1, len(white_eval_clarity))
        black_clarity_avg = sum(black_eval_clarity) / max(1, len(black_eval_clarity))
        
        clarity_metrics['white_evaluation_clarity'] = white_clarity_avg
        clarity_metrics['black_evaluation_clarity'] = black_clarity_avg
        
        return clarity_metrics
    
    def _calculate_concrete_decision_quality(self, board: chess.Board, move: chess.Move, eval_info: Info) -> float:
        """Calculate quality of concrete decision making"""
        # If we have multipv data, we can assess if the most concrete move was chosen
        if not hasattr(eval_info, 'multipv') or not eval_info.multipv:
            return 0.5  # Neutral score if no data
            
        # Get move in UCI format
        move_uci = move.uci()
        
        # Check if it matches any of the top engine moves
        for i, mv_data in enumerate(eval_info.multipv):
            if isinstance(mv_data, dict) and 'move' in mv_data and mv_data['move'] == move_uci:
                # First choice is the most concrete (engine-validated) decision
                return 1.0 - (i * 0.2)  # 1.0 for first choice, 0.8 for second, etc.
        
        return 0.2  # Low score if not in top engine choices
    
    def _is_defending(self, board: chess.Board) -> bool:
        """Check if the position requires defensive play"""
        # Count threats against player's pieces
        player_color = board.turn
        opponent_color = not player_color
        
        # Check for direct threats
        threat_count = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == player_color:
                # Check if attacked by opponent
                if board.is_attacked_by(opponent_color, sq):
                    threat_count += 1
        
        # More than one threat indicates defensive position
        return threat_count > 1 or board.is_check()
    
    def _calculate_defensive_quality(self, prev_board: chess.Board, curr_board: chess.Board, 
                                    move: chess.Move, prev_eval: Info, curr_eval: Info) -> float:
        """
        Calculate quality of defensive play
        Scale: 0.0 (poor defense) to 1.0 (excellent defense)
        """
        # If we're not in check and not threatened, defense is not applicable
        if not self._is_defending(prev_board):
            return 0.5  # Neutral
            
        # Get evaluation change
        prev_cp = prev_eval.cp if hasattr(prev_eval, 'cp') and prev_eval.cp is not None else 0
        curr_cp = curr_eval.cp if hasattr(curr_eval, 'cp') and curr_eval.cp is not None else 0
        
        # Convert to player's perspective
        if prev_board.turn == chess.BLACK:
            prev_cp = -prev_cp
            curr_cp = -curr_cp
        
        # Calculate how much the position improved/deteriorated
        eval_change = curr_cp - prev_cp
        
        # Score defensive quality
        if eval_change > 50:
            # Significantly improved position
            return 1.0
        elif eval_change > 0:
            # Somewhat improved position
            return 0.75
        elif eval_change > -30:
            # Maintained position approximately
            return 0.5
        elif eval_change > -100:
            # Slightly worsened position
            return 0.25
        else:
            # Significantly worsened position
            return 0.0
    
    def _is_objective_choice(self, board: chess.Board, move: chess.Move, eval_info: Info) -> bool:
        """
        Determine if a move is an objective, principle-based choice
        (pragmatist trait)
        """
        # Objective choices are concrete, often materialistic decisions
        # - Captures with clear advantage
        # - Clear defensive moves (addressing direct threats)
        # - Obvious positional improvements
        
        # Check if it's a capturing move with clear advantage
        if board.is_capture(move):
            capturing_piece = board.piece_at(move.from_square)
            captured_piece = board.piece_at(move.to_square)
            
            if capturing_piece and captured_piece:
                # Check if it's a winning capture
                if self.PIECE_VALUES[captured_piece.piece_type] >= self.PIECE_VALUES[capturing_piece.piece_type]:
                    return True
        
        # Check if it's a defensive move addressing a direct threat
        if board.is_check():
            return True  # Any legal move in check is objective (must get out of check)
            
        # Check threatened pieces
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                # If piece is attacked and we moved it to safety
                if board.is_attacked_by(not board.turn, sq) and move.from_square == sq:
                    return True
        
        # Check if it matches the engine's objective evaluation
        if hasattr(eval_info, 'variation') and eval_info.variation and move.uci() in eval_info.variation[:2]:
            return True
            
        return False 