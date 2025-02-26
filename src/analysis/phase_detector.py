# src/analysis/phase_detector.py
import chess
from typing import List, Tuple

class GamePhaseDetector:
    def __init__(self):
        # Bit masks for board regions
        self.FIRST_RANK_MASK = 0xFF
        self.LAST_RANK_MASK = 0xFF << 56
        self.SMALL_SQUARE = 0x0303

    def find_phase_transitions(self, positions: List[chess.Board]) -> Tuple[int, int]:
        """
        Determine the game phase transitions between opening, middlegame, and endgame.
        Uses heuristics based on:
          - Count of major/minor pieces
          - Sparsity of back ranks
          - Mixedness score
        """
        middlegame_start, endgame_start = 0, 0
        
        for i, board in enumerate(positions[1:], start=1):
            move_number = (i + 1) // 2
            majors_minors = self.count_majors_and_minors(board)
            backrank_sparse = self.is_backrank_sparse(board)
            mixedness = self.calculate_mixedness(board)
            
            if middlegame_start == 0 and (majors_minors <= 10 or backrank_sparse or mixedness > 150):
                middlegame_start = move_number + 1
            
            if middlegame_start > 0 and endgame_start == 0 and majors_minors <= 6:
                endgame_start = move_number + 1
                
        if middlegame_start == 0:
            return 0, 0
            
        return middlegame_start, endgame_start

    def count_majors_and_minors(self, board: chess.Board) -> int:
        """Count the number of major and minor pieces (excluding kings and pawns)"""
        pieces = board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.KNIGHT, chess.BLACK)
        pieces |= board.pieces(chess.BISHOP, chess.WHITE) | board.pieces(chess.BISHOP, chess.BLACK)
        pieces |= board.pieces(chess.ROOK, chess.WHITE) | board.pieces(chess.ROOK, chess.BLACK)
        pieces |= board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(chess.QUEEN, chess.BLACK)
        return bin(pieces).count('1')

    def is_backrank_sparse(self, board: chess.Board) -> bool:
        """Check if the back ranks are sparsely populated (less than 4 pieces)"""
        white_back = bin(board.occupied_co[chess.WHITE] & self.FIRST_RANK_MASK).count('1')
        black_back = bin(board.occupied_co[chess.BLACK] & self.LAST_RANK_MASK).count('1')
        return white_back < 4 or black_back < 4

    def calculate_mixedness(self, board: chess.Board) -> int:
        """Calculate a mixedness score based on piece distribution in 2x2 regions"""
        total_score = 0
        for y in range(7):
            for x in range(7):
                region = self.SMALL_SQUARE << (x + 8 * y)
                white_count = bin(board.occupied_co[chess.WHITE] & region).count('1')
                black_count = bin(board.occupied_co[chess.BLACK] & region).count('1')
                total_score += self.score_region(y + 1, white_count, black_count)
        return total_score

    def score_region(self, y: int, white_count: int, black_count: int) -> int:
        """Score a 2x2 board region based on piece distribution"""
        if (white_count, black_count) == (0, 0):
            return 0
        elif (white_count, black_count) == (1, 0):
            return 1 + (8 - y)
        elif (white_count, black_count) == (2, 0):
            return 2 + (y - 2) if y > 2 else 0
        elif (white_count, black_count) == (3, 0):
            return 3 + (y - 1) if y > 1 else 0
        elif (white_count, black_count) == (4, 0):
            return 3 + (y - 1) if y > 1 else 0
        elif (white_count, black_count) == (0, 1):
            return 1 + y
        elif (white_count, black_count) == (1, 1):
            return 5 + abs(3 - y)
        elif (white_count, black_count) == (2, 1):
            return 4 + y
        elif (white_count, black_count) == (3, 1):
            return 5 + y
        elif (white_count, black_count) == (0, 2):
            return 2 + (6 - y) if y < 6 else 0
        elif (white_count, black_count) == (1, 2):
            return 4 + (6 - y)
        elif (white_count, black_count) == (2, 2):
            return 7
        elif (white_count, black_count) == (0, 3):
            return 3 + (7 - y) if y < 7 else 0
        elif (white_count, black_count) == (1, 3):
            return 5 + (6 - y)
        elif (white_count, black_count) == (0, 4):
            return 3 + (7 - y) if y < 7 else 0
        else:
            return 0
