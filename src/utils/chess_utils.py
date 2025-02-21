import chess
from typing import Set

def get_piece_squares(board: chess.Board, piece_type: chess.PieceType, 
                     color: chess.Color) -> Set[chess.Square]:
    """Get squares occupied by specific piece type"""
    return board.pieces(piece_type, color)

def calculate_piece_mobility(board: chess.Board, square: chess.Square) -> int:
    """Calculate number of legal moves for piece"""
    piece = board.piece_at(square)
    if not piece:
        return 0
    moves = 0
    for move in board.legal_moves:
        if move.from_square == square:
            moves += 1
    return moves

def is_center_square(square: chess.Square) -> bool:
    """Check if square is in center (d4,d5,e4,e5)"""
    center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
    return square in center_squares

def get_game_phases(moves: int, mg_start: int, eg_start: int) -> List[float]:
    """Calculate phase lengths as percentages"""
    if moves == 0:
        return [0, 0, 0]
    
    opening = mg_start / moves
    middlegame = (eg_start - mg_start) / moves
    endgame = (moves - eg_start) / moves
    
    return [opening, middlegame, endgame]
