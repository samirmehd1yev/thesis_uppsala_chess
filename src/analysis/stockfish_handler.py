# src/analysis/stockfish_handler.py
from typing import List, Dict, Optional
import chess
from stockfish import Stockfish
from models.data_classes import Info

class StockfishHandler:
    def __init__(self, path: str = "stockfish", depth: int = 16):
        self.stockfish = Stockfish(
            path=path,
            depth=depth,
            parameters={
                "Threads": 7, 
                "Hash": 128,
                "MultiPV": 3,  # Get top 3 moves
                "Minimum Thinking Time": 20
            }
        )
        self.depth = depth


    def evaluate_position(self, board: chess.Board, ply: int) -> Info:
        """
        Evaluate position and return Info object with corrected perspective.
        Args:
            board: The board position to evaluate
            ply: The current ply number
        Returns:
            Info object containing evaluation and best move variation
        """
        self.stockfish.set_fen_position(board.fen())
        eval_dict = self.stockfish.get_evaluation()
        variations = self.stockfish.get_top_moves(3)

        return Info(
            ply=ply,
            eval=eval_dict,
            variation=variations
        )

    def close(self):
        """Clean up Stockfish engine resources"""
        del self.stockfish