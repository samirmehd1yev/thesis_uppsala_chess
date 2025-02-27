# src/analysis/stockfish_handler.py
import chess
import chess.engine
from typing import List, Dict, Optional, Union, Any
from models.data_classes import Info
import logging
import os

logger = logging.getLogger('chess_analyzer')

class StockfishHandler:
    def __init__(self, path: str = "stockfish", depth: int = 16, threads: int = 1, hash_size: int = 128):
        """
        Initialize the StockfishHandler with chess.engine library.
        
        Args:
            path: Path to the Stockfish executable
            depth: Search depth for analysis
            threads: Number of CPU threads to use for analysis
            hash_size: Hash table size in MB
        """
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            # Configure engine parameters - avoid setting managed options
            self.engine.configure({
                "Threads": threads,
                "Hash": hash_size
            })
            self.depth = depth
            self.path = path
        except FileNotFoundError:
            logger.error(f"Stockfish engine not found at path: {path}")
            raise FileNotFoundError(f"Stockfish engine not found at path: {path}")
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish engine: {e}")
            raise

    def get_best_moves(self, board: chess.Board, num_moves: int = 3) -> List[str]:
        """
        Retrieve the best moves for a given position.
        
        Args:
            board: The board position to evaluate
            num_moves: Number of best moves to retrieve
            
        Returns:
            List of best moves in UCI format
        """
        if not board:
            return []
            
        try:
            # Using multipv parameter in analyse directly
            info = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth),
                multipv=num_moves
            )
            
            # Extract moves from each PV line
            moves = []
            
            if isinstance(info, list):
                # Handle multipv result (list of infos)
                # Sort PV lines by score to ensure best moves first
                sorted_pvs = sorted(info, key=lambda x: self._get_score_value(x.get("score")), reverse=board.turn)
                
                for pv_info in sorted_pvs:
                    if "pv" in pv_info and len(pv_info["pv"]) > 0:
                        moves.append(pv_info["pv"][0].uci())
            else:
                # Handle single info dict
                if "pv" in info and len(info["pv"]) > 0:
                    moves.append(info["pv"][0].uci())
            
            return moves
        except Exception as e:
            logger.error(f"Error getting best moves: {e}")
            return []
            
    def _get_score_value(self, score: Optional[chess.engine.Score]) -> int:
        """Helper to convert PovScore to a comparable value for sorting"""
        if score is None:
            return 0
            
        # For mate scores, use a large value
        if score.is_mate():
            mate_score = score.white().mate()
            if mate_score is not None:
                if mate_score > 0:
                    return 10000 - mate_score  # Winning mate (closer mates are better)
                else:
                    return -10000 - mate_score  # Losing mate (further mates are better)
            return 0
        
        # Regular centipawn score - extract using white() to get a consistent perspective
        try:
            return score.white().score()
        except Exception as e:
            logger.warning(f"Error getting score value: {e}")
            return 0

    def _extract_evaluation_dict(self, score: chess.engine.Score) -> Dict[str, Any]:
        """
        Convert a chess.engine.Score object to an evaluation dictionary
        
        Args:
            score: Chess engine score object
            
        Returns:
            Dictionary with evaluation information
        """
        if score.is_mate():
            return {"type": "mate", "value": score.mate()}
        else:
            # Convert score to centipawns from white's perspective
            return {"type": "cp", "value": score.white().score()}

    def evaluate_position(self, board: chess.Board, ply: int) -> Info:
        """
        Evaluate position and return Info object with evaluation and best move variation.
        
        Args:
            board: The board position to evaluate
            ply: The current ply number
        Returns:
            Info object containing evaluation and best move variations
        """
        if not board:
            return Info(ply=ply, eval={"type": "cp", "value": 0}, variation=[])
            
        try:
            # Get analysis with multiple variations
            analysis = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth),
                multipv=3
            )
            
            # Extract best move variations in UCI format
            variations = []
            primary_eval = None
            
            if isinstance(analysis, list):
                # Sort PV lines by score to ensure best moves first
                sorted_pvs = sorted(analysis, key=lambda x: self._get_score_value(x.get("score")), reverse=board.turn)
                
                # Handle multipv result (list of infos)
                if sorted_pvs:
                    primary_eval = sorted_pvs[0]  # First PV is the best line
                    
                    for pv_info in sorted_pvs:
                        if "pv" in pv_info and pv_info["pv"]:
                            variations.append(pv_info["pv"][0].uci())
            else:
                # Handle single info dict
                primary_eval = analysis
                if "pv" in analysis and analysis["pv"]:
                    variations.append(analysis["pv"][0].uci())
            
            # Prepare evaluation dictionary
            eval_dict = {"type": "cp", "value": 0}  # Default value
            
            if primary_eval and "score" in primary_eval:
                eval_dict = self._extract_evaluation_dict(primary_eval["score"])
            
            # Create Info object with evaluation and variations
            info = Info(
                ply=ply,
                eval=eval_dict,
                variation=variations
            )
            
            return info
        except Exception as e:
            logger.error(f"Error evaluating position at ply {ply}: {e}")
            # Return a default Info object in case of error
            return Info(
                ply=ply,
                eval={"type": "cp", "value": 0},
                variation=[]
            )

    def close(self):
        """Clean up engine resources"""
        try:
            if hasattr(self, 'engine'):
                self.engine.quit()
        except Exception as e:
            logger.error(f"Error closing Stockfish engine: {e}")