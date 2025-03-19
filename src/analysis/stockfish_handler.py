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
        try:
            if score.is_mate():
                # For mate scores, we need to extract it differently
                mate_score = None
                try:
                    # Try to get the mate score directly
                    mate_score = score.mate()
                except AttributeError:
                    # If that fails, try to get it from white's perspective
                    try:
                        mate_score = score.white().mate()
                    except Exception as e:
                        logger.warning(f"Error extracting mate score: {e}")
                
                if mate_score is not None:
                    return {"type": "mate", "value": mate_score}
                else:
                    # Fallback to centipawn score if mate extraction fails
                    return {"type": "cp", "value": 0}
            else:
                # Convert score to centipawns from white's perspective
                return {"type": "cp", "value": score.white().score()}
        except Exception as e:
            logger.warning(f"Error extracting evaluation: {e}")
            return {"type": "cp", "value": 0}

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
            # print(analysis)
            # print('\n\n\n\n')
            # wdl = list(chess.engine.PovWdl.white(analysis['wdl']))
            # print(wdl)
            wdl_model = "sf"  # Use Stockfish's WDL model
            wdl = None
            
            # Extract WDL (Win/Draw/Loss) probabilities if possible
            if isinstance(analysis, list) and analysis and "score" in analysis[0]:
                # Get WDL from the best move's score
                score = analysis[0]["score"]
                if hasattr(score, "wdl"):
                    # Get WDL with default parameters (using Stockfish model at ply=30)
                    wdl = score.wdl(model=wdl_model, ply=ply)
                    
                    # print(f"Ply: {ply}")
                    
                    # print(f"Debug: WDL: {wdl}\n\n\n\n")
                    # print(f"Debug: WDL: {wdl}\n\n\n\n")
            
            # Extract best move variations in UCI format
            variations = []
            primary_eval = None
            multipv_data = []  # Store the multipv data
            
            if isinstance(analysis, list):
                # Sort PV lines by score to ensure best moves first
                sorted_pvs = sorted(analysis, key=lambda x: self._get_score_value(x.get("score")), reverse=board.turn)
                
                # Handle multipv result (list of infos)
                if sorted_pvs:
                    primary_eval = sorted_pvs[0]  # First PV is the best line
                    
                    for pv_info in sorted_pvs:
                        if "pv" in pv_info and pv_info["pv"]:
                            variations.append(pv_info["pv"][0].uci())
                            
                            # Store the multipv data
                            score_info = {}
                            if "score" in pv_info:
                                score = pv_info["score"]
                                if score.is_mate():
                                    # For mate scores, we need to extract it differently
                                    mate_score = None
                                    try:
                                        # Try to get the mate score directly
                                        mate_score = score.mate()
                                    except AttributeError:
                                        # If that fails, try to get it from white's perspective
                                        try:
                                            mate_score = score.white().mate()
                                        except Exception as e:
                                            logger.warning(f"Error extracting mate score: {e}")
                                    
                                    if mate_score is not None:
                                        score_info["mate"] = mate_score
                                else:
                                    # Use white().score() to get the centipawn value
                                    score_info["cp"] = score.white().score()
                            
                            multipv_data.append({
                                "move": pv_info["pv"][0].uci(),
                                "score": score_info
                            })
            else:
                # Handle single info dict
                primary_eval = analysis
                if "pv" in analysis and analysis["pv"]:
                    variations.append(analysis["pv"][0].uci())
                    
                    # Store the multipv data for single PV
                    score_info = {}
                    if "score" in analysis:
                        score = analysis["score"]
                        if score.is_mate():
                            # For mate scores, we need to extract it differently
                            mate_score = None
                            try:
                                # Try to get the mate score directly
                                mate_score = score.mate()
                            except AttributeError:
                                # If that fails, try to get it from white's perspective
                                try:
                                    mate_score = score.white().mate()
                                except Exception as e:
                                    logger.warning(f"Error extracting mate score: {e}")
                            
                            if mate_score is not None:
                                score_info["mate"] = mate_score
                        else:
                            # Use white().score() to get the centipawn value
                            score_info["cp"] = score.white().score()
                    
                    multipv_data.append({
                        "move": analysis["pv"][0].uci(),
                        "score": score_info
                    })
            
            # Prepare evaluation dictionary
            eval_dict = {"type": "cp", "value": 0}  # Default value
            
            if primary_eval and "score" in primary_eval:
                eval_dict = self._extract_evaluation_dict(primary_eval["score"])
            
            # Prepare WDL dictionary if available
            wdl_dict = None
            # if wdl:
            #     # The WDL values from Stockfish are in permille (0-1000)
            #     # Convert to 0-1 range for better handling in sharpness calculation
            #     wdl_dict = {
            #         "wins": wdl[0] / 1000.0 ,
            #         "draws": wdl[1] / 1000.0,
            #         "losses": wdl[2] / 1000.0
            #     }
            
            # Create Info object with evaluation and variations
            info = Info(
                ply=ply,
                eval=eval_dict,
                variation=variations,
                wdl=wdl_dict,
                multipv=multipv_data  # Add the multipv data
            )
            
            return info
        except Exception as e:
            logger.error(f"Error evaluating position at ply {ply}: {e}")
            # Return a default Info object in case of error
            return Info(
                ply=ply,
                eval={"type": "cp", "value": 0},
                variation=[],
                wdl=None,
                multipv=[]  # Add empty multipv data
            )

    def close(self):
        """Clean up engine resources"""
        try:
            if hasattr(self, 'engine'):
                self.engine.quit()
        except Exception as e:
            logger.error(f"Error closing Stockfish engine: {e}")