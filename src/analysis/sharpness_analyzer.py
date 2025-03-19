# src/analysis/wdl_sharpness.py
import math
import chess
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from models.data_classes import Info
import subprocess
import json
import os
import logging
import time

class WdlSharpnessAnalyzer:
    """
    Analyzer for measuring position sharpness in chess based on WDL (Win/Draw/Loss) statistics.
    This follows the approach described in the Chess Engine Lab articles, using the LC0's WDL model
    to calculate the sharpness of a position.
    """
    
    def __init__(self, lc0_path=None, network_path=None, nodes=1000):
        """
        Initialize the sharpness analyzer with Leela Chess Zero (LC0) configuration.
        
        Args:
            lc0_path: Path to the LC0 executable. If None, assumes "lc0" is in PATH
            network_path: Path to the neural network file. If None, uses default
            nodes: Number of nodes for LC0 analysis
        """
        self.lc0_path = lc0_path or "lc0"
        self.network_path = network_path
        self.nodes = nodes
        self.logger = logging.getLogger('sharpness_analyzer')
        
        # Check if LC0 is available - raise error if not available
        self.lc0_available = self.check_lc0_available()
        if not self.lc0_available:
            raise RuntimeError("Leela Chess Zero (LC0) is not available. Install LC0 to continue.")
    
    def check_lc0_available(self) -> bool:
        """
        Check if Leela Chess Zero is available in the system.
        
        Returns:
            Boolean indicating if LC0 is available
        """
        try:
            process = subprocess.Popen([self.lc0_path, "--help"], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
            process.communicate(timeout=2)
            return process.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False
    
    def calculate_sharpness(self, 
                           board: chess.Board, 
                           wdl: Tuple[float, float, float]) -> float:
        """
        Calculate the sharpness score based on a formula posted by the
        LC0 team on Twitter.
        
        Args:
            board: The chess board position
            wdl: Tuple of (win, draw, loss) probabilities in range 0-1
            
        Returns:
            A float representing the sharpness score of the position
        """
        # max() to avoid a division by 0, min() to avoid log(0)
        W = min(max(wdl[0], 0.0001), 0.9999)
        L = min(max(wdl[2], 0.0001), 0.9999)

        # Formula based on the entropy of the position
        # Higher entropy = more sharp/critical position
        try:
            return (max(2/(np.log((1/W)-1) + np.log((1/L)-1)), 0))**2
        except Exception as e:
            print(f"Error calculating sharpness: {e}")
            return 0.0
    
    def get_lc0_wdl(self, board: chess.Board) -> Dict[str, float]:
        """
        Get WDL (Win/Draw/Loss) probabilities from Leela Chess Zero for a given position.
        Will throw an error if LC0 is not available or doesn't work.
        
        Args:
            board: Chess board position to analyze
            
        Returns:
            Dictionary with 'wins', 'draws', and 'losses' probabilities (0-1 range)
        """
        # Convert board to FEN
        fen = board.fen()
        
        # Start LC0 process
        self.logger.debug(f"Starting LC0 process with command: {self.lc0_path}")
        try:
            process = subprocess.Popen(
                self.lc0_path,
                universal_newlines=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1
            )
            
            # Initialize UCI mode
            self._send_command(process, "uci")
            output = self._read_until(process, "uciok")
            # print(f"UCI initialization output: {output}")
            
            # Enable WDL output
            self._send_command(process, "setoption name UCI_ShowWDL value true")
            
            # Set neural network if provided
            if self.network_path:
                self._send_command(process, f"setoption name WeightsFile value {self.network_path}")
            
            # Wait for engine to be ready
            self._send_command(process, "isready")
            ready_output = self._read_until(process, "readyok")
            # print(f"Ready output: {ready_output}")
            
            # Set up position
            self._send_command(process, "ucinewgame")
            self._send_command(process, f"position fen {fen}")
            
            # Start analysis with node limit
            self._send_command(process, f"go nodes {self.nodes}")
            
            # Wait for analysis to complete and collect output
            analysis_output = self._read_until(process, "bestmove")
            # print(f"Analysis output sample: {analysis_output[:200]}...")
            
            # Parse the output to extract WDL
            wdl_values = None
            for line in analysis_output.split('\n'):
                if "wdl" in line.lower():
                    # print(f"Found WDL in line: {line}")
                    # Example format: "info ... wdl 350 500 150" (W/D/L in permille)
                    parts = line.split()
                    try:
                        wdl_index = parts.index("wdl")
                        if wdl_index + 3 < len(parts):
                            w = float(parts[wdl_index + 1]) / 1000.0
                            d = float(parts[wdl_index + 2]) / 1000.0
                            l = float(parts[wdl_index + 3]) / 1000.0
                            wdl_values = (w, d, l)
                            # print(f"Parsed WDL values: W={w}, D={d}, L={l}")
                    except (ValueError, IndexError) as e:
                        # print(f"Error parsing WDL data in line: {e}")
                        pass
            
            # Clean up
            self._send_command(process, "quit")
            process.terminate()
            
            # Check if we found WDL values
            if wdl_values:
                return {"wins": wdl_values[0], "draws": wdl_values[1], "losses": wdl_values[2]}
            
            # If we reach here, we couldn't find WDL data
            self.logger.error("Failed to parse WDL data from LC0 output")
            raise RuntimeError("Failed to parse WDL data from LC0 output")
            
        except Exception as e:
            self.logger.error(f"Error communicating with LC0: {e}")
            if process and process.poll() is None:
                process.terminate()
            raise RuntimeError(f"Error communicating with LC0: {e}")
    
    def _send_command(self, process, command):
        """Send a command to the LC0 process"""
        # print(f"Sending command: {command}")
        process.stdin.write(f"{command}\n")
        process.stdin.flush()
    
    def _read_until(self, process, target_text, timeout=10):
        """Read from process until target_text is found or timeout occurs"""
        output = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            line = process.stdout.readline().strip()
            if line:
                output.append(line)
                if target_text in line:
                    break
        
        return "\n".join(output)
    
    def calculate_position_sharpness(self,
                                   board: chess.Board,
                                   eval_info: Info) -> Dict[str, float]:
        """
        Calculate sharpness metrics for a given position using Leela Chess Zero.
        
        Args:
            board: Current board position
            eval_info: Evaluation information from the engine
            
        Returns:
            Dictionary with sharpness scores for each position
        """
        result = {
            'sharpness': 0.0,
            'white_sharpness': 0.0,
            'black_sharpness': 0.0
        }
        
        # Get WDL directly from Leela - no try/except to allow errors to propagate
        wdl_dict = self.get_lc0_wdl(board)
        wins = wdl_dict.get('wins', 0)
        draws = wdl_dict.get('draws', 0)
        losses = wdl_dict.get('losses', 0)
        
        sharpness = self.calculate_sharpness(board, (wins, draws, losses))
        
        # Both sides experience the same objective sharpness in the position
        result['sharpness'] = sharpness
        result['white_sharpness'] = sharpness if board.turn == chess.WHITE else 0.0
        result['black_sharpness'] = sharpness if board.turn == chess.BLACK else 0.0
        print(f"Position sharpness: {sharpness:.2f}, turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        self.logger.debug(f"Position sharpness from LC0: {sharpness:.2f}, turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        return result
    
    def calculate_cumulative_sharpness(self, sharpness_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate cumulative sharpness over a list of sharpness scores.
        
        Args:
            sharpness_scores: List of sharpness score dictionaries
            
        Returns:
            Dictionary with cumulative sharpness scores
        """
        if not sharpness_scores:
            return {'sharpness': 0.0, 'white_sharpness': 0.0, 'black_sharpness': 0.0}
        
        # Separate scores for white and black positions
        white_positions = []
        black_positions = []
        
        for i, score in enumerate(sharpness_scores):
            # Even ply numbers (0, 2, 4...) are positions where it's White's turn to move
            # Odd ply numbers (1, 3, 5...) are positions where it's Black's turn to move
            if i % 2 == 0:
                white_positions.append(score.get('sharpness', 0.0))
            else:
                black_positions.append(score.get('sharpness', 0.0))
        
        # Calculate cumulative values
        white_cumulative = sum(white_positions)
        black_cumulative = sum(black_positions)
        
        # Overall cumulative is the sum of white and black cumulative values
        overall_cumulative = white_cumulative + black_cumulative
        
        return {
            'sharpness': overall_cumulative,
            'white_sharpness': white_cumulative,
            'black_sharpness': black_cumulative
        }
    
    def calculate_sharpness_change(self, 
                                  prev_sharpness: float, 
                                  curr_sharpness: float) -> float:
        """
        Calculate the change in sharpness after a move.
        
        Args:
            prev_sharpness: Sharpness before the move
            curr_sharpness: Sharpness after the move
            
        Returns:
            Change in sharpness (positive means the position got sharper)
        """
        return curr_sharpness - prev_sharpness
    
# Function to print sharpness analysis
def print_wdl_sharpness_analysis(sharpness_scores, colorama_fore, colorama_style):
    """
    Print the WDL sharpness analysis in a formatted way.
    """
    from tabulate import tabulate
    
    print("\n" + "="*80)
    print(f"{colorama_fore.BLUE}{colorama_style.BRIGHT}POSITION SHARPNESS ANALYSIS (WDL MODEL){colorama_style.RESET_ALL}")
    print("="*80)
    
    headers = ["#", "Ply", "Position Sharpness", "Description"]
    rows = []
    
    for i, sharpness in enumerate(sharpness_scores):
        ply = sharpness.get('ply', i)
        move_num = (ply // 2) + 1
        side = "White" if ply % 2 == 0 else "Black"
        move = f"{move_num}.{'' if side == 'White' else '..'}"
        
        # Format sharpness score
        sharpness_value = sharpness.get('sharpness', 0.0)
        
        # Determine color based on sharpness
        if sharpness_value >= 7.0:
            color = colorama_fore.RED
            description = "Extremely sharp"
        elif sharpness_value >= 5.0:
            color = colorama_fore.YELLOW
            description = "Very sharp"
        elif sharpness_value >= 3.0:
            color = colorama_fore.GREEN
            description = "Moderately sharp" 
        elif sharpness_value >= 1.0:
            color = colorama_fore.CYAN
            description = "Somewhat sharp"
        else:
            color = colorama_fore.BLUE
            description = "Quiet position"
        
        sharpness_formatted = f"{color}{sharpness_value:.2f}{colorama_style.RESET_ALL}"
        
        rows.append([move, ply, sharpness_formatted, description])
    
    print(tabulate(rows, headers=headers, tablefmt="pretty"))
    print("="*80)
    
    # Calculate cumulative sharpness
    if sharpness_scores:
        # Separate white and black positions
        white_positions = [s.get('white_sharpness', 0.0) for s in sharpness_scores]
        black_positions = [s.get('black_sharpness', 0.0) for s in sharpness_scores]
        
        # Calculate cumulative values
        white_cumulative = sum(white_positions)
        black_cumulative = sum(black_positions)
        overall_cumulative = white_cumulative + black_cumulative
        
        print(f"\nCumulative Position Sharpness: {overall_cumulative:.2f}")
        print(f"White's Cumulative Position Sharpness: {white_cumulative:.2f} (positions where White is to move)")
        print(f"Black's Cumulative Position Sharpness: {black_cumulative:.2f} (positions where Black is to move)")
        print("="*80)