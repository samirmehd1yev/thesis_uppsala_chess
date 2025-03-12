# src/analysis/leela_handler.py
import chess
import chess.engine
import re
import subprocess
import threading
import time
from typing import List, Dict, Optional, Union, Any, Tuple
from models.data_classes import Info
import logging
import os

logger = logging.getLogger('chess_analyzer')

class LeelaChessHandler:
    def __init__(self, path: str = "lc0", threads: int = 2, network_path: Optional[str] = None):
        """
        Initialize the LeelaChessHandler to interface with Leela Chess Zero engine.
        
        Args:
            path: Path to the Leela Chess Zero executable (default: "lc0")
            threads: Number of CPU threads to use for analysis
            network_path: Optional path to a neural network weights file
        """
        self.path = path
        self.process = None
        self.threads = threads
        self.network_path = network_path
        self.latest_info = {}
        self.latest_score = None
        self.latest_wdl = None
        self.latest_nodes = 0
        self.latest_bestmove = None
        self.readyok_received = False
        
        try:
            self.start_engine()
        except Exception as e:
            logger.error(f"Failed to initialize Leela Chess Zero engine: {e}")
            raise
    
    def start_engine(self):
        """Start the Leela Chess Zero engine process"""
        try:
            self.process = subprocess.Popen(
                self.path,
                universal_newlines=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1
            )
            
            # Set up a thread to read engine output asynchronously
            self.output_thread = threading.Thread(target=self._read_output)
            self.output_thread.daemon = True
            self.output_thread.start()
            
            # Initialize UCI interface
            self.send_command("uci")
            time.sleep(1)  # Give engine time to initialize
            
            # Set engine options
            self.send_command(f"setoption name Threads value {self.threads}")
            self.send_command("setoption name UCI_ShowWDL value true")
            
            if self.network_path:
                self.send_command(f"setoption name WeightsFile value {self.network_path}")
            
            # Tell engine we're ready
            self.send_command("isready")
            time.sleep(0.5)  # Wait for readyok response
            
        except Exception as e:
            logger.error(f"Error starting Leela Chess Zero engine: {e}")
            if self.process:
                self.process.terminate()
            raise
    
    def send_command(self, command):
        """Send a command to the engine"""
        if self.process and self.process.poll() is None:
            try:
                logger.debug(f">>> {command}")
                self.process.stdin.write(f"{command}\n")
                self.process.stdin.flush()
            except Exception as e:
                logger.error(f"Error sending command to engine: {e}")
        else:
            logger.error("Engine not running, cannot send command.")
    
    def _read_output(self):
        """Read and process output from the engine"""
        while self.process and self.process.poll() is None:
            line = self.process.stdout.readline().strip()
            if line:
                logger.debug(f"<<< {line}")
                
                if line.startswith("info") and ("depth" in line or "score" in line):
                    self._parse_info(line)
                elif line.startswith("bestmove"):
                    self.latest_bestmove = line.split()[1]
                elif line == "readyok":
                    self.readyok_received = True
    
    def _parse_info(self, info_str):
        """Parse information output from the engine"""
        # Extract depth
        depth_match = re.search(r'depth (\d+)', info_str)
        if depth_match:
            self.latest_info['depth'] = int(depth_match.group(1))
        
        # Extract score as centipawns
        score_match = re.search(r'score cp ([-\d]+)', info_str)
        if score_match:
            self.latest_score = int(score_match.group(1)) / 100.0
            self.latest_info['score'] = self.latest_score
        
        # Extract WDL values
        wdl_match = re.search(r'wdl (\d+) (\d+) (\d+)', info_str)
        if wdl_match:
            win = int(wdl_match.group(1))
            draw = int(wdl_match.group(2))
            loss = int(wdl_match.group(3))
            self.latest_wdl = (win, draw, loss)
            self.latest_info['wdl'] = self.latest_wdl
        
        # Extract nodes
        nodes_match = re.search(r'nodes (\d+)', info_str)
        if nodes_match:
            self.latest_nodes = int(nodes_match.group(1))
            self.latest_info['nodes'] = self.latest_nodes
    
    def get_wdl(self, board: chess.Board, time_limit: int = 1) -> Tuple[int, int, int]:
        """
        Get WDL (Win/Draw/Loss) probabilities from Leela Chess Zero.
        
        Args:
            board: The chess board position to analyze
            time_limit: Time in seconds to spend on analysis
            
        Returns:
            Tuple of (win, draw, loss) values in the range 0-1000
        """
        self.analyze_position(board, time_limit)
        
        if self.latest_wdl:
            return self.latest_wdl
        
        # Return default values if WDL not available
        return (0, 1000, 0)
    
    def analyze_position(self, board: chess.Board, time_limit: int = 1) -> Dict[str, Any]:
        """
        Analyze a position using Leela Chess Zero.
        
        Args:
            board: The chess board position to analyze
            time_limit: Time in seconds to spend on analysis
            
        Returns:
            Dictionary containing analysis results
        """
        # Reset state for new analysis
        self.latest_info = {}
        self.latest_score = None
        self.latest_wdl = None
        self.latest_nodes = 0
        self.latest_bestmove = None
        
        # Set up new position
        self.send_command("ucinewgame")
        self.send_command(f"position fen {board.fen()}")
        self.send_command(f"go movetime {time_limit * 1000}")
        
        # Wait for analysis to complete
        time.sleep(time_limit + 0.5)
        
        # Return the latest information
        result = {
            'score': self.latest_score,
            'depth': self.latest_info.get('depth'),
            'nodes': self.latest_nodes,
            'wdl': self.latest_wdl,
            'bestmove': self.latest_bestmove
        }
        
        return result
    
    def evaluate_position(self, board: chess.Board, ply: int) -> Info:
        """
        Evaluate a position and return an Info object.
        
        Args:
            board: The chess board position to analyze
            ply: Current ply in the game
            
        Returns:
            Info object containing evaluation data
        """
        analysis = self.analyze_position(board)
        
        # Extract evaluation as a dictionary
        eval_dict = {}
        if analysis.get('score') is not None:
            # Use centipawns for score
            eval_dict = {
                'type': 'cp',
                'value': int(analysis.get('score') * 100)  # Convert to centipawns
            }
        
        # Prepare WDL dictionary
        wdl_dict = None
        if analysis.get('wdl'):
            win, draw, loss = analysis.get('wdl')
            total = win + draw + loss
            
            if total > 0:
                wdl_dict = {
                    "wins": win / total,
                    "draws": draw / total,
                    "losses": loss / total
                }
        
        # Create Info object
        return Info(
            ply=ply,
            eval=eval_dict,
            wdl=wdl_dict,
            lc0_wdl=analysis.get('wdl')  # Store original Leela WDL for reference
        )
    
    def close(self):
        """Terminate the engine process"""
        if self.process and self.process.poll() is None:
            self.send_command("quit")
            self.process.terminate()
            self.process = None
            logger.info("Leela Chess Zero engine terminated.") 