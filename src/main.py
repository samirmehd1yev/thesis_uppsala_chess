from typing import List, Dict
import chess.pgn
from analysis.phase_detector import GamePhaseDetector
from analysis.move_analyzer import MoveAnalyzer
from analysis.stockfish_handler import StockfishHandler
from analysis.clustering import StyleClusterer
from features.extractor import FeatureExtractor
from models.data_classes import FeatureVector

class ChessAnalyzer:
    def __init__(self, stockfish_path: str = None, n_clusters: int = 4):
        self.stockfish = StockfishHandler(stockfish_path) if stockfish_path else None
        self.feature_extractor = FeatureExtractor()
        self.clusterer = StyleClusterer(n_clusters=n_clusters)
        
    def train_clustering(self, pgn_files: List[str]) -> None:
        """Train clustering model on collection of games"""
        feature_vectors = []
        
        for pgn in pgn_files:
            game = chess.pgn.read_game(pgn)
            if not game:
                continue
                
            # Get engine evals if available
            evals = None
            if self.stockfish:
                positions = self.feature_extractor._get_positions(game)
                evals = [self.stockfish.evaluate_position(pos, i) 
                        for i, pos in enumerate(positions)]
            
            # Extract features
            features = self.feature_extractor.extract_features(game, evals)
            feature_vectors.append(features)
            
        # Train clustering
        self.clusterer.fit(feature_vectors)
        
    def analyze_game(self, pgn: str) -> Dict:
        """Analyze single game and assign to cluster"""
        game = chess.pgn.read_game(pgn)
        if not game:
            raise ValueError("Invalid PGN")

        # Get positions and engine evaluations if available
        positions = self.feature_extractor._get_positions(game)
        evals = None
        if self.stockfish:
            evals = [self.stockfish.evaluate_position(pos, i) 
                    for i, pos in enumerate(positions)]

        # Extract features
        features = self.feature_extractor.extract_features(game, evals)
        
        # Get cluster assignment
        cluster = self.clusterer.predict(features)
        
        return {
            'features': features.__dict__,
            'cluster': cluster,
            'eval_stats': self._get_eval_stats(evals) if evals else None
        }

    def analyze_games(self, pgns: List[str]) -> List[Dict]:
        """Analyze multiple games"""
        return [self.analyze_game(pgn) for pgn in pgns]

    def _get_eval_stats(self, evals: List[Dict]) -> Dict:
        """Calculate evaluation statistics"""
        if not evals:
            return None

        mistakes = 0
        blunders = 0
        inaccuracies = 0
        good=0
        total = len(evals) - 1  # -1 because we compare pairs

        for i in range(1, len(evals)):
            judgment = MoveAnalyzer.analyze_move(evals[i-1], evals[i])
            if judgment == Judgment.GOOD:
                good+=1
            elif judgment == Judgment.BLUNDER:
                blunders += 1
            elif judgment == Judgment.MISTAKE:
                mistakes += 1
            elif judgment == Judgment.INACCURACY:
                inaccuracies += 1

        return {
            'accuracy': good / total * 100,
            'blunder_rate': blunders / total * 100,
            'mistake_rate': mistakes / total * 100,
            'inaccuracy_rate': inaccuracies / total * 100
        }

    def close(self):
        """Clean up resources"""
        if self.stockfish:
            self.stockfish.close()

# Example usage
def main():
    # Example game
    test_pgn = """
    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7
    6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5
    """
    
    # Initialize analyzer
    analyzer = ChessAnalyzer(
        stockfish_path="stockfish",
        n_clusters=4
    )
    
    try:
        # Train on some example games first
        training_pgns = [test_pgn]  # In practice, use many games
        analyzer.train_clustering(training_pgns)
        
        # Analyze a game
        results = analyzer.analyze_game(test_pgn)
        
        # Print results
        print("\nAnalysis Results:")
        print("-" * 40)
        
        print("\nFeatures:")
        for name, value in results['features'].items():
            print(f"{name}: {value:.3f}")
        
        print(f"\nAssigned Cluster: {results['cluster']}")
        
        if results['eval_stats']:
            print("\nEvaluation Statistics:")
            print(f"Accuracy: {results['eval_stats']['accuracy']:.1f}%")
            print(f"Blunder Rate: {results['eval_stats']['blunder_rate']:.1f}%")
            print(f"Mistake Rate: {results['eval_stats']['mistake_rate']:.1f}%")
            print(f"Inaccuracy Rate: {results['eval_stats']['inaccuracy_rate']:.1f}%")
            
    finally:
        # Clean up
        analyzer.close()

if __name__ == "__main__":
    main()
