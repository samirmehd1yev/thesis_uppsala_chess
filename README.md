# Chess Game Analyzer

A powerful tool for analyzing chess games with Stockfish, extracting features, and providing insightful judgments on moves.

## Features

- Analyze chess games from PGN files or strings
- Evaluate positions using Stockfish chess engine
- Classify moves as Brilliant, Great, Good, Inaccurate, Mistake, or Blunder
- Detect sacrifices, tactics, and strategic patterns
- Extract game features (opening/middlegame/endgame phases, piece mobility, etc.)
- Generate analysis reports in text or HTML format
- Parallel processing for faster analysis

## Installation

### Prerequisites

- Python 3.7+
- [Stockfish](https://stockfishchess.org/) chess engine installed and accessible in your PATH

### Steps

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Stockfish is installed and available in your PATH or provide the path explicitly

## Usage

### Command Line

```bash
python src/test.py --pgn "path/to/game.pgn" --stockfish "path/to/stockfish" --depth 16
```

#### Options

- `--pgn`: Path to PGN file or PGN string (required)
- `--stockfish`: Path to Stockfish executable (default: "stockfish")
- `--depth`: Analysis depth (default: 16)
- `--threads`: Number of threads per Stockfish instance (default: 1)
- `--hash`: Hash size in MB for Stockfish (default: 128)
- `--cpus`: Number of CPU cores to use (default: CPU count - 1)
- `--output`: Path to output file for analysis report (HTML or TXT)
- `--quiet`: Only output errors (default: false)

### Programmatic Use

```python
from analysis.game_analyzer import analyze_game

# Analyze a game with default settings
pgn_content = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O"
result = analyze_game(pgn_content)

# Or with custom settings
result = analyze_game(
    pgn_content,
    stockfish_path="/usr/local/bin/stockfish",
    analysis_depth=20,
    threads=4
)

# Access analysis components
game = result["game"]
evals = result["evals"]
judgments = result["judgments"]
features = result["features"]
```

## Project Structure

- `src/analysis/`: Core analysis modules
  - `game_analyzer.py`: Main game analysis entry point
  - `move_analyzer.py`: Move classification and judgment
  - `stockfish_handler.py`: Interface with Stockfish engine
  - `phase_detector.py`: Game phase detection
- `src/features/`: Feature extraction modules
  - `extractor.py`: Extract game features
- `src/models/`: Data models and enums
- `src/utils/`: Utility functions
- `src/test.py`: Command-line interface

## Optimizations

The codebase has been optimized for performance and readability:

1. **Parallel Processing**: Uses multiprocessing to analyze positions and moves in parallel
2. **Memory Efficiency**: Minimized redundant data storage and processing
3. **Modular Design**: Clear separation of concerns with GameAnalyzer, MoveAnalyzer, etc.
4. **Type Annotations**: Comprehensive type hints for better IDE support and code quality
5. **Error Handling**: Robust error handling throughout the analysis pipeline
6. **Documentation**: Thorough docstrings and comments for better maintainability

## Example Output

The analyzer provides detailed move-by-move analysis with evaluations and judgments:

```
=================================================================================================
GAME ANALYSIS
=================================================================================================
+-----+-------+-------------+------------+--------+----------+-------------+-------+-------------+------------+--------+----------+-------------+
|  #  | White | Eval Before | Eval After | Change | Judgment |  Top Moves  | Black | Eval Before | Eval After | Change | Judgment |  Top Moves  |
+-----+-------+-------------+------------+--------+----------+-------------+-------+-------------+------------+--------+----------+-------------+
|  1  |   e4  |     0.00    |    +0.35   | +0.35  |   GOOD   | e4, d4, Nf3 |   e5  |    +0.35    |    +0.28   | -0.07  |   GOOD   | e5, c5, e6  |
|  2  |  Nf3  |    +0.28    |    +0.41   | +0.13  |   GOOD   | Nf3, Bc4    |  Nc6  |    +0.41    |    +0.32   | -0.09  |   GOOD   | Nc6, Nf6    |
|  3  |  Bc4  |    +0.32    |    +0.56   | +0.24  |   GOOD   | Bc4, Bb5    |  Nf6  |    +0.56    |    +0.42   | -0.14  |   GOOD   | Nf6, Bc5    |
+-----+-------+-------------+------------+--------+----------+-------------+-------+-------------+------------+--------+----------+-------------+
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stockfish](https://stockfishchess.org/) chess engine
- [python-chess](https://python-chess.readthedocs.io/) library
