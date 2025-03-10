import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib as mpl

# Set up the figure with a specific size - increasing size for better visibility
fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

# Remove axis ticks and labels
ax.set_xlim(0, 12)  # Increased x-range for more space
ax.set_ylim(0, 8)   # Increased y-range for more space
ax.axis('off')

# Title - adjusted position
ax.text(6, 7.5, 'Chess Data Preprocessing Pipeline', 
        fontsize=28, fontweight='bold', ha='center')  # Increased font size

# Create boxes with rounded corners using patches
def create_box(x, y, width, height, title, content, color, edge_color):
    # Main box
    box = patches.FancyBboxPatch(
        (x, y), width, height, boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
        facecolor=color, edgecolor=edge_color, linewidth=2
    )
    ax.add_patch(box)
    
    # Title
    ax.text(x + width/2, y + height - 0.3, title, 
            fontsize=16, fontweight='bold', ha='center')  # Increased font size
    
    # Content
    if isinstance(content, list):
        for i, line in enumerate(content):
            ax.text(x + 0.3, y + height - 0.6 - i * 0.35, line, fontsize=13)  # Increased spacing and font size
    else:
        ax.text(x + width/2, y + height/2, content, fontsize=13, ha='center')  # Increased font size

# Draw arrows
def draw_arrow(start_x, start_y, end_x, end_y):
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2))  # Increased line width

# Data Source box - adjusted positions
create_box(1, 5, 3, 1.5, 'Data Source', 
           ['LumbrasGigaBase', '(15M games, 700K+ players)'], 
           '#e6f2ff', '#3385ff')

# PGN Processing box - adjusted positions and increased size
create_box(6, 5, 3, 1.5, 'PGN Processing', 
           ['• Memory-mapped I/O (mmap)', 
            '• Multi-core parallel processing', 
            '• Regex optimization'], 
           '#e6ffe6', '#33cc33')

# Data Cleaning box - adjusted positions and increased size
create_box(6, 3, 3, 1.5, 'Data Cleaning & Filtering', 
           ['• BOT player removal', 
            '• Time control filtering', 
            '• Invalid game removal'], 
           '#fff2e6', '#ff9933')

# Game Selection box - adjusted positions and increased size
create_box(6, 1, 3, 1.5, 'Game Selection', 
           ['• Prioritize classified players', 
            '• Balanced representation (~100/player)', 
            '• Quality control checks'], 
           '#ffe6f2', '#ff3385')

# Final Dataset box - adjusted positions and increased size
create_box(10, 1, 3, 1.5, 'Final Dataset', 
           ['~5,000 selected games', 
            'Ready for feature extraction'], 
           '#e6e6ff', '#3333cc')

# Process Overview box - adjusted positions and increased size
create_box(1, 1, 3, 3.5, 'Process Overview', '', '#f0f0f0', '#999999')
ax.text(1.3, 4.2, 'Input Format:', fontsize=14, fontweight='bold')
ax.text(1.5, 3.8, '• PGN files', fontsize=13)
ax.text(1.3, 3.4, 'Transformations:', fontsize=14, fontweight='bold')
ax.text(1.5, 3.0, '• Extract metadata', fontsize=13)
ax.text(1.5, 2.6, '• Clean move sequences', fontsize=13)
ax.text(1.5, 2.2, '• Extract evaluations', fontsize=13)
ax.text(1.3, 1.8, 'Output Format:', fontsize=14, fontweight='bold')
ax.text(1.5, 1.4, '• Structured CSV', fontsize=13)

# Key Metrics box - adjusted positions and increased size
create_box(10, 3, 3, 1.5, 'Key Metrics', '', '#f0f0f0', '#999999')
# Add color rectangles as legend
ax.add_patch(patches.Rectangle((10.3, 3.9), 0.25, 0.25, facecolor='#e6f2ff', edgecolor='#3385ff'))
ax.text(10.7, 4.0, 'Initial: 15M games', fontsize=13)
ax.add_patch(patches.Rectangle((10.3, 3.5), 0.25, 0.25, facecolor='#fff2e6', edgecolor='#ff9933'))
ax.text(10.7, 3.6, 'After cleaning: ~10M', fontsize=13)
ax.add_patch(patches.Rectangle((10.3, 3.1), 0.25, 0.25, facecolor='#e6e6ff', edgecolor='#3333cc'))
ax.text(10.7, 3.2, 'Final: ~5K games', fontsize=13)

# Draw arrows connecting the boxes - adjusted positions
# Horizontal arrows
draw_arrow(4, 5.75, 6, 5.75)  # Data Source to PGN Processing
draw_arrow(9, 1.75, 10, 1.75)  # Game Selection to Final Dataset

# Vertical arrows
draw_arrow(7.5, 5, 7.5, 4.5)  # PGN Processing to Data Cleaning
draw_arrow(7.5, 3, 7.5, 2.5)  # Data Cleaning to Game Selection

# Add tight layout and save the figure with higher DPI
plt.tight_layout(pad=1.5)  # Increased padding
plt.savefig('/Users/samir/Desktop/Uppsala/Thesis/thesis_chess_code/src/data_processing/chess_data_pipeline.png', dpi=400, bbox_inches='tight')  # Increased DPI and specified full path
plt.show() 