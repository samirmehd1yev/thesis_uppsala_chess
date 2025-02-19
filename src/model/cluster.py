import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Partial load of ds.
nrows = 10_000_000
df = pd.read_csv("../../data/processed/transformed_data.csv", header=0, nrows=nrows)

# Drop the last game.
df[df["id"] != df["id"].max()]
df["main_eco"] = df["ECO"].str[0]

# Remove bad games.
df = df[df["main_eco"] != "?"]
# undefined = df[df["main_eco"] == "?"]

first_moves = df.groupby("id").first()
counts = (
    first_moves.groupby(["white_username", "main_eco"])
    .size()
    .reset_index(level=[0, 1])
    .rename(columns={0: "count"})
    .pivot(index="white_username", columns="main_eco", values="count")
    .fillna(0)
    .astype(int)
)


# Example
player_with_most_games = first_moves["white_username"].value_counts().idxmax()
counts[counts.index == player_with_most_games]

# ECO types
opening_types = ["A", "B", "C", "D", "E"]
total_counts = counts[opening_types].sum(axis=1)
probs = counts[opening_types].div(total_counts, axis=0).fillna(0)


# Sanity check with random probs.
# randomized_probs = pd.DataFrame(np.random.uniform(0, 1, size=probs.shape), columns=["A", "B", "C", "D", "E"])
#
#
# def softmax(x):
#     exp_x = np.exp(x)
#     return exp_x / exp_x.sum(axis=1, keepdims=True)
#
#
# randomized_probs = pd.DataFrame(softmax(randomized_probs.values), columns=randomized_probs.columns)
#
#
# print(df)
# print(counts)
# probs = randomized_probs

scaler = StandardScaler()
X_scaled = scaler.fit_transform(probs)

kmeans = KMeans(n_clusters=2, random_state=42)
probs["cluster"] = kmeans.fit_predict(X_scaled)
probs["cluster"].value_counts()


fig, axes = plt.subplots(2, 5, figsize=(20, 20))

# Flatten the axes array for easier indexing
axes = axes.flatten()


# Define a function to plot on specific axes
def plot_on_axes(cola, colb, ax):
    sns.scatterplot(x=cola, y=colb, hue="cluster", data=probs, palette="Set1", s=100, ax=ax, alpha=0.7)
    for i in [350, 657, 890]:
        ax.text(probs[cola].iloc[i], probs[colb].iloc[i], f"{probs.iloc[i].name}", fontsize=9, ha="right")
    ax.set_xlabel(f"p of {cola}")
    ax.set_ylabel(f"p of {colb}")
    ax.set_title(f"{cola} vs {colb}")


# Generate all pairs of columns for plotting
pairs = list(itertools.combinations(opening_types, r=2))

# Plot each pair of columns on a specific subplot
for i, (cola, colb) in enumerate(pairs):
    plot_on_axes(cola, colb, axes[i])

# Hide unused subplots
for j in range(len(pairs), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for better visibility
plt.tight_layout()

# Save the figure
plt.savefig('opening_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
