import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

models = [
    ("Tev1 0.8B", "Together AI", 63.5, "#1a1a1a"),
    ("Tev1 4B",   "Together AI", 73.3, "#1a1a1a"),
    ("Nimble 9B", "Bespoke Labs", 75.7, "#1a1a1a"),
    ("Jev 1.13",  "TypeSafe",    76.0, "#d4d0cb"),
]

labels     = [f"{m}\n{p}" for m, p, _, _ in models]
values     = [v for _, _, v, _ in models]
colors     = [c for _, _, _, c in models]

fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.barh(labels, values, color=colors, height=0.55, zorder=2)

# value labels to the right of each bar
for bar, val, color in zip(bars, values, colors):
    ax.text(
        bar.get_width() + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{val}%",
        va="center", ha="left",
        fontsize=11, fontweight="bold",
        color="#1a1a1a",
    )

ax.set_xlim(0, 107)
ax.set_xticks([0, 25, 50, 75])
ax.set_xticklabels(["0", "25", "50", "75"], color="#666")
ax.text(100, -0.7, "100%", ha="right", va="top", color="#666", fontsize=9,
        transform=ax.get_xaxis_transform())

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#ccc")
ax.spines["left"].set_color("#ccc")
ax.tick_params(axis="x", colors="#888", length=0)
ax.tick_params(axis="y", length=0, pad=8)
ax.grid(axis="x", color="#e8e8e8", linewidth=0.8, zorder=1)

# y-axis tick labels: bold model name, gray provider
for label, (model, provider, _, _) in zip(ax.get_yticklabels(), models):
    label.set_text("")  # cleared below via custom text

ax.set_yticklabels([])
for i, (model, provider, _, _) in enumerate(models):
    ax.text(-1, i, model,     ha="right", va="bottom", fontsize=10, fontweight="bold", color="#1a1a1a")
    ax.text(-1, i, provider,  ha="right", va="top",    fontsize=9,  color="#888")

fig.suptitle("Decision models on Ollama", x=0.13, y=1.02, ha="left",
             fontsize=13, fontweight="bold", color="#1a1a1a")
ax.set_title("Bespoke Labs public benchmarks · accuracy, higher is better",
             loc="left", fontsize=9, color="#888", pad=6)

caption = (
    "Mean accuracy across 13 public data sets with human labels, covering 3,880 decisions.\n"
    "Nimble and Tev1 were evaluated on Ollama; Jev 1.13 is from Bespoke Labs' published run."
)
fig.text(0.13, -0.08, caption, fontsize=8, color="#555", va="top")

plt.tight_layout()
plt.savefig("benchmark.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved benchmark.png")
plt.show()
