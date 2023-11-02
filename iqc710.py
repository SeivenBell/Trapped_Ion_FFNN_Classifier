import numpy as np
import matplotlib.pyplot as plt

# Hypothetical marginal distributions
p_x = {"Sunny": 0.5, "Cloudy": 0.3, "Rainy": 0.2}
p_y = {"Outdoors": 0.4, "Indoors": 0.6}

# Hypothetical joint distribution (reflecting some dependence between X and Y)
p_xy = {("Sunny", "Outdoors"): 0.35, 
        ("Sunny", "Indoors"): 0.15,
        ("Cloudy", "Outdoors"): 0.1,
        ("Cloudy", "Indoors"): 0.2,
        ("Rainy", "Outdoors"): 0.05,
        ("Rainy", "Indoors"): 0.15}

# Product of marginals (what the joint distribution would be if X and Y were independent)
p_xp_y = {(x, y): p_x[x] * p_y[y] for x, y in p_xy.keys()}

labels = list(p_xy.keys())
actual_joint_probs = list(p_xy.values())
independent_joint_probs = list(p_xp_y.values())

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width/2, actual_joint_probs, width, label='Actual $p(x, y)$')
rects2 = ax.bar(x + width/2, independent_joint_probs, width, label='Independent $p(x)p(y)$')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probability')
ax.set_title('Comparison between actual joint distribution and independent joint distribution')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

fig.tight_layout()

plt.show()
