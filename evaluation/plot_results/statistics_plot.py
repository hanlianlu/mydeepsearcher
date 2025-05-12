import tkinter as tk
from tkinter import filedialog, simpledialog
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Select the JSON file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select JSON File",
    filetypes=[("JSON files", "*.json")]
)

if not file_path:
    print("No file selected. Exiting.")
    exit()

# Load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract recall values
deepsearcher_recall2 = data["deepsearcher"]["average_recall"]["2"]
deepsearcher_recall5 = data["deepsearcher"]["average_recall"]["5"]
naive_rag_recall2 = data["naive_rag"]["average_recall"]["2"]
naive_rag_recall5 = data["naive_rag"]["average_recall"]["5"]

# Set up the bar chart
metrics = ["Recall@2", "Recall@5"]
deepsearcher_values = [deepsearcher_recall2, deepsearcher_recall5]
naive_rag_values = [naive_rag_recall2, naive_rag_recall5]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, deepsearcher_values, width, label='DeepSearcher', color='#6c8ebf')
rects2 = ax.bar(x + width/2, naive_rag_values, width, label='Naive RAG', color='#a1c084')

# Customize the chart
ax.set_title('Recall Comparison by Metric', fontsize=16, fontweight='bold')
ax.set_xlabel('Recall Metric (@k)', fontsize=14)
ax.set_ylabel('Recall', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(loc='upper left', fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle='--', alpha=0.7)

# Add labels on top of each bar
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# Save the plot with user-input filename
filename = simpledialog.askstring("Input", "Enter filename for the plot (without .png):")
if filename:
    output_file = os.path.join(os.path.dirname(__file__), f"{filename}.png")
else:
    output_file = os.path.join(os.path.dirname(__file__), "recall_plot.png")
plt.savefig(output_file)
print(f"Plot saved as: {output_file}")