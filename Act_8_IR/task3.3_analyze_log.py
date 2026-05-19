import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/query_log.csv")
print("Total queries:", len(df))

# Frequency
freq = df['query_text'].value_counts()
print("\nTop-10 most frequent queries:")
print(freq.head(10))

# Bar chart
plt.figure(figsize=(10,6))
freq.plot(kind='bar')
plt.title("Query Frequency Distribution")
plt.xlabel("Query")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/query_frequency.png")
print("Chart saved to outputs/query_frequency.png")

# Average clicks simulated - already in top_result_id
print("\nAverage results clicked: N/A (simulated)")
