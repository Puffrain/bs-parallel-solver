#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("bench.csv")
t1  = df["wall"].iloc[0]
df["speedup"] = t1 / df["wall"]

plt.plot(df["proc"], df["speedup"], 'o-')
plt.xlabel("#Processes")
plt.ylabel("Speed-up")
plt.title("Strong scaling (CN scheme)")
plt.grid(True)
plt.savefig("speedup.png", dpi=150)
print("Saved speedup.png")

