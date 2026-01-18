#!/usr/bin/env python3
"""Quick and dirty fetch - minimal API call."""
import wandb

api = wandb.Api(timeout=30)
entity = "mllab-ts-universit-di-trieste"
project = "CounterFactualDPG"

print(f"Fetching from {entity}/{project}...")
runs = api.runs(f"{entity}/{project}", per_page=10)

for i, run in enumerate(runs):
    if i >= 10:
        break
    print(f"{i+1}. {run.name} - state={run.state}")
    
print("Done.")
