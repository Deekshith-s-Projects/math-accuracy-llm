import pandas as pd

df = pd.read_pickle('Accuracy Calculation Dataset with LLM Evaluation Results.pkl')

# Accuracy of the algorithm
accuracy = df['llm_matches_human'].mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Median response time captures how most users would experience the tool
median_response_time = df['Time taken to complete the request'].median()
print(f"Median response time: {median_response_time:.2f} seconds")

# Average cost per request
avg_cost = df['prompt_tokens'].mean() * 10/10**6 + \
    df['completion_tokens'].mean() * 30/10**6
# These numbers change based on the model used
# GPT-4-turbo costs $10 / 1M prompt tokens & $30 / 10M completion tokens
print(f"Average cost: ${avg_cost:.3f} / request")
