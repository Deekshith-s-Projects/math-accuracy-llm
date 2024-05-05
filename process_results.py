import pandas as pd

df = pd.read_pickle('Accuracy Calculation Dataset with LLM Evaluation Results.pkl')

# Accuracy of the algorithm
accuracy = df.loc[df['LLM Equivalence Evaluation (Response)'] != 
                  'NOT_APPLICABLE', 'llm_matches_human'].mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Median response time captures how most users would experience the tool
median_response_time = df.loc[
    df['LLM Equivalence Evaluation (Response)'] !='NOT_APPLICABLE', 
        'Time taken to complete the request'].median()
print(f"Median response time: {median_response_time:.2f} seconds")

# Average cost per request
avg_cost = df.loc[df['LLM Equivalence Evaluation (Response)'] !='NOT_APPLICABLE', 
                  'prompt_tokens'].mean() * 10/10**6 + \
    df.loc[df['LLM Equivalence Evaluation (Response)'] !='NOT_APPLICABLE', 
           'completion_tokens'].mean() * 30/10**6
print(f"Average cost: ${avg_cost:.3f} / request")
