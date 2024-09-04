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


# INVESTIGATING THE ERRORS
df_err = df[df['error'].notna()]  # 30 errors
df_incorr = df[~(df['llm_matches_human']) & 
               ~(df['LLM Equivalence Evaluation (Response)'].isin(
                   ['HAS_ERROR', 'NOT_APPLICABLE']))]  # 105
df_fp = df_incorr[df_incorr['human_eval_corrected'] == 'NOT_EQUIVALENT']  # 19 false positives
df_fn = df_incorr[df_incorr['human_eval_corrected'] == 'EQUIVALENT']  # 86 false negatives
df_insuff_cntxt = df[df['LLM Equivalence Evaluation (Response)'] == 
                     'NOT_APPLICABLE']  # 140 
