import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI

from utils import get_agent, evaluate_equivalence, prepare_data
from pprint import pp

load_dotenv()

# llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
llm = ChatOpenAI(model='gpt-4-turbo', temperature=0)
# DONE: Tried modifying the parameters used above e.g., top_p, maybe not temperature
# DONE: - LOW PRIORITY Try exploring other models
# TODO: Try Prompt Engineering
# Clarity and Specificity: Write clear and concise prompts that accurately 
# represent the desired task or question. Avoid ambiguity and focus on the 
# specific information you need from the LLM.
# Instruction and Context: Provide clear instructions and context within the 
# prompt. This helps the LLM understand the purpose of the conversation and the 
# information it should access.
# Examples and References: Include relevant examples or references within the 
# prompt, especially when dealing with complex topics. This can guide the LLM 
# towards a more accurate response.

# TODO: Setup the llm to return the token counts to compute costs

# TODO: - ONLY IF NEEDED - Increasing temp can help get around the limitations 
# of numexpr like unsupported functions like round, but it leads to more errors.
# Can venture into it if really needed since all we need is equivalence in this 
# problem!
agent = get_agent(llm)

df = prepare_data(pd.read_csv("Accuracy Calculation Dataset.csv"))


# Use the agent to evaluate the equivalence using the last dialog
# When the last dialog is not sufficient, mark as "NOT_APPLICABLE" for now

df = df.sample(3)  # for testing purposes
# incorrect_ones = df[df['LLM Equivalence Evaluation (Response)'] != df['Human Evaluation']].index
# df.loc[incorrect_ones, ['llm_eval_results', 'Time taken to complete the request']] = \
#     df.loc[incorrect_ones, :].apply(
#     lambda x: evaluate_equivalence(agent, x['last_question'], 
#                                    x['User Response']), axis=1, result_type ='expand')
x = df.apply(
    lambda x: evaluate_equivalence(agent, x['last_question'], 
                                   x['User Response']), axis=1, 
                                   result_type ='expand')
df[['llm_eval_results', 'Time taken to complete the request', 'prompt_tokens', 
    'completion_tokens']] = x
eval_map = {'Correct': 'EQUIVALENT', 'Incorrect': 'NOT_EQUIVALENT', 
            'Insufficient context': 'NOT_APPLICABLE', 'Has error': 'HAS_ERROR'}
df['LLM Equivalence Evaluation (Response)'] = \
    df['llm_eval_results'].apply(lambda x: x['output']).map(eval_map)
df['error'] = df['llm_eval_results'].apply(lambda x: x.get('error'))
df['llm_matches_human'] = \
    df['LLM Equivalence Evaluation (Response)'] == df['human_eval_corrected']



# TODO: Add a column indicating if image links are used in the last question or the conv history
# TODO: Manual effort to include corrections for the errors in the dataset and NOT_APPLICABLE cases
# in the human_eval_corrected column


# TODO: - LOW PRIORITY - Try using the whole history
# Try using just the last dialog first, and history only when it is not 
# sufficient This enhances both the speed and cost of the algorithm
# Caching may help with faster response times

# ChatOpenAI() allows you to utilize memory to store previous conversation history. 
# This context can be crucial for tasks that require understanding the flow of conversation. 
# Enable memory in the ChatOpenAI constructor and consider passing relevant previous 
# interactions when making new requests.


# TODO: - LOW PRIORITY Try adding image link handling to the agent
