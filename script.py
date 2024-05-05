import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI

from utils import get_agent, evaluate_equivalence, prepare_data
from pprint import pp

load_dotenv()

# llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
llm = ChatOpenAI(model='gpt-4-turbo', temperature=0)

agent = get_agent(llm)

df = prepare_data(pd.read_csv("Accuracy Calculation Dataset.csv"))


df[['llm_eval_results', 'Time taken to complete the request', 'prompt_tokens', 
    'completion_tokens']] = df.apply(lambda x: evaluate_equivalence(
        agent, x['last_question'], x['User Response']), axis=1, 
        result_type ='expand')
eval_map = {'Correct': 'EQUIVALENT', 'Incorrect': 'NOT_EQUIVALENT', 
            'Insufficient context': 'NOT_APPLICABLE', 'Has error': 'HAS_ERROR'}
df['LLM Equivalence Evaluation (Response)'] = \
    df['llm_eval_results'].apply(lambda x: x['output']).map(eval_map)
df['error'] = df['llm_eval_results'].apply(lambda x: x.get('error'))
df['llm_matches_human'] = \
    df['LLM Equivalence Evaluation (Response)'] == df['human_eval_corrected']
# The column 'human_eval_corrected' is added to note the corrected version of 
# 'Human Evaluation' column without messing with it. 
# TODO: This correction needs manual effort though.


# TODO: - LOW PRIORITY - Try using the whole history
# Try using just the last question first, and use history only when it is not 
# sufficient. This enhances both the speed and cost of the algorithm
# Caching may help with faster response times

# ChatOpenAI() allows you to utilize memory to store previous conversation 
# history. This context can be crucial for tasks that require understanding 
# the flow of conversation. Enable memory in the ChatOpenAI constructor and 
# consider passing relevant previous interactions when making new requests.


# TODO: - LOW PRIORITY Try adding image link handling to the agent
