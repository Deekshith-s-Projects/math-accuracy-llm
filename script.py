import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

from utils import get_agent, evaluate_equivalence, prepare_data
from pprint import pp

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
# TODO: - ONLY IF NEEDED - Increasing temp can help get around the limitations 
# of numexpr like unsupported functions like round, but it leads to more errors.
# Can venture into it if really needed since all we need is equivalence in this 
# problem!
agent = get_agent(llm)

df = prepare_data(pd.read_csv("Accuracy Calculation Dataset.csv"))


# Use the agent to evaluate the equivalence using the last dialog
# When the last dialog is not sufficient, mark as "NOT_APPLICABLE" for now

# evaluation = evaluate_equivalence(agent, df['last_question'][20], 
#                                   df['User Response'][20])
df = df.sample(30)  # for testing purposes
df[['llm_eval_results', 'Time taken to complete the request']] = df.apply(
    lambda x: evaluate_equivalence(agent, x['last_question'], 
                                   x['User Response']), axis=1, result_type ='expand')
eval_map = {'Correct': 'EQUIVALENT', 'Incorrect': 'NOT_EQUIVALENT', 
            'Insufficient context': 'NOT_APPLICABLE', 'Has error': 'HAS_ERROR'}
df['LLM Equivalence Evaluation (Response)'] = \
    df['llm_eval_results'].apply(lambda x: x['output']).map(eval_map)
df['error'] = df['llm_eval_results'].apply(lambda x: x.get('error'))

# TODO: - LOW PRIORITY - Try using the whole history
# Try using just the last dialog first, and history only when it is not 
# sufficient This enhances both the speed and cost of the algorithm

# TODO: - LOW PRIORITY Try adding image link handling to the agent
