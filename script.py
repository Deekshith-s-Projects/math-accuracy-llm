import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate

from utils import get_agent, evaluate_equivalence, prepare_data
from pprint import pp

load_dotenv()

# llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
llm = ChatOpenAI(model='gpt-4-turbo', temperature=0)
# TODO: Check if this step can be improved to optimize for the assignment Criteria
# TODO: See if the OpenAI response can be improved by using Prompt Engineering

# TODO: - ONLY IF NEEDED - Increasing temp can help get around the limitations 
# of numexpr like unsupported functions like round, but it leads to more errors.
# Can venture into it if really needed since all we need is equivalence in this 
# problem!
agent = get_agent(llm)

df = prepare_data(pd.read_csv("Accuracy Calculation Dataset.csv"))


# Use the agent to evaluate the equivalence using the last dialog
# When the last dialog is not sufficient, mark as "NOT_APPLICABLE" for now

df = df.sample(30)  # for testing purposes
df[['llm_eval_results', 'Time taken to complete the request']] = df.apply(
    lambda x: evaluate_equivalence(agent, x['last_question'], 
                                   x['User Response']), axis=1, result_type ='expand')
eval_map = {'Correct': 'EQUIVALENT', 'Incorrect': 'NOT_EQUIVALENT', 
            'Insufficient context': 'NOT_APPLICABLE', 'Has error': 'HAS_ERROR'}
df['LLM Equivalence Evaluation (Response)'] = \
    df['llm_eval_results'].apply(lambda x: x['output']).map(eval_map)
df['error'] = df['llm_eval_results'].apply(lambda x: x.get('error'))
df['llm_matches_human'] = \
    df['LLM Equivalence Evaluation (Response)'] == df['human_eval_corrected']



# TODO: Add a column indicating if image links are used in the last question or the user response
# TODO: Manual effort to include corrections for the errors in the dataset and NOT_APPLICABLE cases
# in the human_eval_corrected column




# TODO: - LOW PRIORITY - Try using the whole history
# Try using just the last dialog first, and history only when it is not 
# sufficient This enhances both the speed and cost of the algorithm

# TODO: - LOW PRIORITY Try adding image link handling to the agent
