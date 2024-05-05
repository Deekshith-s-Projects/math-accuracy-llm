import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI

from utils import get_agent, evaluate_equivalence, prepare_data
from pprint import pp


def main():
    # llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
    llm = ChatOpenAI(model='gpt-4-turbo', temperature=0)  
    # GPT-4-turbo shows significant improvement in accuracy over GPT-3.5-turbo.

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
    # Try using just the last question first, and use history only when there is  
    # insufficient context. This saves both time and money. 
    # Also, consider using caching for faster response times overall. 
    # ChatOpenAI() allows you to utilize memory to store previous conversation 
    # history. This helps 


    # TODO: - LOW PRIORITY Try adding image link handling to the agent


    # Saving the results
    df.iloc[:, :10].to_csv('Accuracy Calculation Dataset with LLM Evaluation Results.csv', 
                        index=False)
    df.to_pickle('Accuracy Calculation Dataset with LLM Evaluation Results.pkl')


if __name__ == "__main__":
    load_dotenv()
    main()
