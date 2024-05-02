import pandas as pd
import ast
from dotenv import load_dotenv
from langchain_openai import OpenAI

from utils import get_agent, prepare_data

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
# TODO: Increasing temp can help get around the limitations of numexpr like unsupported 
# functions like round, but it leads to more errors. Can venture into it if really 
# needed since all we need is equivalence in this problem!
agent = get_agent(llm)

df = prepare_data(pd.read_csv("Accuracy Calculation Dataset.csv"))

conv_histories = df['Conversation History'].apply(ast.literal_eval)
last_dialogs = conv_histories.apply(lambda x: x[-1])


# TODO: Try using just the last dialog first, and history only when it is not sufficient
# This enhances both the speed and cost of the algorithm
