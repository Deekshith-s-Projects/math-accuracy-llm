import pandas as pd
import ast
from utils import get_agent, prepare_data


agent = get_agent()

df = prepare_data(pd.read_csv("Accuracy Calculation Dataset.csv"))

conv_histories = df['Conversation History'].apply(ast.literal_eval)
