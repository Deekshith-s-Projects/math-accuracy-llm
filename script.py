import pandas as pd
from utils import get_agent


# df = pd.read_csv("Accuracy Calculation Dataset.csv")
df = pd.read_excel("Accuracy Calculation Dataset.xlsx")  # CSV version has errors due to encoding issues
agent = get_agent()
