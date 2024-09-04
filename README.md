# Evaluating user responses to Math bot questions

## Data

### Errors in the data

Index=84 - Conversation History - the last entry in the list in this string is incomplete, so got it from the row where Index=85 and updated it in the CSV file.

### Other Important Notes

Human Evaluation seems to be incorrect in some cases as mentioned in the Problem Statement. So, added a new column "human_eval_corrected" to incorporate my opinions on the human evaluation. But, this needs a lot of manual effort, so leaving it out of scope of this project. Can be worked on in the future.

## Setup

To set up this project, follow these steps:

1. Install the required dependencies by running `$ pipenv install` or `$ pip install -r requirements.txt` command in this folder after activating your virtual environment
2. Create a .env file with "OPENAI_API_KEY='{add your OpenAI API key here}'" and place it in this folder

## Code / Algorithm

### Approach

Used an LLM agent with 3 Tools for this problem.

- a Reasoning Tool - addresses logical / reasoning questions / problems in a step-by-step logical way
- a Calculator - solves numeric computation problems
- a Symbolic Math Tool - solves questions involving algebraic expressions and equations

The Agent with the help of the above 3 Tools solves the user's math queries in a step-by-step manner, without the need for any specific Math-related training data.

### Code & Data Files

- utils.py
  - prepare_data() - Reading the data file, cleaning it up and adding any additional columns to the table that will be useful for our analysis
  - get_agent() - Creating the LLM Agent + the 3 Tools - Reasoning Tool, Calculator and Symbolic Math Tool
  - evaluate_equivalence() - Evaluating the equivalence between the last bot question and the user response using the above Agent
- main.py - uses a suitable LLM from the OpenAI API along with the above util functions to perform the analysis and store the analysis results
- process_results.py - process the analysis results stored in .pkl file by main.py
- "Accuracy Calculation Dataset.csv" - the raw data file with an additional column added for storing any manual corrections. The manual corrections are out of scope for now.
- "Accuracy Calculation Dataset with LLM Evaluation Results.csv" - created by main.py, based on the above CSV file, to store the analysis results. It has the LLM Evaluation and Time taken columns filled. (I didn't want to mess with the main file in case we redo the analysis)
- "Accuracy Calculation Dataset with LLM Evaluation Results.pkl" - same as the second CSV file above, but has additional columns including a few complex objects, so didn't store in CSV format.

## Results

Given the constraints in the Data / Problem statement i.e.,

- The last question of the bot and the user response together don't have all the context that is present in all the Conversation History,
  - GPT-4-turbo estimates its impact to account for 15% loss in the accuracy mentioned below, but this needs to be validated,
- The Human Evaluation part does seem to have significant number of errors - it is hard to estimate its impact without double-checking by Humans,
- The context present in the image links in the data isn't used,

the results came out to be

- Accuracy: 65.80%
- Median response time: 9.18 seconds (Median response time captures how most users would experience the tool)
- Average cost: $0.018 / request

## Improving the Accuracy

### Fixing errors

- tuple - 8
- lcm - 5
- round - 2
- algebraic expr to LLMMathChain - 9
- unknown format from LLM - word problem format sent to LLMMathChain - 4\*
