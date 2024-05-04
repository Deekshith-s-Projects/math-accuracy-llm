from langchain.chains import LLMMathChain, LLMChain
from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain_community.callbacks.manager import get_openai_callback
from math_prompt import MATH_PROMPT
from time import time
import ast


def get_agent(llm):
    word_problem_template = """You are a reasoning agent tasked with solving the user's logic-based questions.
    Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give
    the final answer. Provide the response in bullet points. Question  {question} Answer"""

    math_assistant_prompt = PromptTemplate(
        input_variables=["question"],
        template=word_problem_template
    )

    word_problem_chain = LLMChain(llm=llm, 
                                  prompt=math_assistant_prompt)
    word_problem_tool = Tool.from_function(name="Reasoning Tool",
                                           func=word_problem_chain.run,
                                           description="Useful for when you need to answer logic-based/reasoning  "
                                                       "questions.",
                                        )

    problem_chain = LLMMathChain.from_llm(llm=llm)
    # problem_chain = LLMMathChain.from_llm(llm=llm, prompt=MATH_PROMPT)
    math_tool = Tool.from_function(name="Calculator",
                                   func=problem_chain.run,
                                   description="Useful for when you need to answer numeric questions. This tool is "
                                               "only for math questions and nothing else. Only input math "
                                               "expressions, without text",
                                   )

    symbolic_math_chain = LLMSymbolicMathChain.from_llm(llm=llm)
    symbolic_math_tool = Tool.from_function(name="Symbolic Math",
                                            func=symbolic_math_chain.run,
                                            description="Useful for when you need to answer symbolic math questions. "
                                                        "This tool is only for symbolic math questions and nothing else. Only "
                                                        "input math expressions that have text",
                                            )

    agent = initialize_agent(
        tools=[math_tool, word_problem_tool, symbolic_math_tool], 
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    return agent


def evaluate_equivalence(agent, question, answer):
    template = PromptTemplate(
        template="""I'm trying to evaluate my answer to a math question. 
        The question is: \"{question}\". My answer is: \"{answer}\". 
        Ignore rounding errors beyond 1 decimal point.
        Is my answer correct, incorrect or do you not have sufficient context?
        Provide your response only as either \"Correct\", \"Incorrect\" or \"Insufficient context\"""",
        input_variables=["question", "answer"]
    )
    prompt = template.format(question=question, answer=answer)
    
    try:
        with get_openai_callback() as cb:
            start = time()
            result = agent(prompt)
            time_lapsed = time() - start
            # Below code is not really relevant to be included in computing the 
            # response time because in real-time scenario, all the student will 
            # need is the response, not the token counts
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
    except Exception as e:
        result = {'error': "Error: " + str(e), 'output': 'Has error'}
        time_lapsed = None
        prompt_tokens = None
        completion_tokens = None
    
    return {'llm_eval_results': result, 'Time taken to complete the request': 
            time_lapsed, 'prompt_tokens': prompt_tokens, 'completion_tokens':
            completion_tokens}


def prepare_data(df):
    # Cleaning
    df['Conversation History'] = df['Conversation History'].str.replace("`", "")

    # Extracting the last dialog / question
    df['conv_history_list'] = df['Conversation History'].apply(ast.literal_eval)
    df['last_dialog'] = df['conv_history_list'].apply(lambda x: x[-1])
    df['last_question'] = df['last_dialog'].apply(lambda x: x['bot'])

    return df
