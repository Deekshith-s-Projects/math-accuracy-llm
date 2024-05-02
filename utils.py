from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from math_prompt import MATH_PROMPT

def get_agent(llm):
    word_problem_template = """You are a reasoning agent tasked with solving the user's logic-based questions.
    Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give
    the final answer. Provide the response in bullet points. Question  {question} Answer"""

    # word_problem_template = """You are a reasoning agent tasked with evaluating 
    # the user's answers against the user's logic-based questions. 
    # Logically arrive at the solution, and be factual. 
    # Evaluate if the user's answer is correct, incorrect, or if you do not have sufficient context.
    # Question  {question} Answer  {answer}  Evaluation  """

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

    # wikipedia = WikipediaAPIWrapper()
    # # Wikipedia Tool
    # wikipedia_tool = Tool(
    #     name="Wikipedia",
    #     func=wikipedia.run,
    #     description="A useful tool for searching the Internet to find information on world events, issues, dates, "
    #                 "years, etc. Worth using for general topics. Use precise questions.",
    # )

    agent = initialize_agent(
        tools=[math_tool, word_problem_tool],  # wikipedia_tool skipped for now
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
        The question is: {question}. My answer is: {answer}. 
        Is my answer correct, incorrect or do you not have sufficient context?
        Provide the response as either 'Correct', 'Incorrect' or 'Insufficient context'""",
        input_variables=["question", "answer"]
    )
    prompt = template.format(question=question, answer=answer)
    try:
        result = agent(prompt)
    except Exception as e:
        result = "Error: " + str(e)
    
    return result


def prepare_data(df):
    df['Conversation History'] = df['Conversation History'].str.replace("`", "")

    return df
