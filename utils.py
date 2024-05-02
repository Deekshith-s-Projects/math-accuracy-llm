from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent

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


def prepare_data(df):
    df['Conversation History'] = df['Conversation History'].str.replace("`", "")

    return df
