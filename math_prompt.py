from langchain.prompts import PromptTemplate

_PROMPT_TEMPLATE = """Translate a math problem into a expression that can be executed using Python's numexpr library. Use the output of running this code to answer the question.

Question: ${{Question with math problem.}}

```text
${{single line mathematical expression that solves the problem}}
```

...numexpr.evaluate(text)...

```output
${{Output of running the code}}
```

Answer: ${{Answer}}


But in the case of round function, use eval() instead of numexpr.evaluate(). 
For example, eval("round(37593**(1/5), 2)") will return 8.22.


Question: ${{Question with math problem.}}

```text
${{single line mathematical expression that solves the problem}}
```

...eval(text)...

```output
${{Output of running the code}}
```

Answer: ${{Answer}}



Begin.

Question: What is 37593 \* 67?

```text
37593 * 67
```

...numexpr.evaluate("37593 \* 67")...

```output
2518731
```

Answer: 2518731

Question: 37593^(1/5)

```text
37593**(1/5)
```

...numexpr.evaluate("37593\*\*(1/5)")...

```output
8.222831614237718
```

Answer: 8.222831614237718


Question: round 7.1999999 to 2 decimal places.

```text
eval("round(7.1999999, 2)")
```

...eval("round(7.1999999, 2)")...

```output
7.20
```

Answer: 7.20


Question: {question}
"""


MATH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)
