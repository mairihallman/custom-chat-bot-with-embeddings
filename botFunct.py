import ast
import math
import warnings

warnings.simplefilter("ignore")
import openai
import pandas as pd
import tiktoken
from IPython.core.display import HTML,display

GPT_MODEL = "gpt-3.5-turbo-16k"
EMBEDDING_MODEL = "text-embedding-ada-002"
encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

def jaccard_ish(a,b):
    a = set(a)
    b = set(b)
    num = len(b)*len(a & b)
    den = len(a | b)
    return(num/den)

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
):
    query_embedding = encoding.encode(query)
    strings_and_relatednesses = [
        (row["text"], jaccard_ish(query_embedding, ast.literal_eval(row["embedding"])))
        for i, row, in df.iterrows()
    ]
    result = sorted(strings_and_relatednesses, key=lambda x: x[1], reverse=True)
    return(result)

def num_tokens(text:str, model: str = GPT_MODEL) -> int:
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
):
    text = strings_ranked_by_relatedness(query, df)
    strings = [i[0] for i in text]
    introduction = """Use the following to answer questions. If that doesn't work, write "I don't know"."""
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_string = f"{string}"
        if(
            num_tokens(message + next_string + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_string
    return message + question

def ask(
    query: str,
    subject: str,
    df: pd.DataFrame,
    model: str = "gpt-4",
    token_budget: int = 8192 - 500,
    print_message: bool = False,
):
   
    message = query_message(query, df, model=model, token_budget=token_budget)

    response = openai.ChatCompletion.create(
        messages = [{"role":"system",
                     "content":f"You answer questions about {subject}"},
                    {"role":"user",
                     "content":message}],
        model = model,
        temperature = 0,
    )

    display(HTML(response["choices"][0]["message"]["content"]))

