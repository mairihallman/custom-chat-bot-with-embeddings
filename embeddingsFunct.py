import openai
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import tiktoken
from IPython.core.display import HTML,display

# GPT_MODEL = "gpt-3.5-turbo-16k"
EMBEDDING_MODEL = "text-embedding-ada-002"
enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

def get_embeddings(path_to_pdf_file, path_to_save_as_csv):
    reader = PdfReader(path_to_pdf_file)
    
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    texts = text_splitter.split_text(raw_text)
    
    
    new = []
    for i in texts:
        t = i.replace("\n"," ")
        new.append(t)
        
    embeddings = [enc.encode(i) for i in new]
        
    df = pd.DataFrame({"text":new,"embedding":embeddings})
    
    df.to_csv(path_to_save_as_csv,index=False)
    
    display(HTML(f"Embeddings saved as: {path_to_save_as_csv}"))
    