import os
import pandas as pd
import textract
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import openai

load_dotenv()

try:
    def loadModel(userQuery):
        #Loading the API KEY
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        loader = PyPDFLoader("./machine_learning_tutorial.pdf")
        pages = loader.load_and_split()
        chunks = pages
        
        #Converting PDF to Text
        doc = textract.process("./machine_learning_tutorial.pdf")

        #Save to .txt and reopen (helps prevent issues)
        with open('text.txt', 'w') as f:
            f.write(doc.decode('utf-8'))

        with open('text.txt', 'r') as f:
            text = f.read()

        # #Create function to count tokens
        # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # def count_tokens(text: str) -> int:
        #     return len(tokenizer.encode(text))

        # #Split text into chunks
        # text_splitter = RecursiveCharacterTextSplitter(
        #     # Set a really small chunk size, just to show.
        #     chunk_size = 512,
        #     chunk_overlap  = 24,
        #     length_function = count_tokens,
        # )

        # chunks = text_splitter.create_documents([text])
        # type(chunks[0]) 

        # # Get embedding model
        # embeddings = OpenAIEmbeddings()

        # # Create vector database
        # db = FAISS.from_documents(chunks, embeddings)

        # # Check similarity search is working
        # query = userQuery
        # docs = db.similarity_search(query)

        # # Create QA chain to integrate similarity search with user queries (answer query from knowledge base)
        # chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        # docs = db.similarity_search(query)

        # result = chain.run(input_documents=docs, question=query)
        # print(result)
        # return result
        
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        print(text)
        
        def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
            messages = [{"role": "user", "content":prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            return response.choices[0].message["content"]
        
        prod_review = text
        
        
        prompt = f"""
        Your task is to answer the question provided, delimited by ** ** on the text ```{text}```
        
        ** What are all the tasks that are due today? **
        """
        
        response = get_completion(prompt)
        return response
except:
    print("error here")
