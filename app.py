import streamlit as st
from langchain.prompts import PromptTemplate , ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
import torch
from dotenv import load_dotenv
load_dotenv()
import os

assert os.environ['GOOGLE_API_KEY'], "Api key is missing in env"

# Model initialization
chat_llm = init_chat_model(model='gemini-2.0-flash', model_provider='google_genai')
chat_llm.invoke('hai, how are you')

system_prompt_template ="""You are AI assistent who explain joke provided by user, what does it mean and where does it come from in which context.
You also tell if the joke is based on any special person or a group.
You should explain what is the core idea behind the joke and why it is considerd to be joke.
Explain what is it based on like , it is racial joke ,or any other related.
if the user provide anything other than the jokes what you heared then just replay that you don't here that joke or you don't know"""

chat_prompt_template = ChatPromptTemplate(
    messages= [
        ('system', system_prompt_template),
        ('user', '{joke}')
    ]
)

output_parser = StrOutputParser()      

chain = chat_prompt_template | chat_llm | output_parser

st.title("JOKE EXPLAINER")
joke = st.text_input(placeholder='tell your joke here', label='')

if joke:
    with st.spinner('Model Generating...'):
        st.write_stream(chain.stream(joke))



