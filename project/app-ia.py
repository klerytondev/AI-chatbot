import os
import tempfile
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser


def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, parser, client = initial_parameters() 

st.set_page_config(
    page_title='Chat PyGPT',
    page_icon='📄',
)
st.header('🤖 Chat com IA 🤖')

with st.sidebar:
    st.header('📄 Upload de arquivos ')
    uploaded_file = st.file_uploader("Adicione seus arquivos", 
                                     type=["pdf", "csv"], 
                                     accept_multiple_files=True)
    
    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]
    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM de sua preferência',
        options=model_options,
    )