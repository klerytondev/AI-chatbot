import os
import pandas as pd
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

persist_directory = 'db'

# Processa o arquivo PDF e retorna os chunks
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read()) # Persisti o arquivo temporário em disco
        temp_file_path = temp_file.name # Recupera o caminho do arquivo temporário em disco C:\Users\user\AppData\Local\Temp\tmp0z7z7z9v.pdf

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Deleta o arquivo temporário
    os.remove(temp_file_path)

    # Divide o texto em chunks
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    # Divide os documentos em chunks
    chunks = text_spliter.split_documents(documents=docs)
    return chunks

# Processa o arquivo CSV e retorna os chunks
def process_csv(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        temp_file.write(file.read()) # Persisti o arquivo temporário em disco
        temp_file_path = temp_file.name # Recupera o caminho do arquivo temporário em disco C:\Users\user\AppData\Local\Temp\tmp0z7z7z9v.csv

    df = pd.read_csv(temp_file_path)
    docs = df.to_string()

    # Deleta o arquivo temporário
    os.remove(temp_file_path)

    # Divide o texto em chunks
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    # Divide o texto em chunks
    chunks = text_spliter.split_text(docs)
    return chunks

# Define o tipo de arquivo e chama a função correspondente
def process_files(file):
    file_name = file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension == '.pdf':
        return process_pdf(file)
    elif file_extension == '.csv':
        return process_csv(file)
    else:
        raise ValueError("Tipo de arquivo não suportado")

# Carrega o vector_store existente ou retorna None caso não exista
def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None

# Adiciona os chunks ao vector_store
def add_to_vector_store(chunks, vector_store=None):
    # Verifica se o vector_store já existe e adiciona os chunks
    if vector_store:
        vector_store.add_documents(chunks)
    # Caso não exista, cria um novo vector_store e adiciona os chunks
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store

# Carrega o vector_store    
vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='Chat PyGPT',
    page_icon='📄',
)
st.header('🤖 Chat com IA 🤖')

# Cria um cache para armazenar os chunks
with st.sidebar:
    st.header('📄 Upload de arquivos ')
    uploaded_file = st.file_uploader("Adicione seus arquivos", 
                                     type=["pdf", "csv"], 
                                     accept_multiple_files=True)
# Processa os arquivos armazenando os chunks    
    if uploaded_file:
        with st.spinner('Processando arquivos...'):
            all_chunks = []
            for file in uploaded_file:
                chunks = process_files(file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )

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
question = st.chat_input('Digite sua pergunta aqui:')

# st.chat_message('user').write(question)