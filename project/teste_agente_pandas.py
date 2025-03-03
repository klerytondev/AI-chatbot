import os
import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import sqlite3
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import tiktoken  # import para contagem de tokens

def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, parser, client = initial_parameters()

def count_tokens(text: str, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def ask_question(model, query, df):
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    system_prompt = '''
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    informe o seguinte: "Não há informações disponíveis"
    e não responda algo que estiver fora do contexto.
    Responda em formato de markdown e com visualizações elaboradas e interativas.
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', query))
    prompt = ChatPromptTemplate.from_messages(messages)
    
    # Contagem dos tokens de entrada
    input_tokens = count_tokens(prompt.format(), model_name="gpt-3.5-turbo-0613")
    st.write("Tokens de entrada: ", input_tokens)
    
    response = agent.run(prompt)
    
    # Contagem dos tokens de saída
    output_tokens = count_tokens(response, model_name="gpt-3.5-turbo-0613")
    st.write("Tokens de saída: ", output_tokens)
    
    return response

def query_database(query):
    conn = sqlite3.connect('caminho/para/seu/banco_de_dados.db')
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Configurações da página
st.set_page_config(
    page_title='Chat PyGPT',
    page_icon='📄',
)
st.title('Chat PyGPT')

# Verifica se as configurações foram definidas
if 'configured' not in st.session_state or not st.session_state.configured:
    st.header('Configuração Inicial')
    aws_key = st.text_input("Digite sua AWS Access Key", type="password", value=os.getenv("AWS_ACCESS_KEY_ID") or "", key="aws_key")
    aws_secret = st.text_input("Digite sua AWS Secret Key", type="password", value=os.getenv("AWS_SECRET_ACCESS_KEY") or "", key="aws_secret")
    
    if st.button("Salvar configurações"):
        if not aws_key or not aws_secret:
            st.warning("Por favor, preencha os dados de configuração.")
        else:
            # Define as variáveis de ambiente
            os.environ["AWS_ACCESS_KEY_ID"] = aws_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret
            
            # Atualiza o estado da sessão para indicar que a configuração foi realizada
            st.session_state.configured = True
            
            st.success("Configurações salvas com sucesso!")
            st.rerun()
else:
    # Página principal de interação com o chat
    st.header('🤖 Chat com IA 🤖')

    # Inicializa o estado da sessão se necessário
    # if 'messages' not in st.session_state:
    #     st.session_state['messages'] = []

    # if 'df_context' not in st.session_state:
    #     st.session_state['df_context'] = query_database('SEU SQL INICIAL AQUI')

    question = st.chat_input('Digite sua pergunta aqui:')

    if question:
        for message in st.session_state.messages:
            st.chat_message(message.get('role')).write(message.get('content'))

        st.chat_message('user').write(question)
        st.session_state.messages.append({'role': 'user', 'content': question})

        with st.spinner('Buscando resposta...'):
            df_context = st.session_state['df_context']
            response = ask_question(
                model=model,
                query=question,
                df=df_context,
            )
            if "Não há informações disponíveis" in response:
                # Nova consulta SQL se necessário
                st.session_state['df_context'] = query_database('SEU SQL DE CONSULTA AQUI')
                df_context = st.session_state['df_context']
                response = ask_question(
                    model=model,
                    query=question,
                    df=df_context,
                )

            st.chat_message('ai').write(response)
            st.session_state.messages.append({'role': 'ai', 'content': response})