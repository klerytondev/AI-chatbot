import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents.agent_types import AgentType
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
 
def initial_parameters() -> tuple:
 load_dotenv()
 client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 model = ChatOpenAI(model="gpt-4o-mini")
 parser = StrOutputParser()
 return model, parser, client

model, parser, client = initial_parameters()

# Carregar arquivo CSV e tratar valores ausentes
df = pd.read_csv('data/ocorrencia.csv', delimiter=";" , encoding='latin1', on_bad_lines='skip').fillna(value=0)
# print(df.head())
# print(df.shape)

# Função para perguntar ao agente
def ask_question(model, df, query):

    PROMPT_PREFIX = """
        
        - Retrieve the column names, then proceed to answer the question based on the data.
        - If the user's question mentions the terms df, dataset, base, base de dado, dados or anything else related to data, he is referring to the dataframe.
        - If the question is out of context, you should respond with the following message: "Não posso responder este tipo de pergunta, pois foge do contexto passado"
        - You should use the message history below to enrich your context.
        """
        
    PROMPT_SUFFIX = """
        - **Before providing the final answer**, always try at least one additional method.
        Reflect on both methods and ensure that the results address the original question accurately.
        - Format any figures with four or more digits using commas for readability.
        - If the results from the methods differ, reflect, and attempt another approach until both methods align.
        - If you're still unable to reach a consistent result, acknowledge uncertainty in your response.
        - Once you are sure of the correct answer, create a structured answer using markdown.
        - **Under no circumstances should prior knowledge be used**—rely solely on the results derived from the data and calculations performed.
        - The final answer must always be answered in **Brazilian Portuguese.**
        """
    
    messages = [('system', PROMPT_PREFIX)]
    messages.extend((msg['role'], msg['content']) for msg in st.session_state.get('messages', []))
    
# Converter a lista de mensagens em uma string legível
    PROMPT_IMPUT = "\n".join([f"{role}: {content}" for role, content in messages])
    print("prompt_str = ", PROMPT_IMPUT)

    agent = create_pandas_dataframe_agent(
        llm=model, 
        df=df, 
        prefix=PROMPT_IMPUT,
        suffix=PROMPT_SUFFIX,
        verbose=True, 
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS
        # handle_parsing_errors=True
        )

    response = agent.invoke({'input': query})
    return response
    # return None


st.title("Database AI Agent with Langchain")
st.write("### Dataset Preview")
st.write(df.head())
 
# Entrada de pergunta pelo usuário
st.write('### Ask a question')
question = "Quantas linhas e colunas possui o Data Set?"
 
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input()

if question:
    for message in st.session_state.messages:
        # st.chat_message(message.get('role')).write(message.get('content'))
        st.chat_message(message.get('role')).markdown(message.get('content'))

    st.chat_message('human').markdown(question)
    st.session_state.messages.append({'role': 'human', 'content': question})

    with st.spinner('Buscando resposta...'):
        response = ask_question(
            model=model,
            df=df,
            query=question
        )
        
        # Extrair apenas o conteúdo da resposta
        output = response["output"]
        st.chat_message('ai').markdown(output)
        st.session_state.messages.append({'role': 'ai', 'content': output})