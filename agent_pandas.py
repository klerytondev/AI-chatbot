import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
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
print(df.head())
print(df.shape)

"""PROMPT_PREFIX =
- Primeiro, ajuste as configurações de exibição do pandas para mostrar todas as colunas.
- Recupere os nomes das colunas e prossiga para responder à pergunta com base nos dados.
- Se a pergunta do usuário mencionar os termos df, dataset, base, base de dados, dados ou qualquer outra coisa relacionada a dados, ele está se referindo ao dataframe.
- Se a pergunta estiver fora de contexto, você deve responder com a seguinte mensagem: "Não posso responder este tipo de pergunta, pois foge do contexto passado"
"""
"""
PROMPT_SUFFIX = 
- **Antes de fornecer a resposta final**, sempre tente pelo menos um método adicional.
 Reflita sobre ambos os métodos e garanta que os resultados abordem a pergunta original com precisão.
- Formate quaisquer números com quatro ou mais dígitos usando vírgulas para facilitar a leitura.
- Se os resultados dos métodos forem diferentes, reflita e tente outra abordagem até que ambos os métodos se alinhem.
- Se você ainda não conseguir chegar a um resultado consistente, reconheça a incerteza em sua resposta.
- Quando tiver certeza da resposta correta, crie uma resposta estruturada usando markdown.
- **Sob nenhuma circunstância deve ser usado conhecimento prévio**—confie somente nos resultados derivados dos dados e cálculos realizados.
- A resposta final deve ser sempre respondida em **português brasileiro.**
"""

PROMPT_PREFIX = """
- First, adjust the pandas display settings to show all columns.
- Retrieve the column names, then proceed to answer the question based on the data.
- If the user's question mentions the terms df, dataset, base, base de dado, dados or anything else related to data, he is referring to the dataframe.
- If the question is out of context, you should respond with the following message: "Não posso responder este tipo de pergunta, pois foge do contexto passado"
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

# Criar agente pandas para manipulação do DataFrame com opção de executar código perigoso
agent = create_pandas_dataframe_agent(
 llm=model, 
 df=df, 
 prefix=PROMPT_PREFIX,
 suffix=PROMPT_SUFFIX,
 verbose=True, 
 allow_dangerous_code=True,
 agent_type=AgentType.OPENAI_FUNCTIONS
 # handle_parsing_errors=True
 )

st.title("Database AI Agent with Langchain")
st.write("### Dataset Preview")
st.write(df.head())
 
# Entrada de pergunta pelo usuário
st.write('### Ask a question')
question = st.text_input(
 "Enter your question about the dataset:",
 "Quantas linhas e colunas possui o Data Set?"
)
 
# Ação ao clicar no botão
if st.button("Run Query"):
 QUERY = question
 res = agent.invoke(QUERY)
 st.write("### Final Answer")
 st.markdown(res["output"])