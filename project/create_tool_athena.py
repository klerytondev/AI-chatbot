import boto3
import re
from langchain_experimental.tools.base import BaseTool

class AthenaQueryTool(BaseTool):
    name = "aws_athena_query_tool"
    description = (
        "Executa uma consulta no AWS Athena. "
        "Forneça os parâmetros no seguinte formato: "
        "'tabela=<nome>, db=<nome>, start=<YYYY-MM-DD>, end=<YYYY-MM-DD>'" # Na string de consulta os padrões para filtro precisam ser passadaos neste padrão
    )

    def _run(self, query: str) -> str:
        # Extrai os parâmetros utilizando regex
        params = dict(re.findall(r'(\w+)=([\S]+)', query))
        table = params.get("tabela")
        db = params.get("db")
        start_date = params.get("start")
        end_date = params.get("end")
        
        if not all([table, db, start_date, end_date]):
            return "Parâmetros insuficientes. Forneça tabela, db, start e end."
        
        # Constrói a query SQL com base nos parâmetros
        sql_query = (
            f"SELECT * FROM {table} "
            f"WHERE date BETWEEN DATE('{start_date}') AND DATE('{end_date}')"
        )
        
        # Cria um cliente AWS Athena. Verifique se a região e outras configurações estão corretas.
        client = boto3.client('athena', region_name='us-east-1')
        
        try:
            response = client.start_query_execution(
                QueryString=sql_query,
                QueryExecutionContext={"Database": db},
                ResultConfiguration={"OutputLocation": "s3://seu-bucket-de-saida/"},
            )
            query_execution_id = response.get("QueryExecutionId", "ID não disponível")
            return f"Consulta executada com sucesso. QueryExecutionId: {query_execution_id}"
        except Exception as e:
            return f"Erro ao executar consulta: {str(e)}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("AthenaQueryTool não suporta execução assíncrona")
    



###############################Chamando a função############################################

from aws_athena_tool import AthenaQueryTool

def ask_question(model, query, df):
    athena_tool = AthenaQueryTool()  # Instancia a ferramenta do Athena

    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        tools=[athena_tool]  # Adiciona a ferramenta extra
    )
    system_prompt = '''
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    informe "Não há informações disponíveis"
    e não responda algo que estiver fora do contexto.
    Responda em formato de markdown e com visualizações elaboradas e interativas.
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', query))
    prompt = ChatPromptTemplate.from_messages(messages)
    response = agent.run(prompt)
    return response
