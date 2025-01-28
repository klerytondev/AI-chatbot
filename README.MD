# IA Chatbot

Este projeto é um chatbot interativo desenvolvido com Streamlit e LangChain, que permite a interação com modelos de linguagem da OpenAI. Ele suporta o upload de arquivos PDF e CSV, processa esses arquivos e utiliza um modelo de linguagem para responder perguntas baseadas no conteúdo dos arquivos.

## Funcionalidades

- Upload de arquivos PDF e CSV.
- Processamento de arquivos e divisão em chunks.
- Armazenamento e recuperação de chunks usando Chroma.
- Interação com modelos de linguagem da OpenAI.
- Respostas em formato markdown com visualizações interativas.

## Instalação

1. Clone o repositório:
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd IA-chatbot
    ```

2. Crie um ambiente virtual e ative-o:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure as variáveis de ambiente:
    Crie um arquivo `.env` na raiz do projeto e adicione sua chave de API da OpenAI:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Uso

1. Execute a aplicação Streamlit:
    ```bash
    streamlit run app-ia.py
    ```

2. No navegador, faça o upload dos arquivos PDF ou CSV que deseja processar.

3. Selecione o modelo de linguagem de sua preferência na barra lateral.

4. Digite sua pergunta no campo de entrada e aguarde a resposta do chatbot.

## Processamento de Arquivos

### Chunking

O processo de chunking divide os documentos em partes menores (chunks) para facilitar a análise e a recuperação de informações. Para arquivos PDF, utilizamos o `PyPDFLoader` para carregar o conteúdo e o `RecursiveCharacterTextSplitter` para dividir o texto em chunks de 1000 caracteres com uma sobreposição de 400 caracteres. Para arquivos CSV, carregamos o conteúdo com o `pandas` e convertemos em uma string, que é então dividida em chunks da mesma forma.

### Vetorização

Os chunks são vetorizados utilizando embeddings gerados pelo `OpenAIEmbeddings`. Esses embeddings são armazenados em um banco de dados Chroma, que permite a recuperação eficiente dos chunks relevantes para responder às perguntas dos usuários.

### Banco de Dados

Utilizamos o Chroma como banco de dados para armazenar e recuperar os embeddings dos chunks. O Chroma é um banco de dados especializado em armazenamento de vetores, permitindo buscas rápidas e eficientes.

## Modelos de Linguagem (LLM)

Utilizamos a API da OpenAI para interagir com modelos de linguagem avançados, como o `gpt-4` e `gpt-3.5-turbo`. Esses modelos são responsáveis por gerar respostas baseadas nos chunks recuperados do banco de dados, proporcionando respostas precisas e contextualmente relevantes.

## Benefícios

- **Interatividade**: Permite a interação direta com modelos de linguagem avançados.
- **Flexibilidade**: Suporta múltiplos formatos de arquivo (PDF e CSV).
- **Persistência**: Armazena e recupera dados de forma eficiente usando Chroma.
- **Visualizações**: Respostas em formato markdown com visualizações interativas, melhorando a experiência do usuário.
- **Customização**: Fácil de configurar e adaptar para diferentes necessidades e modelos de linguagem.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias e correções.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
