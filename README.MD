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
