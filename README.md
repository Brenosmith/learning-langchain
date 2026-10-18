# learning-langchain

Projeto de estudos práticos sobre **LangChain**, explorando a integração com
modelos da OpenAI, modelos hospedados no Hugging Face e modelos locais
executados com Transformers e PyTorch. O projeto também inclui uma interface
de chat interativa desenvolvida com Streamlit.

## Executar

Inicie a aplicação Streamlit com:

```bash
streamlit run source/streamlit_chat.py
```

## Configurar Conexões

Configure as credenciais dos provedores no arquivo `.env` antes de executar.

### 1. Hugging face

Para o exemplo do Hugging Face, crie um token em
<https://huggingface.co/settings/tokens> e adicione ao `.env`:

```dotenv
HUGGINGFACEHUB_API_TOKEN=hf_seu_token
```

Também é possível autenticar pelo terminal:

```bash
hf auth login
```

### 2. OpenAi

Para o exemplo da OpenAi, crie um token em  ,https://platform.openai.com/settings/organization/api-keys> e adicione ao `.env`:

```dotenv
OPENAI_API_KEY=api_key
```
