import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Painel de Análise de Investimentos com Chat")

# --- Título principal ---
st.title("📊 Painel de Análise de Investimentos")
st.markdown("Visão consolidada do mercado financeiro com análises da CrewAI, dados de ações, IPCA, SELIC, PIB, dólar e notícias.")
st.divider()

# --- Carrega variáveis do ambiente ---
load_dotenv()

# --- Chatbot no topo ---
st.header("💬 Converse com o Agente Econômico")

# Initialize chat_model safely
chat_model = None
try:
    chat_model = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM"),
        temperature=0,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_API_VERSION")
    )
except Exception as e:
    st.error(f"Erro ao inicializar o modelo de chat: {e}")
    st.warning("As funcionalidades do chatbot estarão desabilitadas.")

contexto_chat = """
Você é o "Analista Econômico Virtual", um assistente de IA especializado em economia e mercado financeiro brasileiro, com foco em fornecer insights e análises baseadas em dados.

**Seu Perfil:**
- **Especialista em:** Economia brasileira, tendências de mercado, indicadores econômicos (IPCA, SELIC, PIB, Câmbio), análise de ações (foco em volume e notícias relevantes) e interpretação de notícias financeiras.
- **Seu Objetivo:** Ajudar o usuário a entender o cenário econômico, responder perguntas sobre investimentos e finanças de forma clara, objetiva, informativa e consultiva.
- **Seu Tom:** Profissional, analítico, ponderado e educativo. Evite linguagem excessivamente técnica sem explicação. Seja direto, mas completo em suas respostas.

**Contexto Econômico Atual (use como base principal para suas respostas):**
* **Inflação (IPCA):** Observa-se uma tendência de desaceleração nos últimos meses.
* **Taxa de Juros (SELIC):** Atualmente fixada em 10,75% ao ano.
* **Mercado de Ações:** As ações PETR4, VALE3 e WEGE3 apresentam os maiores volumes de negociação recentemente.
* **Cenário Macroeconômico e Notícias:** Atenção para a recente alta nos preços do petróleo e discussões sobre o risco fiscal no país.

**Diretrizes para suas Respostas:**
1.  **Baseie-se nos Dados:** Utilize primordialmente as informações de contexto fornecidas acima. Se uma pergunta extrapolar esses dados, mencione que a informação específica não está no seu contexto atual, mas pode oferecer uma análise geral se aplicável.
2.  **Clareza e Objetividade:** Responda de forma direta e fácil de entender.
3.  **Abordagem Consultiva:** Não se limite a responder; ofereça perspectivas, explique implicações e, quando apropriado, sugira cautela ou pontos de atenção.
4.  **Análise, Não Recomendação:** Você fornece análises e informações, mas NÃO deve dar conselhos de investimento diretos (ex: "compre esta ação" ou "invista nisso"). Em vez disso, explique cenários, riscos e potenciais com base nos dados. Frases como "Considerando o cenário X, um investimento Y pode ter tal comportamento..." são aceitáveis, mas sempre com as devidas ressalvas.
5.  **Atualização dos Dados:** Lembre ao usuário que o cenário econômico é dinâmico e os dados fornecidos no contexto são um retrato do momento.
6.  **Interpretação de Notícias:** Ao comentar notícias, foque nos seus potenciais impactos econômicos e nos ativos mencionados.
7.  **Seja Proativo:** Se uma pergunta for simples, tente agregar valor com um breve contexto adicional relevante.

Exemplo de interação desejada:
Usuário: "Com a SELIC a 10,75%, ainda vale a pena investir em renda fixa?"
Você: "Com a taxa SELIC em 10,75% ao ano, a renda fixa permanece uma modalidade de investimento atrativa, especialmente para perfis mais conservadores, pois oferece retornos nominais consideráveis. Títulos atrelados à SELIC, como o Tesouro SELIC, acompanham essa taxa, proporcionando liquidez e baixo risco. É importante considerar também a inflação (IPCA) para calcular o ganho real. A desaceleração do IPCA, mencionada no nosso contexto, pode favorecer o rendimento real desses investimentos. No entanto, a decisão de investir deve sempre considerar seus objetivos financeiros, perfil de risco e o cenário econômico completo, incluindo discussões sobre risco fiscal que podem afetar as expectativas futuras para juros e inflação."

Agora, responda à pergunta do usuário.
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

pergunta_cliente = st.text_input("Digite sua pergunta sobre investimentos ou economia:")

if pergunta_cliente and chat_model:
    mensagens = [SystemMessage(content=contexto_chat)]
    for troca in st.session_state.chat_history:
        mensagens.append(HumanMessage(content=troca["pergunta"]))
        # Langchain typically expects AIMessage for bot responses in history for some models,
        # but SystemMessage can work depending on the model and library version.
        # If issues arise, consider changing this to AIMessage for 'resposta'.
        mensagens.append(SystemMessage(content=troca["resposta"]))
    mensagens.append(HumanMessage(content=pergunta_cliente))

    try:
        resposta = chat_model(mensagens).content
        st.session_state.chat_history.append({"pergunta": pergunta_cliente, "resposta": resposta})

        st.markdown("### 🧠 Resposta do Agente:")
        st.write(resposta)
    except Exception as e:
        st.error(f"Erro ao obter resposta do agente: {e}")

elif pergunta_cliente and not chat_model:
    st.warning("O modelo de chat não está configurado. Não é possível processar a pergunta.")

if chat_model:
    with st.expander("📜 Histórico da conversa", expanded=False):
        for i, troca in enumerate(st.session_state.chat_history):
            st.markdown(f"**Você:** {troca['pergunta']}")
            st.markdown(f"**Agente:** {troca['resposta']}")
            if i < len(st.session_state.chat_history) - 1:
                 st.markdown("---")
st.divider()

# --- Funções de carregamento ---
@st.cache_data
def carregar_relatorio_md(caminho_arquivo):
    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            return f.read()
    return "Relatório não encontrado. Execute a CrewAI primeiro."

@st.cache_data
def carregar_csv(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        return f"Arquivo {os.path.basename(caminho_arquivo)} não encontrado."
    try:
        df = pd.read_csv(caminho_arquivo)
        if df.empty:
            return f"Arquivo {os.path.basename(caminho_arquivo)} está vazio."
        return df
    except pd.errors.EmptyDataError: # Specific error for empty CSV
        return f"Arquivo {os.path.basename(caminho_arquivo)} não contém dados para parsear."
    except Exception as e:
        return f"Erro ao carregar {os.path.basename(caminho_arquivo)}: {e}"

# --- Caminhos dos arquivos ---
ARQUIVO_RELATORIO_AGENTES = "data/relatorio_indicacao_acoes.md"
ARQUIVO_ACOES = "data/top_10_acoes.csv"
ARQUIVO_INDICADORES_ECONOMICOS = "data/indicadores_economicos.csv" # Assuming it's in the same directory or provide full path "data/indicadores_economicos.csv"
ARQUIVO_NOTICIAS = "data/noticias_investimentos.csv"

# --- Painel principal ---
st.header("📊 Análises Detalhadas")
st.divider()

# --- Relatório dos agentes ---
st.subheader("🤖 Relatório da Análise dos Agentes (CrewAI)")
relatorio_agentes = carregar_relatorio_md(ARQUIVO_RELATORIO_AGENTES)
with st.expander("Clique para ver o relatório completo", expanded=False):
    st.markdown(relatorio_agentes, unsafe_allow_html=True)
st.divider()

# --- Ações e Índices Econômicos em colunas ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Top 10 Ações (Últimos 20 dias)")
    df_acoes = carregar_csv(ARQUIVO_ACOES)
    if isinstance(df_acoes, pd.DataFrame):
        if 'ticker' not in df_acoes.columns:
            st.error(f"Coluna 'ticker' não encontrada no arquivo {os.path.basename(ARQUIVO_ACOES)}.")
        else:
            st.markdown(f"Dados de {len(df_acoes['ticker'].unique())} ações carregados.")
            lista_tickers = sorted(df_acoes['ticker'].unique())
            if lista_tickers:
                ticker_selecionado = st.selectbox("Selecione uma ação para ver o gráfico:", lista_tickers)
                if ticker_selecionado:
                    df_ticker = df_acoes[df_acoes['ticker'] == ticker_selecionado].copy()
                    
                    date_col_acao = None
                    if 'Unnamed: 0' in df_ticker.columns and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_ticker['Unnamed: 0'], errors='coerce')):
                        date_col_acao = 'Unnamed: 0'
                    elif 'Data' in df_ticker.columns and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_ticker['Data'], errors='coerce')):
                        date_col_acao = 'Data'
                    elif 'Date' in df_ticker.columns and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_ticker['Date'], errors='coerce')):
                        date_col_acao = 'Date'
                    # Add more potential date column names if necessary
                    
                    if date_col_acao:
                        df_ticker['data_plot'] = pd.to_datetime(df_ticker[date_col_acao], errors='coerce')
                        df_ticker.dropna(subset=['data_plot'], inplace=True)
                        df_ticker.set_index('data_plot', inplace=True)
                        df_ticker.sort_index(inplace=True)
                    else:
                        st.warning("Coluna de data não identificada ou não conversível para o gráfico de ações. Verifique se existe uma coluna como 'Data', 'Date' ou 'Unnamed: 0' (com datas).")

                    if 'fechamento' in df_ticker.columns:
                        if not df_ticker.empty and date_col_acao:
                            st.line_chart(df_ticker['fechamento'])
                        elif not date_col_acao:
                             pass # Warning already shown
                        else:
                            st.info(f"Não há dados de fechamento para plotar para {ticker_selecionado} após processamento.")
                    else:
                        st.warning("Coluna 'fechamento' não encontrada para a ação selecionada.")
                    
                    with st.expander(f"Ver tabela de dados - {ticker_selecionado}", expanded=False):
                        st.dataframe(df_acoes[df_acoes['ticker'] == ticker_selecionado], height=300)
            else:
                st.info(f"Nenhum ticker disponível no arquivo {os.path.basename(ARQUIVO_ACOES)}.")
    elif isinstance(df_acoes, str):
        st.error(df_acoes)
    # No explicit else for df_acoes is None, as carregar_csv now returns a string for not found.

with col2:
    st.subheader("📉 Índices Econômicos (IPCA, SELIC, PIB, Dólar, etc.)")
    df_indicadores = carregar_csv(ARQUIVO_INDICADORES_ECONOMICOS)
    
    if isinstance(df_indicadores, pd.DataFrame):
        # Check for required columns based on the CSV structure
        required_cols = ['data', 'valor', 'indicador']
        if not all(col in df_indicadores.columns for col in required_cols):
            st.error(f"O arquivo {os.path.basename(ARQUIVO_INDICADORES_ECONOMICOS)} deve conter as colunas: {', '.join(required_cols)}.")
        else:
            try:
                # Convert 'data' column to datetime
                df_indicadores['data'] = pd.to_datetime(df_indicadores['data'], errors='coerce')
                df_indicadores.dropna(subset=['data'], inplace=True) # Remove rows where date conversion failed

                if df_indicadores.empty:
                    st.warning("Não há dados válidos após a conversão de datas.")
                else:
                    lista_de_indicadores = sorted(df_indicadores['indicador'].unique())
                    
                    if not lista_de_indicadores:
                        st.warning("Nenhum indicador único encontrado na coluna 'indicador'.")
                    else:
                        indicador_selecionado = st.selectbox(
                            "Selecione o índice para visualização:",
                            lista_de_indicadores,
                            help="Escolha um dos indicadores econômicos disponíveis no arquivo."
                        )

                        if indicador_selecionado:
                            df_plot = df_indicadores[df_indicadores['indicador'] == indicador_selecionado].copy()
                            
                            if df_plot.empty:
                                st.info(f"Não há dados para o indicador '{indicador_selecionado}'.")
                            else:
                                # Ensure 'valor' column is numeric
                                if not pd.api.types.is_numeric_dtype(df_plot['valor']):
                                    st.warning(f"A coluna 'valor' para o indicador '{indicador_selecionado}' não é numérica. Tentando converter...")
                                    df_plot['valor'] = pd.to_numeric(df_plot['valor'], errors='coerce')
                                    df_plot.dropna(subset=['valor'], inplace=True) # Drop rows where conversion failed
                                
                                if df_plot.empty or df_plot['valor'].isnull().all():
                                     st.info(f"Não há valores numéricos válidos para plotar para o indicador '{indicador_selecionado}'.")
                                else:
                                    df_plot.sort_values(by='data', inplace=True)
                                    df_plot.set_index('data', inplace=True)
                                    st.line_chart(df_plot['valor'])

                                    with st.expander(f"Ver tabela de dados - {indicador_selecionado}", expanded=False):
                                        st.dataframe(df_indicadores[df_indicadores['indicador'] == indicador_selecionado], height=300)
                        else:
                            st.info("Por favor, selecione um indicador para visualização.")
            except Exception as e:
                st.error(f"Erro ao processar os dados dos indicadores econômicos: {e}")
                
    elif isinstance(df_indicadores, str): # Error message or "not found" from carregar_csv
        st.error(df_indicadores)
    # No explicit else for df_indicadores is None, as carregar_csv now returns a string.

st.divider()

# --- Notícias Recentes ---
st.subheader("📰 Top 10 Notícias de Investimento")
df_noticias = carregar_csv(ARQUIVO_NOTICIAS)
if isinstance(df_noticias, pd.DataFrame):
    if 'titulo' in df_noticias.columns and 'link' in df_noticias.columns:
        for _, row in df_noticias.head(min(10, len(df_noticias))).iterrows():
            st.markdown(f"### {row['titulo']}")
            if pd.notna(row['link']) and str(row['link']).strip() and str(row['link']).lower() not in ['nan', 'na', 'n/a']:
                st.markdown(f"[Ler notícia completa]({row['link']})")
                st.caption(f"Link: {row['link']}")
            else:
                st.caption("Link não disponível.")
            st.markdown("---")
    else:
        st.warning(f"Colunas 'titulo' e 'link' não encontradas no arquivo {os.path.basename(ARQUIVO_NOTICIAS)}. Exibindo primeiras 10 linhas se disponíveis.")
        st.dataframe(df_noticias.head(10))
elif isinstance(df_noticias, str):
    st.error(df_noticias)

# --- Rodapé ---
st.sidebar.info(f"Painel atualizado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}")