
import requests
import pandas as pd
from time import sleep
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("ALPHA_VANTAGE_API_KEY")

# Top 10 ações da B3 por volume - definidas manualmente
top_10_acoes = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "BBAS3", "B3SA3", "WEGE3", "RENT3", "MGLU3"]

def buscar_dados_acao_alpha_vantage(ticker_b3, api_key, num_registros=10): # Adicionado num_registros
    ticker = ticker_b3 + ".SA"
    # outputsize=compact ainda busca 100 pontos, o que é bom para depois selecionarmos os últimos N
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=compact"

    response = requests.get(url)
    if response.status_code != 200:
        # Adiciona um tratamento para o erro 503, comum quando a API está ocupada
        if response.status_code == 503:
            print(f"[{ticker_b3}] Servidor Alpha Vantage indisponível (503). Tentando novamente em alguns segundos...")
            sleep(30) # Espera um pouco mais em caso de 503
            response = requests.get(url) # Tenta novamente
            if response.status_code != 200:
                 raise Exception(f"[{ticker_b3}] Erro {response.status_code} após nova tentativa.")
        else:
            raise Exception(f"[{ticker_b3}] Erro {response.status_code}")


    data = response.json()
    # Verifica se a chave "Time Series (Daily)" existe e também se há alguma mensagem de erro/limite
    if "Note" in data or "Information" in data: # Checa por mensagens de limite de API
        print(f"[{ticker_b3}] Nota da API: {data.get('Note', data.get('Information', 'Limite de API provavelmente atingido.'))}")
        return None
    if "Time Series (Daily)" not in data:
        print(f"[{ticker_b3}] Sem dados 'Time Series (Daily)' na resposta. Resposta completa: {data}")
        return None

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.columns = ["abertura", "alta", "baixa", "fechamento", "volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True) # Garante que o índice (data) está em ordem crescente
    df["ticker"] = ticker_b3
    
    # Seleciona os últimos 'num_registros' (os mais recentes)
    df = df.tail(num_registros)
    
    return df

df_total = pd.DataFrame()

for ativo in top_10_acoes:
    try:
        print(f"🔄 Coletando {ativo}...")
        # Agora a função buscar_dados_acao_alpha_vantage buscará 20 registros por padrão
        df = buscar_dados_acao_alpha_vantage(ativo, api_key) 
        # Se quisesse um número diferente, poderia passar: buscar_dados_acao_alpha_vantage(ativo, API_KEY, num_registros=25)

        if df is not None and not df.empty: # Adicionado not df.empty para segurança
            df_total = pd.concat([df_total, df])
            print(f"✅ {ativo} adicionado com {len(df)} registros.")
        elif df is not None and df.empty:
            print(f"⚠️ {ativo} retornou um DataFrame vazio após o processamento (pode ser que não haja 20 dias de dados após o filtro).")
        # Se df for None, a mensagem já foi impressa dentro da função
        
        sleep(15) # Mantém o sleep para respeitar os limites da API Alpha Vantage (5 chamadas/minuto, 500/dia na gratuita)
    except Exception as e:
        print(f"❌ Erro com {ativo}: {e}")
if not df_total.empty:
    df_total.to_csv("../data/top_10_acoes.csv", index=True, encoding="utf-8-sig")
    print(f"📁 Arquivo final salvo com {len(df_total)} linhas.")
else:
    print("ℹ️ Nenhum dado foi coletado para salvar no arquivo CSV.")
