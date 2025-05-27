# main.py

import os

print("🔁 Executando coleta de indicadores econômicos...")
os.system("python scripts/indicadores_economicos.py")

print("📈 Executando coleta das ações da Bovespa...")
os.system("python scripts/acoes_bovespa.py")

print("📰 Executando coleta de notícias econômicas...")
os.system("python scripts/noticias.py")

print("🧠 Executando análise dos agentes econômicos (CrewAI)...")
os.system("python scripts/agentes_economicos.py")

print("🚀 Iniciando dashboard Streamlit...")
os.system("streamlit run streamlit/dashboard.py")
