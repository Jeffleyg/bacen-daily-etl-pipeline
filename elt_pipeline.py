import requests
import pandas as pd
from datetime import datetime, timedelta

SERIES_BACEN = {
    "dolar_ptax": 1,        # Taxa de câmbio - Dólar comercial (venda)
    "selic_diaria": 11,      # Taxa de juros - Selic acumulada no mês / diária
    "cdi_diario": 12,        # Taxa CDI diária
    "ouro_grama": 4          # Preço do ouro
}

def fetch_bacen_series(code: int, start_date: str) -> pd.DataFrame:
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start_date}"
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if not df.empty:
                df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
                df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                return df
    except Exception as e:
        print(f"Erro ao extrair série {code}: {e}")
    return pd.DataFrame()

def run_pipeline():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extraindo dados do Banco Central...")
    data_inicio = (datetime.now() - timedelta(days=3*365)).strftime("%d/%m/%Y")
    
    df_merged = None
    for name, code in SERIES_BACEN.items():
        df_serie = fetch_bacen_series(code, data_inicio)
        if not df_serie.empty:
            df_serie = df_serie.rename(columns={"valor": name})
            if df_merged is None:
                df_merged = df_serie
            else:
                df_merged = pd.merge(df_merged, df_serie, on="data", how="outer")
    
    if df_merged is not None and not df_merged.empty:
        df_merged = df_merged.sort_values("data").reset_index(drop=True)
        df_merged = df_merged.ffill()
        df_merged['dolar_mm_7d'] = df_merged['dolar_ptax'].rolling(window=7, min_periods=1).mean()
        df_merged['dolar_mm_30d'] = df_merged['dolar_ptax'].rolling(window=30, min_periods=1).mean()
        df_merged['dolar_var_pct'] = df_merged['dolar_ptax'].pct_change() * 100
        
        df_merged.to_parquet("bacen_daily_master.parquet", index=False)
        print(f"✅ Arquivo bacen_daily_master.parquet gerado com sucesso! ({len(df_merged):,} registros)")
    else:
        print("❌ Falha na extração de dados do BACEN.")

if __name__ == "__main__":
    run_pipeline()