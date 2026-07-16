import yfinance as yf
import numpy as np
import os
from pathlib import Path

# Could probably make this less hardcoded. 

def fetch_save(tickers: list, conditional: list, start: str, end: str, 
               save_dir: str, name: str, cond_name: str):
    
    # potentially: independent dropna can misalign, but its minor i can fix later. 
    data = yf.download(tickers, start, end, interval='1wk')['Close'].dropna()
    cond_data = yf.download(conditional, start, end, interval='1wk')['Close'].dropna()

    # Because we are saving in npz files we need to save the names seperately
    symbols = np.array(data.columns, dtype=str)
    data = data.values
    cond_data = cond_data.values
    
    # save data
    data_path = os.path.join(save_dir, name)
    np.savez(file=data_path, close_data=data, tickers=symbols)

    # Save condtionals
    cond_path = os.path.join(save_dir, cond_name)
    np.savez(file=cond_path, conditional_data=cond_data) # using savez here because if i expand we'll need tickers param

# this will need to be updated if there are more conditionals
def load_raw_data(data_path: str, stock_dataset: str, cond_dataset: str):
    data = np.load(os.path.join(data_path, stock_dataset))
    prices = data['close_data']
    tickers = data['tickers']

    cond = np.load(os.path.join(data_path, cond_dataset))
    cond = cond['conditional_data']
    # N = number of stocks, T = Number of weeks
    # tickers= (N,)
    # prices = (T, N)
    # cond   = (T, N)

    return tickers, prices, cond 

if __name__ == '__main__':
    START = '2015-01-01'
    END = '2025-01-01'
    CONDITIONAL = ['^VIX'] # list because I may want to try multiple
    
    TICKERS = sorted([
    "MMM", "AMZN", "AXP", "AMGN", "AAPL", 
    "BA", "CAT", "CVX", "CSCO", "KO", 
    "DIS", "GS", "HD", "HON", "IBM", 
    "JNJ", "JPM", "MCD", "MRK", "MSFT", 
    "NKE", "NVDA", "PG", "CRM", "SHW", 
    "TRV", "UNH", "VZ", "V", "WMT"
])
    
    SAVE_DIR = Path('data/raw').resolve()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    fetch_save(
        tickers=TICKERS,
        conditional=CONDITIONAL,
        start=START,
        end=END,
        save_dir=SAVE_DIR,
        name='DOW.npz',
        cond_name='VIX.npz'
    )
