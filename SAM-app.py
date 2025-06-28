import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import ADXIndicator
from ta.momentum import TRIXIndicator

print("TA version:", ta.__version__)

# --- Functie om data op te halen ---
def fetch_data(ticker, interval):
    # Intervalspecifieke periode instellen
    if interval == "15m":
        period = "30d"
    elif interval == "1h":
        period = "720d"
    elif interval == "4h":
        period = "360d"
    elif interval == "1d":
        period = "20y"
    else:
        period = "25y"

    # Data ophalen
    df = yf.download(ticker, interval=interval, period=period)

    # Verwijder rijen zonder volume of zonder koersverandering
    df = df[
        (df["Volume"] > 0) &
        ((df["Open"] != df["Close"]) | (df["High"] != df["Low"]))
    ]

    # Zorg dat de index datetime is en verwijder ongeldige datums
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # Vul ontbrekende waarden op met vorige/volgende geldige waarde
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

    return df

from datetime import timedelta

def bepaal_grafiekperiode(interval):
    if interval == "15m":
        return timedelta(days=5)
    elif interval == "1h":
        return timedelta(days=5)
    elif interval == "4h":
        return timedelta(days=60)
    elif interval == "1d":
        return timedelta(days=120)
    else:
        return timedelta(weeks=260)  # bijv. bij weekly/monthly data


# --- SAM Indicatorberekeningen ---
def calculate_sam(df):
    df = df.copy()

    # Basiskolommen
    # --- SAMG op basis van Weighted Moving Averages + Crossovers nodig ---
    def weighted_moving_average(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    
    # --- SAMK: candlestick score op basis van patronen Open/Close ---

    df["c1"] = df["Close"] > df["Open"]
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
    df["c3"] = df["Close"] > df["Close"].shift(1)
    df["c4"] = df["Close"].shift(1) > df["Close"].shift(2)
    df["c5"] = df["Close"] < df["Open"]
    df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
    df["c7"] = df["Close"] < df["Close"].shift(1)
    df["c8"] = df["Close"].shift(1) < df["Close"].shift(2)

    df["SAMK"] = 0.0  # standaard

    # Positieve patronen
    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False),
        "SAMK"
    ] = 1.25

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c3"] & df["c4"]).fillna(False),
        "SAMK"
    ] = 1.0

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c3"]).fillna(False),
        "SAMK"
    ] = 0.5

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] | df["c3"]).fillna(False),
        "SAMK"
    ] = 0.25

    # Negatieve patronen
    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c6"] & df["c7"] & df["c8"]).fillna(False),
        "SAMK"
    ] = -1.25

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c7"] & df["c8"]).fillna(False),
        "SAMK"
    ] = -1.0

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c7"]).fillna(False),
        "SAMK"
    ] = -0.5

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] | df["c7"]).fillna(False),
        "SAMK"
    ] = -0.25
    
#    df["c1"] = df["Close"] > df["Open"]
#    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
#    df["c3"] = df["Close"].shift(2) > df["Open"].shift(2)
#    df["c4"] = df["Close"].shift(3) > df["Open"].shift(3)
#    df["c5"] = df["Close"].shift(4) > df["Open"].shift(4)
#    df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
#    df["c7"] = df["Close"].shift(2) < df["Open"].shift(2)

    # SAMK oud
#    df["SAMK"] = 0
#    df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
 #   df.loc[(df["c1"] & df["c6"] & df["c7"]).fillna(False), "SAMK"] = -1

    # --- SAMG (WMA-based trendanalyse met crossovers) ---
    df["WMA18"] = weighted_moving_average(df["Close"], 18)
    df["WMA35"] = weighted_moving_average(df["Close"], 35)
    df["WMA18_shifted"] = df["WMA18"].shift(1)
    df["WMA35_shifted"] = df["WMA35"].shift(1)

    df["SAMG"] = 0.0

    # Kleine trendbewegingen omhoog en omlaag
    df.loc[
        (df["WMA18"] > df["WMA18_shifted"] * 1.0015) & (df["WMA18"] > df["WMA18_shifted"]),
        "SAMG"
    ] = 0.5

    df.loc[
        (df["WMA18"] < df["WMA18_shifted"] * 1.0015) & (df["WMA18"] > df["WMA18_shifted"]),
        "SAMG"
    ] = -0.5

    df.loc[
        (df["WMA18"] > df["WMA18_shifted"] / 1.0015) & (df["WMA18"] <= df["WMA18_shifted"]),
        "SAMG"
    ] = 0.5

    df.loc[
        (df["WMA18"] < df["WMA18_shifted"] / 1.0015) & (df["WMA18"] <= df["WMA18_shifted"]),
        "SAMG"
    ] = -0.5

    # Grote crossovers tussen WMA18 en WMA35
    df.loc[
        (df["WMA18_shifted"] < df["WMA35_shifted"]) & (df["WMA18"] > df["WMA35"]),
        "SAMG"
    ] = 1.0

    df.loc[
        (df["WMA18_shifted"] > df["WMA35_shifted"]) & (df["WMA18"] < df["WMA35"]),
        "SAMG"
    ] = -1.0

#  samg oud
#    df["Change"] = df["Close"].pct_change()
#    df["SAMG"] = 0
#    df.loc[df["Change"] > 0.03, "SAMG"] = 1
#    df.loc[df["Change"] < -0.03, "SAMG"] = -1

    # SAMT oud
#    df["SMA5"] = df["Close"].rolling(window=5).mean()
#    df["SMA20"] = df["Close"].rolling(window=20).mean()
#    df["SAMT"] = 0
 #   df.loc[df["SMA5"] > df["SMA20"], "SAMT"] = 1
#    df.loc[df["SMA5"] < df["SMA20"], "SAMT"] = -1
# --- SAMT op basis van Weighted Moving Averages 6 en 80 ---
#    def weighted_moving_average(series, window): --> al eerder gedefineerd
 #       weights = np.arange(1, window + 1)
  #      return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

    df["WMA6"] = weighted_moving_average(df["Close"], 6)
    df["WMA6_shifted"] = df["WMA6"].shift(1)
    df["WMA80"] = weighted_moving_average(df["Close"], 80)

    df["SAMT"] = 0.0  # standaardwaarde

    df.loc[
        (df["WMA6"] > df["WMA6_shifted"]) & (df["WMA6"] > df["WMA80"]),
        "SAMT"
    ] = 0.5

    df.loc[
        (df["WMA6"] > df["WMA6_shifted"]) & (df["WMA6"] <= df["WMA80"]),
        "SAMT"
    ] = 0.25

    df.loc[
        (df["WMA6"] <= df["WMA6_shifted"]) & (df["WMA6"] <= df["WMA80"]),
        "SAMT"
    ] = -0.75

    df.loc[
        (df["WMA6"] <= df["WMA6_shifted"]) & (df["WMA6"] > df["WMA80"]),
        "SAMT"
    ] = -0.5
    
    # SAMD
    # --- SAMD op basis van DI+ en DI- ---
    high_series = df["High"].squeeze()
    low_series = df["Low"].squeeze()
    close_series = df["Close"].squeeze()

    adx = ADXIndicator(high=high_series, low=low_series, close=close_series, window=14)

    df["DI_PLUS"] = adx.adx_pos()
    df["DI_MINUS"] = adx.adx_neg()
  #  df["SAMD"] = 0.0  # begin met 0.0

    # Epsilon-drempels instellen
    epsilonneg = 10.0  # vrijwel afwezig andere richting
    epsilonpos = 30.0  # sterke richting
    df["SAMD"] = 0.0
    # 1️⃣ Sterke positieve richting
    df.loc[(df["DI_PLUS"] > epsilonpos) & (df["DI_MINUS"] <= epsilonneg), "SAMD"] = 1.0
    # 2️⃣ Sterke negatieve richting
    df.loc[(df["DI_MINUS"] > epsilonpos) & (df["DI_PLUS"] <= epsilonneg), "SAMD"] = -1.0
    # 3️⃣ Lichte positieve richting
    df.loc[(df["DI_PLUS"] > df["DI_MINUS"]) & (df["DI_MINUS"] > epsilonneg), "SAMD"] = 0.5
    # 4️⃣ Lichte negatieve richting
    df.loc[(df["DI_MINUS"] > df["DI_PLUS"]) & (df["DI_PLUS"] > epsilonneg), "SAMD"] = -0.5


    # samd oud
#    df["daily_range"] = df["High"] - df["Low"]
 #   avg_range = df["daily_range"].rolling(window=14).mean()
#    df["SAMD"] = 0
#    df.loc[df["daily_range"] > avg_range, "SAMD"] = 1
#    df.loc[df["daily_range"] < avg_range, "SAMD"] = -1

    # SAMM
    # ✅ Correcte MACD-berekening met ta
    close_series = df["Close"].squeeze()  # Zorg dat het een Series is
    macd_ind = ta.trend.MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)

    df["MACD"] = macd_ind.macd()
    df["SIGNAL"] = macd_ind.macd_signal()

    # ✅ Detecteer MACD crossovers voor SAMM
    prev_macd = df["MACD"].shift(1)
    prev_signal = df["SIGNAL"].shift(1)

    conditions = [
        (prev_macd < prev_signal) & (df["MACD"] > df["SIGNAL"]),
        (df["MACD"] > df["SIGNAL"]),
        (prev_macd > prev_signal) & (df["MACD"] < df["SIGNAL"]),
        (df["MACD"] <= df["SIGNAL"])
    ]
    keuzes = [1.0, 0.5, -1.0, -0.5]

    df["SAMM"] = np.select(conditions, keuzes, default=0.0)

    # samm oud   
  #  df["SMA10"] = df["Close"].rolling(window=10).mean()
  #  df["SMA50"] = df["Close"].rolling(window=50).mean()
 #   df["SAMM"] = 0
  #  df.loc[df["SMA10"] > df["SMA50"], "SAMM"] = 1
 #   df.loc[df["SMA10"] < df["SMA50"], "SAMM"] = -1

    # SAMX
#    from ta.momentum import TRIXIndicator

    # --- SAMX op basis van TRIX ---
    close_series = df["Close"].squeeze()
    trix_ind = TRIXIndicator(close=close_series, window=15)

    df["TRIX"] = trix_ind.trix()
    df["TRIX_PREV"] = df["TRIX"].shift(1)
    df["SAMX"] = 0.0  # Standaardwaarde

    # Sterke opwaartse trend
    df.loc[(df["TRIX"] > 0) & (df["TRIX"] > df["TRIX_PREV"]), "SAMX"] = 0.75

    # Zwakke opwaartse trend
    df.loc[(df["TRIX"] > 0) & (df["TRIX"] <= df["TRIX_PREV"]), "SAMX"] = 0.5

    # Sterke neerwaartse trend
    df.loc[(df["TRIX"] < 0) & (df["TRIX"] < df["TRIX_PREV"]), "SAMX"] = -0.75

    # Zwakke neerwaartse trend
    df.loc[(df["TRIX"] < 0) & (df["TRIX"] >= df["TRIX_PREV"]), "SAMX"] = -0.5

    # SAMX OUD
#    df["Momentum"] = df["Close"] - df["Close"].shift(3)
#    df["SAMX"] = 0
#    df.loc[df["Momentum"] > 0, "SAMX"] = 1
 #   df.loc[df["Momentum"] < 0, "SAMX"] = -1

    # Totale SAM
    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)

    return df
    

# --- Advies en rendementen ---
def determine_advice(df, threshold):
    df = df.copy()
    df["Trend"] = df["SAM"].rolling(window=12).mean() # cruciaal oorspronkelijk 3, maar sam 12
    df["TrendChange"] = df["Trend"] - df["Trend"].shift(1)

    df["Advies"] = np.nan
    df.loc[df["TrendChange"] > threshold, "Advies"] = "Kopen"
    df.loc[df["TrendChange"] < -threshold, "Advies"] = "Verkopen"
    df["Advies"] = df["Advies"].ffill()

    df["AdviesGroep"] = (df["Advies"] != df["Advies"].shift()).cumsum()
    rendementen = []
    sam_rendementen = []

    groepen = list(df.groupby("AdviesGroep"))

    for i in range(len(groepen)):
        _, groep = groepen[i]
        advies = groep["Advies"].iloc[0]

        start = groep["Close"].iloc[0]
        eind = None

        if i < len(groepen) - 1:
            # Eerste koers van volgende groep
            eind = groepen[i + 1][1]["Close"].iloc[0]
        else:
            # Laatste koers van deze groep
            eind = groep["Close"].iloc[-1]

        # ✅ Veiligheid: omzetting en check
        try:
            start = float(start)
            eind = float(eind)
            if start != 0.0:
                markt_rendement = (eind - start) / start
                sam_rendement = markt_rendement if advies == "Kopen" else -markt_rendement
            else:
                markt_rendement = 0.0
                sam_rendement = 0.0
        except Exception:
            markt_rendement = 0.0
            sam_rendement = 0.0

        # 🔁 Voeg altijd exacte lengte toe
        rendementen.extend([markt_rendement] * len(groep))
        sam_rendementen.extend([sam_rendement] * len(groep))

    # ✅ Laatste check: gelijke lengte
    if len(rendementen) != len(df):
        raise ValueError(f"Lengte mismatch: rendementen={len(rendementen)}, df={len(df)}")

    df["Markt-%"] = rendementen
    df["SAM-%"] = sam_rendementen

    # Huidig advies bepalen
    if "Advies" in df.columns and df["Advies"].notna().any():
        huidig_advies = df["Advies"].dropna().iloc[-1]
    else:
        huidig_advies = "Niet beschikbaar"

    return df, huidig_advies
    
# --- Streamlit UI ---
st.title("SAM Trading Indicator")

# --- Volledige tickerlijsten ---
aex_tickers = {
    "ABN.AS": "ABN AMRO",
    "ADYEN.AS": "Adyen",
    "AGN.AS": "Aegon",
    "AD.AS": "Ahold Delhaize",
    "AKZA.AS": "Akzo Nobel",
    "MT.AS": "ArcelorMittal",
    "ASM.AS": "ASMI",
    "ASML.AS": "ASML",
    "ASRNL.AS": "ASR Nederland",
    "BESI.AS": "BESI",
    "DSFIR.AS": "DSM-Firmenich",
    "GLPG.AS": "Galapagos",
    "HEIA.AS": "Heineken",
    "IMCD.AS": "IMCD",
    "INGA.AS": "ING Groep",
    "TKWY.AS": "Just Eat Takeaway",
    "KPN.AS": "KPN",
    "NN.AS": "NN Group",
    "PHIA.AS": "Philips",
    "PRX.AS": "Prosus",
    "RAND.AS": "Randstad",
    "REN.AS": "Relx",
    "SHELL.AS": "Shell",
    "UNA.AS": "Unilever",
    "WKL.AS": "Wolters Kluwer"
}

dow_tickers = {
    'MMM': '3M', 'AXP': 'American Express', 'AMGN': 'Amgen', 'AAPL': 'Apple', 'BA': 'Boeing',
    'CAT': 'Caterpillar', 'CVX': 'Chevron', 'CSCO': 'Cisco', 'KO': 'Coca-Cola', 'DIS': 'Disney',
    'GS': 'Goldman Sachs', 'HD': 'Home Depot', 'HON': 'Honeywell', 'IBM': 'IBM', 'INTC': 'Intel',
    'JPM': 'JPMorgan Chase', 'JNJ': 'Johnson & Johnson', 'MCD': 'McDonaldâ€™s', 'MRK': 'Merck',
    'MSFT': 'Microsoft', 'NKE': 'Nike', 'PG': 'Procter & Gamble', 'CRM': 'Salesforce',
    'TRV': 'Travelers', 'UNH': 'UnitedHealth', 'VZ': 'Verizon', 'V': 'Visa', 'WMT': 'Walmart',
    'DOW': 'Dow', 'RTX': 'RTX Corp.', 'WBA': 'Walgreens Boots'
}
nasdaq_tickers = {
    'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'AAPL': 'Apple', 'AMZN': 'Amazon', 'META': 'Meta',
    'NFLX': 'Netflix', 'GOOG': 'Google', 'GOOGL': 'Alphabet', 'TSLA': 'Tesla', 'CSCO': 'Cisco',
    'INTC': 'Intel', 'ADBE': 'Adobe', 'CMCSA': 'Comcast', 'PEP': 'PepsiCo', 'COST': 'Costco',
    'AVGO': 'Broadcom', 'QCOM': 'Qualcomm', 'TMUS': 'T-Mobile', 'TXN': 'Texas Instruments',
    'AMAT': 'Applied Materials'
}

ustech_tickers = {
    "SMCI": "Super Micro Computer",
    "PLTR": "Palantir",
    "ORCL": "Oracle",
    "SNOW": "Snowflake",
    "NVDA": "NVIDIA",
    "AMD": "AMD",
    "MDB": "MongoDB",
    "DDOG": "Datadog",
    "CRWD": "CrowdStrike",
    "ZS": "Zscaler",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "GOOGL": "Alphabet (GOOGL)",
    "MSFT": "Microsoft"
}
eurostoxx_tickers = {
    'ASML.AS': 'ASML Holding', 'AIR.PA': 'Airbus', 'BAS.DE': 'BASF', 'BAYN.DE': 'Bayer',
    'BNP.PA': 'BNP Paribas', 'MBG.DE': 'Mercedes-Benz Group', 'ENEL.MI': 'Enel',
    'ENGI.PA': 'Engie', 'IBE.MC': 'Iberdrola', 'MC.PA': 'LVMH', 'OR.PA': 'L’Oréal',
    'PHIA.AS': 'Philips', 'SAN.PA': 'Sanofi', 'SAP.DE': 'SAP', 'SIE.DE': 'Siemens',
    'SU.PA': 'Schneider Electric', 'TTE.PA': 'TotalEnergies', 'VIV.PA': 'Vivendi',
    'AD.AS': 'Ahold Delhaize', 'CRH.L': 'CRH', 'DPW.DE': 'Deutsche Post', 'IFX.DE': 'Infineon',
    'ITX.MC': 'Inditex', 'MT.AS': 'ArcelorMittal', 'RI.PA': 'Pernod Ricard', 'STLA.MI': 'Stellantis',
    'UN01.DE': 'Uniper'
}
# --- Toevoeging tickers AMX & Crypto ---
amx_tickers = {
    "AMG.AS": "AMG", "ARCAD.AS": "Arcadis", "BAMNB.AS": "BAM Groep",
    "BPOST.AS": "BPost", "FAGR.AS": "Fagron", "FUR.AS": "Fugro", "KENDR.AS": "Kendrion",
    "SBMO.AS": "SBM Offshore", "TKWY.AS": "Just Eat", "VASTN.AS": "Vastned Retail"
}

crypto_tickers = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "BNB-USD": "BNB", "XRP-USD": "XRP", "DOGE-USD": "Dogecoin"
}
# --- Update tab labels en bijbehorende mapping ---
tabs_mapping = {
    "🇺🇸 Dow Jones": dow_tickers,
    "🇺🇸 Nasdaq": nasdaq_tickers,
    "🇺🇸 US Tech": ustech_tickers,
    "🇪🇺 Eurostoxx": eurostoxx_tickers,
    "🇳🇱 AEX": aex_tickers,
    "🇳🇱 AMX": amx_tickers,
    "🌐 Crypto": crypto_tickers
}

tab_labels = list(tabs_mapping.keys())
selected_tab = st.radio("Kies beurs", tab_labels, horizontal=True)

valutasymbool = {
    "🇳🇱 AEX": "€ ",
    "🇳🇱 AMX": "€ ",
    "🇺🇸 Dow Jones": "$ ",
    "🇺🇸 Nasdaq": "$ ",
    "🇪🇺 Eurostoxx": "€ ",
    "🇺🇸 US Tech": "$ ",
    "🌐 Crypto": "",  # Geen symbool
}.get(selected_tab, "")

#- Data ophalen voor dropdown live view ---
#def get_live_ticker_data(tickers_dict):
# --- Data ophalen voor dropdown live view ---
def get_live_ticker_data(tickers_dict):
    tickers = list(tickers_dict.keys())
    data = yf.download(tickers, period="1d", interval="1d", progress=False, group_by='ticker')
    result = []

    for ticker in tickers:
        try:
            last = data[ticker]['Close'].iloc[-1]
            prev = data[ticker]['Open'].iloc[-1]
            change = (last - prev) / prev * 100
            kleur = "#00FF00" if change > 0 else "#FF0000" if change < 0 else "#808080"
            naam = tickers_dict[ticker]
            result.append((ticker, naam, last, change, kleur))
        except Exception:
            continue

    return result

# --- Weergave dropdown met live info ---
live_info = get_live_ticker_data(tabs_mapping[selected_tab])
# --- Dropdown dictionary voorbereiden ---
dropdown_dict = {}  # key = ticker, value = (display_tekst, naam)

for t, naam, last, change, kleur in live_info:
    emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
    display = f"{t} - {naam} | {valutasymbool}{last:.2f} {emoji} {change:+.2f}%"
    dropdown_dict[t] = (display, naam)

# --- Bepalen van de juiste default key voor selectie
# Herstel vorige selectie als deze nog bestaat
default_ticker_key = st.session_state.get(f"ticker_select_{selected_tab}")
if default_ticker_key not in dropdown_dict:
    default_ticker_key = list(dropdown_dict.keys())[0]  # fallback naar eerste

# --- Dropdown zelf ---
selected_ticker = st.selectbox(
    f"Selecteer {selected_tab} ticker:",
    options=list(dropdown_dict.keys()),  # alleen tickers als stabiele optie-key
    format_func=lambda x: dropdown_dict[x][0],  # toon koers etc.
    key=f"ticker_select_{selected_tab}",
    index=list(dropdown_dict.keys()).index(default_ticker_key)
)

# --- Ophalen ticker info ---
ticker = selected_ticker
ticker_name = dropdown_dict[ticker][1]


# --- Opgehaalde waarden ---
#ticker = selected_ticker
#ticker_name = dropdown_dict[ticker][1]

# --- Live koers opnieuw ophalen voor de geselecteerde ticker ---
try:
    live_data = yf.download(ticker, period="1d", interval="1d", progress=False)
    last = live_data["Close"].iloc[-1]
except Exception:
    last = 0.0  # fallback


# --- Andere instellingen ---
# --- Intervalopties ---
interval_optie = st.selectbox(
    "Kies de interval",
    ["Dagelijks", "Wekelijks", "4-uur", "1-uur", "15-minuten"]
)

# Vertaal gebruikerskeuze naar Yahoo Finance intervalcode
interval_mapping = {
    "Dagelijks": "1d",
    "Wekelijks": "1wk",
    "4-uur": "4h",
    "1-uur": "1h",
    "15-minuten": "15m"
}

interval = interval_mapping[interval_optie]

# de gevoeligheid slider
thresh = st.slider("Gevoeligheid van trendverandering", 0.01, 0.5, 0.1, step=0.02)

# Berekening
df = fetch_data(ticker, interval)
df = calculate_sam(df)
#df = determine_advice(df, threshold=thresh)
#df, huidig_advies = determine_advice(df, threshold=thresh)
df, huidig_advies = determine_advice(df, threshold=thresh)
#df, huidig_advies = determine_advice(df, threshold=thresh)
# debugging tools
st.subheader("🔍 SAM Debug-tabel (laatste 8 rijen)")
st.dataframe(
    df[["Close", "SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX", "SAM"]].tail(180),
    use_container_width=True
)
#st.caption(f"SAM-componenten gemiddeld: "
#           f"SAMK={df['SAMK'].mean():+.2f}, "
 #          f"SAMG={df['SAMG'].mean():+.2f}, "
#           f"SAMT={df['SAMT'].mean():+.2f}, "
#           f"SAMD={df['SAMD'].mean():+.2f}, "
 #          f"SAMM={df['SAMM'].mean():+.2f}, "
 #          f"SAMX={df['SAMX'].mean():+.2f}, "
  #         f"SAM totaal={df['SAM'].mean():+.2f}")

# Grafieken
#st.subheader(f"SAM-indicator en trend voor {ticker}")
# Huidige advies ophalen
# huidig_advies = df["Advies"].dropna().iloc[-1]

# Kleur bepalen op basis van advies
advies_kleur = "green" if huidig_advies == "Kopen" else "red" if huidig_advies == "Verkopen" else "gray"

# Titel met kleur en grootte tonen
st.markdown(
#    f"""
#    <h3>SAM-indicator en trend voor <span style='color:#3366cc'>{ticker_name} - ${last:.2f}</span></h3>
#    <h2 style='color:{advies_kleur}'>Huidig advies: {huidig_advies}</h2>
#    """,
#    unsafe_allow_html=True
#)
#st.markdown(
    f"""
    <h3>SAM-indicator en trend voor <span style='color:#3366cc'>{ticker_name}</span></h3>
    <h2 style='color:{advies_kleur}'>Huidig advies: {huidig_advies}</h2>
    """,
    unsafe_allow_html=True
)

import matplotlib.pyplot as plt
import streamlit as st

# --- Grafiek met SAM en Trend ---

# Bepaal de weergaveperiode op basis van interval
grafiek_periode = bepaal_grafiekperiode(interval)

# Bepaal cutoff-datum
cutoff_datum = df.index.max() - grafiek_periode

# Filter alleen grafiekdata
df_grafiek = df[df.index >= cutoff_datum].copy()

# --- Grafiek met SAM en Trend ---
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(df_grafiek.index, df_grafiek["SAM"], color="lightblue", label="SAM")
ax1.axhline(y=0, color="black", linewidth=1, linestyle="--")  # nullijn
ax2 = ax1.twinx()
ax2.plot(df_grafiek.index, df_grafiek["Trend"], color="red", label="Trend")
ax1.set_ylabel("SAM")
ax2.set_ylabel("Trend")
fig.tight_layout()
st.pyplot(fig)

# --- Tabel met signalen en rendement ---
st.subheader("Laatste signalen en rendement")

# Kolommen selecteren en formatteren
kolommen = ["Close", "Advies", "SAM", "Trend", "Markt-%", "SAM-%"]
#tabel = df[kolommen].dropna().tail(30).round(3).copy()
tabel = df[kolommen].dropna().copy()
tabel = tabel.sort_index(ascending=False).head(20)

# Datumkolom aanmaken vanuit index
if not isinstance(tabel.index, pd.DatetimeIndex):
    tabel.index = pd.to_datetime(tabel.index, errors="coerce")
tabel = tabel[~tabel.index.isna()]
tabel["Datum"] = tabel.index.strftime("%d-%m-%Y")

# Zet kolomvolgorde
tabel = tabel[["Datum"] + kolommen]

# Afronding en formatting
if selected_tab == "🌐 Crypto":
    tabel["Close"] = tabel["Close"].map("{:.3f}".format)
else:
    tabel["Close"] = tabel["Close"].map("{:.2f}".format)
# ✅ Eerst de juiste berekening behouden als float (NIET afronden!)
tabel["Markt-%"] = tabel["Markt-%"].astype(float) * 100
tabel["SAM-%"] = tabel["SAM-%"].astype(float) * 100

# ✅ Daarna afzonderlijke kolommen voor weergave formatteren
tabel["Markt-% weergave"] = tabel["Markt-%"].map("{:+.2f}%".format)
tabel["SAM-% weergave"] = tabel["SAM-%"].map("{:+.2f}%".format)
tabel["Trend Weergave"] = tabel["Trend"].map("{:+.3f}".format)

tabel = tabel[["Datum", "Close", "Advies", "SAM", "Trend Weergave", "Markt-% weergave", "SAM-% weergave"]]
tabel = tabel.rename(columns={
    "Markt-% weergave": "Markt-%",
    "SAM-% weergave": "SAM-%",
    "Trend Weergave": "Trend"
})

#tabel["SAM"] = tabel["SAM"].map("{:.2f}".format)
#tabel["Trend"] = tabel["Trend"].map("{:.3f}".format)

#tabel["Markt-%"] = (tabel["Markt-%"].astype(float) * 100).map("{:+.2f}%".format)
#tabel["SAM-%"] = (tabel["SAM-%"].astype(float) * 100).map("{:+.2f}%".format)


# HTML-rendering
html = """
<style>
    table {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    th {
        background-color: #004080;
        color: white;
        padding: 6px;
        text-align: center;
    }
    td {
        border: 1px solid #ddd;
        padding: 6px;
        text-align: right;
        background-color: #f9f9f9;
        color: #222222;
    }
    tr:nth-child(even) td {
        background-color: #eef2f7;
    }
    tr:hover td {
        background-color: #d0e4f5;
    }
</style>
<table>
    <thead>
        <tr>
            <th style='width: 110px;'>Datum</th>
            <th style='width: 80px;'>Close</th>
            <th style='width: 90px;'>Advies</th>
            <th style='width: 60px;'>SAM</th>
            <th style='width: 70px;'>Trend</th>
            <th style='width: 90px;'>Markt-%</th>
            <th style='width: 90px;'>SAM-%</th>
        </tr>
    </thead>
    <tbody>
"""

# Voeg rijen toe aan de tabel
for _, row in tabel.iterrows():
    html += "<tr>"
    for value in row:
        html += f"<td>{value}</td>"
    html += "</tr>"

html += "</tbody></table>"

#Weergave in Streamlit
st.markdown(html, unsafe_allow_html=True)


# 📊 3. Backtestfunctie: sluit op close van nieuw signaal
## 📊 3. Backtestfunctie: sluit op close van nieuw signaal
# ✅ 1. Data inladen of ontvangen (voorbeeld)
# ✅ 2. Data voorbereiden voor backtest (vereist kolom 'Advies')
df_signalen = df.copy()
if "Advies" not in df_signalen.columns:
    st.error("Kolom 'Advies' ontbreekt in de data.")
    st.stop()

##--- Toevoeging: Backtestfunctie ---

from datetime import date
import pandas as pd
import streamlit as st

# âœ… Zorg dat de index datetime is
df = df.copy()
df.index = pd.to_datetime(df.index)

# ðŸ—“ï¸ 1. Datumkeuze
st.subheader("Vergelijk Marktrendement en SAM-rendement")

# â±ï¸ Aangepaste startdatum = 1 januari huidig jaar
current_year = date.today().year
default_start = date(current_year, 1, 1)
default_end = df.index.max().date()

start_date = st.date_input("Startdatum analyse", default_start)
end_date = st.date_input("Einddatum analyse", default_end)

# ðŸ“… 2. Filter op periode
df_period = df.loc[
    (df.index.date >= start_date) & (df.index.date <= end_date)
].copy()

# ðŸ§¹ Flatten MultiIndex indien nodig
if isinstance(df_period.columns, pd.MultiIndex):
    df_period.columns = [
        "_".join([str(i) for i in col if i]) for col in df_period.columns
    ]

# ðŸ” Zoek geldige 'Close'-kolom
close_col = next(
    (col for col in df_period.columns if col.lower().startswith("close")), None
)

if not close_col:
    st.error("âŒ Geen geldige 'Close'-kolom gevonden in deze dataset.")
    st.stop()

# ðŸ“ˆ Marktrendement (Buy & Hold)
df_period[close_col] = pd.to_numeric(df_period[close_col], errors="coerce")
df_valid = df_period[close_col].dropna()

marktrendement = None
if len(df_valid) >= 2 and df_valid.iloc[0] != 0.0:
    koers_start = df_valid.iloc[0]
    koers_eind = df_valid.iloc[-1]
    marktrendement = ((koers_eind - koers_start) / koers_start) * 100

# âœ… Signaalkeuze geforceerd op Beide
signaalkeuze = "Beide"

# ðŸŽ¯ Filter alleen op Kopen/Verkopen
advies_col = "Advies"
df_signalen = df_period[df_period[advies_col].isin(["Kopen", "Verkopen"])].copy()

# ðŸ“Š 3. Backtestfunctie: sluit op close van nieuw signaal
def bereken_sam_rendement(df_signalen, signaal_type="Beide", close_col="Close"):
    rendementen = []
    trades = []
    entry_price = None
    entry_date = None
    entry_type = None

    type_map = {"Koop": "Kopen", "Verkoop": "Verkopen", "Beide": "Beide"}
    mapped_type = type_map.get(signaal_type, "Beide")

   # for datum, row in df_signalen.iterrows():
    #    advies = row["Advies"]
     #   close = row[close_col]

    for i in range(len(df_signalen)):
        advies = df_signalen["Advies"].iloc[i]
        close = df_signalen[close_col].iloc[i]
        datum = df_signalen.index[i]

        # ✅ STAP 1: Forceer eerste trade bij begin selectie
        # ✅ STAP 1: Forceer eerste geldige trade
        if entry_type is None and advies in ["Kopen", "Verkopen"]:
            if mapped_type == "Beide" or advies == mapped_type:
                entry_type = advies
                entry_price = close
                entry_date = datum
                continue  # ⬅️ Na eerste entry niets anders doen in deze iteratie
    
     #   if i == 0 and advies in ["Kopen", "Verkopen"]:
      #      entry_type = advies
       #     entry_price = close
     #       entry_date = datum
   #         continue
#

  #      if entry_type is None:
  #          if mapped_type == "Beide" or advies == mapped_type:
  #              entry_type = advies
    #            entry_price = close
   #             entry_date = datum
        else:
            if advies != entry_type and (mapped_type == "Beide" or entry_type == mapped_type):
                # Sluit trade op nieuwe signaal
                sluit_datum = datum
                sluit_close = close

                if entry_type == "Kopen":
                    rendement = (sluit_close - entry_price) / entry_price * 100
                else:
                    rendement = (entry_price - sluit_close) / entry_price * 100

                rendementen.append(rendement)
                trades.append({
                    "Type": entry_type,
                    "Open datum": entry_date.date(),
                    "Open prijs": round(entry_price, 2),
                    "Sluit datum": sluit_datum.date(),
                    "Sluit prijs": round(sluit_close, 2),
                    "Rendement (%)": round(rendement, 2)
                })

                # Eventueel nieuwe trade openen
                if mapped_type == "Beide" or advies == mapped_type:
                    entry_type = advies
                    entry_price = close
                    entry_date = datum
                else:
                    entry_type = None
                    entry_price = None
                    entry_date = None

    # Forceer sluiting op einddatum
    if entry_type and entry_price is not None:
        laatste_datum = df_signalen.index[-1]
        laatste_koers = df_signalen[close_col].iloc[-1]

        if entry_type == "Kopen":
            rendement = (laatste_koers - entry_price) / entry_price * 100
        else:
            rendement = (entry_price - laatste_koers) / entry_price * 100

        rendementen.append(rendement)
        trades.append({
            "Type": entry_type,
            "Open datum": entry_date.date(),
            "Open prijs": round(entry_price, 2),
            "Sluit datum": laatste_datum.date(),
            "Sluit prijs": round(laatste_koers, 2),
            "Rendement (%)": round(rendement, 2)
        })

    sam_rendement = sum(rendementen) if rendementen else 0.0
    return sam_rendement, trades, rendementen

# âœ… 4. Berekening
sam_rendement, trades, rendementen = bereken_sam_rendement(df_signalen, signaalkeuze, close_col)

# ✅ 6. Visueel weergeven
col1, col2 = st.columns(2)
col1.metric("Marktrendement (Buy & Hold)", f"{marktrendement:+.2f}%" if marktrendement is not None else "n.v.t.")
col2.metric("📊 SAM-rendement", f"{sam_rendement:+.2f}%" if isinstance(sam_rendement, (int, float)) else "n.v.t.")

if trades:
    df_trades = pd.DataFrame(trades)

    df_trades["SAM-% Koop"] = df_trades.apply(
        lambda row: row["Rendement (%)"] if row["Type"] == "Kopen" else None, axis=1)
    df_trades["SAM-% Verkoop"] = df_trades.apply(
        lambda row: row["Rendement (%)"] if row["Type"] == "Verkopen" else None, axis=1)
    df_trades["Markt-%"] = df_trades.apply(
        lambda row: round(((row["Sluit prijs"] - row["Open prijs"]) / row["Open prijs"]) * 100, 2), axis=1)

    df_trades = df_trades.rename(columns={"Rendement (%)": "SAM-% tot."})
    df_trades = df_trades[[
        "Open datum", "Open prijs", "Sluit datum", "Sluit prijs",
        "Markt-%", "SAM-% tot.", "SAM-% Koop", "SAM-% Verkoop"]]

    aantal_trades = len(df_trades)
    aantal_koop = df_trades["SAM-% Koop"].notna().sum()
    aantal_verkoop = df_trades["SAM-% Verkoop"].notna().sum()
    rendement_totaal = df_trades["SAM-% tot."].sum()
    rendement_koop = df_trades["SAM-% Koop"].sum(skipna=True)
    rendement_verkoop = df_trades["SAM-% Verkoop"].sum(skipna=True)
    aantal_succesvol = (df_trades["SAM-% tot."] > 0).sum()
    aantal_succesvol_koop = (df_trades["SAM-% Koop"] > 0).sum()
    aantal_succesvol_verkoop = (df_trades["SAM-% Verkoop"] > 0).sum()

    st.caption(f"Aantal afgeronde **trades**: **{aantal_trades}**, totaal resultaat SAM-%: **{rendement_totaal:+.2f}%**, aantal succesvol: **{aantal_succesvol}**")
    st.caption(f"Aantal **koop** trades: **{aantal_koop}**, SAM-% koop: **{rendement_koop:+.2f}%**, succesvol: **{aantal_succesvol_koop}**")
    st.caption(f"Aantal **verkoop** trades: **{aantal_verkoop}**, SAM-% verkoop: **{rendement_verkoop:+.2f}%**, succesvol: **{aantal_succesvol_verkoop}**")

    def kleur_positief_negatief(val):
        if pd.isna(val): return "color: #808080"
        if val > 0: return "color: #008000"
        if val < 0: return "color: #FF0000"
        return "color: #808080"

    kleurbare_kolommen = ["Markt-%", "SAM-% tot.", "SAM-% Koop", "SAM-% Verkoop"]
    toon_alle = st.toggle("Toon alle trades", value=False)
    df_display = df_trades if toon_alle or len(df_trades) <= 12 else df_trades.iloc[-12:]

    styler = df_display.style.format({col: "{:+.2f}%" for col in kleurbare_kolommen})
    styler = styler.applymap(kleur_positief_negatief, subset=kleurbare_kolommen)
    st.dataframe(styler, use_container_width=True)
else:
    st.info("ℹ️ Geen trades gevonden binnen de geselecteerde periode.")
    
    
















# wit










# with
