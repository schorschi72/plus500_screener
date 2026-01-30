# -*- coding: utf-8 -*-
"""
Multi-Symbol Screener & Backtest (Haltedauer & Skalierung) – Plus500-Finder & Hebel-Filter

Wissenschaftliche Strenge:
  A) Datum-basierte, purged Cross-Validation im Screener (kein Cross-Section-Leakage)
  B) Trailing-Stop & Entry-Sizing mit ATR vom Vortag (kein Intraday-Look-Ahead)
  C) Richtungsentscheidung ohne KI rein regelbasiert (SHORTs wieder möglich)
  D) Sharpe auf Tages-Equity-Returns (sqrt(252))
  E) Portfolio-Risikodeckel (max. simultanes Gesamtrisiko in Vielfachen des Einztrisikos)
  F) Time-Exit-Text an Entry-Datum (Open t+1) ausgerichtet
  G) Optionale Feature-Normierung (z-Score je Symbol)
  H) Walk-Forward-Qualitätsmetriken (Accuracy, AUC, Brier) im Backtest-Tab

Robustheit:
  - Mehrfach-Kandidaten je DataSymbol (kommagetrennt); erster funktionierender Ticker wird genutzt.
  - GOLD-Fallback: 'GC=F' -> 'XAUUSD=X'; SILBER-Fallback: 'SI=F' -> 'XAGUSD=X'; PALLADIUM: 'PL=F' -> 'XPDUSD=X'.
  - Index-Fallbacks auf ETF-Proxies (z. B. ^STOXX50E -> EXW1.DE; ^FCHI -> E40.PA; ^GDAXI -> EXS1.DE).
  - Normalisierung problematischer Ticker (z. B. '^FTSEMIB' -> 'FTSEMIB.MI').
  - Synthetische FX (GBPUSD) aus GBPEUR × EURUSD, wenn Direktticker ausfällt.
  - „🎛️ Code‑1‑Modus“: 1‑Tages‑Horizont, Haltedauer=1, min_score=0.55, ATRN=0.3–4.0%, SHORT an, Normierung aus.

🆕 Live-Preis (Beta):
  - Optionaler Live-Schalter in der Sidebar.
  - Holt den aktuellsten Intraday-Minutenpreis (1m/5m/fast_info Fallback).
  - Patcht nur HEUTE den Close (Features/ATR bleiben Vortag → kein Intraday-Look-Ahead).
  - Optionaler Auto-Refresh (streamlit_autorefresh falls vorhanden; HTML-Fallback deaktiviert).

NEU (Gating-Logik, Stabilitäts-Patches):
  - Alle Einstellungen zuerst in einer Form übernehmen („✅ Einstellungen übernehmen“).
  - Panel/Daten werden nur per Button „🔁 Daten laden/aktualisieren“ gebaut.
  - Screener wird nur per Button „🚦 Screener jetzt ausführen“ gerechnet.
  - Backtest startet nur per Button „🚀 Backtest starten“.
  - Auto-Refresh pausiert automatisch während Screener/Backtest; Ergebnisse werden persistiert.
  - Sidebar-Button „🔄 Auto‑Refresh wieder aktivieren“ hebt die Pause wieder auf.

Trader‑QoL (neu):
  - Mindesthistorie je Ticker konfigurierbar (Default 20).
  - MACD_hist_prev‑Fallback: first valid row nutzt prev=hist statt Komplett‑Drop.
  - Diagnose-Deckung: „Gewählt vs Panel/Roh/Picks“.
  - ATRN‑Band im Rule‑Check optional an/aus.
  - Klarere Hinweise bei leeren Rohsignalen.
  - Symbol‑Quick‑Diagnose (letzte 3 Zeilen je Titel).
"""

import os
import re
import html
import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss

# yfinance-Logs leiser stellen
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="Multi-Symbol Screener (Plus500-Finder & Hebel-Filter)", layout="wide")

# --- Sticky Top-Leiste: Styles ---
st.markdown("""
<style>
:root { --topbar-height: 96px; }

/* Light/Dark Theme Farben */
@media (prefers-color-scheme: light) {
  :root { --topbar-bg: rgba(255,255,255,0.92); --topbar-fg: #111827; --topbar-border: rgba(0,0,0,0.06); }
}
@media (prefers-color-scheme: dark) {
  :root { --topbar-bg: rgba(13,17,23,0.85); --topbar-fg: #E5E7EB; --topbar-border: rgba(255,255,255,0.07); }
}

/* Container der Top-Leiste (sticky) */
#topbar {
  position: sticky;
  top: 0;
  z-index: 9999;
  background: var(--topbar-bg);
  color: var(--topbar-fg);
  border-bottom: 1px solid var(--topbar-border);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  padding: 12px 16px;
  border-radius: 10px;
  margin-bottom: 8px;
}

/* Layout der Karten in der Leiste */
#topbar .row {
  display: flex;
  gap: 12px;
  align-items: stretch;
  flex-wrap: wrap;
}
#topbar .cell { flex: 1 1 320px; min-width: 280px; }

/* Karten/Chips im Topbar */
#topbar .chip {
  display:flex; gap:10px; align-items:flex-start;
  border-radius:10px;
  border:1px solid var(--topbar-border);
  padding:10px 12px;
  font-size:0.95rem;
  background: transparent;
}
#topbar .chip .dot { font-size:1.15rem; line-height:1; }

/* Bullet-Liste in Leiste */
#topbar .reasons { margin: 6px 0 0 0; padding: 0 0 0 1.1em; }
#topbar .reasons li { margin: 0.2em 0; }

/* Links (Placeholder) */
#topbar a { color: inherit; text-decoration: underline dotted; }
</style>
""", unsafe_allow_html=True)

# =========================
# Tooltip-Funktion
# =========================
def hover_info_icon(text: str) -> str:
    safe = html.escape(text, quote=True)
    return f"<span title=\"{safe}\" style=\"cursor:help\"> ℹ️</span>"

# =========================
# HELP / Glossar
# =========================
HELP = {
    # Auswahl / Katalog
    "history_years": "Lookback für den Daten‑Download. Mehr Historie = robustere Features & stabileres Modell.",
    "hold_days": "Maximale Haltedauer eines Trades. Spätester Exit H Tage nach Entry (Entry = Open t+1).",
    "scale_mode": "Skalierung Stop/TP/Trailing: √Zeit, linear oder keine.",
    "forecast_horizon": "Wenn an: KI‑Label über Haltedauer H, sonst 1‑Tag‑Label.",

    # Konto / Risiko / Filter
    "equity": "Kontogröße (Equity) für Positionsgrößen & Backtest.",
    "risk_pct": "Positionsgröße = (Equity × Risiko%) ÷ (Stopdistanz × Punktewert).",
    "enable_short": "SHORTs erlauben.",
    "min_score": "FinalScore‑Schwelle: 0.7×AI_Prob + 0.3×Rule_OK.",
    "time_exit": "Schließt Position spätestens nach H Tagen.",
    "trailing": "Trailing mit ATR vom Vortag (kein Intraday‑Look‑Ahead).",
    "atrn_minmax": "ATRN‑Band (ATR/Close). Code‑1: 0.3–4.0%.",
    "cost_per_trade": "Fixe Kosten je Trade.",
    "stop_first": "Stop vor TP bei gleichzeitiger Berührung (konservativ).",
    "use_ai_bt": "KI (RandomForest) in Walk‑Forward nutzen.",
    "retrain_every": "Retraining‑Frequenz (Tage).",
    "fast_mode": "Schnellerer Backtest (kürzeres Fenster, weniger Features).",
    "feature_norm": "z‑Score Normierung je Symbol.",
    "max_port_risk": "Risikodeckel (× Einzetrisiko).",

    # Plus500
    "acct_type": "Konto‑Typ (Retail/Pro) steuert Hebelgrenzen.",
    "point_value": "CFD‑Punktewert: PnL = (Exit − Entry) × Units × Punktewert.",

    # Code‑1
    "code1_mode": "Preset: H=1, Score=0.55, ATRN 0.3–4.0%, SHORT an, Normierung aus.",

    # Live
    "live_toggle": "Heutigen Close mit Live‑Minutenpreis patchen.",
    "auto_refresh": "Auto‑Refresh für Live‑Patch.",
    "live_entry": "Entry im Plan = Live‑Preis (nur Anzeige).",

    # Trader‑Erweitert
    "use_atrn_filter": "ATRN‑Band im Rule‑Check anwenden (an/aus).",
    "min_rows_first_ok": "Minimale Historie (Zeilen) für ersten funktionierenden Ticker (Default 20).",
}

# =========================
# Spalten-Tooltips
# =========================
COL_HELP: Dict[str, str] = {
    "Plus500Name": "Originaler Plus500-Name.",
    "OrigSymbol": "Plus500-Name so wie gewählt.",
    "Symbol": "Verwendeter Yahoo‑Ticker.",
    "Category": "Kategorie (Indices/FX/…)",
    "Date": "Schlusskurs‑Tag.",
    "Open": "Tages‑Open.",
    "High": "Tages‑High.",
    "Low": "Tages‑Low.",
    "Close": "Tages‑Close (oder Live‑Patch).",
    "EMA20": "EMA(20).",
    "EMA50": "EMA(50).",
    "RSI7": "RSI(7).",
    "RSI14": "RSI(14).",
    "MACD": "MACD‑Linie.",
    "MACD_signal": "MACD‑Signal.",
    "MACD_hist": "MACD‑Histogramm.",
    "MACD_hist_prev": "MACD‑Hist vom Vortag.",
    "ATR14": "ATR(14).",
    "ATRN": "ATR/Close.",
    "ATRN_%": "ATRN in %.",
    "VolZ20": "Volumen‑ZScore(20).",
    "ROC3": "3‑Tages‑ROC.",
    "BreakoutUp": "Close > Vor‑High.",
    "BreakoutDn": "Close < Vor‑Low.",
    "Ret_1D": "Tagesrendite.",
    "Ret_1D_fwd": "Rendite t->t+1 (Label).",
    "FwdRet_H": "Vorwärtsrendite über H.",
    "Label": "1=Up, 0=Down.",
    "AI_Prob_Up": "KI‑Wahrscheinlichkeit Up.",
    "AI_Prob": "Für LONG=AI_Prob_Up, SHORT=1−AI_Prob_Up.",
    "Direction": "LONG/SHORT/NO‑TRADE.",
    "Rule_OK": "Regeln erfüllt (1/0).",
    "FinalScore": "0–1 Score.",
    "EntryPrice_used": "Entry‑Preis (Live wenn aktiv).",
    "Stop": "Stop‑Level.",
    "TakeProfit": "TP‑Level.",
    "Units_suggested": "Vorgeschlagene Stückzahl.",
    "Time_Exit_By": "Spätester Zeit‑Exit.",
    "Trailing": "Trailing‑Beschreibung.",
    "EntryDate": "Entry‑Datum.",
    "EntryPrice": "Entry‑Preis.",
    "ATR14_atEntry": "ATR am Vortag des Entry.",
    "StopInit": "Initialer Stop.",
    "TPInit": "Initialer TP.",
    "ExitDate": "Exit‑Datum.",
    "ExitPrice": "Exit‑Preis.",
    "ExitReason": "Grund des Exits.",
    "DaysHeld": "Kalendertage im Trade.",
    "PnL": "PnL (nach Kosten).",
    "R": "Ergebnis in R‑Einheiten.",
    "Win": "Trade win?",
    "Trades": "Trades je Titel.",
    "WinRate": "Trefferquote.",
    "AvgR": "Durchschnitts‑R.",
    "TotalPnL": "Summe PnL.",
    "Vol-Zone": "ATRN‑Zone.",
}

def build_col_config(df: pd.DataFrame) -> Dict[str, "st.column_config.Column"]:
    cfg: Dict[str, "st.column_config.Column"] = {}
    if df is None or df.empty or not hasattr(st, "column_config"):
        return cfg
    for col in df.columns:
        help_txt = COL_HELP.get(col, "")
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if hasattr(st.column_config, "DatetimeColumn"):
                    cfg[col] = st.column_config.DatetimeColumn(col, help=help_txt)
                else:
                    cfg[col] = st.column_config.Column(col, help=help_txt)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if col.endswith("%") or col.lower().endswith("_pct"):
                    if hasattr(st.column_config, "NumberColumn"):
                        cfg[col] = st.column_config.NumberColumn(col, help=help_txt, format="%.3f")
                    else:
                        cfg[col] = st.column_config.Column(col, help=help_txt)
                else:
                    if hasattr(st.column_config, "NumberColumn"):
                        cfg[col] = st.column_config.NumberColumn(col, help=help_txt)
                    else:
                        cfg[col] = st.column_config.Column(col, help=help_txt)
            else:
                cfg[col] = st.column_config.Column(col, help=help_txt)
        except Exception:
            pass
    return cfg

# =========================
# Utils
# =========================
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii").strip().upper()

def _to_naive_dt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    try:
        out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out

def _now_berlin():
    try:
        return pd.Timestamp.now(tz="Europe/Berlin")
    except Exception:
        return pd.Timestamp.now()

def status_chip(color: str, text_md: str) -> str:
    colors = {
        "green": ("#065f46", "#d1fae5", "🟢"),
        "orange": ("#92400e", "#fde68a", "🟠"),
        "red": ("#7f1d1d", "#fecaca", "🔴"),
        "gray": ("#374151", "#e5e7eb", "⚪"),
    }
    fg, bg, dot = colors.get(color, colors["gray"])
    return (
        f"<div style='display:flex;gap:10px;align-items:flex-start;"
        f"background:{bg};color:{fg};border-radius:10px;padding:10px 12px;"
        f"border:1px solid rgba(0,0,0,0.06);font-size:0.95rem;'>"
        f"<div style='font-size:1.1rem;'>{dot}</div>"
        f"<div style='line-height:1.35'><div><strong>Daten‑Qualität:</strong> {color.upper()}</div><div>{text_md}</div></div>"
        f"</div>"
    )

def assess_data_quality(panel: pd.DataFrame, use_live_now: bool) -> dict:
    out = {"status": "gray", "reasons": [], "detail": pd.DataFrame()}
    if panel is None or panel.empty:
        out["status"] = "red"
        out["reasons"].append("Panel ist leer (kein Symbol/keine Features).")
        return out

    now = _now_berlin()
    today = now.normalize().date()
    try:
        latest_date = pd.to_datetime(panel["Date"]).dt.tz_localize(None).max().date()
    except Exception:
        latest_date = pd.to_datetime(panel["Date"]).max().date()

    core_feats = ["Close","EMA20","EMA50","RSI7","RSI14","MACD_hist","ATR14","ATRN"]
    present = [c for c in core_feats if c in panel.columns]
    nan_frac = panel[present].isna().mean().mean() if present else 1.0

    depth = panel.groupby("OrigSymbol")["Date"].count()
    shallow_symbols = depth[depth < 150].index.tolist()

    if use_live_now:
        out["status"] = "orange"
        out["reasons"].append("Live‑Preis aktiv: Heutiger Close gepatcht → Intraday (noch nicht final).")
    else:
        if latest_date < today:
            out["status"] = "green"
            out["reasons"].append(f"EOD‑Daten vollständig (letztes Datum = {latest_date}).")
        elif latest_date == today:
            if now.hour > 22 or (now.hour == 22 and now.minute >= 15):
                out["status"] = "green"
                out["reasons"].append("Heute nach 22:15 → Daten sehr wahrscheinlich final.")
            else:
                out["status"] = "orange"
                out["reasons"].append("Heutiges Datum vor 22:15 → Daten können unvollständig sein.")
        else:
            out["status"] = "red"
            out["reasons"].append("Zukünftiges Datum erkannt (Zeit-/Datenfehler).")

    if nan_frac > 0.02 and out["status"] != "red":
        out["status"] = "orange"
    if nan_frac > 0.10:
        out["status"] = "red"
    out["reasons"].append(f"NaN‑Anteil in Kernfeatures: {nan_frac*100:.2f}%.")

    if shallow_symbols:
        out["reasons"].append(
            f"Wenig Historie (<150 Zeilen) bei: {', '.join(shallow_symbols[:6])}" +
            (" …" if len(shallow_symbols) > 6 else "")
        )

    last_per_sym = (panel.sort_values(["OrigSymbol","Date"])
                          .groupby("OrigSymbol")
                          .tail(1)[["OrigSymbol","Symbol","Date","Close","ATR14","ATRN"]]
                          .copy())
    last_per_sym["Date"] = pd.to_datetime(last_per_sym["Date"]).dt.date
    last_per_sym["OK_Features"] = last_per_sym[["Close","ATR14","ATRN"]].notna().all(axis=1)
    out["detail"] = last_per_sym.sort_values("OrigSymbol")
    return out

def render_quality_box(q: dict):
    color = q.get("status", "gray")
    lines = q.get("reasons", [])
    text = "  \n".join(f"• {l}" for l in lines) if lines else "Keine Gründe angegeben."
    st.markdown(status_chip(color, text), unsafe_allow_html=True)
    with st.expander("🔎 Qualitäts‑Details je Symbol", expanded=False):
        detail_df = q.get("detail", pd.DataFrame())
        st.dataframe(detail_df, use_container_width=True, column_config=build_col_config(detail_df))

def detect_session(now_ts: pd.Timestamp) -> str:
    h, m = now_ts.hour, now_ts.minute
    if (h == 7 and m >= 0) or (8 <= h < 9) or (h == 9 and m <= 0):
        return "morning"
    if (h == 22 and m >= 15) or (22 <= h <= 23):
        return "evening"
    return "regular"

def recommend_by_session(picks: pd.DataFrame, catalog_df: pd.DataFrame) -> pd.DataFrame:
    if picks is None or picks.empty or catalog_df is None or catalog_df.empty:
        return pd.DataFrame()

    now = _now_berlin()
    sess = detect_session(now)

    cat_map = catalog_df.set_index("Plus500Name")["Category"].to_dict()
    df = picks.copy()
    df["Category"] = df["OrigSymbol"].map(cat_map).fillna("")

    if sess == "morning":
        keep_cat = {"Indices","Forex","Commodities"}
        prefer_names = {"GERMANY 40","SWISS 20","EURO STOXX 50","EUR/USD","GOLD","BRENT OIL","OIL","NATURAL GAS"}
        df["prio"] = (
            df["OrigSymbol"].isin(prefer_names).astype(int)*2
            + df["Category"].isin(keep_cat).astype(int)
        )
        rec = df[df["Category"].isin(keep_cat)].copy()
    elif sess == "evening":
        keep_cat = {"Indices","Commodities","Forex"}
        prefer_names = {"US 500","US-TECH 100","GOLD","SILVER","OIL","BRENT OIL","NATURAL GAS","EUR/USD","USD/JPY"}
        df["prio"] = (
            df["OrigSymbol"].isin(prefer_names).astype(int)*2
            + df["Category"].isin(keep_cat).astype(int)
        )
        rec = df[df["Category"].isin(keep_cat)].copy()
    else:
        df["prio"] = 1
        rec = df.copy()

    if rec.empty:
        return rec

    def atr_ok(a):
        try:
            return 0.003 <= float(a) <= 0.04
        except Exception:
            return False

    rec["ATR_ok"] = rec["ATRN"].apply(atr_ok).astype(int)
    rec = rec.sort_values(["prio","ATR_ok","FinalScore"], ascending=[False, False, False])
    cols = [c for c in ["OrigSymbol","Category","Direction","FinalScore","AI_Prob","ATRN","EntryPrice_used","Stop","TakeProfit","Units_suggested"] if c in rec.columns]
    return rec[cols].head(12).reset_index(drop=True)

def auto_export_watchlist(picks: pd.DataFrame,
                          filename_prefix: str = "watchlist_plus500",
                          window_start: str = "22:15",
                          window_end: str = "23:30") -> Optional[str]:
    if picks is None or picks.empty:
        return None

    now = _now_berlin()
    today = now.normalize()
    ws_h, ws_m = map(int, window_start.split(":"))
    we_h, we_m = map(int, window_end.split(":"))
    ws = today + pd.Timedelta(hours=ws_h, minutes=ws_m)
    we = today + pd.Timedelta(hours=we_h, minutes=we_m)

    if not (ws <= now <= we):
        return None

    export_cols = [c for c in [
        "OrigSymbol","Symbol","Date","Direction","FinalScore","AI_Prob","Close",
        "EMA20","EMA50","RSI7","RSI14","MACD_hist","ATR14","ATRN",
        "EntryPrice_used","Stop","TakeProfit","Units_suggested","Time_Exit_By","Trailing"
    ] if c in picks.columns]
    if not export_cols:
        return None

    date_tag = now.strftime("%Y-%m-%d_%H%M")
    fname = f"{filename_prefix}_{date_tag}.csv"
    try:
        picks[export_cols].to_csv(fname, index=False, encoding="utf-8")
        return fname
    except Exception:
        return None

# --------- Yahoo-Symbol-Normalisierung ----------
def normalize_yahoo_symbol(sym: str) -> str:
    s = (sym or "").strip()
    u = s.upper()

    # Indizes
    if u in {"^FTSEMIB", "FTSE MIB", "FTSE-MIB", "ITA40", "ITA 40"}:
        return "FTSEMIB.MI"
    if u in {"STOXX50E", "^EUROSTOXX50", "EURO STOXX 50"}:
        return "^STOXX50E"
    if u in {"CAC40", "CAC 40"}:
        return "^FCHI"
    if u in {"DAX", "GER40", "GERMANY 40", "GERMANY40"}:
        return "^GDAXI"

    # Metalle / Commodities
    if u in {"SI=F", "SILVER FUTURES", "SILVER"}:
        return "SI=F"
    if u in {"PL=F", "PALLADIUM", "PALLADIUM FUTURES"}:
        return "PL=F"
    if u in {"XAU", "XAUUSD", "XAUUSD=X", "GOLD"}:
        return "XAUUSD=X"

    # FX
    if u in {"GBPUSD", "GBP/USD"}:
        return "GBPUSD=X"

    return s

# =========================
# Plus500-Finder: Katalog/Mapping
# =========================
@dataclass
class P5Item:
    Plus500Name: str
    Category: str
    DataSymbol: str
    DefaultPointValue: float
    MaxLeverage_Retail: float
    MaxLeverage_Pro: float
    Notes: str = ""

P5_DEFAULTS: List[P5Item] = [
    # Indices
    P5Item("GERMANY 40","Indices","^GDAXI",1.0,20,300,"DAX Index Proxy"),
    P5Item("SWISS 20","Indices","^SSMI",0.1,20,300,"SMI Proxy"),
    P5Item("US 500","Indices","^GSPC",1.0,20,300,"S&P 500 Proxy"),
    P5Item("US-TECH 100","Indices","^NDX",1.0,20,300,"NASDAQ 100 Proxy"),
    P5Item("UK 100","Indices","^FTSE",1.0,20,300,"FTSE 100 Proxy"),
    P5Item("ITALY 40","Indices","FTSEMIB.MI",1.0,20,300,"FTSE MIB Proxy"),
    # Commodities
    P5Item("OIL","Commodities","CL=F",1.0,10,150,"WTI Crude Futures"),
    P5Item("BRENT OIL","Commodities","BZ=F",1.0,10,150,"Brent Futures"),
    P5Item("NATURAL GAS","Commodities","NG=F",1.0,10,150,"Henry Hub NG"),
    P5Item("GOLD","Commodities","GC=F,XAUUSD=X",1.0,20,150,"Gold Futures -> Spot Fallback"),
    P5Item("SILVER","Commodities","SI=F,XAGUSD=X",1.0,10,150,"Silver Futures -> Spot Fallback"),
    # Forex
    P5Item("EUR/USD","Forex","EURUSD=X",100000.0,30,300,"FX Major"),
    P5Item("USD/JPY","Forex","USDJPY=X",100000.0,30,300,"FX Major"),
    # Crypto
    P5Item("BITCOIN/USD","Crypto","BTC-USD",1.0,2,20,"Spot Proxy"),
    # Shares
    P5Item("TESLA","Shares","TSLA",1.0,5,20,"US Share"),
    P5Item("APPLE","Shares","AAPL",1.0,5,20,"US Share"),
]

CATALOG_FILE = "plus500_catalog.csv"

def load_catalog() -> pd.DataFrame:
    if os.path.exists(CATALOG_FILE):
        df = pd.read_csv(CATALOG_FILE)
    else:
        df = pd.DataFrame([vars(x) for x in P5_DEFAULTS])
    for col in ["Plus500Name","Category","DataSymbol","Notes"]:
        if col not in df.columns: df[col] = ""
    for col in ["DefaultPointValue","MaxLeverage_Retail","MaxLeverage_Pro"]:
        if col not in df.columns: df[col] = np.nan
    df["_KEY"] = df["Plus500Name"].apply(_norm)
    return df

# =========================
# GATING: Sidebar-Form – Alle Einstellungen zuerst übernehmen
# =========================
st.sidebar.header("🧭 Plus500‑Finder")

with st.sidebar.form("params_form", clear_on_submit=False):
    catalog_df = load_catalog()

    # Kategorie
    cats = ["Alle"] + sorted(catalog_df["Category"].dropna().unique().tolist())
    sel_cat = st.selectbox("Kategorie", options=cats, index=0)

    # Konto‑Typ & Hebel‑Band
    st.markdown("### Konto‑Typ & Hebel‑Filter")
    acct_type = st.radio("Konto‑Typ", ["Retail (ESMA)", "Professional"], index=0, help=HELP["acct_type"])
    min_leverage, max_leverage = st.slider("Hebel-Band (Min–Max)", min_value=2, max_value=300, value=(2, 300), step=1)
    lev_col = "MaxLeverage_Pro" if acct_type == "Professional" else "MaxLeverage_Retail"

    # Kandidaten
    candidates = catalog_df.copy() if sel_cat == "Alle" else catalog_df[catalog_df["Category"] == sel_cat].copy()
    lev = pd.to_numeric(candidates[lev_col], errors="coerce")
    candidates = candidates[(lev >= float(min_leverage)) & (lev <= float(max_leverage))].sort_values(["Category","Plus500Name"]).reset_index(drop=True)

    # Multi-Select
    def make_search_options(df: pd.DataFrame) -> Dict[str, str]:
        lab2name = {}
        for _, r in df.iterrows():
            name = str(r["Plus500Name"]); cat = str(r.get("Category","") or ""); sym = str(r.get("DataSymbol","") or "")
            label = f"[{cat}] {name}" + (f" — ({sym})" if sym else "")
            lab2name[label] = name
        return lab2name

    lab2name = make_search_options(candidates)
    search_labels = sorted(lab2name.keys())
    sel_all = st.checkbox("Alle Titel auswählen", value=False)
    selected_labels = st.multiselect("Titel suchen & auswählen (tippbar)", options=search_labels, default=(search_labels if sel_all else []))
    selected_plus500 = [lab2name[l] for l in selected_labels]

    # Manuelle Ticker
    with st.expander("Zusätzliche direkte Ticker (optional)", expanded=False):
        manual_raw = st.text_area("Direkte Daten-Ticker (kommasepariert; z. B. CL=F, GC=F, FTSEMIB.MI)", value="", height=60)

    # Historie & Haltedauer & Skalierung & Forecast
    history_years = st.slider("Historie (Jahre)", 1, 10, 5, 1, help=HELP["history_years"])
    hold_days = st.slider("Haltedauer (Tage, Time-Exit)", 1, 10, 2, 1, help=HELP["hold_days"])
    scale_mode = st.selectbox("Skalierung Stop/TP/Trailing", ["√Zeit (empfohlen)", "linear", "keine"], index=0, help=HELP["scale_mode"])
    forecast_horizon_days = st.checkbox("KI-Prognose auf Haltedauer H (statt 1 Tag)", value=True, help=HELP["forecast_horizon"])

    # Weitere Einstellungen
    with st.expander("Weitere Einstellungen", expanded=False):
        account_equity   = st.number_input("Kontogröße", min_value=1000.0, value=10000.0, step=500.0, format="%.2f", help=HELP["equity"])
        risk_per_trade   = st.slider("Risiko pro Trade (%)", 0.1, 1.5, 0.75, 0.05, help=HELP["risk_pct"]) / 100.0
        enable_short     = st.checkbox("Short-Signale aktivieren", value=True, help=HELP["enable_short"])
        min_score        = st.slider("Mindest-Score", 0.50, 0.85, 0.55, 0.01, help=HELP["min_score"])
        use_time_exit    = st.checkbox("Zeit-Exit aktiv", value=True, help=HELP["time_exit"])
        use_trailing     = st.checkbox("Trailing-Stop aktiv", value=True, help=HELP["trailing"])
        atrn_min_pct     = st.slider("ATRN Minimum (%)", 0.0, 6.0, 0.3, 0.1, help=HELP["atrn_minmax"])
        atrn_max_pct     = st.slider("ATRN Maximum (%)", 0.0, 6.0, 4.0, 0.1, help=HELP["atrn_minmax"])
        cost_per_trade   = st.number_input("Pauschale Kosten je Trade", min_value=0.0, value=0.0, step=0.5, help=HELP["cost_per_trade"])
        stop_first       = st.checkbox("Konservatives Fill (Stop vor TP)", value=True, help=HELP["stop_first"])
        use_ai_in_bt     = st.checkbox("KI im Backtest (Walk‑Forward)", value=True, help=HELP["use_ai_bt"])
        retrain_every_n  = st.slider("Re-Train-Frequenz (Tage)", 1, 20, 5, 1, help=HELP["retrain_every"])
        fast_mode        = st.checkbox("⚡ Fast Mode (schneller)", value=True, help=HELP["fast_mode"])
        feature_norm     = st.checkbox("Feature‑Normierung (z‑Score je Symbol)", value=False, help=HELP["feature_norm"])
        max_total_risk_multiple = st.slider("Max. Gesamt‑Risiko (× Einzetrisiko)", 1.0, 5.0, 2.0, 0.5, help=HELP["max_port_risk"])

    # Code-1‑Modus
    code1_mode = st.checkbox("🎛️ Code‑1‑Modus (Angleichen an Code 1)", value=False, help=HELP["code1_mode"])

    # Live-Preis
    st.markdown("### Live‑Preis (Beta)")
    use_live_now = st.checkbox("📡 Live‑Preis verwenden", value=False, help=HELP["live_toggle"])
    use_live_entry_in_plan = st.checkbox("Entry im Trade‑Plan = Live‑Preis (nur Anzeige)", value=True, help=HELP["live_entry"])
    auto_refresh_on = st.checkbox("Auto‑Refresh aktivieren", value=False, help=HELP["auto_refresh"])
    live_interval_sec = st.slider("Auto‑Refresh Intervall (Sek.)", 10, 300, 60, 10)

    # Backtest-Zeitraum
    bt_default_start = (pd.Timestamp.today() - pd.DateOffset(years=min(5, history_years))).date()
    bt_start, bt_end = st.date_input("Backtest-Zeitraum", value=(bt_default_start, pd.Timestamp.today().date()))

    # (NEU) Diagnose / Erweitert
    with st.expander("🔧 Diagnose / Erweitert", expanded=False):
        use_atrn_filter = st.checkbox("ATRN‑Band im Rule‑Check anwenden", value=True, help=HELP["use_atrn_filter"])
        min_rows_first_ok = st.number_input("Minimale Historie (Zeilen) je Kandidat", min_value=10, max_value=60, value=20, step=1, help=HELP["min_rows_first_ok"])

    debug_mode = st.checkbox("🪛 Debug-Ausgaben", value=False)

    # Form-Submit
    submitted = st.form_submit_button("✅ Einstellungen übernehmen")

# Übernommene Settings einfrieren
if submitted:
    st.session_state["params"] = dict(
        sel_cat=sel_cat, acct_type=acct_type, min_leverage=min_leverage, max_leverage=max_leverage, lev_col=lev_col,
        selected_plus500=selected_plus500, manual_raw=manual_raw,
        history_years=history_years, hold_days=hold_days, scale_mode=scale_mode,
        forecast_horizon_days=forecast_horizon_days, account_equity=account_equity, risk_per_trade=risk_per_trade,
        enable_short=enable_short, min_score=min_score, use_time_exit=use_time_exit, use_trailing=use_trailing,
        atrn_min_pct=atrn_min_pct, atrn_max_pct=atrn_max_pct, cost_per_trade=cost_per_trade, stop_first=stop_first,
        use_ai_in_bt=use_ai_in_bt, retrain_every_n=retrain_every_n, fast_mode=fast_mode, feature_norm=feature_norm,
        max_total_risk_multiple=max_total_risk_multiple, code1_mode=code1_mode,
        use_live_now=use_live_now, use_live_entry_in_plan=use_live_entry_in_plan,
        auto_refresh_on=auto_refresh_on, live_interval_sec=live_interval_sec,
        bt_start=pd.to_datetime(bt_start), bt_end=pd.to_datetime(bt_end),
        debug_mode=debug_mode,
        use_atrn_filter=use_atrn_filter, min_rows_first_ok=int(min_rows_first_ok)
    )
    st.session_state["params_ready"] = True

# Hinweis & Ablauf
st.markdown("### 🚦 Ablauf: 1) Einstellungen übernehmen → 2) Daten laden → 3) Screener/Backtest")
if not st.session_state.get("params_ready"):
    st.info("Bitte zuerst in der Sidebar **„✅ Einstellungen übernehmen“** klicken. Danach kannst du Daten laden.")
    st.stop()

# =========================
# Sitzungs-Flags & Persistenz
# =========================
st.session_state.setdefault("refresh_paused", False)
st.session_state.setdefault("latest_signals", pd.DataFrame())
st.session_state.setdefault("screener_reports", "")
st.session_state.setdefault("latest_picks_plan", pd.DataFrame())
st.session_state.setdefault("latest_rec_df", pd.DataFrame())
st.session_state.setdefault("last_signals_all", pd.DataFrame())
st.session_state.setdefault("last_backtest_trades", pd.DataFrame())
st.session_state.setdefault("last_backtest_summary", pd.DataFrame())
st.session_state.setdefault("last_backtest_eq", pd.Series(dtype="float64"))
st.session_state.setdefault("last_live_patch_ts", pd.Timestamp(0, tz="Europe/Berlin"))

# =========================
# Punktewerte (CFD)
# =========================
params = st.session_state["params"]
catalog_df = load_catalog()

symbol_rows = catalog_df[catalog_df["Plus500Name"].isin(params["selected_plus500"])].copy()
for t in re.split(r"[\s,;]+", params["manual_raw"].strip()):
    if t:
        symbol_rows = pd.concat([symbol_rows, pd.DataFrame([{
            "Plus500Name": t.strip(), "Category": "Manual",
            "DataSymbol": t.strip(), "DefaultPointValue": 1.0,
            "MaxLeverage_Retail": np.nan, "MaxLeverage_Pro": np.nan,
            "Notes": "Manual Ticker", "_KEY": _norm(t)
        }])], ignore_index=True)

st.sidebar.markdown("### Punktewerte je Titel (CFD)")
point_values: Dict[str, float] = st.session_state.get("point_values", {})
for _, r in symbol_rows.iterrows():
    orig = str(r["Plus500Name"]).strip()
    default_pv = r.get("DefaultPointValue", 1.0)
    try:
        default_pv = float(default_pv) if np.isfinite(default_pv) else 1.0
    except Exception:
        default_pv = 1.0
    val = st.sidebar.number_input(f"{orig} Punktewert", min_value=0.01, value=point_values.get(orig, default_pv), step=0.1, help=HELP["point_value"])
    point_values[orig] = float(val)
st.session_state["point_values"] = point_values

# =========================
# Auto-Refresh Steuerung
# =========================
st.sidebar.markdown("### 🔄 Auto‑Refresh Steuerung")
if st.session_state.get("refresh_paused"):
    st.sidebar.caption("Auto‑Refresh ist aktuell **pausiert**.")
resume_refresh = st.sidebar.button("🔄 Auto‑Refresh wieder aktivieren")
if resume_refresh:
    st.session_state["refresh_paused"] = False
    st.experimental_rerun()

# =========================
# Daten laden / Reset
# =========================
START_DATE = (pd.Timestamp.today() - pd.DateOffset(years=params["history_years"])).strftime("%Y-%m-%d")

symbols_map: Dict[str, str] = {}
for _, r in symbol_rows.iterrows():
    orig = str(r["Plus500Name"]).strip()
    data = str(r["DataSymbol"]).strip()
    if orig and data:
        symbols_map[orig] = data

colA, colB = st.columns([1,1])
with colA:
    do_load = st.button("🔁 Daten laden/aktualisieren", type="primary", key="btn_load")
with colB:
    do_reset = st.button("🧹 Reset (Panel/Signale)", key="btn_reset")

if do_reset:
    for k in ["panel", "signals_all", "latest_signals", "screener_reports",
              "latest_picks_plan", "latest_rec_df", "last_signals_all",
              "last_backtest_trades", "last_backtest_summary", "last_backtest_eq"]:
        st.session_state.pop(k, None)
    st.success("Zurückgesetzt. Bitte erneut Daten laden.")
    st.stop()

# --- Diagnose-Expander: yfinance-Test & Cache-Clear ---
with st.expander("🩺 Diagnose: yfinance-Test & Cache", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Test: Lade ^GSPC (S&P 500)", key="btn_diag_spx"):
            try:
                test = yf.download("^GSPC", period="5y", interval="1d", auto_adjust=True, progress=False)
                st.write("Zeilen:", len(test))
                st.dataframe(test.tail(5))
                if len(test) < 50:
                    st.warning("Wenig Zeilen → möglicherweise Netzwerk/Firewall/Rate-Limit.")
                else:
                    st.success("yfinance ok. Problem wahrsch. bei Mapping/Lookback/Filter.")
            except Exception as e:
                st.error(f"yfinance Fehler: {e}")
    with c2:
        if st.button("Cache leeren (cache_data.clear)", key="btn_clear_cache"):
            st.cache_data.clear()
            st.success("Cache geleert. Bitte erneut laden.")

# =========================
# Technische Indikatoren & Daten
# =========================
def ema(s: pd.Series, span: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    gain = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    loss = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = gain / (loss.replace(0, np.nan))
    return (100 - (100/(1+rs))).fillna(50.0)

def macd(s: pd.Series, fast: int=12, slow: int=26, signal: int=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    s = pd.to_numeric(s, errors="coerce")
    ef, es = ema(s, fast), ema(s, slow)
    line = ef-es
    sig  = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist

def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int=14) -> pd.Series:
    h = pd.to_numeric(h, errors="coerce"); l = pd.to_numeric(l, errors="coerce"); c = pd.to_numeric(c, errors="coerce")
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def volume_zscore(v: pd.Series, n: int=20) -> pd.Series:
    v = pd.to_numeric(v, errors="coerce")
    ma = v.rolling(n, min_periods=n).mean()
    sd = v.rolling(n, min_periods=n).std(ddof=0)
    return ((v - ma) / sd.replace(0, np.nan)).fillna(0.0)

def roc(s: pd.Series, n: int=3) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.pct_change(n)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history_single(symbol: str, start: str) -> pd.DataFrame:
    """Robuster History-Download mit Fallbacks; toleranter Clean."""
    sym = normalize_yahoo_symbol(symbol)

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

        close = df.get("Close")
        if close is None and df.get("Adj Close") is not None:
            df["Close"] = df["Adj Close"]

        if "Close" in df.columns:
            for c in ["Open","High","Low"]:
                if c not in df.columns:
                    df[c] = df["Close"]

        for c in ["Open","High","Low","Close","Adj Close","Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        core = [c for c in ["Open","High","Low","Close"] if c in df.columns]
        if not core:
            return pd.DataFrame()

        df = df.dropna(subset=core, how="any")
        if df.empty:
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        try: df.index = df.index.tz_localize(None)
        except Exception: pass
        return df

    def _try_download(**kwargs) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(sym, auto_adjust=True, progress=False, **kwargs)
            df = _clean(df)
            return df if not df.empty else None
        except Exception:
            return None

    errors = []

    # start-basiert
    df = _try_download(start=start)
    if df is not None and len(df) >= 30:
        return df
    if df is None: errors.append("start")

    # period=max
    df = _try_download(period="max", interval="1d")
    if df is not None and len(df) >= 30:
        try: df = df[df.index >= pd.to_datetime(start)]
        except Exception: pass
        if not df.empty: return df
    if df is None: errors.append("period=max")

    # Ticker().history
    try:
        t = yf.Ticker(sym)
        hist = t.history(period="max", interval="1d", auto_adjust=True)
        df = _clean(hist)
        if df is not None and not df.empty and len(df) >= 30:
            try: df = df[df.index >= pd.to_datetime(start)]
            except Exception: pass
            if not df.empty: return df
        else:
            errors.append("Ticker.history")
    except Exception:
        errors.append("Ticker.history")

    # synthetisches GBPUSD
    if sym.upper() == "GBPUSD=X":
        try:
            g_eur = fetch_history_single("GBPEUR=X", start)
            e_usd = fetch_history_single("EURUSD=X", start)
            if not g_eur.empty and not e_usd.empty:
                idx = g_eur.index.intersection(e_usd.index)
                if len(idx) > 0:
                    px = (g_eur.loc[idx, "Close"] * e_usd.loc[idx, "Close"]).dropna()
                    if not px.empty:
                        out = pd.DataFrame({"Open": px, "High": px, "Low": px, "Close": px, "Volume": 0.0}, index=idx)
                        return _clean(out)
        except Exception:
            pass
        errors.append("synthetic GBPUSD")

    st.warning(f"⚠️ Download fehlgeschlagen für '{symbol}' → normalisiert: '{sym}'. Versuche: {', '.join(errors)}")
    return pd.DataFrame()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    close = pd.to_numeric(d.get("Close"), errors="coerce")
    high  = pd.to_numeric(d.get("High"), errors="coerce")
    low   = pd.to_numeric(d.get("Low"), errors="coerce")

    base_mask = close.notna() & high.notna() & low.notna()
    d = d.loc[base_mask].copy()
    if d.empty:
        return pd.DataFrame()

    vol = pd.to_numeric(d.get("Volume", pd.Series(index=d.index, dtype="float64")), errors="coerce")

    d["EMA20"], d["EMA50"] = ema(close,20), ema(close,50)
    m_line, m_sig, m_hist = macd(close)
    d["MACD"], d["MACD_signal"], d["MACD_hist"] = m_line, m_sig, m_hist
    d["RSI7"], d["RSI14"] = rsi(close,7), rsi(close,14)
    d["ATR14"] = atr(high, low, close, 14)
    d["ATRN"]  = d["ATR14"] / close
    d["VolZ20"]= volume_zscore(vol, 20) if "Volume" in d.columns else 0.0
    d["ROC3"]  = roc(close, 3)
    d["Dist_EMA20"] = (close - d["EMA20"]) / d["EMA20"]
    d["Dist_EMA50"] = (close - d["EMA50"]) / d["EMA50"]
    d["BreakoutUp"] = (close > d["High"].shift(1)).astype(int)
    d["BreakoutDn"] = (close < d["Low"].shift(1)).astype(int)
    d["Ret_1D"]     = close.pct_change(1)
    d["Ret_1D_fwd"] = close.pct_change(1).shift(-1)

    needed = ["Close","EMA20","EMA50","RSI7","RSI14","MACD_hist","ATR14","ATRN"]
    d = d.dropna(subset=needed).copy()

    d.index.name = "Date"
    out = d.reset_index()
    out["Date"] = _to_naive_dt(out["Date"])
    return out

def explode_candidates(data_sym: str, orig_name: str) -> List[str]:
    cands: List[str] = []
    if data_sym:
        for s in str(data_sym).split(","):
            s = s.strip()
            if s:
                cands.append(normalize_yahoo_symbol(s))

    norm_orig = _norm(orig_name)

    # ETF/Spot-Fallbacks
    if any(cs.upper()=="^STOXX50E" for cs in cands) or norm_orig in {"EURO STOXX 50","EUROSTOXX 50","STOXX50E"}:
        if "EXW1.DE" not in cands: cands.append("EXW1.DE")
    if any(cs.upper()=="^FCHI" for cs in cands) or norm_orig in {"FRANCE 40","CAC 40","CAC40"}:
        if "E40.PA" not in cands: cands.append("E40.PA")
    if any(cs.upper()=="^GDAXI" for cs in cands) or norm_orig in {"GERMANY 40","GER40","DAX"}:
        if "EXS1.DE" not in cands: cands.append("EXS1.DE")
    if any(cs.upper()=="SI=F" for cs in cands) or norm_orig in {"SILVER","XAG","XAGUSD"}:
        if "XAGUSD=X" not in cands: cands.append("XAGUSD=X")
    if any(cs.upper()=="PL=F" for cs in cands) or norm_orig in {"PALLADIUM","XPD","XPDUSD"}:
        if "XPDUSD=X" not in cands: cands.append("XPDUSD=X")
    if any(cs.upper()=="RB=F" for cs in cands) or norm_orig in {"RBOB","GASOLINE"}:
        if "UGA" not in cands: cands.append("UGA")
    if (norm_orig in {"GOLD","XAU","XAUUSD","GOLD/USD"} or any(cs.upper()=="GC=F" for cs in cands)):
        if "XAUUSD=X" not in cands: cands.append("XAUUSD=X")
    return list(dict.fromkeys(cands))

def fetch_first_ok(candidates: List[str], start: str, min_rows:int=20) -> Tuple[pd.DataFrame, Optional[str]]:
    """Nimmt den ersten Kandidaten mit >= min_rows Zeilen Historie (Default 20)."""
    for s in candidates:
        df = fetch_history_single(s, start)
        if df is not None and not df.empty and len(df) >= int(min_rows):
            return df, s
    return pd.DataFrame(), None

def build_panel_dynamic_mapped(symbol_map: Dict[str,str], start: str, debug: bool=False, min_rows:int=20) -> pd.DataFrame:
    frames = []
    failures = []
    for orig, data_sym in symbol_map.items():
        try:
            candidates = explode_candidates(data_sym, orig)
            if debug:
                st.write(f"{orig} Kandidaten: {candidates}")
            df, used_symbol = fetch_first_ok(candidates, start, min_rows=min_rows)
            if df is None or df.empty:
                failures.append(f"{orig}: Download leer/zu kurz (Kandidaten: {candidates}, min_rows={min_rows})")
                continue
            if len(df) < min_rows:
                failures.append(f"{orig}: Zu wenig Historie ({len(df)}) für {used_symbol} (min_rows={min_rows})")
                continue
            feat = make_features(df)
            if feat.empty:
                failures.append(f"{orig}: Feature-Erzeugung ergab leer (nach NaN-Filter)")
                continue
            feat["Symbol"] = used_symbol
            feat["OrigSymbol"] = orig
            frames.append(feat)
            if debug: st.write(f"FEAT {orig}->{used_symbol}: {feat.shape}")
        except Exception as e:
            failures.append(f"{orig}->{data_sym}: Fehler: {e}")

    if failures:
        with st.expander("⚠️ Daten-Fehler / Skipped Symbole (öffnet automatisch)", expanded=True):
            for m in failures:
                st.error(m)

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = _to_naive_dt(panel["Date"])
    panel = panel.sort_values(["OrigSymbol","Date"]).reset_index(drop=True)
    # (NEU) MACD_prev‑Fallback: erste Zeile je Symbol bekommt prev=hist
    panel["MACD_hist_prev"] = panel.groupby("OrigSymbol")["MACD_hist"].shift(1)
    first_idx = panel.groupby("OrigSymbol")["Date"].transform("idxmin") == panel.index
    panel.loc[first_idx, "MACD_hist_prev"] = panel.loc[first_idx, "MACD_hist"]
    panel = panel.dropna(subset=["MACD_hist_prev"]).reset_index(drop=True)
    return panel

def fetch_live_last(symbol: str) -> Optional[float]:
    """Holt den aktuellsten Intraday-Minutenpreis (1m/5m/fast_info Fallback)."""
    sym = normalize_yahoo_symbol(symbol)

    def _last_close(df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty or "Close" not in df.columns: return None
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if s.empty: return None
        v = float(s.iloc[-1])
        return v if np.isfinite(v) else None

    try:
        df = yf.download(sym, period="1d", interval="1m", progress=False, auto_adjust=True)
        v = _last_close(df)
        if v is not None: return v
    except Exception:
        pass
    try:
        df = yf.download(sym, period="5d", interval="5m", progress=False, auto_adjust=True)
        v = _last_close(df)
        if v is not None: return v
    except Exception:
        pass
    try:
        t = yf.Ticker(sym)
        v = getattr(getattr(t, "fast_info", None), "last_price", None)
        if v is None:
            hist = t.history(period="1d", interval="1m", auto_adjust=True)
            v = _last_close(hist)
        return float(v) if (v is not None and np.isfinite(v)) else None
    except Exception:
        return None

def inject_live_close(panel: pd.DataFrame, use_live: bool) -> pd.DataFrame:
    """Patcht HEUTE den Close durch Live-Minutenpreis. Features bleiben Vortag."""
    if not use_live or panel is None or panel.empty:
        return panel
    df = panel.copy()
    last_date = df["Date"].max()
    mask_today = (df["Date"] == last_date)
    if not mask_today.any():
        return df

    patched = 0
    for orig in df.loc[mask_today, "OrigSymbol"].unique():
        try:
            sym = df.loc[(df["OrigSymbol"] == orig) & (df["Date"] == last_date), "Symbol"].iloc[0]
        except Exception:
            continue
        live = fetch_live_last(sym)
        if live is not None and np.isfinite(live):
            sel = (df["OrigSymbol"] == orig) & (df["Date"] == last_date)
            df.loc[sel, "Close"] = float(live)
            patched += 1
    if patched > 0:
        st.caption(f"Live‑Patch: {patched} Titel aktualisiert (Close@heute).")
    else:
        st.caption("Live‑Patch: Kein aktueller Preis verfügbar (verwende Close vom Vortag).")
    return df

def maybe_inject_live(panel: pd.DataFrame, use_live: bool, interval_sec: int) -> pd.DataFrame:
    """Debounced Live-Patching."""
    if panel is None or panel.empty or not use_live:
        return panel
    now = _now_berlin()
    last = st.session_state.get("last_live_patch_ts", pd.Timestamp(0, tz="Europe/Berlin"))
    try:
        elapsed = (now - last).total_seconds()
    except Exception:
        elapsed = float('inf')
    if elapsed >= float(interval_sec):
        out = inject_live_close(panel, True)
        st.session_state["last_live_patch_ts"] = now
        return out
    else:
        return panel

# =========================
# Regeln (mit optionalem ATRN‑Band)
# =========================
def rule_long(r) -> bool:
    prev = r["MACD_hist_prev"] if pd.notna(r["MACD_hist_prev"]) else r["MACD_hist"]
    core = (r["RSI7"]>55) and (r["RSI14"]>50) and (r["Close"]>r["EMA20"]) and (r["EMA20"]>r["EMA50"]) and (r["MACD_hist"]>prev)
    if st.session_state.get("params", {}).get("use_atrn_filter", True):
        return core and (r["ATRN"]>=atrn_min) and (r["ATRN"]<=atrn_max)
    return core

def rule_short(r) -> bool:
    prev = r["MACD_hist_prev"] if pd.notna(r["MACD_hist_prev"]) else r["MACD_hist"]
    core = (r["RSI7"]<45) and (r["RSI14"]<50) and (r["Close"]<r["EMA20"]) and (r["EMA20"]<r["EMA50"]) and (r["MACD_hist"]<prev)
    if st.session_state.get("params", {}).get("use_atrn_filter", True):
        return core and (r["ATRN"]>=atrn_min) and (r["ATRN"]<=atrn_max)
    return core

# =========================
# Feature-Normierung (optional)
# =========================
def zscore_by_symbol(df: pd.DataFrame, cols: List[str], win: int = 252) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu = out.groupby("OrigSymbol")[c].transform(lambda s: s.rolling(win, min_periods=win).mean())
        sd = out.groupby("OrigSymbol")[c].transform(lambda s: s.rolling(win, min_periods=win).std(ddof=0))
        out[c + "_z"] = (out[c] - mu) / (sd.replace(0, np.nan))
    return out

def _apply_feature_norm(df: pd.DataFrame, feats: List[str], enabled: bool) -> Tuple[pd.DataFrame, List[str]]:
    if not enabled:
        return df, feats
    df_norm = zscore_by_symbol(df, feats)
    feats_z = [f + "_z" for f in feats]
    return df_norm, feats_z

# =========================
# KI: Screener & Walk-Forward
# =========================
def train_and_score(panel: pd.DataFrame, allow_short: bool=True, debug: bool=False,
                    use_horizon: bool=True, horizon_days: int=2, norm_features: bool=False):
    if panel is None or panel.empty:
        return pd.DataFrame(), "Keine Daten."
    base_features = ["RSI7","RSI14","MACD","MACD_signal","MACD_hist",
                     "EMA20","EMA50","Dist_EMA20","Dist_EMA50",
                     "ATRN","VolZ20","ROC3","BreakoutUp","BreakoutDn","Ret_1D"]
    df = panel.copy().sort_values(["OrigSymbol","Date"])
    if use_horizon and horizon_days >= 1:
        df["FwdRet_H"] = df.groupby("OrigSymbol", group_keys=False)["Close"].pct_change(horizon_days).shift(-horizon_days)
        df["Label"] = (df["FwdRet_H"] > 0).astype(int)
    else:
        df["Label"] = (df["Ret_1D_fwd"] > 0).astype(int)

    df, features = _apply_feature_norm(df, base_features, norm_features)
    df = df.dropna(subset=features + ["Label"])

    unique_dates = np.sort(df["Date"].unique())
    n_splits = min(5, max(2, len(unique_dates)//60))
    if n_splits < 2:
        return pd.DataFrame(), "Zu wenig Daten für CV."
    fold_sizes = len(unique_dates) // (n_splits + 1)

    reports, models = [], []
    for k in range(1, n_splits + 1):
        train_end_idx = k * fold_sizes
        test_end_idx  = (k + 1) * fold_sizes if (k + 1) * fold_sizes <= len(unique_dates) else len(unique_dates)
        if train_end_idx <= 0 or test_end_idx - train_end_idx < 5:
            continue
        train_dates = unique_dates[:train_end_idx]
        test_dates  = unique_dates[train_end_idx:test_end_idx]

        Xtr = df[df["Date"].isin(train_dates)][features]
        ytr = df[df["Date"].isin(train_dates)]["Label"]
        Xte = df[df["Date"].isin(test_dates)][features]
        yte = df[df["Date"].isin(test_dates)]["Label"]
        if len(Xtr) < 200 or len(Xte) < 50:
            continue

        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=6, random_state=42, n_jobs=1)
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        rep = classification_report(yte, y_pred, digits=3)
        reports.append(f"Fold {k}\n{rep}")
        models.append(clf)

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()
    latest["AI_Prob_Up"] = 0.5 if not models else models[-1].predict_proba(latest[features])[:, 1]

    dirs, rules = [], []
    for _, r in latest.iterrows():
        long_ok = rule_long(r); short_ok = rule_short(r) if allow_short else False
        if long_ok and (r["AI_Prob_Up"] >= 0.5): dirs.append("LONG");  rules.append(1)
        elif short_ok and (r["AI_Prob_Up"] < 0.5): dirs.append("SHORT"); rules.append(1)
        else: dirs.append("NO-TRADE"); rules.append(0)
    latest["Direction"] = dirs
    latest["Rule_OK"]   = rules
    latest["AI_Prob"]   = np.where(latest["Direction"]=="LONG", latest["AI_Prob_Up"],
                             np.where(latest["Direction"]=="SHORT", 1.0-latest["AI_Prob_Up"], 0.0))
    latest["FinalScore"]= 0.7*latest["AI_Prob"] + 0.3*latest["Rule_OK"]
    return latest.sort_values(["FinalScore","OrigSymbol"], ascending=[False, True]), "\n\n".join(reports) if reports else "Keine aussagekräftigen CV-Folds."

def get_model_config(fast: bool, retrain_user: int) -> Dict:
    if fast:
        return {"effective_retrain_every": max(retrain_user,10),
                "training_window_days": 520,
                "n_estimators":120, "min_samples_leaf":7, "n_jobs":1,
                "feature_subset":["RSI7","RSI14","MACD_hist","Dist_EMA20","Dist_EMA50","ATRN","ROC3","Ret_1D"]}
    else:
        return {"effective_retrain_every": retrain_user,
                "training_window_days": None,
                "n_estimators":300, "min_samples_leaf":5, "n_jobs":1,
                "feature_subset":None}

def walkforward_signals(panel: pd.DataFrame, allow_short: bool, use_ai: bool, cfg: Dict,
                        use_horizon: bool=True, horizon_days: int=2, norm_features: bool=False) -> pd.DataFrame:
    if panel is None or panel.empty: return pd.DataFrame()
    full = ["RSI7","RSI14","MACD","MACD_signal","MACD_hist","EMA20","EMA50",
            "Dist_EMA20","Dist_EMA50","ATRN","VolZ20","ROC3","BreakoutUp","BreakoutDn","Ret_1D"]
    df = panel.copy().sort_values(["OrigSymbol","Date"])
    df["Date"] = _to_naive_dt(df["Date"])
    if use_horizon and horizon_days >= 1:
        df["FwdRet_H"] = df.groupby("OrigSymbol", group_keys=False)["Close"].pct_change(horizon_days).shift(-horizon_days)
        df["Label"] = (df["FwdRet_H"] > 0).astype(int)
    else:
        df["Label"] = (df["Ret_1D_fwd"] > 0).astype(int)

    feats = cfg.get("feature_subset") or full
    df, feats = _apply_feature_norm(df, feats, norm_features)
    df = df.dropna(subset=feats + ["Label"])

    dates = sorted(pd.to_datetime(df["Date"]).dt.tz_localize(None).unique())
    out = []; last_trained = None; clf = None
    for d in dates:
        d = pd.to_datetime(d).tz_localize(None)
        train_mask = df["Date"] < d
        if cfg.get("training_window_days"):
            start_win = d - pd.Timedelta(days=cfg["training_window_days"])
            train_mask = train_mask & (df["Date"] >= start_win)
        test_mask = df["Date"] == d
        if train_mask.sum() < 100 or test_mask.sum() == 0: continue

        if use_ai:
            need_train = (last_trained is None) or ((d - pd.to_datetime(last_trained)).days >= cfg["effective_retrain_every"])
            if need_train:
                Xtr = df.loc[train_mask, feats]; ytr = df.loc[train_mask, "Label"]
                clf = RandomForestClassifier(n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"], random_state=42, n_jobs=cfg["n_jobs"])
                clf.fit(Xtr, ytr); last_trained = d

        day = df.loc[test_mask].copy()
        day["AI_Prob_Up"] = 0.5 if (not use_ai or clf is None) else clf.predict_proba(day[feats])[:, 1]

        dirs, rules = [], []
        for _, r in day.iterrows():
            long_ok = rule_long(r); short_ok = rule_short(r) if allow_short else False
            if use_ai:
                if long_ok and (r["AI_Prob_Up"] >= 0.5): dirs.append("LONG");  rules.append(1)
                elif short_ok and (r["AI_Prob_Up"] < 0.5): dirs.append("SHORT"); rules.append(1)
                else: dirs.append("NO-TRADE"); rules.append(0)
            else:
                if long_ok: dirs.append("LONG"); rules.append(1)
                elif short_ok: dirs.append("SHORT"); rules.append(1)
                else: dirs.append("NO-TRADE"); rules.append(0)

        day["Direction"] = dirs
        day["Rule_OK"]   = rules
        day["AI_Prob"]   = np.where(day["Direction"]=="LONG", day["AI_Prob_Up"],
                               np.where(day["Direction"]=="SHORT", 1.0-day["AI_Prob_Up"], 0.0))
        day["FinalScore"]= 0.7*day["AI_Prob"] + 0.3*day["Rule_OK"]
        out.append(day)

    return pd.concat(out).reset_index(drop=True) if out else pd.DataFrame()

# =========================
# Tradeplan & Backtest
# =========================
def compute_multipliers(hold_days: int, mode: str, base_stop: float = 1.0, base_tp: float = 1.8) -> Tuple[float, float, float]:
    H = max(int(hold_days), 1)
    if mode.startswith("√"):
        f = float(np.sqrt(H))
    elif mode.startswith("linear"):
        f = float(H)
    else:
        f = 1.0
    return base_stop*f, base_tp*f, 1.0*f

def compute_trade_plan(row, equity, risk_frac, pv_map: Dict[str,float],
                       use_time_exit=True, use_trailing=True,
                       hold_days: int = 2,
                       stop_mult: float = 1.0, tp_mult: float = 1.8, trail_mult: float = 1.0,
                       live_entry_price: Optional[float] = None) -> pd.Series:
    sym = row.get("OrigSymbol", row.get("Symbol"))
    base_close = float(row["Close"])
    close = float(live_entry_price) if (live_entry_price is not None and np.isfinite(live_entry_price)) else base_close

    atr_val = float(row.get("ATR14", np.nan))
    if not np.isfinite(atr_val) or atr_val <= 0:
        return pd.Series({
            "Stop": np.nan, "TakeProfit": np.nan, "Units_suggested": 0.0,
            "ATR": np.nan, "ATRN_%": float(row.get("ATRN", np.nan))*100 if pd.notna(row.get("ATRN", np.nan)) else np.nan,
            "Time_Exit_By": "Keine Berechnung (ATR NaN)", "Trailing": "Keine Berechnung (ATR NaN)",
            "EntryPrice_used": close
        })
    stop_dist = atr_val * stop_mult
    tp_dist   = atr_val * tp_mult

    if row["Direction"] == "LONG":
        stop_level = close - stop_dist
        tp_level   = close + tp_dist
    elif row["Direction"] == "SHORT":
        stop_level = close + stop_dist
        tp_level   = close - tp_dist
    else:
        return pd.Series({
            "Stop": np.nan, "TakeProfit": np.nan, "Units_suggested": 0.0,
            "ATR": atr_val, "ATRN_%": float(row.get("ATRN", np.nan))*100 if pd.notna(row.get("ATRN", np.nan)) else np.nan,
            "Time_Exit_By": "", "Trailing": "", "EntryPrice_used": close
        })

    pv = float(pv_map.get(sym, 1.0))
    denom = stop_dist * pv
    units = (equity * risk_frac) / denom if denom > 0 and np.isfinite(denom) else 0.0

    try:
        entry_date = pd.to_datetime(row["Date"]).tz_localize(None) + pd.Timedelta(days=1)
    except Exception:
        entry_date = None

    time_exit_text = "Deaktiviert"
    if use_time_exit:
        if entry_date is not None:
            exit_by = entry_date + pd.Timedelta(days=int(hold_days))
            time_exit_text = f"Spätester Exit: bis {exit_by:%Y-%m-%d} (≈ {hold_days} Tage ab Entry)"
        else:
            time_exit_text = f"Spätester Exit: ≈ {int(hold_days)} Tage ab Entry"

    trailing_text = ("Aktiv: LONG → max(Initial, HH − {m:.2f}×ATR_prev); SHORT → min(Initial, LL + {m:.2f}×ATR_prev)"
                     .format(m=trail_mult)) if use_trailing else "Deaktiviert"

    return pd.Series({
        "Stop": stop_level, "TakeProfit": tp_level, "Units_suggested": max(float(units), 0.0),
        "ATR": atr_val, "ATRN_%": float(row.get("ATRN", np.nan))*100 if pd.notna(row.get("ATRN", np.nan)) else np.nan,
        "Time_Exit_By": time_exit_text, "Trailing": trailing_text,
        "EntryPrice_used": close
    })

def first_touch_exit(high: float, low: float, stop: float, tp: float, direction: str, stop_first: bool=True):
    if direction=="LONG":
        hit_stop = low<=stop; hit_tp = high>=tp
        if hit_stop and hit_tp: return (True, stop if stop_first else tp, "Stop&TP_SameDay")
        if hit_stop: return (True, stop, "Stop")
        if hit_tp:   return (True, tp, "TP")
        return (False, None, "")
    else:
        hit_stop = high>=stop; hit_tp = low<=tp
        if hit_stop and hit_tp: return (True, stop if stop_first else tp, "Stop&TP_SameDay")
        if hit_stop: return (True, stop, "Stop")
        if hit_tp:   return (True, tp, "TP")
        return (False, None, "")

def backtest(panel: pd.DataFrame, signals: pd.DataFrame, equity_start: float, risk_frac: float,
             pv_map: Dict[str,float], min_score_local: float, use_time_exit: bool, use_trailing: bool,
             cost_per_trade: float=0.0, stop_first: bool=True,
             bt_start: Optional[pd.Timestamp]=None, bt_end: Optional[pd.Timestamp]=None,
             hold_days: int = 2, stop_mult: float = 1.0, tp_mult: float = 1.8, trail_mult: float = 1.0,
             max_total_risk_multiple: float = 2.0):
    if panel is None or panel.empty or signals is None or signals.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64")

    df = panel.copy(); sig = signals.copy()
    df["Date"] = _to_naive_dt(df["Date"])
    sig["Date"] = _to_naive_dt(sig["Date"])

    if bt_start is not None: df=df[df["Date"]>=bt_start]; sig=sig[sig["Date"]>=bt_start]
    if bt_end   is not None: df=df[df["Date"]<=bt_end];   sig=sig[sig["Date"]<=bt_end]
    sig = sig[(sig["Rule_OK"]==1) & (sig["FinalScore"]>=min_score_local) & (sig["Direction"]!="NO-TRADE")].copy()
    if sig.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64")

    df = df.sort_values(["OrigSymbol","Date"])
    symbols_list = sorted(df["Symbol"].unique().tolist())
    equities = []
    trades = []
    equity = float(equity_start)
    open_positions = {s: None for s in symbols_list}

    # OHLC + ATR, plus VORTAGS-ATR
    ohlc = df.set_index(["Symbol","Date"])[["Open","High","Low","Close","ATR14"]].copy()
    ohlc["ATR14_prev"] = ohlc.groupby(level=0)["ATR14"].shift(1)

    all_dates = sorted(df["Date"].unique())

    for d in all_dates:
        # 1) EXITS & Trailing
        for sym in symbols_list:
            pos = open_positions.get(sym)
            if pos is None: continue
            if (sym, d) not in ohlc.index: continue
            day = ohlc.loc[(sym, d)]
            if use_trailing:
                atr_for_trail = float(day.get("ATR14_prev", np.nan))
                if np.isfinite(atr_for_trail):
                    if pos["Direction"]=="LONG":
                        pos["HH"] = max(pos["HH"], float(day["High"]))
                        pos["Stop"] = max(pos["Stop"], pos["HH"] - trail_mult * atr_for_trail)
                    else:
                        pos["LL"] = min(pos["LL"], float(day["Low"]))
                        pos["Stop"] = min(pos["Stop"], pos["LL"] + trail_mult * atr_for_trail)

            hit, px, reason = first_touch_exit(float(day["High"]), float(day["Low"]),
                                               pos["Stop"], pos["TP"], pos["Direction"], stop_first)
            exit_today = False; exit_price=None; exit_reason=None
            if hit:
                exit_today=True; exit_price=float(px); exit_reason=reason
            else:
                if use_time_exit:
                    days_held=(pd.to_datetime(d)-pd.to_datetime(pos["EntryDate"])).days
                    if days_held>=int(hold_days):
                        exit_today=True; exit_price=float(day["Close"]); exit_reason=f"TimeExit_D{int(hold_days)}_Close"
            if exit_today:
                pv = float(pv_map.get(pos.get("OrigSymbol", sym), 1.0))
                pnl = ((exit_price-pos["EntryPrice"]) if pos["Direction"]=="LONG" else (pos["EntryPrice"]-exit_price)) * pos["Units"] * pv
                pnl -= float(cost_per_trade)
                R = pnl / pos["RiskAmount"] if pos["RiskAmount"]!=0 else np.nan
                equity += pnl
                trades.append({
                    "OrigSymbol":pos.get("OrigSymbol", sym), "Symbol":sym, "Direction":pos["Direction"],
                    "EntryDate":pos["EntryDate"],"EntryPrice":pos["EntryPrice"],
                    "ATR14_atEntry":pos["ATR_entry"],"StopInit":pos["StopInit"],"TPInit":pos["TPInit"],"Units":pos["Units"],
                    "ExitDate":d,"ExitPrice":exit_price,"ExitReason":exit_reason,
                    "DaysHeld":(pd.to_datetime(d)-pd.to_datetime(pos["EntryDate"])).days,
                    "PnL":pnl,"R":R
                })
                open_positions[sym]=None

        # 2) ENTRIES (Signale Vortag -> Open heute)
        prev_day = pd.to_datetime(d) - pd.Timedelta(days=1)
        day_signals = sig[sig["Date"]==prev_day].copy()
        if not day_signals.empty:
            current_risk = sum((pos or {}).get("RiskAmount", 0.0) for pos in open_positions.values() if pos)
            risk_unit = equity * risk_frac
            max_risk_budget = max_total_risk_multiple * risk_unit

            day_signals = day_signals.sort_values(["FinalScore","OrigSymbol"], ascending=[False, True])

            for _, srow in day_signals.iterrows():
                sym = srow["Symbol"]
                if open_positions.get(sym) is not None:
                    continue
                if (sym, d) not in ohlc.index:
                    continue
                if (current_risk + risk_unit) > max_risk_budget:
                    break

                o = ohlc.loc[(sym, d)]
                entry_price = float(o["Open"])

                atr_prev = float(o.get("ATR14_prev", np.nan))
                if not np.isfinite(atr_prev) or atr_prev<=0:
                    continue

                stop_dist=stop_mult * atr_prev
                tp_dist=tp_mult * atr_prev
                if srow["Direction"]=="LONG":
                    stop_level=entry_price-stop_dist; tp_level=entry_price+tp_dist
                else:
                    stop_level=entry_price+stop_dist; tp_level=entry_price-tp_dist

                pv = float(pv_map.get(srow.get("OrigSymbol", sym), 1.0))
                denom = stop_dist * pv
                units = (equity * risk_frac) / denom if denom>0 and np.isfinite(denom) else 0.0
                if units <= 0:
                    continue

                open_positions[sym] = {
                    "OrigSymbol": srow.get("OrigSymbol", sym),
                    "Direction":srow["Direction"],"EntryDate":d,"EntryPrice":entry_price,"ATR_entry":atr_prev,
                    "Stop":stop_level,"TP":tp_level,"StopInit":stop_level,"TPInit":tp_level,
                    "Units":float(units),"RiskAmount":risk_unit,
                    "HH":float(o["High"]), "LL":float(o["Low"])
                }
                current_risk += risk_unit

        equities.append((d, equity))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        eq_series = pd.Series([equity_start], index=[df["Date"].min()], dtype="float64")
        return trades_df, pd.DataFrame(), eq_series

    trades_df["Win"] = trades_df["PnL"]>0
    summary = trades_df.groupby("OrigSymbol", as_index=False).agg(
        Trades=("OrigSymbol","count"),
        WinRate=("Win","mean"),
        AvgR=("R","mean"),
        TotalPnL=("PnL","sum")
    )

    eq_df = pd.DataFrame(equities, columns=["Date","Equity"]).drop_duplicates("Date")
    eq_df["Date"]=pd.to_datetime(eq_df["Date"]).dt.tz_localize(None)
    eq_series = eq_df.set_index("Date")["Equity"].sort_index()
    return trades_df, summary, eq_series

# =========================
# Daten laden (nur per Button)
# =========================
if do_load:
    if not symbols_map:
        st.error("Bitte Titel im Multi‑Select auswählen (oder manuelle Ticker hinzufügen).")
        st.stop()

    with st.spinner("Lade Daten & berechne Features …"):
        @st.cache_data(show_spinner=False, ttl=3600)
        def _build_panel_cached(symbols_map, start, debug, min_rows):
            return build_panel_dynamic_mapped(symbols_map, start, debug, min_rows=min_rows)

        panel = _build_panel_cached(symbols_map, START_DATE, params["debug_mode"], params.get("min_rows_first_ok", 20))
        panel = maybe_inject_live(panel, params["use_live_now"], int(params["live_interval_sec"]))
        if panel is None or panel.empty:
            st.error("Keine Daten/Features. Prüfe Auswahl/Lookback/Verbindung.")
            st.stop()
        st.session_state["panel"] = panel
        st.success("Panel geladen. Du kannst jetzt im Screener/Backtest fortfahren.")

# Panel aus Session holen
panel = st.session_state.get("panel")
if panel is None or panel.empty:
    st.warning("Noch keine Daten geladen. Bitte „🔁 Daten laden/aktualisieren“ klicken.")
    st.stop()

# =========================
# Optional Auto‑Refresh (nur Info/Timer)
# =========================
if params["use_live_now"] and params["auto_refresh_on"]:
    if not st.session_state.get("refresh_paused", False):
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore
            st_autorefresh(interval=int(params["live_interval_sec"]) * 1000, key="auto_refresh_live")
            st.caption(f"Auto‑Refresh aktiv (alle {params['live_interval_sec']}s).")
        except Exception:
            st.warning("Installiere `streamlit_autorefresh` für sanfte Re‑Runs.")
    else:
        st.caption("⏸️ Auto‑Refresh pausiert (während Screener/Backtest).")

# =========================
# Topbar: Qualität & Session‑Hint
# =========================
quality = assess_data_quality(panel, params["use_live_now"])

def _next_actions(q: dict) -> str:
    sess = detect_session(_now_berlin())
    if q["status"] == "green":
        if sess == "evening": return "Daten final. Jetzt handeln/planen (22:15–23:30) empfohlen."
        if sess == "morning": return "Vortagsdaten final – morgens checken & planen."
        return "Daten ok. Für finale Signale ab 22:15 erneut prüfen."
    if q["status"] == "orange": return "Teilweise final (Intraday/vor 22:15). Später erneut ausführen."
    return "Unzureichend. Prüfe Auswahl/Historie/Netzwerk oder warte auf Abschluss."

dot = {"green":"🟢","orange":"🟠","red":"🔴"}.get(quality["status"], "⚪")
reasons_list = "".join(f"<li>{html.escape(str(r))}</li>" for r in quality.get("reasons", []))
reasons_html = f"<ul class='reasons'>{reasons_list}</ul>" if reasons_list else "<div>Keine Hinweise.</div>"
_now = _now_berlin()
session_name = {"morning":"Morgen‑Session (EU/FX/Metalle)","evening":"Abend‑Session (US/Commodities/FX)","regular":"Neutral (Top‑Scores)"}[detect_session(_now)]

topbar_html = f"""
<div id="topbar">
  <div class="row">
    <div class="cell">
      <div class="chip">
        <div class="dot">{dot}</div>
        <div>
          <div><strong>Daten‑Qualität: {quality['status'].upper()}</strong></div>
          {reasons_html}
        </div>
      </div>
    </div>
    <div class="cell">
      <div class="chip">
        <div class="dot">🧭</div>
        <div>
          <div><strong>Session:</strong> {session_name} <span style="opacity:.7;">({_now.tz_convert('Europe/Berlin'):%H:%M})</span></div>
          <div style="margin-top:.35rem;"><strong>Nächste Schritte:</strong> {html.escape(_next_actions(quality))}</div>
        </div>
      </div>
    </div>
  </div>
</div>
"""
st.markdown(topbar_html, unsafe_allow_html=True)

# Sidebar: Daten-Qualität kompakt
st.sidebar.markdown("### ✅ Daten‑Qualität")
render_quality_box(quality)

# =========================
# Eff. Variablen (Code‑1‑Modus)
# =========================
eff_hold_days = 1 if params["code1_mode"] else params["hold_days"]
eff_forecast_horizon = False if params["code1_mode"] else params["forecast_horizon_days"]
eff_min_score = 0.55 if params["code1_mode"] else params["min_score"]
eff_atrn_min_pct = 0.3 if params["code1_mode"] else params["atrn_min_pct"]
eff_atrn_max_pct = 4.0 if params["code1_mode"] else params["atrn_max_pct"]
eff_enable_short = True if params["code1_mode"] else params["enable_short"]
eff_feature_norm = False if params["code1_mode"] else params["feature_norm"]
atrn_min, atrn_max = eff_atrn_min_pct/100.0, eff_atrn_max_pct/100.0

def compute_multipliers(hold_days: int, mode: str, base_stop: float = 1.0, base_tp: float = 1.8) -> Tuple[float, float, float]:
    H = max(int(hold_days), 1)
    if mode.startswith("√"):
        f = float(np.sqrt(H))
    elif mode.startswith("linear"):
        f = float(H)
    else:
        f = 1.0
    return base_stop*f, base_tp*f, 1.0*f

stop_mult, tp_mult, trail_mult = compute_multipliers(eff_hold_days, params["scale_mode"])

# =========================
# Diagnose „Gewählt vs. Panel/Roh/Picks“
# =========================
with st.expander("🧩 Diagnose: Gewählt vs. Panel/Rohsignale/Picks", expanded=True):
    selected = pd.Index(params.get("selected_plus500", []), dtype="object")
    panel_syms = pd.Index(panel["OrigSymbol"].unique()) if (panel is not None and not panel.empty) else pd.Index([])
    latest_df = st.session_state.get("latest_signals", pd.DataFrame())
    raw_syms = pd.Index(latest_df["OrigSymbol"].unique()) if (latest_df is not None and not latest_df.empty) else pd.Index([])
    picks_plan = st.session_state.get("latest_picks_plan", pd.DataFrame())
    pick_syms = pd.Index(picks_plan["OrigSymbol"].unique()) if (picks_plan is not None and not picks_plan.empty) else pd.Index([])

    diag = pd.DataFrame({"Plus500Name": selected})
    diag["in_Panel"]      = diag["Plus500Name"].isin(panel_syms)
    diag["in_Rohsignale"] = diag["Plus500Name"].isin(raw_syms)
    diag["in_Picks"]      = diag["Plus500Name"].isin(pick_syms)
    diag["Status"] = np.select(
        [
            ~diag["in_Panel"],
            diag["in_Panel"] & ~diag["in_Rohsignale"],
            diag["in_Rohsignale"] & ~diag["in_Picks"],
            diag["in_Picks"],
        ],
        [
            "❌ nicht im Panel (Download/Features)",
            "⚠️ im Panel, aber kein Rohsignal (Datum/Live)",
            "ℹ️ Rohsignal ohne Pick (Rules/Score/ATRN/Short)",
            "✅ in Picks",
        ],
        default="–"
    )
    st.dataframe(diag, use_container_width=True, column_config=build_col_config(diag))

# =========================
# UI – Tabs
# =========================
tab_screener, tab_vol, tab_backtest = st.tabs(["🔎 Screener", "📊 Volatilität", "🧪 Backtest"])

# =========================
# 🔎 Screener
# =========================
with tab_screener:
    st.markdown('<a id="screener"></a>', unsafe_allow_html=True)
    st.markdown("## 📈 Multi‑Symbol Screener (Haltedauer & Skalierung)" + hover_info_icon("Rohsignale + KI‑Score; Picks nach Score & Rules."), unsafe_allow_html=True)

    # Quick‑Diagnose je Symbol
    with st.expander("🔎 Quick‑Diagnose je Symbol (letzte 3 Zeilen)"):
        name = st.text_input("Plus500-Name (z. B. GERMANY 40, GOLD, EUR/USD)")
        if name:
            dbg = panel[panel["OrigSymbol"].str.upper() == name.strip().upper()]
            if dbg.empty:
                st.info("Kein Panel-Eintrag – prüfe Mapping/Download.")
            else:
                show = dbg.sort_values("Date").tail(3)
                st.dataframe(show, use_container_width=True, column_config=build_col_config(show))

    run_screener = st.button("🚦 Screener jetzt ausführen", key="btn_screener")
    if run_screener:
        st.session_state["refresh_paused"] = True

        panel_for_screener = maybe_inject_live(panel, params["use_live_now"], int(params["live_interval_sec"]))
        latest, reports = train_and_score(
            panel_for_screener, allow_short=eff_enable_short,
            use_horizon=eff_forecast_horizon, horizon_days=eff_hold_days, norm_features=eff_feature_norm
        )
        st.session_state["latest_signals"] = latest
        st.session_state["screener_reports"] = reports

        st.markdown("**Heutige Roh‑Signale (vor Filter)**")
        if latest.empty:
            last_panel_date = pd.to_datetime(panel["Date"]).max().date()
            st.info(f"Keine Rohdaten sichtbar. Panel letzter Tag = {last_panel_date}. Prüfe Live-Schalter oder lade später (ab 22:15) erneut.")
        else:
            cols = ["OrigSymbol","Symbol","Date","Close","EMA20","EMA50","RSI7","RSI14","MACD_hist","MACD_hist_prev","ATR14","ATRN","VolZ20","AI_Prob_Up","Direction","Rule_OK","AI_Prob","FinalScore"]
            show_cols = [c for c in cols if c in latest.columns]
            st.dataframe(latest[show_cols], use_container_width=True, column_config=build_col_config(latest[show_cols]))
            st.caption("AI_Prob_Up = KI‑Wahrscheinlichkeit für Anstieg (für SHORT wird 1 − AI_Prob_Up genutzt).")

        st.markdown("**✅ Handelssignale & Tradeplan**")
        picks = latest[(latest["Rule_OK"]==1) & (latest["FinalScore"]>=eff_min_score) & (latest["Direction"]!="NO-TRADE")].copy()
        if picks.empty:
            st.info(f"Keine handelbaren Signale (Score ≥ {eff_min_score:.2f}" + (f", ATRN {atrn_min*100:.1f}–{atrn_max*100:.1f}% aktiv im Rule‑Check)." if params.get("use_atrn_filter", True) else ", ATRN-Band derzeit NICHT im Rule‑Check)."))
            st.session_state["latest_picks_plan"] = pd.DataFrame()
            st.session_state["latest_rec_df"] = pd.DataFrame()
        else:
            pv_map = st.session_state.get("point_values", {})
            def _live_entry_for_row(sym: str) -> Optional[float]:
                return fetch_live_last(sym) if (params["use_live_now"] and params["use_live_entry_in_plan"]) else None

            plan = picks.apply(lambda r: compute_trade_plan(
                r, params["account_equity"], params["risk_per_trade"], pv_map,
                use_time_exit=params["use_time_exit"], use_trailing=params["use_trailing"],
                hold_days=eff_hold_days, stop_mult=stop_mult, tp_mult=tp_mult, trail_mult=trail_mult,
                live_entry_price=_live_entry_for_row(r["Symbol"])
            ), axis=1)
            picks = pd.concat([picks, plan], axis=1)
            picks["AI_Prob"]=picks["AI_Prob"].round(3); picks["FinalScore"]=picks["FinalScore"].round(3); picks["ATRN_%"]=(picks["ATRN"]*100).round(3)

            cols2 = ["OrigSymbol","Symbol","Date","Direction","FinalScore","AI_Prob","Close","EMA20","EMA50","RSI7","RSI14","MACD_hist","ATR14","ATRN_%","VolZ20","EntryPrice_used","Stop","TakeProfit","Units_suggested","Time_Exit_By","Trailing"]
            show_cols2 = [c for c in cols2 if c in picks.columns]
            df_plan = picks[show_cols2]
            st.dataframe(df_plan, use_container_width=True, column_config=build_col_config(df_plan))
            st.download_button("Watchlist CSV", data=df_plan.to_csv(index=False).encode("utf-8"), file_name="watchlist_plus500_finder.csv", mime="text/csv")
            st.session_state["latest_picks_plan"] = df_plan

            # Empfehlungen
            st.markdown("### 🧭 Empfehlungen je Tageszeit")
            _now_local = _now_berlin()
            session_name = {"morning":"Morgen‑Session (EU/FX/Metalle)","evening":"Abend‑Session (US/Commodities/FX)","regular":"Neutral (Top‑Scores)"}[detect_session(_now_local)]
            rec_df = recommend_by_session(picks, catalog_df)
            st.caption(f"Aktuell erkannt: **{session_name}** – lokale Zeit ({_now_local.tz_convert('Europe/Berlin'):%H:%M}).")
            if rec_df is None or rec_df.empty:
                st.info("Keine passenden Empfehlungen für diese Session.")
            else:
                st.dataframe(rec_df, use_container_width=True, column_config=build_col_config(rec_df))
            st.session_state["latest_rec_df"] = rec_df

        with st.expander("🔧 Cross‑Validation Reports (datum‑basiert, purged)"):
            st.text(st.session_state.get("screener_reports",""))

        with st.expander("🔎 GOLD‑Diagnose (letzte 3 Zeilen)"):
            latest_dbg = st.session_state.get("latest_signals", pd.DataFrame())
            if not latest_dbg.empty:
                gold_mask = latest_dbg["OrigSymbol"].str.upper().isin(["GOLD","XAU","XAUUSD","GOLD/USD"])
                gold_dbg = latest_dbg.loc[gold_mask, ["OrigSymbol","Symbol","Date","Close","EMA20","EMA50","RSI7","RSI14","MACD_hist","MACD_hist_prev","ATR14","ATRN","VolZ20","AI_Prob_Up","Direction","Rule_OK","AI_Prob","FinalScore"]].sort_values("Date").tail(3)
                st.dataframe(gold_dbg, use_container_width=True, column_config=build_col_config(gold_dbg))

        with st.sidebar.expander("📤 Abend‑Auto‑Export (Watchlist)", expanded=False):
            auto_export_toggle = st.checkbox("Abend‑Auto‑Export aktivieren (22:15–23:30)", value=False)
            fname_prefix = st.text_input("Datei‑Präfix", value="watchlist_plus500")
            start_win = st.text_input("Fenster‑Start (HH:MM)", value="22:15")
            end_win   = st.text_input("Fenster‑Ende (HH:MM)",  value="23:30")
            if auto_export_toggle:
                picks_for_export = st.session_state.get("latest_picks_plan", pd.DataFrame())
                if picks_for_export is None or picks_for_export.empty:
                    st.warning("Kein Inhalt zum Exportieren (picks leer).")
                else:
                    exported = auto_export_watchlist(picks_for_export, filename_prefix=fname_prefix, window_start=start_win, window_end=end_win)
                    if exported:
                        st.success(f"Gespeichert: {exported}")
                        try:
                            st.download_button("Exportierte Datei herunterladen", data=open(exported, "rb").read(), file_name=exported, mime="text/csv")
                        except Exception:
                            pass
                    else:
                        st.info("Warte auf Export‑Zeitfenster …")

    else:
        latest = st.session_state.get("latest_signals", pd.DataFrame())
        reports = st.session_state.get("screener_reports", "")
        if latest is not None and not latest.empty:
            st.info("Zeige zuletzt berechnete Screener-Ergebnisse (persistiert).")
            cols = ["OrigSymbol","Symbol","Date","Close","EMA20","EMA50","RSI7","RSI14","MACD_hist","MACD_hist_prev","ATR14","ATRN","VolZ20","AI_Prob_Up","Direction","Rule_OK","AI_Prob","FinalScore"]
            show_cols = [c for c in cols if c in latest.columns]
            st.dataframe(latest[show_cols], use_container_width=True, column_config=build_col_config(latest[show_cols]))

            st.markdown("**✅ Handelssignale & Tradeplan**")
            df_plan = st.session_state.get("latest_picks_plan", pd.DataFrame())
            if df_plan is None or df_plan.empty:
                st.info("Keine persistierten handelbaren Signale.")
            else:
                st.dataframe(df_plan, use_container_width=True, column_config=build_col_config(df_plan))
                st.download_button("Watchlist CSV", data=df_plan.to_csv(index=False).encode("utf-8"), file_name="watchlist_plus500_finder.csv", mime="text/csv")
            with st.expander("🔧 Cross‑Validation Reports (datum‑basiert, purged)"):
                st.text(reports)
        else:
            st.info("Klicke auf **„🚦 Screener jetzt ausführen“**, nachdem die Daten geladen wurden.")

# =========================
# 📊 Volatilität
# =========================
with tab_vol:
    st.markdown('<a id="vol"></a>', unsafe_allow_html=True)
    st.markdown("## 📊 Volatilitäts‑Monitor" + hover_info_icon("Aktuelle ATRN‑Werte & Verlauf"), unsafe_allow_html=True)

    latest_date = panel["Date"].max()
    latest_rows = panel[panel["Date"]==latest_date].copy()
    if latest_rows.empty:
        st.info("Keine aktuellen Volatilitäts‑Daten.")
    else:
        vol_table = latest_rows[["OrigSymbol","Symbol","Close","ATR14","ATRN"]].copy()
        vol_table["ATRN_%"] = (vol_table["ATRN"]*100).round(3)

        def zone(v):
            if pd.isna(v): return "n/a"
            if v<0.003:    return "zu niedrig (<0.3%)"
            if v>0.04:     return "zu hoch (>4.0%)"
            return "ok (0.3–4.0%)"

        vol_table["Vol-Zone"] = vol_table["ATRN"].apply(zone)
        df_vol = vol_table[["OrigSymbol","Symbol","Close","ATR14","ATRN_%","Vol-Zone"]].sort_values("ATRN_%", ascending=False)
        st.dataframe(df_vol, use_container_width=True, column_config=build_col_config(df_vol))
        st.caption("ATR14 = absolute Tages‑Range; ATRN_% = ATR/Preis×100.")

        st.markdown("**ATRN% – Ranking**")
        st.bar_chart(vol_table.set_index("OrigSymbol")["ATRN_%"])

        st.markdown("**ATRN‑Verlauf (letzte ~90 Tage)**")
        recent = panel[panel["Date"] >= (latest_date - pd.Timedelta(days=90))]
        if not recent.empty:
            pivot = recent.pivot_table(index="Date", columns="OrigSymbol", values="ATRN")
            st.line_chart(pivot)

# =========================
# 🧪 Backtest
# =========================
with tab_backtest:
    st.markdown('<a id="backtest"></a>', unsafe_allow_html=True)
    st.markdown("## 🧪 Walk‑Forward Backtest" + hover_info_icon("Training bis Vortag, Signale je Tag, Simulation mit Stop/TP/Trailing/Time‑Exit."), unsafe_allow_html=True)

    cfg = get_model_config(params["fast_mode"], params["retrain_every_n"])
    st.caption(("Fast Mode: ~2 Jahre Fenster, selteneres Retraining, reduziertes Feature‑Set.")
               if params["fast_mode"] else ("Standard Mode: volle Historie, häufigeres Retraining, volles Feature‑Set."))

    bt_start = params["bt_start"]; bt_end = params["bt_end"]

    run_bt = st.button("🚀 Backtest starten", type="primary", key="btn_backtest")
    if run_bt:
        st.session_state["refresh_paused"] = True

        with st.spinner("Trainiere & simuliere…"):
            signals_all = walkforward_signals(
                panel, allow_short=eff_enable_short, use_ai=params["use_ai_in_bt"], cfg=cfg,
                use_horizon=eff_forecast_horizon, horizon_days=eff_hold_days, norm_features=eff_feature_norm
            )
            trades_df, summary_df, eq = backtest(
                panel=panel, signals=signals_all,
                equity_start=params["account_equity"], risk_frac=params["risk_per_trade"],
                pv_map=st.session_state.get("point_values", {}), min_score_local=eff_min_score,
                use_time_exit=params["use_time_exit"], use_trailing=params["use_trailing"],
                cost_per_trade=params["cost_per_trade"], stop_first=params["stop_first"],
                bt_start=bt_start, bt_end=bt_end,
                hold_days=eff_hold_days, stop_mult=stop_mult, tp_mult=tp_mult, trail_mult=trail_mult,
                max_total_risk_multiple=params["max_total_risk_multiple"]
            )

            st.session_state["last_signals_all"] = signals_all
            st.session_state["last_backtest_trades"] = trades_df
            st.session_state["last_backtest_summary"] = summary_df
            st.session_state["last_backtest_eq"] = eq

            if trades_df is None or trades_df.empty or eq is None or eq.empty:
                st.info("Keine Trades im gewählten Zeitraum / unter den Filtern.")
            else:
                eq = eq.sort_index()
                total_return = (eq.iloc[-1]/eq.iloc[0]-1.0) if len(eq)>=2 else 0.0
                days_span = (eq.index.max()-eq.index.min()).days if len(eq)>=2 else 1
                years = max(days_span/365.25, 1e-6)
                cagr = (1+total_return)**(1/years)-1 if total_return>-0.999 else -1.0

                daily_ret = eq.pct_change().dropna()
                sharpe = (daily_ret.mean()/(daily_ret.std()+1e-12))*np.sqrt(252) if len(daily_ret)>=5 else np.nan

                roll_max = eq.cummax(); dd = (eq/roll_max)-1.0; max_dd = float(dd.min()) if len(dd) else 0.0

                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Total Return", f"{total_return*100:,.2f}%")
                c2.metric("CAGR", f"{cagr*100:,.2f}%")
                c3.metric("Win‑Rate", f"{(trades_df['Win'].mean()*100):.1f}%")
                c4.metric("Sharpe (täglich)", f"{sharpe:,.2f}" if np.isfinite(sharpe) else "n/a")
                c5.metric("Max Drawdown", f"{max_dd*100:.1f}%")

                st.markdown("**Equity‑Kurve**")
                st.line_chart(eq)

                if summary_df is not None and not summary_df.empty:
                    st.markdown("**Leistung je Plus500‑Titel**")
                    st.dataframe(summary_df, use_container_width=True, column_config=build_col_config(summary_df))

                st.markdown("**Trades (Detail)**")
                show_cols = ["OrigSymbol","Symbol","Direction","EntryDate","EntryPrice","ATR14_atEntry","StopInit","TPInit","ExitDate","ExitPrice","ExitReason","DaysHeld","PnL","R"]
                show_cols = [c for c in show_cols if c in trades_df.columns]
                st.dataframe(trades_df[show_cols].sort_values(["EntryDate","OrigSymbol"]), use_container_width=True, column_config=build_col_config(trades_df[show_cols]))

                st.download_button("Trades als CSV", data=trades_df.to_csv(index=False).encode("utf-8"), file_name="backtest_trades_plus500.csv", mime="text/csv")

                st.markdown("### 🔬 Walk‑Forward Klassifikationsmetriken")
                lbl_df = panel.copy()
                lbl_df["Date"] = _to_naive_dt(lbl_df["Date"])
                if eff_forecast_horizon:
                    H = max(1, eff_hold_days)
                    lbl_df["FwdRet_H"] = lbl_df.groupby("OrigSymbol", group_keys=False)["Close"].pct_change(H).shift(-H)
                    lbl_df["Label"] = (lbl_df["FwdRet_H"] > 0).astype(int)
                else:
                    lbl_df["Label"] = (lbl_df["Ret_1D_fwd"] > 0).astype(int)
                lbl_df = lbl_df[["OrigSymbol","Symbol","Date","Label"]]

                sig_cols = ["OrigSymbol","Symbol","Date","AI_Prob_Up","Direction","Rule_OK"]
                sig_base = (signals_all[sig_cols].copy()
                            if set(sig_cols).issubset(signals_all.columns) else pd.DataFrame(columns=sig_cols))
                sig_base["Date"] = _to_naive_dt(sig_base["Date"])

                sig_eval = sig_base.merge(lbl_df, on=["OrigSymbol","Symbol","Date"], how="inner")

                if not sig_eval.empty and sig_eval["Rule_OK"].sum()>0:
                    mask_period = (sig_eval["Date"]>=bt_start) & (sig_eval["Date"]<=bt_end)
                    sig_eval = sig_eval[mask_period].copy()
                    sig_eval = sig_eval[sig_eval["Direction"]!="NO-TRADE"].copy()

                    if not sig_eval.empty:
                        y_true = sig_eval["Label"].values.astype(int)
                        prob_up = sig_eval["AI_Prob_Up"].values.astype(float)
                        p_trade = np.where(sig_eval["Direction"]=="LONG", prob_up,
                                           np.where(sig_eval["Direction"]=="SHORT", 1.0-prob_up, 0.5))
                        y_pred = (p_trade >= 0.5).astype(int)
                        acc = (y_pred == y_true).mean()
                        try:   auc = roc_auc_score(y_true, p_trade)
                        except Exception: auc = np.nan
                        try:   brier = brier_score_loss(y_true, p_trade)
                        except Exception: brier = np.nan

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Accuracy (Signale)", f"{acc*100:,.2f}%")
                        c2.metric("AUC (Signale)", f"{auc:,.3f}" if np.isfinite(auc) else "n/a")
                        c3.metric("Brier (Signale)", f"{brier:,.4f}" if np.isfinite(brier) else "n/a")

                        sig_eval["YYYY-MM"] = pd.to_datetime(sig_eval["Date"]).dt.to_period("M").astype(str)
                        monthly_acc = sig_eval.assign(correct=(y_pred==y_true)).groupby("YYYY-MM")["correct"].mean()
                        st.markdown("**Monatliche Accuracy (Signale)**")
                        st.bar_chart(monthly_acc)
                    else:
                        st.info("Keine auswertbaren Signale für Metriken im gewählten Zeitraum.")
                else:
                    st.info("Nicht genug Signale/Labels für Klassifikationsmetriken.")
    else:
        trades_df = st.session_state.get("last_backtest_trades", pd.DataFrame())
        summary_df = st.session_state.get("last_backtest_summary", pd.DataFrame())
        eq = st.session_state.get("last_backtest_eq", pd.Series(dtype="float64"))

        if trades_df is not None and not trades_df.empty and eq is not None and not eq.empty:
            st.info("Zeige die zuletzt berechneten Backtest-Ergebnisse (persistiert).")
            eq = eq.sort_index()
            total_return = (eq.iloc[-1]/eq.iloc[0]-1.0) if len(eq)>=2 else 0.0
            days_span = (eq.index.max()-eq.index.min()).days if len(eq)>=2 else 1
            years = max(days_span/365.25, 1e-6)
            cagr = (1+total_return)**(1/years)-1 if total_return>-0.999 else -1.0

            daily_ret = eq.pct_change().dropna()
            sharpe = (daily_ret.mean()/(daily_ret.std()+1e-12))*np.sqrt(252) if len(daily_ret)>=5 else np.nan

            roll_max = eq.cummax(); dd = (eq/roll_max)-1.0; max_dd = float(dd.min()) if len(dd) else 0.0

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total Return", f"{total_return*100:,.2f}%")
            c2.metric("CAGR", f"{cagr*100:,.2f}%")
            c3.metric("Win‑Rate", f"{(trades_df['Win'].mean()*100):.1f}%")
            c4.metric("Sharpe (täglich)", f"{sharpe:,.2f}" if np.isfinite(sharpe) else "n/a")
            c5.metric("Max Drawdown", f"{max_dd*100:.1f}%")

            st.markdown("**Equity‑Kurve**")
            st.line_chart(eq)

            if summary_df is not None and not summary_df.empty:
                st.markdown("**Leistung je Plus500‑Titel**")
                st.dataframe(summary_df, use_container_width=True, column_config=build_col_config(summary_df))

            st.markdown("**Trades (Detail)**")
            show_cols = ["OrigSymbol","Symbol","Direction","EntryDate","EntryPrice","ATR14_atEntry","StopInit","TPInit","ExitDate","ExitPrice","ExitReason","DaysHeld","PnL","R"]
            show_cols = [c for c in show_cols if c in trades_df.columns]
            df_trades = trades_df[show_cols].sort_values(["EntryDate","OrigSymbol"])
            st.dataframe(df_trades, use_container_width=True, column_config=build_col_config(df_trades))
            st.download_button("Trades als CSV", data=trades_df.to_csv(index=False).encode("utf-8"),
                               file_name="backtest_trades_plus500.csv", mime="text/csv")
        else:
            st.info("Klicke auf **„🚀 Backtest starten“**, nachdem die Daten geladen wurden.")

st.caption("Hinweis: Diese App ist für Bildungs- und Researchzwecke. Keine Anlageberatung.")