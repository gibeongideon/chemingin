#!/usr/bin/env python3
"""Export the XAU champion backtest to Excel: daily rows (inputs) + formula-driven summaries.

Engine: `v5_xau_champion_lifts.engine_v` formula, reproduced here so the POSITION is available
(XAU H4, `champion_signal`, 10% vol target, EWM vol hl 42 bars, buffer 0.1, next-bar fills,
cost = half the spread floored at $0.448 + $0.10 slippage). Daily rows from 2016-01-04 (2015 is
signal warm-up).

    python scripts/v5_champion_excel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_xau_champion_lifts as L  # noqa: E402
from scripts.v5_xau_turn_prob import load_h4  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "champion_anatomy" / "champion_backtest.xlsx"
START = "2016-01-01"
CAPITAL = 10_000          # starting equity ($), editable in Summary!B2
F = "Arial"


def backtest() -> pd.DataFrame:
    df = load_h4(L.COST_USD)
    close = df.close
    ret = close.pct_change()
    fc = champion_signal(close)
    vol = ret.ewm(halflife=L.VOL_HL, min_periods=20).std() * np.sqrt(L.ANN_H4)
    pos = (fc.clip(-2, 2) * (L.TARGET_VOL / vol)).clip(-L.MAX_LEV, L.MAX_LEV)
    band = (L.BUFFER * (L.TARGET_VOL / vol).clip(0, L.MAX_LEV)).values
    p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=pos.index).shift(1).fillna(0.0)
    cost = (df.spread_px / 2 + L.SLIP_USD) / close
    net = (pos * ret - pos.diff().abs().fillna(0.0) * cost).fillna(0.0)
    g = net.groupby(net.index.normalize())
    d = pd.DataFrame({"champ": g.apply(lambda x: (1 + x).prod() - 1),
                      "gold_close": close.groupby(close.index.normalize()).last(),
                      "fc_prev": fc.groupby(fc.index.normalize()).last().shift(1),
                      "lev_prev": pos.groupby(pos.index.normalize()).last().shift(1),
                      "turnover": pos.diff().abs().groupby(pos.index.normalize()).sum()})
    d["gold"] = d.gold_close.pct_change()
    return d.loc[START:].dropna()


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        x = ws.cell(row=row, column=c)
        x.font = Font(name=F, bold=True, color="FFFFFF")
        x.fill = PatternFill("solid", fgColor="1F3864")
        x.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def main() -> None:
    d = backtest()
    n = len(d)
    last = n + 1                                      # last data row on the Daily sheet
    wb = Workbook()

    # ---------------------------------------------------------------- Daily (inputs + running calcs)
    ws = wb.active
    ws.title = "Daily"
    hdr = ["Date", "Year", "Month", "Gold close ($)", "Forecast (prev day, 0-2)",
           "Leverage (prev day, x)", "Champion daily return", "Gold daily return",
           "Champion equity ($)", "Equity peak", "Drawdown", "Gold B&H equity ($)",
           "Year-to-date peak", "Drawdown within year", "ln(1+champ)", "ln(1+gold)", "Forecast state",
           "Turnover (x notional)"]
    ws.append(hdr)
    style_header(ws, 1, len(hdr))
    for k, (t, r) in enumerate(d.iterrows()):
        i = k + 2
        ws.append([t.to_pydatetime(), f"=YEAR(A{i})", f"=MONTH(A{i})", float(r.gold_close),
                   float(r.fc_prev), float(r.lev_prev), float(r.champ), float(r.gold),
                   f"=Summary!$B$2*(1+G{i})" if i == 2 else f"=I{i-1}*(1+G{i})",
                   f"=I{i}" if i == 2 else f"=MAX(J{i-1},I{i})",
                   f"=I{i}/J{i}-1",
                   f"=Summary!$B$2*(1+H{i})" if i == 2 else f"=L{i-1}*(1+H{i})",
                   f"=I{i}" if i == 2 else f"=IF(B{i}=B{i-1},MAX(M{i-1},I{i}),I{i})",
                   f"=I{i}/M{i}-1",
                   f"=LN(1+G{i})", f"=LN(1+H{i})",
                   f'=IF(E{i}<0.2,"1 resting",IF(E{i}<0.8,"2 building",IF(E{i}<1.5,"3 trend","4 strong trend")))',
                   float(r.turnover)])
    blue = Font(name=F, color="0000FF")
    black = Font(name=F)
    for row in ws.iter_rows(min_row=2, max_row=last):
        for c in row:
            c.font = blue if c.column in (1, 4, 5, 6, 7, 8, 18) else black
        row[0].number_format = "yyyy-mm-dd"
        row[3].number_format = "#,##0.00"
        row[4].number_format = "0.00"
        row[5].number_format = "0.00x"
        for j in (6, 7, 10, 13):
            row[j].number_format = "0.00%"
        for j in (8, 9, 11, 12):
            row[j].number_format = "$#,##0"
        row[17].number_format = "0.000"
    ws.freeze_panes = "B2"
    for j, w in enumerate([11, 6, 6, 11, 11, 11, 11, 11, 12, 10, 10, 12, 11, 11, 10, 10, 14, 11], 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws["A1"].comment = Comment("Columns in BLUE are backtest outputs from scripts/v5_champion_excel.py "
                               "(engine = v5_xau_champion_lifts.engine_v). Black columns are formulas.", "model")

    D = "Daily!"
    rng = lambda col: f"{D}${col}$2:${col}${last}"  # noqa: E731

    # ---------------------------------------------------------------- Summary
    s = wb.create_sheet("Summary", 0)
    s["A1"] = "XAU champion — long-only trend follower, backtest summary"
    s["A1"].font = Font(name=F, bold=True, size=14)
    s.append(["Starting capital ($)", CAPITAL, "", "INPUT — change this and every $ figure recalculates"])
    s.append(["Metric", "Champion", "Gold buy & hold", "Note"])
    style_header(s, 3, 4)
    rows = [
        ("Start date", f"=MIN({rng('A')})", f"=MIN({rng('A')})", "after a 2015 signal warm-up"),
        ("End date", f"=MAX({rng('A')})", f"=MAX({rng('A')})", ""),
        ("Years", "=(B5-B4)/365.25", "=(C5-C4)/365.25", ""),
        ("Final equity ($)", f"={D}I{last}", f"={D}L{last}", "compounded from the starting capital"),
        ("CAGR", "=(B7/$B$2)^(1/B6)-1", "=(C7/$B$2)^(1/C6)-1", ""),
        ("Total profit ($)", "=B7-$B$2", "=C7-$B$2", ""),
        ("Largest drawdown in $", f"=MIN(Daily!$U$2:$U${last})", "", "deepest fall from a prior equity peak"),
        ("Annual volatility", f"=STDEV({rng('G')})*SQRT(252)", f"=STDEV({rng('H')})*SQRT(252)", ""),
        ("Sharpe (daily, x sqrt 252)", f"=AVERAGE({rng('G')})/STDEV({rng('G')})*SQRT(252)",
         f"=AVERAGE({rng('H')})/STDEV({rng('H')})*SQRT(252)", "risk-free = 0"),
        ("Max drawdown", f"=MIN({rng('K')})", "=MIN(Gold_DD)", "gold DD from the GoldDD helper column"),
        ("Return / max drawdown", "=B8/-B13", "=C8/-C13", "Calmar-style"),
        ("% positive days", f'=COUNTIF({rng("G")},">0")/COUNT({rng("G")})',
         f'=COUNTIF({rng("H")},">0")/COUNT({rng("H")})', ""),
        ("Average up day", f'=AVERAGEIF({rng("G")},">0")', f'=AVERAGEIF({rng("H")},">0")', ""),
        ("Average down day", f'=AVERAGEIF({rng("G")},"<0")', f'=AVERAGEIF({rng("H")},"<0")', ""),
        ("Average leverage (x)", f"=AVERAGE({rng('F')})", 1, "gold B&H = fully invested"),
        ("Max leverage (x)", f"=MAX({rng('F')})", 1, ""),
        ("Turnover per year (x notional)", f"=SUM({rng('R')})/B6", "", "how often the book is traded"),
        ("Gold at the CHAMPION's volatility: Sharpe", "", "=C12", "Sharpe is unchanged by leverage"),
        ("Gold at the CHAMPION's volatility: CAGR", "",
         "=EXP(AVERAGE(Daily!$P$2:$P$%d)*252*B11/C11)-1" % last, "approx: log drift scaled to champion vol"),
    ]
    for r in rows:
        s.append(list(r))
    pct = {8, 11, 13, 15, 16, 17, 22}
    usd = {7, 9, 10}
    for i in range(4, 4 + len(rows)):
        for c in ("B", "C"):
            x = s[f"{c}{i}"]
            x.font = Font(name=F)
            x.number_format = ("yyyy-mm-dd" if i in (4, 5) else "0.0%" if i in pct
                               else "$#,##0;($#,##0)" if i in usd
                               else "0.00x" if i in (18, 19, 20) else "0.00")
        s[f"A{i}"].font = Font(name=F)
        s[f"D{i}"].font = Font(name=F, italic=True, color="666666")
    s["B2"].font = Font(name=F, color="0000FF", bold=True)
    s["B2"].fill = PatternFill("solid", fgColor="FFFF00")
    s["B2"].number_format = "$#,##0"
    s["A2"].font = Font(name=F, bold=True)
    s["D2"].font = Font(name=F, italic=True, color="666666")
    s.append([])
    s.append(["Assumptions (hardcoded in the backtest engine)"])
    s[f"A{s.max_row}"].font = Font(name=F, bold=True)
    for a in ["Market: XAUUSD H4 bars (data/XAUUSD_H4_long.csv), aggregated to daily rows",
              "Signal: 0.5*conc(max(EWMAC_mid+, Breakout_fast+)) + 0.5*conc(Breakout_fast+) + 0.15, clipped 0..2 (LONG-ONLY)",
              "Sizing: forecast x 10% target vol / EWM vol (half-life 42 H4 bars), buffer 0.1, max 8x",
              "Fills: next H4 bar; cost = half the spread (floored at a live $0.448) + $0.10 slippage per unit traded",
              "No swap/financing charged (≈ -1.6%/yr on this sleeve at live rates; see V5_FINDINGS §3q)",
              "Source: scripts/v5_champion_excel.py, generated 2026-11-06"]:
        s.append([a])
        s[f"A{s.max_row}"].font = Font(name=F, color="0000FF")
    s.column_dimensions["A"].width = 40
    for c in "BC":
        s.column_dimensions[c].width = 16
    s.column_dimensions["D"].width = 42

    # gold drawdown helper (defined name)
    ws.cell(row=1, column=19, value="Gold peak")
    ws.cell(row=1, column=20, value="Gold drawdown")
    ws.cell(row=1, column=21, value="Champion drawdown ($)")
    ws.cell(row=1, column=22, value="Champion daily P&L ($)")
    style_header(ws, 1, 22)
    for i in range(2, last + 1):
        for col, f in ((21, f"=I{i}-J{i}"), (22, f"=I{i}-Summary!$B$2" if i == 2 else f"=I{i}-I{i-1}")):
            c = ws.cell(row=i, column=col, value=f)
            c.font = black
            c.number_format = "$#,##0;[Red]($#,##0)"
    for i in range(2, last + 1):
        ws.cell(row=i, column=19, value=f"=L{i}" if i == 2 else f"=MAX(S{i-1},L{i})").font = black
        c = ws.cell(row=i, column=20, value=f"=L{i}/S{i}-1")
        c.font = black
        c.number_format = "0.00%"
    from openpyxl.workbook.defined_name import DefinedName  # noqa: PLC0415
    wb.defined_names["Gold_DD"] = DefinedName("Gold_DD", attr_text=f"Daily!$T$2:$T${last}")

    # ---------------------------------------------------------------- Yearly
    y = wb.create_sheet("Yearly", 1)
    y.append(["Year", "Champion return", "Gold return", "Champion - gold", "Avg forecast (0-2)",
              "Avg leverage (x)", "Worst drawdown within year", "Days", "Champion up days %",
              "Champion profit ($)", "Equity at year end ($)"])
    style_header(y, 1, 11)
    years = sorted(d.index.year.unique())
    for k, yr in enumerate(years):
        i = k + 2
        y.append([str(yr),
                  f"=EXP(SUMIFS({rng('O')},{rng('B')},VALUE(A{i})))-1",
                  f"=EXP(SUMIFS({rng('P')},{rng('B')},VALUE(A{i})))-1",
                  f"=B{i}-C{i}",
                  f"=AVERAGEIFS({rng('E')},{rng('B')},VALUE(A{i}))",
                  f"=AVERAGEIFS({rng('F')},{rng('B')},VALUE(A{i}))",
                  f"=_xlfn.MINIFS({rng('N')},{rng('B')},VALUE(A{i}))",
                  f"=COUNTIFS({rng('B')},VALUE(A{i}))",
                  f'=COUNTIFS({rng("B")},VALUE(A{i}),{rng("G")},">0")/H{i}',
                  f"=SUMIFS(Daily!$V$2:$V${last},{rng('B')},VALUE(A{i}))",
                  f"=Summary!$B$2+SUM($J$2:J{i})"])
    for row in y.iter_rows(min_row=2, max_row=len(years) + 1):
        for c in row:
            c.font = Font(name=F)
        for j in (1, 2, 3, 6, 8):
            row[j].number_format = "0.0%;[Red]-0.0%"
        row[4].number_format = "0.00"
        row[5].number_format = "0.00x"
        row[9].number_format = row[10].number_format = "$#,##0;[Red]($#,##0)"
    for j, w in enumerate([8, 14, 12, 14, 14, 13, 16, 8, 14, 16, 18], 1):
        y.column_dimensions[get_column_letter(j)].width = w
    y.freeze_panes = "B2"

    # ---------------------------------------------------------------- Monthly grid
    m = wb.create_sheet("Monthly", 2)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m.append(["Year"] + months + ["Year"])
    style_header(m, 1, 14)
    for k, yr in enumerate(years):
        i = k + 2
        row = [str(yr)]
        for mo in range(1, 13):
            row.append(f'=IF(COUNTIFS({rng("B")},VALUE($A{i}),{rng("C")},{mo})=0,"",'
                       f'EXP(SUMIFS({rng("O")},{rng("B")},VALUE($A{i}),{rng("C")},{mo}))-1)')
        row.append(f"=Yearly!B{i}")
        m.append(row)
    for row in m.iter_rows(min_row=2, max_row=len(years) + 1):
        for c in row:
            c.font = Font(name=F)
            c.number_format = "0.0%;[Red]-0.0%"
        row[13].font = Font(name=F, bold=True, color="008000")
    from openpyxl.formatting.rule import ColorScaleRule  # noqa: PLC0415
    m.conditional_formatting.add(f"B2:M{len(years)+1}", ColorScaleRule(
        start_type="num", start_value=-0.08, start_color="F8696B", mid_type="num", mid_value=0,
        mid_color="FFFFFF", end_type="num", end_value=0.08, end_color="63BE7B"))
    m.column_dimensions["A"].width = 8
    n_rows = len(years) + 1
    m.append([])
    m.append(["Positive months", f'=COUNTIF(B2:M{n_rows},">0")/COUNT(B2:M{n_rows})'])
    m.append(["Best month", f"=MAX(B2:M{n_rows})"])
    m.append(["Worst month", f"=MIN(B2:M{n_rows})"])
    for i in range(n_rows + 2, n_rows + 5):
        m[f"A{i}"].font = Font(name=F, bold=True)
        m[f"B{i}"].font = Font(name=F)
        m[f"B{i}"].number_format = "0.0%"
    m.column_dimensions["A"].width = 16

    # ---------------------------------------------------------------- Forecast states
    st = wb.create_sheet("By forecast state", 3)
    st.append(["Forecast state (yesterday's forecast)", "Share of days", "Avg leverage (x)",
               "Champion mean daily return", "Gold mean daily return", "Share of champion P&L",
               "Champion up-day %"])
    style_header(st, 1, 7)
    for k, nm in enumerate(["1 resting", "2 building", "3 trend", "4 strong trend"]):
        i = k + 2
        st.append([nm, f'=COUNTIF({rng("Q")},A{i})/COUNT({rng("G")})',
                   f'=AVERAGEIF({rng("Q")},A{i},{rng("F")})',
                   f'=AVERAGEIF({rng("Q")},A{i},{rng("G")})',
                   f'=AVERAGEIF({rng("Q")},A{i},{rng("H")})',
                   f'=SUMIF({rng("Q")},A{i},{rng("O")})/SUM({rng("O")})',
                   f'=COUNTIFS({rng("Q")},A{i},{rng("G")},">0")/COUNTIF({rng("Q")},A{i})'])
    for row in st.iter_rows(min_row=2, max_row=5):
        for c in row:
            c.font = Font(name=F)
        row[1].number_format = row[5].number_format = row[6].number_format = "0.0%"
        row[2].number_format = "0.00x"
        row[3].number_format = row[4].number_format = "0.000%"
    st.append([])
    st.append(["States: resting < 0.2 (the 0.15 long tilt), building 0.2-0.8, trend 0.8-1.5, strong "
               "1.5-2. Uses YESTERDAY's forecast, i.e. what was known before the day's move."])
    st[f"A{st.max_row}"].font = Font(name=F, italic=True, color="666666")
    for j, w in enumerate([34, 12, 14, 18, 18, 18, 14], 1):
        st.column_dimensions[get_column_letter(j)].width = w

    # ---------------------------------------------------------------- Charts
    ch = wb.create_sheet("Charts", 1)
    c1 = LineChart()
    c1.title = "Equity ($, starting $10,000): champion vs gold buy & hold"
    c1.y_axis.title = "$"
    c1.height, c1.width = 9, 24
    c1.add_data(Reference(ws, min_col=9, min_row=1, max_row=last), titles_from_data=True)
    c1.add_data(Reference(ws, min_col=12, min_row=1, max_row=last), titles_from_data=True)
    c1.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
    c1.x_axis.number_format = "yyyy"
    ch.add_chart(c1, "A1")
    c2 = LineChart()
    c2.title = "Drawdown: champion vs gold"
    c2.height, c2.width = 7, 24
    c2.add_data(Reference(ws, min_col=11, min_row=1, max_row=last), titles_from_data=True)
    c2.add_data(Reference(ws, min_col=20, min_row=1, max_row=last), titles_from_data=True)
    c2.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
    c2.y_axis.number_format = "0%"
    ch.add_chart(c2, "A20")
    c3 = LineChart()
    c3.title = "Leverage the champion carried (x equity)"
    c3.height, c3.width = 7, 24
    c3.add_data(Reference(ws, min_col=6, min_row=1, max_row=last), titles_from_data=True)
    c3.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
    ch.add_chart(c3, "A35")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(OUT, n, "daily rows")


if __name__ == "__main__":
    main()
