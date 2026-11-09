#!/usr/bin/env python3
"""Build the §3dp blog post (HTML) and working paper (HTML -> PDF) from docs/paper/stats.json.

    python scripts/v5_champion_paper_assets.py     # stats + figures first
    python scripts/v5_champion_writeups.py
Outputs: docs/blog/gold-champion-15-years.html, docs/paper/gold-trend-cme-2011-2026.{html,pdf}
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "docs" / "paper"
S = json.loads((P / "stats.json").read_text())
C, K, G = S["streams"]["champion"], S["streams"]["const"], S["streams"]["gold"]
YRS = S["by_year"]


def svg(name: str) -> str:
    s = (P / "figs" / f"{name}.svg").read_text()
    s = re.sub(r"<\?xml.*?\?>", "", s, flags=re.S)
    s = re.sub(r"<!DOCTYPE.*?>", "", s, flags=re.S)
    s = re.sub(r'width="[\d.]+pt" height="[\d.]+pt"', 'width="100%"', s, count=1)
    return s


def pct(x: float, d: int = 1, sign: bool = True) -> str:
    return f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%"


def cls(x: float) -> str:
    return "pos" if x > 0 else "neg" if x < 0 else ""


def year_rows() -> str:
    out = []
    for y, r in YRS.items():
        out.append(f"<tr><td>{y}{'*' if y == '2026' else ''}</td>" + "".join(
            f'<td class="{cls(r[k])}">{pct(r[k])}</td>' for k in ("champion", "const", "gold")) + "</tr>")
    return "".join(out)


ERA = [  # from data/v5_runs/champion_h4_cme/report.txt (§3dp)
    ("Out-of-sample, Jun 2011 – Dec 2014", (4.4, 0.18, -15.5), (-13.1, -0.34, -28.0), (-21.4, -0.24, -40.9)),
    ("Gold bear market, Sep 2011 – Dec 2015", (-5.6, -0.12, -15.5), (-22.4, -0.56, -29.0), (-42.2, -0.58, -45.0)),
    ("Design era, Jan 2015 – Nov 2026", (171.3, 0.83, -18.9), (118.3, 0.83, -12.9), (142.8, 0.53, -30.1)),
    ("Full sample, Jun 2011 – Nov 2026", (183.2, 0.69, -18.9), (89.6, 0.52, -29.0), (90.7, 0.32, -45.0)),
]


def era_rows() -> str:
    out = []
    for name, *cells in ERA:
        out.append(f'<tr class="grp"><td colspan="4">{name}</td></tr>')
        for lab, (tot, sr, dd) in zip(("Champion", "Constant long, same vol", "Gold buy &amp; hold"), cells):
            out.append(f'<tr{" class=hl" if lab == "Champion" else ""}><td>{lab}</td>'
                       f'<td class="{cls(tot)}">{pct(tot)}</td><td>{sr:+.2f}</td><td>{dd:.1f}%</td></tr>')
    return "".join(out)


def stat_rows() -> str:
    rows = [("Total return", lambda s: pct(s["total"], 0)), ("CAGR", lambda s: pct(s["cagr"])),
            ("Annualised volatility", lambda s: pct(s["vol"], 1, False)),
            ("Sharpe ratio", lambda s: f"{s['sharpe']:.2f}"),
            ("Sharpe 95% CI (block bootstrap)", lambda s: f"[{s['sharpe_ci'][0]:.2f}, {s['sharpe_ci'][1]:.2f}]"),
            ("Sortino ratio", lambda s: f"{s['sortino']:.2f}"),
            ("Maximum drawdown", lambda s: f"{s['maxdd']:.1f}%"),
            ("Calmar (CAGR / max DD)", lambda s: f"{s['calmar']:.2f}"),
            ("Longest time under water", lambda s: f"{s['underwater']:.1f} yrs"),
            ("Positive months", lambda s: pct(s["pct_pos_months"], 0, False)),
            ("Best / worst month", lambda s: f"{s['best_month']:+.1f}% / {s['worst_month']:+.1f}%"),
            ("Skewness of monthly returns", lambda s: f"{s['skew_monthly']:+.2f}")]
    return "".join(f"<tr><td>{n}</td>" + "".join(f"<td>{f(x)}</td>" for x in (C, K, G)) + "</tr>" for n, f in rows)


def cost_rows() -> str:
    return "".join(f"<tr><td>{float(c):g} bp</td><td>{v:.2f}</td></tr>" for c, v in S["cost_sweep"].items())


DVC, DVG, EXP, BF = S["diff_vs_const"], S["diff_vs_gold"], S["exposure"], S["bear_forecast"]

# =============================================================================================== BLOG
BLOG = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold Trend Stress Test</title>
<meta name="description" content="A long-only gold trend strategy re-tested on fifteen years of CME gold futures, including the 2011-2015 crash it was never tuned on.">
<style>
:root{{--bg:#fbfaf7;--fg:#1d1c1a;--muted:#6b675f;--rule:#e3dfd6;--card:#ffffff;--accent:#b5791a;--good:#2e7d4f;--bad:#b23b3b;--tint:#f5efe3}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151412;--fg:#ece8df;--muted:#a39d91;--rule:#2e2c28;--card:#1d1c19;--accent:#e3a33c;--good:#6fc191;--bad:#e07a7a;--tint:#24211b}}}}
:root[data-theme="dark"]{{--bg:#151412;--fg:#ece8df;--muted:#a39d91;--rule:#2e2c28;--card:#1d1c19;--accent:#e3a33c;--good:#6fc191;--bad:#e07a7a;--tint:#24211b}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:18px/1.68 Georgia,"Iowan Old Style","Times New Roman",serif}}
main{{max-width:740px;margin:0 auto;padding:48px 16px 90px}}
.kicker{{font:600 13px/1 system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}}
h1{{font-size:clamp(30px,5.4vw,46px);line-height:1.1;margin:12px 0 14px;letter-spacing:-.01em}}
.dek{{font-size:20px;color:var(--muted);margin:0 0 12px}}
.meta{{font:14px system-ui,sans-serif;color:var(--muted);border-bottom:1px solid var(--rule);padding-bottom:22px;margin-bottom:30px}}
h2{{font-size:26px;margin:52px 0 12px;line-height:1.25}}
h3{{font:650 17px/1.3 system-ui,sans-serif;margin:28px 0 8px}}
p{{margin:0 0 18px}} a{{color:var(--accent)}}
.tldr{{background:var(--tint);border-radius:10px;padding:18px 22px;margin:0 0 30px}}
.tldr h2{{margin:0 0 8px;font:700 13px system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}}
.tldr ul{{margin:0;padding-left:20px}} .tldr li{{margin-bottom:6px}}
.toc{{font:15px/1.7 system-ui,sans-serif;columns:2;column-gap:28px;margin:0 0 10px;padding-left:20px}}
figure{{margin:30px 0;background:#fff;border:1px solid var(--rule);border-radius:10px;padding:14px 12px 10px}}
figure svg{{display:block;height:auto}}
figcaption{{font:14px/1.5 system-ui,sans-serif;color:#5f5b53;margin-top:8px}}
.tablewrap{{overflow-x:auto;margin:22px 0}}
table{{border-collapse:collapse;width:100%;font:15px/1.4 system-ui,sans-serif;min-width:460px}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
thead th{{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
tr.hl td{{font-weight:650}} tr.grp td{{font-weight:650;color:var(--accent);padding-top:16px;border-bottom:none}}
.pos{{color:var(--good)}} .neg{{color:var(--bad)}}
.callout{{border-left:3px solid var(--accent);padding:4px 0 4px 18px;margin:28px 0;font-size:21px;line-height:1.5}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}
.stat{{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:14px}}
.stat b{{display:block;font:700 26px/1.1 system-ui,sans-serif;color:var(--accent)}}
.stat span{{font:13px/1.4 system-ui,sans-serif;color:var(--muted)}}
ol li,ul li{{margin-bottom:10px}}
details{{border-bottom:1px solid var(--rule);padding:12px 0}} summary{{cursor:pointer;font-weight:650}}
details p{{margin:10px 0 4px}}
.note{{font:14px/1.6 system-ui,sans-serif;color:var(--muted);border-top:1px solid var(--rule);margin-top:52px;padding-top:20px}}
</style></head>
<body><main>
<div class="kicker">Backtest review · Gold</div>
<h1>Fifteen years and one gold crash: an honest stress test of a gold trend strategy</h1>
<p class="dek">Our gold strategy looked excellent on the data it was built on. We re-ran it, unchanged, on fifteen years of exchange data, including a 45% crash it had never seen. This is everything we found, good and bad.</p>
<div class="meta">November 2026 · about 15 minutes to read</div>

<div class="tldr"><h2>The short version</h2><ul>
<li><strong>It survived the crash.</strong> In 2011–2014, years it was never tuned on, gold lost 21% and the strategy made 4%, with a third of gold's drawdown.</li>
<li><strong>Its headline Sharpe of 1.08 was flattered.</strong> On fifteen years of clean data it is <strong>{C['sharpe']:.2f}</strong>, holding costs included. Gold's is {G['sharpe']:.2f}.</li>
<li><strong>Its strength is limiting losses, not predicting gold.</strong> Worst drawdown {C['maxdd']:.0f}%, against {G['maxdd']:.0f}% for gold. Its timing advantage over simply holding gold at the same risk is too small to be statistically meaningful.</li>
<li><strong>Realistic expectation:</strong> about 7–10% a year at standard risk, a worst drawdown near 20%, and roughly three-year stretches below a previous high.</li>
</ul></div>

<ol class="toc">
<li>Why we re-tested it</li><li>How the strategy works</li><li>The data, and how we checked it</li>
<li>The out-of-sample test</li><li>Inside the crash</li><li>Fifteen years in numbers</li>
<li>Year by year</li><li>Is it skill or luck?</li><li>Costs</li><li>What didn't help</li>
<li>What to expect</li><li>FAQ</li></ol>

<h2>1. Why we re-tested it</h2>
<p>Every trading strategy looks good on the data used to design it. Its settings are chosen because they worked on that data, so its backtest is partly a measure of how well it fits the past. The only honest test is data it has never seen.</p>
<p>Our gold strategy, which we call "the champion", had two weaknesses there:</p>
<ul>
<li><strong>Its headline Sharpe of about 1.08 came from 2015 onward.</strong> That is the same period we used to choose its settings, and gold mostly rose over it. A long-only strategy looks good in a bull market almost by default.</li>
<li><strong>Our older long gold history was stitched together from two brokers' price feeds.</strong> That kind of splice had already produced two fake results in our research, so we couldn't trust it for this.</li>
</ul>
<p>An earlier test ran a daily-bar version back to 2002 and suggested the strategy survived the 2011–2015 crash. But the live strategy runs on 4-hour bars, and clean 4-hour gold data before 2015 wasn't available. So the question stayed open: is the 4-hour strategy genuinely good, or did it look good because of when we tested it?</p>

<h2>2. How the strategy works</h2>
<p>It's deliberately simple, and every rule is fixed in advance:</p>
<h3>Long only</h3>
<p>It buys gold or holds a small amount. It never bets on gold falling. Across our research, the short side of gold lost money in every family of signals we tried. Gold's long-run upward drift makes selling it a losing habit.</p>
<h3>Two trend signals, combined</h3>
<ul>
<li><strong>Trend:</strong> compares fast and slow moving averages of price at three speeds, from about 16 to 256 trading days. The further the fast averages sit above the slow ones, the stronger the trend.</li>
<li><strong>Breakout:</strong> measures where today's price sits within its recent range over 10, 20 and 40 days. Near the top of the range scores high.</li>
</ul>
<p>The two are combined into a <em>forecast</em> between 0 and 2. Strong signals are boosted and weak ones shrunk (the score is raised to the power 1.5), so the strategy commits when trends are clear and stays light when they're not.</p>
<h3>A small resting position</h3>
<p>Even with no trend, the forecast doesn't drop below about 0.15. It keeps a token long position, so it's never fully out when a new uptrend starts.</p>
<h3>Sizing by volatility</h3>
<p>The position is sized so the account's risk stays near a 10% yearly volatility target. When gold becomes jumpy, it holds less; when gold is calm, it holds more. Over fifteen years the position averaged {EXP['mean']:.2f}× the account (median {EXP['median']:.2f}×), and only 5% of the time was it above {EXP['p95']:.1f}×.</p>
<h3>Checked every 4 hours, traded only when needed</h3>
<p>It recalculates every 4 hours, but only trades when the target position has moved meaningfully away from what it holds. That keeps costs down.</p>

<h2>3. The data, and how we checked it</h2>
<p>We used <strong>hourly CME gold futures bars from June 2010 to November 2026</strong>, from Databento. It's one source, the exchange where gold's price is actually set. Futures contracts expire every few months, so we stitched consecutive contracts together and removed the artificial price jump at each switch.</p>
<p>Before trusting it, we checked it against the gold our live strategy actually trades at a broker:</p>
<ul>
<li>Hour by hour, the two prices move together with a correlation of <strong>0.986</strong>. One hour apart, the correlation is about zero, so the timestamps line up.</li>
<li>On the years both cover (2017 onward), the strategy's daily results on the two data sources correlate at <strong>0.95</strong>.</li>
</ul>
<p>One difference matters. A futures price already includes the cost of holding gold, which is roughly the interest rate. A broker charges that cost separately, as overnight swap. So these results include the cost of holding, and on recent years, when interest rates were high, they come out lower than our broker-data backtest (Sharpe 0.86 against 1.12 for 2017 onward).</p>
<p>The test starts in <strong>June 2011</strong>, because the slowest trend signal needs about a year of history before it can trade. Every comparison below starts on the same date.</p>

<figure>{svg("equity")}
<figcaption><strong>Growth of $10,000, June 2011 – November 2026</strong> (log scale). The shaded band is gold's 2011–2015 bear market. The champion ends at about ${10000 * (1 + C['total'] / 100):,.0f}, always-long at the same risk at ${10000 * (1 + K['total'] / 100):,.0f}, and gold buy-and-hold at ${10000 * (1 + G['total'] / 100):,.0f}.</figcaption></figure>

<h2>4. The out-of-sample test: 2011 to 2014</h2>
<p>The strategy's settings were chosen on data from 2015 onward, so everything before 2015 is a genuine out-of-sample test. It's also the worst stretch for gold in a generation.</p>
<div class="tablewrap"><table>
<thead><tr><th>Jun 2011 – Dec 2014</th><th>Return</th><th>Sharpe</th><th>Worst drawdown</th></tr></thead>
<tbody>
<tr class="hl"><td>Champion</td><td class="pos">+4.4%</td><td>+0.18</td><td>−15.5%</td></tr>
<tr><td>Always long, same risk</td><td class="neg">−13.1%</td><td>−0.34</td><td>−28.0%</td></tr>
<tr><td>Gold buy &amp; hold</td><td class="neg">−21.4%</td><td>−0.24</td><td>−40.9%</td></tr>
</tbody></table></div>
<p>Gold lost a fifth of its value and the strategy made a small profit, with a drawdown well under half of gold's. A 0.18 Sharpe is nothing to celebrate on its own. What matters is that the strategy <em>held its ground</em> while everything that simply owned gold got hurt.</p>

<h2>5. Inside the crash</h2>
<p>How did it avoid the damage? It didn't predict the crash. It noticed the uptrend had stopped and stayed small:</p>
<div class="stats">
<div class="stat"><b>{BF['resting']:.0f}%</b><span>of 4-hour checks in 2011–15 spent at the small resting position</span></div>
<div class="stat"><b>{BF['mean']:.2f}</b><span>average forecast in the bear market (scale 0–2)</span></div>
<div class="stat"><b>{EXP['bear_mean']:.2f}×</b><span>average position in the bear, against {EXP['post_mean']:.2f}× from 2016 on</span></div>
</div>
<figure>{svg("bear_forecast")}
<figcaption><strong>Gold price (top) and the strategy's forecast (bottom), 2011–2016.</strong> Each rally produced a brief burst of conviction. When the rally failed, the forecast fell back to its 0.15 floor within weeks. Losses came from those false starts, but each one was small.</figcaption></figure>
<p class="callout">It didn't need to know the crash was coming. It only needed to notice the uptrend had stopped, and then stay small.</p>
<p>Notice the trade-off in the chart. Every failed rally cost something, which is why the strategy still lost about 6% across the full bear (Sep 2011 – Dec 2015). Always-long at the same risk lost 22%, and gold lost 42%.</p>

<h2>6. Fifteen years in numbers</h2>
<div class="tablewrap"><table>
<thead><tr><th>Jun 2011 – Nov 2026</th><th>Champion</th><th>Always long</th><th>Gold</th></tr></thead>
<tbody>{stat_rows()}</tbody></table></div>
<p>Three things stand out:</p>
<ul>
<li><strong>Fewer than half its months are positive</strong> ({C['pct_pos_months']:.0f}%), yet it more than doubles the always-long return. That's what trend following looks like: many small losses, a few large gains. Its best month (+{C['best_month']:.1f}%) is nearly three times the size of its worst ({C['worst_month']:.1f}%), and its returns lean strongly to the upside (skewness +{C['skew_monthly']:.1f}).</li>
<li><strong>Its drawdowns are far shallower</strong> than gold's, and it recovers faster: its longest stretch below a previous high was {C['underwater']:.1f} years, against {G['underwater']:.1f} years for gold.</li>
<li><strong>It runs slightly above its 10% volatility target</strong> ({C['vol']:.1f}%), because its strongest signals size up to twice the base position.</li>
</ul>
<figure>{svg("drawdown")}
<figcaption><strong>Drawdown from the previous peak.</strong> Gold spent most of 2013–2019 more than 20% below its 2011 high. The champion's worst drop was {C['maxdd']:.1f}%.</figcaption></figure>
<figure>{svg("rolling_sharpe")}
<figcaption><strong>Rolling 3-year Sharpe ratio.</strong> The strategy's quality varies a lot with the market. It was negative for three-year windows ending in 2014–2015, mostly between 0.5 and 0.9 from 2016 to 2023, and above 1 only in 2025–2026. Expect that range.</figcaption></figure>

<h2>7. Year by year</h2>
<div class="tablewrap"><table>
<thead><tr><th>Year</th><th>Champion</th><th>Always long</th><th>Gold</th></tr></thead>
<tbody>{year_rows()}</tbody></table></div>
<p style="font:14px system-ui,sans-serif;color:var(--muted)">* 2011 from June; 2026 to 6 November. Returns are at the 10% volatility target, so they're not directly comparable to gold's raw percentage moves.</p>
<p>The strategy was positive in {S['pct_years_positive']:.0f}% of years and beat gold in {S['pct_years_beat_gold']:.0f}% of them. Its worst calendar year was 2013 (−7.3%), when gold fell 28%. It lags badly in a straight-up boom, as in 2025, when gold rose 56% and the strategy made 23%, because it keeps its risk constant rather than chasing the move.</p>

<h2>8. Is it skill or luck?</h2>
<p>Fifteen years sounds like a lot, but for a strategy that trades long trends it holds only a handful of independent episodes. So we asked how confident we can be, using a bootstrap: resampling the daily returns in monthly blocks thousands of times.</p>
<ul>
<li><strong>The strategy's Sharpe ratio</strong> lies between {C['sharpe_ci'][0]:.2f} and {C['sharpe_ci'][1]:.2f} with 95% confidence. It's clearly above zero, but the range is wide.</li>
<li><strong>Against gold buy-and-hold:</strong> the strategy's Sharpe is higher in about {100 - 100 * DVG['p_le0']:.0f}% of resamples. That's strong evidence of a better risk-adjusted return than simply owning gold.</li>
<li><strong>Against always-long at the same risk:</strong> higher in only about {100 - 100 * DVC['p_le0']:.0f}% of resamples. That's suggestive, not proof. A good part of the improvement over gold comes from <em>sizing by volatility</em> rather than from <em>timing</em>.</li>
</ul>
<p>Our honest reading: the strategy reliably <strong>turns gold into a smoother investment</strong>, with similar or better returns and much smaller crashes. Whether its trend timing adds much beyond that can't be proven from fifteen years of data.</p>

<h2>9. Costs</h2>
<p>A strategy that only works with near-zero costs is fragile. This one trades rarely, turning over about {S['turnover_per_year']:.0f}× its account value a year, so costs barely matter:</p>
<div class="tablewrap"><table style="min-width:260px;max-width:360px">
<thead><tr><th>Cost per trade</th><th>Sharpe</th></tr></thead>
<tbody>{cost_rows()}</tbody></table></div>
<p>Ten times our baseline cost lowers the Sharpe from 0.69 to 0.58. Costs aren't what decides this strategy; holding costs (swap) matter more. A swap-free account would add back roughly the interest rate a year on the gold held.</p>

<h2>10. What didn't help</h2>
<p>We've spent months trying to improve this strategy. Almost nothing beat the simple version once compared fairly, meaning at the same risk level and against random-timing benchmarks:</p>
<ul>
<li><strong>Market-regime filters</strong> (trading only when trend strength or volatility looks favourable) cut returns and made drawdowns worse.</li>
<li><strong>Short selling</strong> lost money in every form.</li>
<li><strong>Adding more trend assets</strong> diluted the result. Gold plus bitcoin was the best pair we found.</li>
<li><strong>Volume and order-flow signals</strong> ("footprint" charts built from real buy and sell volume) added nothing once price was accounted for.</li>
<li><strong>Machine-learning timing</strong> (gradient boosting, LSTMs, a transformer trained on candles) didn't beat random entry timing.</li>
<li><strong>Taking profits early or scaling out</strong> raised the win rate but lowered total returns. Trend followers make their money on the trades they let run.</li>
</ul>
<p>The one lever that clearly worked was <strong>lowering the cost of holding</strong>: swap-free gold accounts improved results more than any signal tweak.</p>

<h2>11. What to expect if you run it</h2>
<ul>
<li><strong>Returns:</strong> about 7–10% a year at a 10% volatility target, before broker swap. Scale the target up or down to change both return and drawdown proportionally.</li>
<li><strong>Drawdowns:</strong> expect a peak-to-trough fall of around 15–20% at some point, and several years with small losses.</li>
<li><strong>Patience:</strong> twice in fifteen years it spent about three years below a previous high: October 2012 to February 2016, and August 2020 to March 2024.</li>
<li><strong>Underperformance in booms:</strong> in a year like 2025, simply holding gold will beat it by a wide margin. That's the price of insurance against a year like 2013.</li>
<li><strong>Don't intervene.</strong> Most of the return comes from a few long trends. Closing positions early to "bank" gains removes exactly those.</li>
</ul>

<h2>12. FAQ</h2>
<details><summary>Why compare against "always long at the same risk"?</summary><p>Because the strategy changes two things at once: <em>when</em> it holds gold (timing) and <em>how much</em> (volatility sizing). Holding a constant long position with the same volatility sizing isolates the timing. Beating gold is easy if you only control risk; beating the constant-long version shows the timing itself helps.</p></details>
<details><summary>Why 4-hour bars?</summary><p>The live strategy runs on them. A daily-bar version tested back to 2002 gave a similar full-period Sharpe (0.64) and the same crash protection, so the result isn't specific to the bar size.</p></details>
<details><summary>Isn't fifteen years enough to prove it?</summary><p>For trend following, not quite. The strategy's returns come from a small number of multi-month trends, so fifteen years holds perhaps a dozen independent bets. That's why the confidence interval on the Sharpe is so wide.</p></details>
<details><summary>Would it have worked before 2010?</summary><p>The daily version, tested 2002–2026, kept the same character: similar returns to holding gold at the same risk, with roughly half the drawdown. It lost 11% in the 2011–15 bear while gold lost 42%.</p></details>
<details><summary>Why not also bet on gold falling?</summary><p>We tested it extensively. Shorting gold lost money in every signal family over the long run, because gold drifts upward over time and short positions pay the holding cost the other way. Long-only roughly doubled the Sharpe of the long/short versions.</p></details>

<div class="note"><strong>Method.</strong> Data: CME Globex gold futures (GC, most-active contract by volume), hourly bars via Databento, June 2010 – November 2026, back-adjusted at contract rolls, grouped into 4-hour bars on the broker server clock (New York + 7h). Strategy: long-only blend of an EWMAC trend forecast (16/64, 32/128, 64/256 days) and a breakout forecast (10, 20, 40 days), concentrated with exponent 1.5, floor 0.15, cap 2. Sizing: 10% annual volatility target, exponentially weighted volatility (half-life 42 bars), 10% no-trade buffer. Costs 1 basis point per trade (sweep shown above). Futures returns already include holding costs. Confidence intervals: moving-block bootstrap, 21-day blocks, 2,000 resamples. Full working paper and code available on request.<br><br>
<strong>Disclaimer.</strong> This is a backtest: past simulated results, not live trading and not investment advice. Real trading adds slippage, broker costs and execution risk. Past performance does not guarantee future results.</div>
</main></body></html>"""

# =============================================================================================== PAPER
PAPER = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Long-Only Gold Trend Following Out of Sample</title>
<style>
@page{{size:A4;margin:24mm 22mm 22mm 22mm;@bottom-center{{content:counter(page);font:10pt "Latin Modern Roman","CMU Serif","Times New Roman",serif}}}}
body{{font:11pt/1.55 "Latin Modern Roman","CMU Serif","Computer Modern","Times New Roman",serif;color:#111;margin:0}}
.title{{text-align:center;margin-top:8mm}}
.title h1{{font-size:20pt;line-height:1.25;margin:0 0 14px;font-weight:700}}
.authors{{font-size:12pt;margin-bottom:6px}} .aff{{font-size:9pt;color:#333}} .date{{font-size:9.5pt;margin-top:10px}}
hr.thick{{border:0;border-top:1.4pt solid #000;margin:16px 0}}
.abstract{{margin:18px 9mm 10px;font-size:10pt;text-align:justify}} .abstract h2{{text-align:center;font-size:11pt;margin:0 0 6px}}
.kw{{margin:10px 9mm 0;font-size:9pt}}
h2.sec{{font-size:14pt;margin:22px 0 8px;break-after:avoid}} h3.sub{{font-size:11.5pt;margin:16px 0 6px;break-after:avoid}}
.subt{{font-size:14pt;margin:-4px 0 16px;white-space:nowrap}}
p{{margin:0 0 10px;text-align:justify}}
.eq{{text-align:center;margin:10px 0;font-style:italic}} .eq span{{float:right;font-style:normal}}
table{{border-collapse:collapse;margin:8px auto;font-size:9.5pt}}
table.bk{{border-top:1.2pt solid #000;border-bottom:1.2pt solid #000}}
table.bk thead th{{border-bottom:.6pt solid #000;padding:4px 9px;font-weight:600}}
table.bk td{{padding:3px 9px;text-align:right}} table.bk td:first-child,table.bk th:first-child{{text-align:left}}
tr.grp td{{font-style:italic;padding-top:7px}} tr.hl td{{font-weight:600}}
.cap{{font-size:9.5pt;margin:4px 6mm 16px;text-align:justify}} .cap b{{font-weight:700}}
figure{{margin:12px 0 4px;break-inside:avoid}} figure svg{{width:100%;height:auto}}
.tbl{{break-inside:avoid}}
.refs p{{font-size:9.5pt;padding-left:18px;text-indent:-18px;margin-bottom:5px}}
.pb{{break-before:page}}
.pos,.neg{{}}
</style></head><body>
<div class="title">
<hr class="thick">
<h1>Long-Only Gold Trend Following<br>Through a Bear Market</h1><div class="subt">An Out-of-Sample Test on CME Gold Futures, 2011–2026</div>
<div class="authors">Gibeon Kipngeno</div>
<div class="aff">Trader36 Research</div>
<div class="date">Working paper · First version: November 7, 2026</div>
<hr class="thick">
</div>
<div class="abstract"><h2>Abstract</h2>
We test a long-only, volatility-targeted trend-following strategy on gold, designed and calibrated on broker data from 2015 onward, using fifteen years of single-provenance CME gold futures (GC) hourly data. The pre-2015 period, which contains the 2011–2015 gold bear market, is a genuine out-of-sample test. From June 2011 to December 2014 the strategy returned +4.4% (Sharpe 0.18, maximum drawdown −15.5%), while gold fell 21.4% (maximum drawdown −40.9%) and a constant long position at the same volatility target lost 13.1%. Over the full sample (June 2011 – November 2026), net of costs and implicit financing, the strategy earned 8.1% a year with a Sharpe ratio of {C['sharpe']:.2f} (95% block-bootstrap interval {C['sharpe_ci'][0]:.2f}–{C['sharpe_ci'][1]:.2f}) and a maximum drawdown of {C['maxdd']:.1f}%, against {G['sharpe']:.2f} and {G['maxdd']:.1f}% for buy-and-hold. The in-sample Sharpe of about 1.08 is not reproduced. Most of the improvement over buy-and-hold is attributable to volatility targeting: the timing component's advantage over a volatility-matched constant long position (paired t = +1.00; bootstrap P(ΔSharpe ≤ 0) = {DVC['p_le0']:.2f}) is not statistically significant, whereas its drawdown reduction is large and stable across regimes. We conclude that the strategy's robust contribution is downside control rather than return predictability.
</div>
<div class="kw"><b>Keywords:</b> <i>Trend following, Time-series momentum, Gold, Volatility targeting, Out-of-sample testing, Futures</i><br><b>JEL Classification:</b> G11, G14, G17, Q02</div>

<h2 class="sec pb">1&nbsp;&nbsp;Introduction</h2>
<p>Backtests of trading rules are evaluated on the same data used to choose them, and their performance is therefore biased upward by an amount that grows with the number of configurations tried (Bailey &amp; López de Prado, 2014; Harvey, Liu &amp; Zhu, 2016). For a long-only strategy on an asset with a strong upward drift, the bias has a second source: a sample dominated by a bull market rewards any rule that is mostly long. Gold from 2015 to 2026, which rose from roughly $1,100 to above $4,000 an ounce, is such a sample.</p>
<p>This paper studies a strategy built under exactly these conditions. It combines exponentially weighted moving-average crossover and breakout forecasts, is long-only, concentrates its exposure when signals are strong, and targets a constant volatility. On broker data from 2015 onward it achieved a Sharpe ratio of approximately 1.02–1.10 at 4-hour frequency. Two problems prevented us from validating that figure on earlier data. First, a pre-2015 4-hour gold series was available only as a splice of two brokers' feeds, a construction that had previously produced spurious results in our research. Second, an earlier validation at daily frequency back to 2002 could not speak directly to the 4-hour implementation.</p>
<p>We resolve both problems with exchange data. CME Globex gold futures hourly bars from June 2010 are single-provenance, exchange-timestamped, and, once mapped to the broker's server clock, correlate at 0.986 with the instrument the strategy trades. Because the strategy's parameters were fixed using post-2015 data, every observation from June 2011 to December 2014 is out of sample. That window includes gold's largest drawdown since the early 1980s.</p>
<p>Our contributions are threefold. (i) We report a clean out-of-sample test of a 4-hour gold trend strategy through a 45% bear market. (ii) We decompose its performance relative to buy-and-hold into a volatility-targeting component and a timing component, and show that only the former is statistically robust. (iii) We quantify the gap between in-sample and out-of-sample Sharpe for a strategy of this kind, which is useful for setting realistic expectations.</p>

<h2 class="sec">2&nbsp;&nbsp;Related literature</h2>
<p>Time-series momentum, the tendency of an asset's own past returns to predict its future returns, is documented across asset classes by Moskowitz, Ooi &amp; Pedersen (2012), and over more than a century of data by Hurst, Ooi &amp; Pedersen (2017). Gold appears in these studies as one commodity among many; dedicated evidence is thinner, and the metal's role as a hedge or safe haven (Baur &amp; Lucey, 2010) and its weak link to inflation over practical horizons (Erb &amp; Harvey, 2013) make its long-run return distribution distinctive. Volatility targeting, which scales exposure inversely to recent volatility, has been shown to raise risk-adjusted returns in several asset classes (Moreira &amp; Muir, 2017). The EWMAC and breakout forecasts used here follow the practitioner construction of Carver (2015). Our inference relies on the moving-block bootstrap (Künsch, 1989) and takes seriously the multiple-testing concerns of Bailey &amp; López de Prado (2014) and Harvey, Liu &amp; Zhu (2016).</p>

<h2 class="sec">3&nbsp;&nbsp;Data</h2>
<h3 class="sub">3.1&nbsp;&nbsp;Source and construction</h3>
<p>We use hourly OHLCV bars for CME Globex gold futures (root GC) from Databento's GLBX.MDP3 dataset, covering 7 June 2010 to 6 November 2026 ({S['bars_h4']:,} 4-hour bars after resampling). The continuous series selects the most-actively-traded contract each day. Because consecutive contracts trade at different prices, we construct a back-adjusted series by setting the return to zero on every bar where the underlying contract changes, and scale open, high and low prices by the same adjustment factor as the close. The series contains 81 contract changes.</p>
<h3 class="sub">3.2&nbsp;&nbsp;Clock alignment</h3>
<p>The strategy operates on 4-hour bars stamped on the broker's server clock, which is New York local time plus seven hours. We convert the exchange's UTC timestamps to this clock, accounting for daylight-saving changes, and resample hourly bars into 4-hour bars with the same boundaries as the live feed.</p>
<h3 class="sub">3.3&nbsp;&nbsp;Validation against the traded instrument</h3>
<p>Hourly log returns of the adjusted futures series correlate with those of the broker's spot gold CFD at 0.986 at lag zero, and at −0.009 and −0.008 at lags of minus and plus one hour, confirming alignment. Daily returns correlate at 0.995. Running the strategy separately on the two data sources over their common period (2017–2026) yields daily strategy returns that correlate at 0.951.</p>
<h3 class="sub">3.4&nbsp;&nbsp;Financing</h3>
<p>A futures position's return already nets out the cost of carry, which for gold is close to the short-term interest rate. A spot CFD instead charges financing separately as a daily swap. We add no further financing charge, so our results correspond approximately to a CFD account that pays swap at the cash rate. Over 2017–2026 this explains most of the gap between the strategy's Sharpe on futures data (0.86) and on swap-free broker data (1.12), as short-term rates were elevated for much of 2022–2025.</p>

<h2 class="sec">4&nbsp;&nbsp;Strategy and benchmarks</h2>
<h3 class="sub">4.1&nbsp;&nbsp;Forecasts</h3>
<p>Let <i>P<sub>t</sub></i> denote the price at the close of 4-hour bar <i>t</i>, with six bars per trading day. The trend forecast averages three EWMAC rules with fast/slow spans of (16, 64), (32, 128) and (64, 256) days:</p>
<div class="eq">raw<sub>f,s</sub>(t) = [ EMA<sub>f</sub>(P)<sub>t</sub> − EMA<sub>s</sub>(P)<sub>t</sub> ] / ( P<sub>t</sub> σ<sub>t</sub> )<span>(1)</span></div>
<p>where σ<sub>t</sub> is the exponentially weighted standard deviation of returns (span 36 bars). Each rule is normalised by the expanding mean of its own absolute value, using only past data, so that an average-strength signal has magnitude one. The three are averaged, scaled by 1.3, and capped at ±2. The breakout forecast averages three rules with look-backs <i>N</i> of 10, 20 and 40 days:</p>
<div class="eq">raw<sub>N</sub>(t) = EMA<sub>N/4</sub>( 4 · [ P<sub>t</sub> − mid<sub>N</sub>(t) ] / range<sub>N</sub>(t) )<span>(2)</span></div>
<p>where mid and range are the midpoint and width of the rolling <i>N</i>-day high–low channel. The same normalisation applies, with scale 1.2 and cap ±2.</p>
<h3 class="sub">4.2&nbsp;&nbsp;Long-only concentration</h3>
<p>With trend forecast <i>E<sub>t</sub></i> and breakout forecast <i>B<sub>t</sub></i>, and the concentration operator <i>c</i>(<i>x</i>) = normalised ( max(<i>x</i>, 0) )<sup>1.5</sup>, the final forecast is</p>
<div class="eq">F<sub>t</sub> = ½ [ 0.8 c( max(E<sub>t</sub><sup>+</sup>, B<sub>t</sub><sup>+</sup>) ) + 0.15 ] + ½ [ 0.8 c(B<sub>t</sub>) + 0.15 ], &nbsp;clipped to [0, 2].<span>(3)</span></div>
<p>The constant 0.15 is a resting long position held in the absence of any trend. The exponent 1.5 concentrates exposure in strong-signal states.</p>
<h3 class="sub">4.3&nbsp;&nbsp;Position sizing and execution</h3>
<p>The target position, as a multiple of equity, is <i>w<sub>t</sub></i> = <i>F<sub>t</sub></i> · τ / σ̂<sub>t</sub>, with τ = 10% annualised and σ̂<sub>t</sub> the exponentially weighted volatility of 4-hour returns (half-life 42 bars, annualised with 1,512 bars per year), capped at 8. A no-trade buffer of 10% of τ/σ̂<sub>t</sub> suppresses small rebalances. Positions decided at the close of bar <i>t</i> earn the return of bar <i>t</i>+1. Transaction costs are 1 basis point of notional per unit of turnover; Section 6 varies this from 0 to 10 bp.</p>
<h3 class="sub">4.4&nbsp;&nbsp;Benchmarks and sample split</h3>
<p>We compare against (a) gold buy-and-hold on the adjusted futures series, and (b) a constant long position with <i>F<sub>t</sub></i> ≡ 1 passed through the identical volatility-targeting engine. Benchmark (b) isolates the contribution of timing from that of volatility scaling. Because the slowest EWMAC rule requires roughly one year of data, all series begin on 1 June 2011. The out-of-sample period is June 2011 – December 2014; the design period is 2015 onward.</p>

<h2 class="sec">5&nbsp;&nbsp;Results</h2>
<h3 class="sub">5.1&nbsp;&nbsp;Out-of-sample and sub-period performance</h3>
<div class="tbl"><table class="bk"><thead><tr><th>Period / strategy</th><th>Total return</th><th>Sharpe</th><th>Max drawdown</th></tr></thead><tbody>{era_rows()}</tbody></table>
<p class="cap"><b>Table 1.</b> Performance by period. Sharpe ratios are annualised from daily returns. Strategies are scaled to a 10% annual volatility target; gold buy-and-hold is unscaled. The out-of-sample period precedes all data used to design the strategy.</p></div>
<p>Table 1 reports the central result. In the out-of-sample period the strategy returned +4.4% while gold lost 21.4% and the volatility-matched constant long lost 13.1%. Its maximum drawdown (−15.5%) was 38% of gold's. Over the bear market proper (September 2011 – December 2015) the strategy lost 5.6% against 42.2% for gold. In the design period it delivered a Sharpe of 0.83, equal to the constant long's, with a larger drawdown. Over the full sample its Sharpe of 0.69 exceeds both benchmarks, at well under half of gold's maximum drawdown.</p>
<figure>{svg("equity")}</figure>
<p class="cap"><b>Figure 1.</b> Growth of $10,000, June 2011 – November 2026, log scale. Shading marks the 2011–2015 gold bear market.</p>

<h3 class="sub">5.2&nbsp;&nbsp;Full-sample risk and return</h3>
<div class="tbl"><table class="bk"><thead><tr><th></th><th>Champion</th><th>Constant long</th><th>Gold B&amp;H</th></tr></thead><tbody>{stat_rows()}</tbody></table>
<p class="cap"><b>Table 2.</b> Full-sample statistics, June 2011 – November 2026, net of 1 bp costs. Confidence intervals: moving-block bootstrap with 21-day blocks and 2,000 resamples.</p></div>
<p>The strategy's return profile is characteristic of trend following: only {C['pct_pos_months']:.0f}% of months are positive, but monthly returns are strongly right-skewed (skewness +{C['skew_monthly']:.2f}, against +{K['skew_monthly']:.2f} for the constant long), with a best month of +{C['best_month']:.1f}% and a worst of {C['worst_month']:.1f}%. Realised volatility ({C['vol']:.1f}%) exceeds the 10% target because the forecast can reach 2.</p>
<figure>{svg("drawdown")}</figure>
<p class="cap"><b>Figure 2.</b> Drawdown from the running maximum.</p>

<h3 class="sub">5.3&nbsp;&nbsp;Behaviour in the bear market</h3>
<p>Between September 2011 and December 2015 the forecast averaged {BF['mean']:.2f}, sat at the resting level (below 0.2) on {BF['resting']:.0f}% of bars, and reached strong-trend territory (≥ 1.5) on {BF['strong']:.0f}%. Average exposure fell to {EXP['bear_mean']:.2f}× equity, against {EXP['post_mean']:.2f}× from 2016 onward. Figure 3 shows the mechanism: each counter-trend rally produced a short burst of exposure, which was unwound within weeks when the breakout failed. The strategy's bear-market losses are the sum of these false starts, each small relative to the decline it avoided.</p>
<figure>{svg("bear_forecast")}</figure>
<p class="cap"><b>Figure 3.</b> Adjusted gold futures price (top) and the strategy forecast F<sub>t</sub> (bottom), 2011–2016. The dashed line marks the 0.15 resting level.</p>

<h3 class="sub">5.4&nbsp;&nbsp;Timing versus volatility scaling</h3>
<p>To isolate timing, we compare the strategy's daily returns with those of the constant long rescaled to the same realised volatility. The mean difference has a paired <i>t</i>-statistic of +1.00 over the full sample (+1.29 out of sample, +1.39 in the bear market, +0.34 in the design period). A block bootstrap of the Sharpe-ratio difference gives a 95% interval of [{DVC['lo']:+.2f}, {DVC['hi']:+.2f}] and P(ΔSharpe ≤ 0) = {DVC['p_le0']:.2f}. Against gold buy-and-hold the corresponding interval is [{DVG['lo']:+.2f}, {DVG['hi']:+.2f}] with P(ΔSharpe ≤ 0) = {DVG['p_le0']:.3f}. The evidence that the strategy improves on owning gold is therefore considerably stronger than the evidence that its timing improves on a volatility-scaled constant position. Timing contributed mainly in the bear market, where it reduced drawdown; in the design period it added nothing to the Sharpe ratio.</p>
<figure>{svg("rolling_sharpe")}</figure>
<p class="cap"><b>Figure 4.</b> Rolling three-year Sharpe ratio (756 trading days). It was negative for windows ending in 2014–2015, between roughly 0.5 and 0.9 for windows ending 2016–2023, and above 1 only for windows ending in 2025–2026. The two longest drawdown spells lasted 3.3 years (October 2012 – February 2016) and 2.9 years (August 2020 – March 2024).</p>

<h2 class="sec">6&nbsp;&nbsp;Robustness</h2>
<div class="tbl"><table class="bk"><thead><tr><th>One-way cost</th><th>Sharpe</th></tr></thead><tbody>{cost_rows()}</tbody></table>
<p class="cap"><b>Table 3.</b> Full-sample Sharpe ratio as a function of the transaction cost per unit of turnover. Average turnover is {S['turnover_per_year']:.0f}× equity a year.</p></div>
<p><i>Transaction costs.</i> Table 3 shows that the result is insensitive to execution costs: at ten times the baseline cost the Sharpe declines by about 0.11. Financing, which the futures data embed, is the more material cost for a long-only gold strategy.</p>
<p><i>Data provenance.</i> On the common period 2017–2026 the strategy's returns on futures and on broker data correlate at 0.951. The difference in Sharpe (0.86 versus 1.12) is consistent with the financing embedded in futures and absent from the swap-free broker backtest.</p>
<p><i>Bar frequency and longer history.</i> In a companion test, a daily-frequency version of the same rules, with all speeds expressed in days, was run on continuous gold futures from 2002. Over 2002–2026 it achieved a Sharpe of 0.64 with a maximum drawdown of −17.3%, against 0.69 and −44.4% for buy-and-hold and 0.75 and −32.6% for a volatility-matched constant long, and it lost 11.2% in the 2011–2015 bear market while gold lost 42.0%. The 4-hour and daily results agree: similar risk-adjusted return to a volatility-scaled long position, with roughly half its drawdown.</p>
<figure>{svg("exposure")}</figure>
<p class="cap"><b>Figure 5.</b> Daily average position as a multiple of equity. Dashed line: sample mean ({EXP['mean']:.2f}×).</p>

<h2 class="sec">7&nbsp;&nbsp;Limitations</h2>
<p>First, fifteen years contains few independent trend episodes, so the Sharpe ratio's confidence interval ([{C['sharpe_ci'][0]:.2f}, {C['sharpe_ci'][1]:.2f}]) is wide. Second, although the parameters were fixed on post-2015 data, the strategy family was selected after extensive research on that same data, and this selection is not fully captured by the out-of-sample split. Third, back-adjusted futures differ from spot exposure through the term structure; our validation suggests the difference is small at the frequencies studied. Fourth, we model costs proportionally and ignore market impact, which is negligible at retail size in gold futures but not at institutional scale. Fifth, the strategy is single-asset; diversification across trend-following markets would change its risk profile substantially.</p>

<h2 class="sec">8&nbsp;&nbsp;Conclusion</h2>
<p>A long-only, volatility-targeted gold trend strategy designed on 2015–2026 data survives a clean out-of-sample test through the 2011–2015 bear market, preserving capital while gold lost a fifth of its value. Over fifteen years its Sharpe ratio is {C['sharpe']:.2f} with financing included, well below the in-sample estimate near 1.1 but roughly double that of buy-and-hold, and its maximum drawdown is less than half of gold's. Its advantage over a volatility-matched constant long position is directionally consistent but not statistically significant. The robust value of the strategy lies in drawdown control: it delivers gold's long-run return with materially smaller and shorter losses. For practitioners, the gap between the in-sample and out-of-sample Sharpe ratios is a reminder to size and evaluate such strategies on out-of-sample expectations.</p>

<h2 class="sec">References</h2>
<div class="refs">
<p>Bailey, D. H., &amp; López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. <i>Journal of Portfolio Management</i>, 40(5), 94–107.</p>
<p>Baur, D. G., &amp; Lucey, B. M. (2010). Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold. <i>Financial Review</i>, 45(2), 217–229.</p>
<p>Carver, R. (2015). <i>Systematic Trading: A unique new method for designing trading and investing systems.</i> Harriman House.</p>
<p>Erb, C. B., &amp; Harvey, C. R. (2013). The golden dilemma. <i>Financial Analysts Journal</i>, 69(4), 10–42.</p>
<p>Harvey, C. R., Liu, Y., &amp; Zhu, H. (2016). … and the cross-section of expected returns. <i>Review of Financial Studies</i>, 29(1), 5–68.</p>
<p>Hurst, B., Ooi, Y. H., &amp; Pedersen, L. H. (2017). A century of evidence on trend-following investing. <i>Journal of Portfolio Management</i>, 44(1), 15–29.</p>
<p>Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. <i>Annals of Statistics</i>, 17(3), 1217–1241.</p>
<p>Moreira, A., &amp; Muir, T. (2017). Volatility-managed portfolios. <i>Journal of Finance</i>, 72(4), 1611–1644.</p>
<p>Moskowitz, T. J., Ooi, Y. H., &amp; Pedersen, L. H. (2012). Time series momentum. <i>Journal of Financial Economics</i>, 104(2), 228–250.</p>
</div>

<h2 class="sec pb">Appendix A&nbsp;&nbsp;Annual returns</h2>
<div class="tbl"><table class="bk"><thead><tr><th>Year</th><th>Champion</th><th>Constant long</th><th>Gold B&amp;H</th></tr></thead><tbody>{year_rows()}</tbody></table>
<p class="cap"><b>Table A1.</b> Calendar-year returns in percent. * 2011 from 1 June; 2026 to 6 November. The strategy was positive in {S['pct_years_positive']:.0f}% of years and outperformed gold in {S['pct_years_beat_gold']:.0f}%.</p></div>
<p><i>Replication.</i> Data: Databento GLBX.MDP3, ohlcv-1h, symbol GC.v.0. Code: <span style="font-family:monospace;font-size:9pt">scripts/fetch_databento_ohlcv.py</span>, <span style="font-family:monospace;font-size:9pt">scripts/v5_champion_h4_cme.py</span>, <span style="font-family:monospace;font-size:9pt">scripts/v5_champion_paper_assets.py</span>; signal definition in <span style="font-family:monospace;font-size:9pt">src/v5/xau_dual_signals.py</span>.</p>
<p style="font-size:9pt;margin-top:18px"><i>Disclaimer.</i> This paper reports simulated historical results. It is not investment advice. Past performance, simulated or actual, does not guarantee future results.</p>
</body></html>"""


def main() -> None:
    blog = ROOT / "docs" / "blog" / "gold-champion-15-years.html"
    blog.parent.mkdir(parents=True, exist_ok=True)
    blog.write_text(BLOG)
    html = P / "gold-trend-cme-2011-2026.html"
    html.write_text(PAPER)
    pdf = P / "gold-trend-cme-2011-2026.pdf"
    subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", f"file://{html}"], check=True, capture_output=True, timeout=180)
    print(blog, len(BLOG))
    print(pdf, pdf.stat().st_size)


if __name__ == "__main__":
    main()
