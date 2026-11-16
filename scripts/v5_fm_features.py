#!/usr/bin/env python3
"""FOUNDATION-MODEL FEATURES for the intraday day-signal study (V5_FINDINGS 3ew).

User, 2026-11-16: "i still cannot believe simple LLM model cannot assist, as a feature, maybe we
need better data manipulation". Prior LLM attempts here: Claude reading numeric summaries (3n),
a hand-made 45-token candle alphabet + n-gram (3cc), a small BERT trained on gold alone (3dl).
None used a PRETRAINED model. This script produces zero-shot, one-step-ahead forecasts from

  * Kronos-small  (arXiv 2508.02739): a candle-native tokenizer + decoder pretrained on ~12B
    K-lines from 45 exchanges. Code vendored unmodified in third_party/kronos (MIT).
  * Chronos-Bolt-small (Amazon): a generic pretrained time-series model.

It is deliberately BLIND TO OUTCOMES: it writes features only, indexed by the bar being
predicted. Scoring is in scripts/v5_fm_day_signal.py. Design and bars are fixed in
data/v5_runs/intraday_day_signal/PREREGISTRATION.md (commit 264ebb8).

Frames (bar being predicted <- context of bars that closed before it opened):
  D1  17:00 ET -> 17:00 ET day            S8  8h ET bars, predict the [08-16) NY bar
  H1  60-min bars                         F24 24 consecutive H1 bars ending at a ZigZag fire

    venv/bin/python scripts/v5_fm_features.py chronos oanda          # minutes
    venv/bin/python scripts/v5_fm_features.py kronos  oanda D1,S8    # ~1h on 8 CPU threads
    venv/bin/python scripts/v5_fm_features.py kronos  verify         # fast path == reference
Needs: einops, chronos-forecasting (a venv with --system-site-packages is enough).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "intraday_day_signal"
K_CTX, K_S, K_TOP_P, K_T = 256, 32, 0.9, 1.0          # Kronos: README defaults, S fixed in the prereg
C_CTX = 512                                            # Chronos context
H1_FROM = "2024-07-01"                                 # Kronos H1 window (compute)
NY = "America/New_York"


# ------------------------------------------------------------------ frames (no outcome is touched)
def _et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize("UTC").tz_convert(NY).tz_localize(None)


def _agg(d: pd.DataFrame, key, min_n: int) -> pd.DataFrame:
    g = d.groupby(key, sort=True)
    b = pd.DataFrame(dict(open=g.open.first(), high=g.high.max(), low=g.low.min(), close=g.close.last(),
                          n=g.close.size(), t_open=g.t.first(), t_close=g.t.last()))
    return b[b.n >= min_n]


def load_base(source: str) -> tuple[pd.DataFrame, int]:
    """Finest bars for a source, UTC index, with the minimum bar counts that make a D1 / S8 bar."""
    if source == "oanda":
        d = pd.read_parquet(ROOT / "data/oanda_hf/XAUUSD_M1.parquet")
        d.index = pd.to_datetime(d.time, unit="s")
        return d[["open", "high", "low", "close"]].astype("float64"), 60
    from scripts.v5_zigzag_panel_probs import load_cme  # noqa: PLC0415
    return load_cme("GC")[["open", "high", "low", "close"]], 1


def build_frames(source: str) -> dict[str, pd.DataFrame]:
    """Each frame: OHLC bars in time order with `t_open` (UTC stamp of the first base bar) and a
    boolean `target` column marking the bars to be predicted."""
    d, per_hour = load_base(source)
    d = d.assign(t=d.index)
    et = _et(d.index)
    day = (et + pd.Timedelta(hours=7)).normalize()                 # 17:00 ET opens the next day
    D1 = _agg(d, day, 10 * per_hour).assign(target=True)
    D1.index.name = "day"
    S8 = _agg(d, [pd.Index(et.normalize(), name="date"), pd.Index(et.hour // 8, name="blk")], per_hour)
    S8 = S8.assign(target=S8.index.get_level_values("blk") == 1)   # the [08-16) ET bar = NY session
    out = {"D1": D1, "S8": S8}
    if source == "oanda":
        h = pd.read_parquet(ROOT / "data/oanda_hf/XAUUSD_H1.parquet")
        h.index = pd.to_datetime(h.time, unit="s")
    else:
        h = d
    H1 = h[["open", "high", "low", "close"]].astype("float64").assign(t_open=h.index, target=True)
    out["H1"] = H1
    return out


def fire_frames(src: str) -> tuple[pd.DataFrame, np.ndarray]:
    """H1 bars of a ZigZag source and the bar positions of its variant-B + breaker fires (exactly
    the event set of scripts/v5_zigzag_prob_sizing.py; every qualifying fire bar, not only the
    ones that become trades, so the feature exists whichever slot rule is applied later)."""
    from scripts.v5_zigzag_fall_gate import circuit, daily  # noqa: PLC0415
    from scripts.v5_zigzag_long_history import OUT as ZZ, SRC, THR, load  # noqa: PLC0415
    cfg = SRC[src]
    g = load(src, cfg["gold"])
    p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
    fire = np.isfinite(p) & (p >= THR) & ~circuit(g, daily(g), 1.0, "day") & (g.index >= pd.Timestamp(cfg["start"]))
    return g, np.flatnonzero(fire)


def blocks_24(g: pd.DataFrame, i: int, n_blocks: int) -> np.ndarray | None:
    """n_blocks bars of 24 consecutive H1 bars each, the last one ENDING at position i."""
    lo = i + 1 - 24 * n_blocks
    if lo < 0:
        return None
    w = g.iloc[lo:i + 1]
    o, h, l, c = (w[k].to_numpy().reshape(n_blocks, 24) for k in ("open", "high", "low", "close"))
    return np.column_stack([o[:, 0], h.max(1), l.min(1), c[:, -1]])


# ------------------------------------------------------------------ Kronos
def _stamps(t: pd.DatetimeIndex) -> np.ndarray:
    return np.column_stack([t.minute, t.hour, t.weekday, t.day, t.month]).astype(np.float32)


class KronosNext:
    """One autoregressive step = the next candle, S samples per decision. Identical to the
    reference `auto_regressive_inference` for pred_len = 1, except that the S samples of a
    decision share ONE transformer pass (they share the context) and the tokenizer decodes only
    the last position with cached prefix keys/values. `verify()` asserts both against the
    reference code path."""

    def __init__(self, threads: int = 8):
        import torch  # noqa: PLC0415
        sys.path.insert(0, str(ROOT / "third_party" / "kronos"))
        from model import Kronos, KronosTokenizer  # noqa: PLC0415
        from model.kronos import sample_from_logits  # noqa: PLC0415
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        self.torch, self.sample = torch, sample_from_logits
        self.tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
        self.model = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()

    # -- pieces
    def _rope(self, inv_freq, L):
        torch = self.torch
        f = torch.einsum("i,j->ij", torch.arange(L).type_as(inv_freq), inv_freq)
        e = torch.cat((f, f), dim=-1)
        return e.cos(), e.sin()

    @staticmethod
    def _rot(x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + self_cat(-x2, x1) * sin

    def decode_last(self, pre_ctx, post_ctx, pre_c, post_c):
        """Decoded candle at position T for S candidates that share the T context tokens."""
        torch, F, tok = self.torch, self.torch.nn.functional, self.tok
        T, S = pre_ctx.shape[0], pre_c.shape[0]
        zp = tok.post_quant_embed(tok.indices_to_bits([pre_ctx[None], post_ctx[None]], half=True))
        zc = tok.post_quant_embed(tok.indices_to_bits([pre_c[:, None], post_c[:, None]], half=True))
        d = zp.shape[-1]
        for layer in tok.decoder:
            att = layer.self_attn
            H, hd = att.n_heads, att.head_dim
            cos, sin = self._rope(att.rotary.inv_freq, T + 1)
            xp, xc = layer.norm1(zp), layer.norm1(zc)
            pr = lambda lin, x: lin(x).view(x.shape[0], x.shape[1], H, hd).transpose(1, 2)  # noqa: E731
            qP, kP, vP = pr(att.q_proj, xp), pr(att.k_proj, xp), pr(att.v_proj, xp)
            qC, kC, vC = pr(att.q_proj, xc), pr(att.k_proj, xc), pr(att.v_proj, xc)
            qP, kP = self._rot(qP, cos[:T], sin[:T]), self._rot(kP, cos[:T], sin[:T])
            qC, kC = self._rot(qC, cos[T:], sin[T:]), self._rot(kC, cos[T:], sin[T:])
            aP = F.scaled_dot_product_attention(qP, kP, vP, is_causal=True)
            aC = F.scaled_dot_product_attention(qC, torch.cat([kP.expand(S, -1, -1, -1), kC], 2),
                                                torch.cat([vP.expand(S, -1, -1, -1), vC], 2))
            zp = zp + att.out_proj(aP.transpose(1, 2).reshape(1, T, d))
            zc = zc + att.out_proj(aC.transpose(1, 2).reshape(S, 1, d))
            zp, zc = zp + layer.ffn(layer.norm2(zp)), zc + layer.ffn(layer.norm2(zc))
        return tok.head(zc)[:, 0]

    def s2_logits_last(self, ctx, s1):
        """s2 logits at the last position for S sampled s1 tokens that share one context.
        Mirrors model.decode_s2: the cross-attention query is the s1 embedding (a length-1
        sequence, so its rotary position is 0 = identity) over the context's hidden states."""
        torch, F, m = self.torch, self.torch.nn.functional, self.model
        ca = m.dep_layer.cross_attn
        H, hd, S, T = ca.n_heads, ca.head_dim, s1.shape[0], ctx.shape[0]
        q = ca.q_proj(m.embedding.emb_s1(s1)).view(S, 1, H, hd).transpose(1, 2)
        k = ca.k_proj(ctx).view(1, T, H, hd).transpose(1, 2).expand(S, -1, -1, -1)
        v = ca.v_proj(ctx).view(1, T, H, hd).transpose(1, 2).expand(S, -1, -1, -1)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(S, ca.d_model)
        return m.head.cond_forward(m.dep_layer.norm(ctx[-1][None] + ca.out_proj(a)))

    def next_candles(self, x: np.ndarray, st: np.ndarray, S: int = K_S) -> np.ndarray:
        """x [B,T,4] raw OHLC, st [B,T,5] stamps -> [B,S,4] sampled next candles in price units."""
        torch = self.torch
        with torch.inference_mode():
            x6 = np.concatenate([x, np.zeros(x.shape[:2] + (2,))], -1).astype(np.float32)
            mu, sd = x6.mean(1, keepdims=True), x6.std(1, keepdims=True)
            xn = torch.from_numpy(np.clip((x6 - mu) / (sd + 1e-5), -5, 5))
            pre, post = self.tok.encode(xn, half=True)
            s1_logits, ctx = self.model.decode_s1(pre, post, torch.from_numpy(st))
            out = np.empty((x.shape[0], S, 4))
            for b in range(x.shape[0]):
                s1 = self.sample(s1_logits[b, -1][None].repeat(S, 1), temperature=K_T, top_k=0, top_p=K_TOP_P)[:, 0]
                s2 = self.sample(self.s2_logits_last(ctx[b], s1), temperature=K_T, top_k=0, top_p=K_TOP_P)[:, 0]
                z = self.decode_last(pre[b], post[b], s1, s2).numpy()
                out[b] = (z * (sd[b] + 1e-5) + mu[b])[:, :4]
        return out

    def verify(self) -> None:
        """Fast path vs the vendored reference, on random-walk candles."""
        torch = self.torch
        rng = np.random.default_rng(1)
        B, T, S = 3, 64, 8
        c = 2000 * np.exp(np.cumsum(rng.normal(0, 0.003, (B, T)), 1))
        o = np.roll(c, 1, 1); o[:, 0] = c[:, 0]
        x = np.stack([o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c], -1)
        x6 = np.concatenate([x, np.zeros((B, T, 2))], -1).astype(np.float32)
        xn = torch.from_numpy(np.clip((x6 - x6.mean(1, keepdims=True)) / (x6.std(1, keepdims=True) + 1e-5), -5, 5))
        st = torch.from_numpy(np.stack([rng.integers(0, hi, (B, T)) for hi in (60, 24, 7, 28, 12)], -1).astype(np.float32))
        with torch.inference_mode():
            pre, post = self.tok.encode(xn, half=True)
            s1_logits, ctx = self.model.decode_s1(pre, post, st)
            worst = 0.0
            for b in range(B):
                s1 = torch.from_numpy(rng.integers(0, 1024, S))
                s2 = torch.from_numpy(rng.integers(0, 1024, S))
                ref_s2 = self.model.decode_s2(ctx[b][None].expand(S, -1, -1).contiguous(), s1[:, None])[:, -1, :]
                d1 = float((ref_s2 - self.s2_logits_last(ctx[b], s1)).abs().max())
                full = [torch.cat([pre[b][None].expand(S, -1), s1[:, None]], 1),
                        torch.cat([post[b][None].expand(S, -1), s2[:, None]], 1)]
                ref = self.tok.decode(full, half=True)[:, -1]
                d2 = float((ref - self.decode_last(pre[b], post[b], s1, s2)).abs().max())
                worst = max(worst, d1, d2)
                print(f"  decision {b}: max|s2 logits diff| {d1:.2e}   max|decoded candle diff| {d2:.2e}")
        assert worst < 1e-4, f"fast path differs from the reference by {worst:.2e}"
        print(f"VERIFY OK: fast path == reference (max diff {worst:.2e})")


def self_cat(a, b):
    import torch  # noqa: PLC0415
    return torch.cat((a, b), dim=-1)


def summarise(samples: np.ndarray, last: np.ndarray, pre: str) -> dict:
    """samples [B,S,4] OHLC, last [B] = last context close."""
    o, h, l, c = (samples[..., k] for k in range(4))
    L = last[:, None]
    return {f"{pre}mu": (c / L - 1).mean(1), f"{pre}med": np.median(c / L - 1, 1), f"{pre}pup": (c > L).mean(1),
            f"{pre}rng": ((h - l) / L).mean(1), f"{pre}skew": (((h - L) - (L - l)) / L).mean(1)}


def kronos_frame(kn: KronosNext, bars: pd.DataFrame, targets: np.ndarray, tag: str, batch: int = 24) -> pd.DataFrame:
    X = bars[["open", "high", "low", "close"]].to_numpy()
    ST = _stamps(pd.DatetimeIndex(bars.t_open))
    targets = targets[targets >= K_CTX]
    rows, t0 = [], time.time()
    for s in range(0, len(targets), batch):
        tg = targets[s:s + batch]
        x = np.stack([X[i - K_CTX:i] for i in tg])
        st = np.stack([ST[i - K_CTX:i] for i in tg])
        f = summarise(kn.next_candles(x, st), x[:, -1, 3], "k_")
        rows.append(pd.DataFrame(f, index=tg))
        if (s // batch) % 10 == 0:
            done = s + len(tg)
            print(f"  kronos {tag}: {done:,}/{len(targets):,}  {done / (time.time() - t0):.1f}/s  "
                  f"eta {(len(targets) - done) / max(done / (time.time() - t0), 1e-9) / 60:.0f} min", flush=True)
    return pd.concat(rows)


# ------------------------------------------------------------------ Chronos
def chronos_frame(pipe, close: np.ndarray, targets: np.ndarray, batch: int = 256) -> pd.DataFrame:
    import torch  # noqa: PLC0415
    targets = targets[targets >= 64]
    rows = []
    for s in range(0, len(targets), batch):
        tg = targets[s:s + batch]
        ctx = np.full((len(tg), C_CTX), np.nan, dtype=np.float32)
        for r, i in enumerate(tg):
            w = close[max(0, i - C_CTX):i]
            ctx[r, C_CTX - len(w):] = w                           # left-padded with NaN = "missing"
        q, _ = pipe.predict_quantiles(torch.from_numpy(ctx), prediction_length=1, quantile_levels=[0.1, 0.5, 0.9])
        q = q[:, 0, :].numpy().astype(float)
        last = close[tg - 1]
        rows.append(pd.DataFrame({"c_mu": q[:, 1] / last - 1, "c_rng": (q[:, 2] - q[:, 0]) / last,
                                  "c_skew": (q[:, 2] + q[:, 0] - 2 * q[:, 1]) / last}, index=tg))
    return pd.concat(rows)


def chronos_pipe(threads: int = 8):
    import torch  # noqa: PLC0415
    from chronos import BaseChronosPipeline  # noqa: PLC0415
    torch.set_num_threads(threads)
    return BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)


# ------------------------------------------------------------------ driver
def save(df: pd.DataFrame, bars: pd.DataFrame, name: str) -> None:
    out = df.copy()
    out.insert(0, "t_open", pd.DatetimeIndex(bars.t_open)[df.index])
    out.insert(1, "last_close", bars.close.to_numpy()[df.index - 1])
    out.index.name = "pos"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"fm_{name}.parquet")
    print(f"saved fm_{name}.parquet: {len(out):,} rows, {out.t_open.iloc[0]} -> {out.t_open.iloc[-1]}", flush=True)


def main() -> None:
    model, source = sys.argv[1], sys.argv[2]
    want = sys.argv[3].split(",") if len(sys.argv) > 3 else ["D1", "S8", "H1"]
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    if model == "kronos" and source == "verify":
        return KronosNext(threads).verify()
    eng = KronosNext(threads) if model == "kronos" else chronos_pipe(threads)
    if model == "kronos":
        eng.verify()
    if source in ("oanda", "cme"):
        fr = build_frames(source)
        for name in want:
            bars = fr[name].reset_index(drop=True)
            tg = np.flatnonzero(bars.target.to_numpy())
            if name == "H1" and model == "kronos":
                tg = tg[pd.DatetimeIndex(bars.t_open)[tg] >= pd.Timestamp(H1_FROM)]
            df = (kronos_frame(eng, bars, tg, f"{source}/{name}") if model == "kronos"
                  else chronos_frame(eng, bars.close.to_numpy(), tg))
            save(df, bars, f"{model}_{source}_{name}")
    else:                                                       # zz_cme / zz_switch: features at fires
        src = source.split("_")[1]
        g, fires = fire_frames(src)
        rows = {}
        t0 = time.time()
        for n, i in enumerate(fires):
            if model == "kronos":
                b = blocks_24(g, int(i), K_CTX)
                if b is None:
                    continue
                st = _stamps(g.index[int(i) + 1 - 24 * K_CTX:int(i) + 1:24])[None]
                rows[g.index[i]] = {k: float(v[0]) for k, v in summarise(eng.next_candles(b[None], st), b[-1:, 3], "k_").items()}
            else:
                nb = min(C_CTX, (int(i) + 1) // 24)
                b = blocks_24(g, int(i), nb)
                df = chronos_frame(eng, np.append(b[:, 3], np.nan), np.array([nb]))
                rows[g.index[i]] = df.iloc[0].to_dict() | {"n_blocks": nb}
            if n % 200 == 0:
                print(f"  {model} fires {src}: {n:,}/{len(fires):,}  {(n + 1) / (time.time() - t0):.1f}/s", flush=True)
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "t_fire"
        out.to_parquet(OUT / f"fm_{model}_zz_{src}.parquet")
        print(f"saved fm_{model}_zz_{src}.parquet: {len(out):,} fires", flush=True)


if __name__ == "__main__":
    sys.exit(main())
