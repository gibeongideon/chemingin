#!/usr/bin/env python3
"""KRONOS, PAPER PROTOCOL (V5_FINDINGS 3ew, PREREG_ADDENDUM_1.md, commit 5ad0167).
Multi-step sampled paths exactly as the authors evaluate (arXiv 2508.02739 Tables 6 and 8):
T = 0.6, top-p = 0.9, N = 10 paths, forecast = mean decoded path; H1 look-back 80 / horizon 12,
D1 look-back 90 / horizon 10. OHLC only (the paper's own forex setting). Blind to outcomes.

The reference `auto_regressive_inference` re-runs the whole window for every generated bar. This
keeps per-layer keys/values instead; `verify` asserts the cached next-token logits equal the
vendored model's on every generated position.

    venv/bin/python scripts/v5_fm_kronos_paths.py small verify
    venv/bin/python scripts/v5_fm_kronos_paths.py small oanda KPd,KPn,KPh 4
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "kronos"))
from scripts.v5_fm_features import NY, OUT, _stamps, build_frames  # noqa: E402

T_, TOP_P, N_PATH, POST = 0.6, 0.9, 10, "2024-07-01"
SH, NSH = (int(v) for v in (sys.argv[6].split("/") if len(sys.argv) > 6 else ("0", "1")))   # shard i of n
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 1      # compute only: sample every k-th H1 decision


class KronosPaths:
    def __init__(self, size: str, threads: int):
        import torch  # noqa: PLC0415
        from model import Kronos, KronosTokenizer  # noqa: PLC0415
        from model.kronos import sample_from_logits  # noqa: PLC0415
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        self.torch, self.F, self.sample = torch, torch.nn.functional, sample_from_logits
        self.tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
        self.m = Kronos.from_pretrained(f"NeoQuasar/Kronos-{size}").eval()

    @staticmethod
    def _rot(x, cos, sin):
        import torch  # noqa: PLC0415
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def _cs(self, att, a, b):
        torch = self.torch
        f = torch.einsum("i,j->ij", torch.arange(a, b).type_as(att.rotary.inv_freq), att.rotary.inv_freq)
        e = torch.cat((f, f), dim=-1)
        return e.cos(), e.sin()

    def _embed(self, s1, s2, stamp):
        return self.m.embedding([s1, s2]) + self.m.time_emb(stamp)

    def _step(self, x, cache, p0):
        """x [B,L,d] new positions starting at absolute index p0; cache = per-layer (K,V) or None."""
        torch, F = self.torch, self.F
        B, Ln, d = x.shape
        new = []
        for li, layer in enumerate(self.m.transformer):
            att = layer.self_attn
            H, hd = att.n_heads, att.head_dim
            xn = layer.norm1(x)
            pr = lambda lin: lin(xn).view(B, Ln, H, hd).transpose(1, 2)  # noqa: E731
            q, k, v = pr(att.q_proj), pr(att.k_proj), pr(att.v_proj)
            cos, sin = self._cs(att, p0, p0 + Ln)
            q, k = self._rot(q, cos, sin), self._rot(k, cos, sin)
            if cache is None:
                K, V = k, v
                a = F.scaled_dot_product_attention(q, K, V, is_causal=True)
            else:
                K, V = torch.cat([cache[li][0], k], 2), torch.cat([cache[li][1], v], 2)
                a = F.scaled_dot_product_attention(q, K, V)             # one new position sees everything before it
            x = x + att.out_proj(a.transpose(1, 2).reshape(B, Ln, d))
            x = x + layer.ffn(layer.norm2(x))
            new.append((K, V))
        return self.m.norm(x), new

    def _kv(self, hid):
        """Cross-attention keys/values of final hidden states [B,L,d] (cached across steps by the caller)."""
        ca = self.m.dep_layer.cross_attn
        B, L, _ = hid.shape
        return (ca.k_proj(hid).view(B, L, ca.n_heads, ca.head_dim).transpose(1, 2),
                ca.v_proj(hid).view(B, L, ca.n_heads, ca.head_dim).transpose(1, 2))

    def _s2(self, hid, s1, kv=None):
        """hid [B,L,d] final hidden states (or only the last one when kv is given), s1 [B] -> s2 logits.
        Mirrors model.decode_s2 with a length-1 query."""
        F, m = self.F, self.m
        ca = m.dep_layer.cross_attn
        B, d = hid.shape[0], hid.shape[2]
        q = ca.q_proj(m.embedding.emb_s1(s1)).view(B, 1, ca.n_heads, ca.head_dim).transpose(1, 2)
        k, v = kv if kv is not None else self._kv(hid)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, d)
        return m.head.cond_forward(m.dep_layer.norm(hid[:, -1] + ca.out_proj(a)))

    def paths(self, x: np.ndarray, st_x: np.ndarray, st_y: np.ndarray, n: int = N_PATH):
        """x [T,4] raw OHLC, stamps [T,5] and [H,5] -> (paths [n,H,4] in price units, embedding [d])."""
        torch = self.torch
        with torch.inference_mode():
            x6 = np.concatenate([x, np.zeros((len(x), 2))], -1).astype(np.float32)
            mu, sd = x6.mean(0), x6.std(0)
            xn = torch.from_numpy(np.clip((x6 - mu) / (sd + 1e-5), -5, 5))[None]
            pre, post = self.tok.encode(xn, half=True)
            T, H = pre.shape[1], len(st_y)
            sx, sy = torch.from_numpy(st_x)[None], torch.from_numpy(st_y)
            h, cache = self._step(self._embed(pre, post, sx), None, 0)
            emb = h[0, -1].numpy().copy()
            cache = [(K.expand(n, -1, -1, -1), V.expand(n, -1, -1, -1)) for K, V in cache]
            hid = h[:, -1:].expand(n, -1, -1)
            kc, vc = (u.expand(n, -1, -1, -1) for u in self._kv(h))
            g1, g2 = torch.empty(n, H, dtype=torch.long), torch.empty(n, H, dtype=torch.long)
            for j in range(H):
                s1 = self.sample(self.m.head(hid[:, -1]), temperature=T_, top_k=0, top_p=TOP_P)[:, 0]
                s2 = self.sample(self._s2(hid, s1, (kc, vc)), temperature=T_, top_k=0, top_p=TOP_P)[:, 0]
                g1[:, j], g2[:, j] = s1, s2
                if j < H - 1:
                    hn, cache = self._step(self._embed(s1[:, None], s2[:, None], sy[j][None, None].expand(n, 1, -1)), cache, T + j)
                    hid = hn
                    kn, vn = self._kv(hn)
                    kc, vc = torch.cat([kc, kn], 2), torch.cat([vc, vn], 2)
            z = self.tok.decode([torch.cat([pre.expand(n, -1), g1], 1), torch.cat([post.expand(n, -1), g2], 1)], half=True)
            out = z[:, -H:, :4].numpy() * (sd[:4] + 1e-5) + mu[:4]
        return out, emb

    def verify(self) -> None:
        torch = self.torch
        rng = np.random.default_rng(2)
        T, H, B = 40, 6, 3
        s1 = torch.from_numpy(rng.integers(0, 1024, (B, T + H)))
        s2 = torch.from_numpy(rng.integers(0, 1024, (B, T + H)))
        st = torch.from_numpy(np.stack([rng.integers(0, hi, (B, T + H)) for hi in (60, 24, 7, 28, 12)], -1).astype(np.float32))
        worst = 0.0
        with torch.inference_mode():
            ref_logits, ref_ctx = self.m.decode_s1(s1, s2, st)
            h, cache = self._step(self._embed(s1[:, :T], s2[:, :T], st[:, :T]), None, 0)
            hid = h
            for j in range(H):
                worst = max(worst, float((self.m.head(hid[:, -1]) - ref_logits[:, T + j - 1]).abs().max()))
                probe = torch.from_numpy(rng.integers(0, 1024, B))
                ref2 = self.m.decode_s2(ref_ctx[:, :T + j].contiguous(), probe[:, None])[:, -1]
                worst = max(worst, float((self._s2(hid, probe) - ref2).abs().max()))
                hn, cache = self._step(self._embed(s1[:, T + j:T + j + 1], s2[:, T + j:T + j + 1], st[:, T + j:T + j + 1]), cache, T + j)
                hid = torch.cat([hid, hn], 1)
        assert worst < 1e-3, f"cached path differs from the reference by {worst:.2e}"
        print(f"VERIFY OK: cached generation == reference logits on {H} positions (max diff {worst:.2e})", flush=True)


def jobs(source: str, kind: str):
    """Yields (key timestamp, context OHLC, context stamps, future stamps)."""
    fr = build_frames(source)
    if kind == "KPd":
        b = fr["D1"].reset_index(drop=True)
        look, hor, idx = 90, 10, np.arange(90, len(b))
    else:
        b = fr["H1"].reset_index(drop=True)
        look, hor = 80, 12
        t = pd.DatetimeIndex(b.t_open)
        et = t.tz_localize("UTC").tz_convert(NY).tz_localize(None)
        idx = np.flatnonzero(et.hour == 8) if kind == "KPn" else np.flatnonzero(t >= pd.Timestamp(POST))
        idx = idx[idx >= look]
        if kind == "KPh" and STRIDE > 1:
            idx = idx[::STRIDE]
    X = b[["open", "high", "low", "close"]].to_numpy()
    tt = pd.DatetimeIndex(b.t_open)
    if os.environ.get("KP_FROM"):                                 # compute only: restrict to recent decisions
        idx = idx[tt[idx] >= pd.Timestamp(os.environ["KP_FROM"])]
    ST = _stamps(tt)
    step = tt[1:] - tt[:-1]
    med = step.to_series().median()
    for i in idx:
        fut = ST[i:i + hor]
        if len(fut) < hor:                                        # the last decisions: nominal future stamps
            ext = pd.DatetimeIndex([tt[-1] + med * (q + 1) for q in range(hor - len(fut))])
            fut = np.vstack([fut, _stamps(ext)]) if len(fut) else _stamps(ext)
        yield tt[i], X[i - look:i], ST[i - look:i], fut


def main() -> None:
    size, source = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    eng = KronosPaths(size, threads)
    eng.verify()
    if source == "verify":
        return
    for kind in sys.argv[3].split(","):
        rows, embs, t0 = [], [], time.time()
        for n, (key, x, sx, sy) in enumerate(jobs(source, kind)):
            if n % NSH != SH:
                continue
            p, e = eng.paths(x, sx, sy)
            c = p[..., 3].mean(0) / x[-1, 3] - 1                  # mean path, close, vs the last context close
            rows.append((key, x[-1, 3], c[0], c[7] if len(c) > 7 else np.nan, c[-1], p[..., 3].mean(0).mean() / x[-1, 3] - 1,
                         float(((p[..., 1].max(1) - p[..., 2].min(1)) / x[-1, 3]).mean())))
            if kind == "KPh" and size == "small":
                embs.append(e.astype(np.float32))
            if n % 700 == SH:
                print(f"  {size} {source} {kind}: {n:,}  {(n + 1) / (time.time() - t0):.1f}/s", flush=True)
        df = pd.DataFrame(rows, columns=["t_open", "last_close", "mu1", "mu8", "muH", "mu_avg", "rngH"]).set_index("t_open")
        df.to_parquet(OUT / f"kp_{size}_{source}_{kind}_s{SH}.parquet")
        if embs:
            np.savez_compressed(OUT / f"kp_{size}_{source}_{kind}_emb_s{SH}.npz", emb=np.vstack(embs), t=df.index.values.astype("int64"))
        print(f"saved kp_{size}_{source}_{kind}: {len(df):,} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
