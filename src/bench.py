"""Tiny benchmark: is CPU or Apple-GPU (MPS) faster for this small Q-network?"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .dqn import QNet
from .features import N_FEATURES


def bench_devices(batch=128, n=300):
    torch.set_num_threads(1)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    out = {}
    for dev in devices:
        net, tgt = QNet().to(dev), QNet().to(dev)
        opt = torch.optim.Adam(net.parameters(), 5e-4)
        s, s2 = torch.randn(batch, N_FEATURES, device=dev), torch.randn(batch, N_FEATURES, device=dev)
        a = torch.randint(0, 5, (batch,), device=dev)
        r = torch.randn(batch, device=dev)
        x = torch.randn(32, N_FEATURES)

        def update():
            q = net(s).gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                y = r + 0.95 * tgt(s2).gather(1, net(s2).argmax(1, keepdim=True)).squeeze(1)
            loss = F.mse_loss(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss.item()

        for _ in range(30):
            update()
        t = time.time()
        for _ in range(n):
            update()
        upd = (time.time() - t) / n * 1e3
        t = time.time()
        for _ in range(n):
            with torch.no_grad():
                net(x.to(dev)).argmax(1).cpu().numpy()
        act = (time.time() - t) / n * 1e3
        out[dev] = {"update_ms": round(upd, 3), "act_ms": round(act, 3)}
    best = min(out, key=lambda d: out[d]["update_ms"] + out[d]["act_ms"])
    return best, out
