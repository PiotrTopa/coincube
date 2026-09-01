"""3D signed dressed walk (Phase W): single-carrier sector, gauge-signed paths.

Bit dynamics: split-step X/Y/Z cycles of rule-891 conditional propagation, each axis
against its own medium. The fermionic content is entirely in the path-sign gauge
(base / ks / stall / ks+stall), which never alters the bit dynamics.

Architecture note (J26 follow-up): with media streaming only during their own axis
cycle, each medium is frozen 4 of 6 sub-steps and the walk DECELERATES (partial
frozen-trap; measured: per-cycle increments 1.69 -> 0.63 and falling). Default is
therefore ``continuous=True``: every medium streams every sub-step, so the carrier
meets fresh bits at each engagement and per-axis ballisticity is restored.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as xp
    GPU = True
except Exception:  # pragma: no cover
    import numpy as xp
    GPU = False


def make_evolve(L: int, E: int, TCYC: int, continuous: bool = True,
                axes=(0, 1, 2), lane_media: bool = False):
    """Returns evolve(q, gauge, seed) -> (traj, signs), lists per cycle (len TCYC+1).

    gauge substrings: "ks" applies Kogut-Susskind axis phases per hop
    (eta_x = +1, eta_y = (-1)^x, eta_z = (-1)^(x+y)); "stall" applies sigma = -1 per
    stall event.

    axes: which axes get carrier sub-steps (controls: (0,) = 1D restriction).
    lane_media: medium for axis a depends on coordinate a only (constant across the
    transverse plane), so the carrier's transverse motion never changes which bits it
    engages -- per-axis dynamics is then exactly the 1D rule-891 co-motion.
    """

    def evolve(q: float, gauge: str, seed: int = 3):
        r = np.random.default_rng(seed)
        if lane_media:
            lanes = r.random((E, 3, L)) < q
            med = xp.empty((E, 3, L, L, L), dtype=bool)
            med[:, 0] = xp.asarray(lanes[:, 0])[:, :, None, None] * xp.ones(
                (1, 1, L, L), dtype=bool)
            med[:, 1] = xp.asarray(lanes[:, 1])[:, None, :, None] * xp.ones(
                (1, L, 1, L), dtype=bool)
            med[:, 2] = xp.asarray(lanes[:, 2])[:, None, None, :] * xp.ones(
                (1, L, L, 1), dtype=bool)
        else:
            med = xp.asarray(r.random((E, 3, L, L, L)) < q)
        pos = xp.zeros((E, 3), dtype=xp.int64) + L // 2
        sign = xp.ones(E, dtype=xp.int64)
        traj = [pos.copy()]
        sgns = [sign.copy()]
        idx = xp.arange(E)

        def stream_medium(b: int, o: int) -> None:
            m = xp.moveaxis(med[:, b], 1 + b, -1)
            if o:
                m = xp.roll(m, -1, axis=-1)
            m0 = m[..., 0::2].copy()
            m[..., 0::2] = m[..., 1::2]
            m[..., 1::2] = m0
            if o:
                m = xp.roll(m, 1, axis=-1)
            med[:, b] = xp.moveaxis(m, -1, 1 + b)

        def substep(a: int, o: int) -> None:
            nonlocal pos, sign
            u = pos[:, a]
            base = (u - o) % L
            partner = ((base ^ 1) + o) % L
            u0 = (base - (base % 2) + o) % L
            u1 = (u0 + 1) % L
            sel0 = [pos[:, 0], pos[:, 1], pos[:, 2]]
            sel1 = [pos[:, 0], pos[:, 1], pos[:, 2]]
            sel0[a] = u0
            sel1[a] = u1
            b0 = med[idx, a, sel0[0], sel0[1], sel0[2]]
            b1 = med[idx, a, sel1[0], sel1[1], sel1[2]]
            par_odd = (b0.astype(xp.int64) + b1.astype(xp.int64)) % 2 == 1
            move = ~par_odd
            if "stall" in gauge:
                sign *= xp.where(par_odd, -1, 1)
            if "ks" in gauge and a > 0:
                if a == 1:
                    eta = 1 - 2 * (pos[:, 0] % 2)
                else:
                    eta = 1 - 2 * ((pos[:, 0] + pos[:, 1]) % 2)
                sign *= xp.where(move, eta, 1)
            newpos = pos.copy()
            newpos[:, a] = xp.where(move, partner, u)
            pos = newpos
            # media stream AFTER the carrier reads (1D table semantics)
            if continuous:
                for b in range(3):
                    stream_medium(b, o)
            else:
                stream_medium(a, o)

        for _ in range(TCYC):
            for a in axes:
                for o in (0, 1):
                    substep(a, o)
            traj.append(pos.copy())
            sgns.append(sign.copy())
        return traj, sgns

    return evolve


"""W3 quaternionic coin walk --------------------------------------------------

Annealed winner of scripts/w3_transfer_scan.py (v2), verified exactly isotropic
in scripts/w3_cone_verify.py:

    C_x = XZ(x)1,  C_y = Z(x)XZ,  C_z = -X(x)XZ      (conversion signed perms)
    d_x = diag(Z(x)1), d_y = diag(Z(x)Z), d_z = diag(1(x)Z)   (move directions)

Coin channel index = 2*b1 + b2. The C's form a real quaternion triple
(pairwise anticommuting, squaring to -I): the real irrep of Cl(0,3).

Quenched realization: lane media (1D per axis, class-locked interleaved
streaming), coin-dictated move direction, conversion + sign on odd lane parity.
NOTE (legality, deferred to W3c): coin-dictated direction is species-dependent
streaming; its block-partition realization (stall re-alignment) is an open
design item — this walker is the *instrument* for the quenched cone question.
"""

_X = np.array([[0.0, 1], [1, 0]])
_Z = np.diag([1.0, -1])
_XZ = _X @ _Z
COIN_C = [np.kron(_XZ, np.eye(2)), np.kron(_Z, _XZ), -np.kron(_X, _XZ)]
COIN_D = [np.diag(np.kron(_Z, np.eye(2))).astype(np.int64),
          np.diag(np.kron(_Z, _Z)).astype(np.int64),
          np.diag(np.kron(np.eye(2), _Z)).astype(np.int64)]


def _perm_sign(c: np.ndarray):
    """Signed permutation matrix -> (perm[j] = i, sign[j]) with c[i, j] != 0."""
    perm = np.argmax(np.abs(c), axis=0)
    sign = c[perm, np.arange(c.shape[1])].astype(np.int64)
    return perm, sign


def make_coin_evolve(L: int, E: int, TCYC: int):
    """Quenched quaternionic coin walk on lane media. Returns evolve(q, seed).

    evolve -> (traj, coins, sgns): per-cycle lists of (E,3) positions, (E,)
    coin channels, (E,) signs. All walkers start at centre, coin channel 0.
    """
    perms, signs = zip(*(_perm_sign(c) for c in COIN_C))
    perms = [xp.asarray(p) for p in perms]
    signs = [xp.asarray(s) for s in signs]
    ds = [xp.asarray(d) for d in COIN_D]

    def evolve(q: float, seed: int = 3):
        r = np.random.default_rng(seed)
        lanes = xp.asarray(r.random((E, 3, L)) < q)
        pos = xp.zeros((E, 3), dtype=xp.int64) + L // 2
        coin = xp.zeros(E, dtype=xp.int64)
        sign = xp.ones(E, dtype=xp.int64)
        traj, coins, sgns = [pos.copy()], [coin.copy()], [sign.copy()]
        idx = xp.arange(E)

        def substep(a: int, o: int) -> None:
            nonlocal pos, coin, sign
            u = pos[:, a]
            base = (u - o) % L
            u0 = (base - (base % 2) + o) % L
            u1 = (u0 + 1) % L
            par_odd = lanes[idx, a, u0] ^ lanes[idx, a, u1]
            move = ~par_odd
            newpos = pos.copy()
            newpos[:, a] = xp.where(move, (u + ds[a][coin]) % L, u)
            pos = newpos
            sign = xp.where(par_odd, sign * signs[a][coin], sign)
            coin = xp.where(par_odd, perms[a][coin], coin)
            # lane a streams AFTER the carrier reads (1D table semantics),
            # interleaved schedule: only during its own axis sub-steps (J27)
            m = lanes[:, a]
            if o:
                m = xp.roll(m, -1, axis=-1)
            m0 = m[:, 0::2].copy()
            m[:, 0::2] = m[:, 1::2]
            m[:, 1::2] = m0
            if o:
                m = xp.roll(m, 1, axis=-1)
            lanes[:, a] = m

        for _ in range(TCYC):
            for a in (0, 1, 2):
                for o in (0, 1):
                    substep(a, o)
            traj.append(pos.copy())
            coins.append(coin.copy())
            sgns.append(sign.copy())
        return traj, coins, sgns

    return evolve


def evolve_field(L: int, R: int, TCYC: int, q: float, seed: int = 3,
                 lane_media: bool = False, annealed: bool = False,
                 launch: int = 0):
    """Exact single-particle signed FIELD of the coin walk, ensemble-averaged.

    Per medium realization the signed path sum is a linear field evolution
    (single-carrier sector), so this is walker-exact with infinite walkers:
    G'[c](x) = G[c](x - d_c) on even parity at the source block, plus
    sign_c * G[perm(c)](x) on odd parity (conversion, stall). Media: 3D product
    Bernoulli per axis, streaming along their own axis during their own axis
    sub-steps (legal clustering vacuum -- the coin carries direction memory, so
    no class-locking is wanted; lane_media=True kept for the cross-check that
    frozen co-moving parities kill the cone).

    Returns G(t, 4, L, L, L), mean over R realizations, delta launch channel 0.
    """
    perms, signs = zip(*(_perm_sign(c) for c in COIN_C))
    r = np.random.default_rng(seed)
    acc = np.zeros((TCYC + 1, 4, L, L, L))

    for _ in range(R):
        if lane_media:
            lanes = r.random((3, L)) < q
            med = np.empty((3, L, L, L), dtype=bool)
            med[0] = lanes[0][:, None, None]
            med[1] = lanes[1][None, :, None]
            med[2] = lanes[2][None, None, :]
        else:
            med = r.random((3, L, L, L)) < q
        med = xp.asarray(med)
        g = xp.zeros((4, L, L, L))
        g[launch, L // 2, L // 2, L // 2] = 1.0
        out = [g.copy()]

        def parity(a: int, o: int):
            m = med[a]
            partner = xp.roll(m, -1, axis=a)
            par = m ^ partner                      # parity seen from block-left
            # both sites of a block see the same parity: take it from the
            # block-left site and broadcast to the right site
            left = (xp.arange(L) - o) % 2 == 0
            shape = [1, 1, 1]
            shape[a] = L
            leftmask = left.reshape(shape)
            par_left = xp.where(leftmask, par, xp.roll(par, 1, axis=a))
            return par_left                        # bool: True = odd parity

        def substep(a: int, o: int):
            nonlocal g
            if annealed:
                # i.i.d. Bernoulli(q) parity field: the annealed model verbatim
                # (here q plays the role of the conversion weight p directly)
                par_odd = xp.asarray(r.random((L, L, L)) < q)
            else:
                par_odd = parity(a, o)
            new = xp.zeros_like(g)
            for c in range(4):
                d = int(COIN_D[a][c])
                src_ok = ~par_odd                  # even parity at source: move
                moved = xp.roll(xp.where(src_ok, g[c], 0.0), d, axis=a)
                new[c] += moved
                new[perms[a][c]] += signs[a][c] * xp.where(par_odd, g[c], 0.0)
            g = new
            # stream medium a (own axis sub-steps only), after the carrier reads
            m = xp.moveaxis(med[a], a, -1)
            if o:
                m = xp.roll(m, -1, axis=-1)
            m0 = m[..., 0::2].copy()
            m[..., 0::2] = m[..., 1::2]
            m[..., 1::2] = m0
            if o:
                m = xp.roll(m, 1, axis=-1)
            med[a] = xp.moveaxis(m, -1, a)

        for _ in range(TCYC):
            for a in (0, 1, 2):
                for o in (0, 1):
                    substep(a, o)
            out.append(g.copy())
        stack = xp.stack(out)
        acc += stack.get() if GPU else stack
    return acc / R


def gfield_coin(traj, coins, sgns, L: int) -> np.ndarray:
    """G(t, c, r): signed indicator per cycle, resolved by coin channel."""
    E = traj[0].shape[0]
    out = np.zeros((len(traj), 4, L, L, L))
    for t, (p, c, s) in enumerate(zip(traj, coins, sgns)):
        p = p.get() if GPU else p
        c = c.get() if GPU else c
        s = s.get() if GPU else s
        np.add.at(out[t], (c, p[:, 0], p[:, 1], p[:, 2]), s)
    return out / E


def gfield(traj, sgns, L: int) -> np.ndarray:
    """G(t, r) on the grid: mean signed indicator per cycle."""
    out = np.zeros((len(traj), L, L, L))
    for t, (p, s) in enumerate(zip(traj, sgns)):
        p = p.get() if GPU else p
        s = s.get() if GPU else s
        np.add.at(out[t], (p[:, 0], p[:, 1], p[:, 2]), s)
    return out / p.shape[0]
