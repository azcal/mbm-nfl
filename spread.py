"""
Spread pricing, shared by the NBA and NFL boards.

Built conditional on the winner, so it cannot disagree with the moneyline head:

    P(margin > k), k >= 0  =  P(home) * P(|M| > k | home won)
    P(margin > k), k <  0  =  P(home) + P(away) * P(|M| < |k| | away won)

At k = 0 this collapses to P(home) exactly.

|M| is a discrete distribution over 1..JMAX: a normal kernel around a fitted mean, times a
per-integer weight learned from history. Nothing is hardcoded. The weights come back smooth
and monotone in NBA, where they correct the skew of a floored, right-tailed quantity, and
spiky in NFL, where they also carry the 3-7 scoring structure. Same code, two behaviours,
because the fit decides rather than a branch.

Tested against a t kernel with fitted degrees of freedom, which was marginally worse alone
and a dead tie with weights applied, on both sports. Not used.

The magnitude head carries a win_home indicator. Without it, flipping features symmetrically
strips out home advantage and the model predicts identical winning margins for mirrored
inputs, which underprices the home side.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

JMAX = 70
W_CLIP = (0.15, 6.0)


def flip(X, cols):
    """Rewrite a feature frame from the other side's point of view."""
    out = X.copy()
    for c in cols:
        if c.endswith("_d"):
            out[c] = -X[c]
        elif c.startswith("h_") and ("a_" + c[2:]) in cols:
            out[c] = X["a_" + c[2:]]
        elif c.startswith("a_") and ("h_" + c[2:]) in cols:
            out[c] = X["h_" + c[2:]]
    return out


def _weights(absm, mu, sd):
    j = np.arange(1, JMAX + 1)
    emp = np.array([(absm == v).mean() for v in j])
    nor = stats.norm.pdf((j[None, :] - mu[:, None]) / sd).mean(axis=0)
    nor = nor / nor.sum()
    emp = emp / max(emp.sum(), 1e-9)
    return np.clip(np.where(nor > 1e-6, emp / np.maximum(nor, 1e-6), 1.0), *W_CLIP)


def roughness(w):
    """Mean distance from each weight to the average of its neighbours. Around 0.08 for a
    smooth skew correction, around 0.5 when genuine discrete structure is present. If a
    board's value drifts upward over time the fit has started chasing noise."""
    return float(np.mean([abs(w[i] - (w[i-1] + w[i+1]) / 2) for i in range(1, 30)]))


def fit(df, feats, tr, C, target="margin"):
    """Fit both heads plus the magnitude weights. Returns a dict to pass to price()."""
    y = (df.loc[:, target] > 0).astype(int)
    scw = StandardScaler().fit(df.loc[tr, feats].values)
    mw = LogisticRegression(C=C, max_iter=3000).fit(
        scw.transform(df.loc[tr, feats].values), y[tr].values)

    X = df.loc[tr, feats]
    won_home = y[tr].values == 1
    Xw = X.where(pd.Series(won_home, index=X.index), flip(X, feats))
    Xw = Xw.assign(win_home=won_home.astype(float))
    absm = df.loc[tr, target].abs().values
    scm = StandardScaler().fit(Xw.values)
    mm = Ridge(alpha=5.0).fit(scm.transform(Xw.values), absm)
    mu = np.clip(mm.predict(scm.transform(Xw.values)), 1.0, None)
    sd = float(np.std(absm - mu, ddof=1))
    w = _weights(absm, mu, sd)
    return {"scw": scw, "mw": mw, "scm": scm, "mm": mm, "sd": sd, "w": w,
            "feats": feats, "roughness": roughness(w)}


def price(H, X, ph=None):
    """Per-game probabilities. Returns P(home wins) and the two magnitude distributions.

    Pass ph to anchor on the caller's own moneyline number. Without it this refits an
    internal winner head, which is trained on a slightly different mask and so lands a
    fraction away from the probability the runner logs. That fraction is enough to put the
    median line on the opposite side of zero from the moneyline on near-even games, which
    reads as the two heads contradicting each other.
    """
    if ph is None:
        ph = H["mw"].predict_proba(H["scw"].transform(X[H["feats"]].values))[:, 1]
    ph = np.asarray(ph, float)
    Xh = X[H["feats"]].assign(win_home=1.0)
    Xa = flip(X[H["feats"]], H["feats"]).assign(win_home=0.0)
    j = np.arange(1, JMAX + 1)

    def pmf(F):
        mu = np.clip(H["mm"].predict(H["scm"].transform(F.values)), 1.0, None)
        P = stats.norm.pdf((j[None, :] - mu[:, None]) / H["sd"]) * H["w"][None, :]
        return P / P.sum(axis=1, keepdims=True)
    return ph, pmf(Xh), pmf(Xa)


def cover(ph, Ph, Pa, k):
    """P(home covers a line of k in margin space). k > 0 means home must win by more."""
    j = np.arange(1, JMAX + 1)[None, :]
    k = np.asarray(k, float)
    k = k[:, None] if k.ndim else np.full((len(ph), 1), float(k))
    home_by_more = (Ph * (j > k)).sum(axis=1)
    away_by_less = (Pa * (j < -k)).sum(axis=1)
    return np.where(k[:, 0] >= 0, ph * home_by_more, ph + (1 - ph) * away_by_less)


def median_line(ph, Ph, Pa, grid=np.arange(-40.5, 41.0, 1.0)):
    """The half-point line where P(home covers) is closest to 0.5.

    This is what a spread line is, and it is what must be displayed. The mean margin is a
    different quantity: on a skewed game the two sit on opposite sides of zero, so a team
    can be favoured on the moneyline while the mean points the other way. The median cannot
    do that, because P(margin > 0) is the moneyline number by construction.

    The grid is half points only, matching how books quote and guaranteeing the line never
    lands on a whole number where a game could push.

    Nearest to 0.5 rather than the last point still clearing it. The crossing almost never
    lands exactly on a half point, so taking the last one that clears biases every line
    toward the home side and outcomes stop splitting evenly around it.
    """
    P = np.column_stack([cover(ph, Ph, Pa, k) for k in grid])
    # P is flat across zero, because a margin of exactly 0 is impossible, so "nearest to
    # 0.5" can pick either side arbitrarily and land opposite the moneyline. Search only
    # the side the moneyline is on, which restores the guarantee that the two agree.
    want = np.where(np.asarray(ph, float) >= 0.5, 1.0, -1.0)
    ok = np.sign(grid)[None, :] == want[:, None]
    return grid[np.where(ok, np.abs(P - 0.5), np.inf).argmin(axis=1)]


def expected_margin(ph, Ph, Pa):
    j = np.arange(1, JMAX + 1)
    return ph * (Ph * j).sum(axis=1) - (1 - ph) * (Pa * j).sum(axis=1)


def ladder(ph_i, Ph_i, Pa_i, centre, min_odds, span=5, step=1.0):
    """Rows of (quoted home line, P(home covers), odds needed, P(away covers), odds needed).

    A book quotes the handicap added to a team's score, which is the negative of the margin
    threshold for the home side. Getting that flip wrong hands you the opposite bet.
    """
    rows = []
    lo = np.floor(centre) - span + 0.5
    for n in range(int(2 * span) + 1):
        k = lo + n * step
        p = float(cover(np.array([ph_i]), Ph_i[None, :], Pa_i[None, :], k)[0])
        rows.append((-k, p, min_odds(p), 1 - p, min_odds(1 - p)))
    return rows
