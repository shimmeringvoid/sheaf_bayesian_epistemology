#!/usr/bin/env python3
"""
Companion numerical verification, macro exporter and figure suite (Version 8.6) for:
"Truth as a Structural Invariant: A Sheaf-Bayesian Epistemology for Superintelligence"
Author: Rafael Espericueta (ShimmeringVoid.org)

Script history: Versions 1-6 were drafted with Gemini (Google).  The referee patch on Version 6
(adopted as Version 7) and the Version 8 revision were written by Claude (Anthropic).  Version 8.2
added the ingredient experiment for the three-stage variant.  Version 8.5 follows a blind third-party
check of Version 8.4 (ChatGPT, OpenAI): Layer 1 is now marginalised exactly instead of sampled, with
a count dynamic programme contributed by Gemini; the precision-division arm of the signature table
shares the two-stage filter's Stage 1; and two guards cover paths the experiments never enter.  Only
the Layer-1 numbers (Tables 13-15 and one trajectory of the Proposition 4 check) and one row label
differ from Version 8.2.  Version 8.6 follows the full text of that review, which reached Claude after
Version 8.5.  It adds checks and changes no earlier number: a duality-gap certificate for every Stage 1
fit, the sweeps rerun with a longer EM, per-seed signs of the log-posterior gaps, two isolation runs
(layout, verified-source boundary), a blind start for log Z in Experiment 4, and a few derived values.

What the script computes.  Every table body and every inline number of Section 6 is exported as a
LaTeX macro with the version-neutral prefix \res, so the manuscript holds no hand-typed result.

  Experiment 1  Testimony saturation, the unanimity dip, exact subset-DP provenance inference, the
                mirror ablation, and the growth of corroboration with K unanimous sources.
  Experiment 2  SO(2) sheaf filtering on 20 vertices: baseline estimators, placements, the probit cap,
                checks of Propositions 3 and 4, the distance and offset-ratio sweeps, the coupling
                sweep, the ingredients of the three-stage variant, the capture map, Layer-1 signature
                scenarios with exact inverse cluster sizes, the verified-source boundary, and the
                combined filter under a clone majority.  Three filters are compared:
                  two-stage    weighted spatial-median fit on H^0_tau, then exponent-weighted
                               Student-t EM from the fixed-scale warm start (the paper's filter);
                  three-stage  the same fit, an edge coupling that admits the map error, and EM
                               started from the fit (shown as a comparison);
                  uncentred    the Version 6 filter, kept as a baseline.
  Experiment 3  Dial-free spacetime sheaves, and the two-step Hodge rule on a triangulated annulus
                (exact decision rates from non-central chi-square laws; no simulation).
  Experiment 4  Symbolic DAG consilience and the objective-by-sampling ablation.

Usage:   python3 simulate_sheaf_bayes_v8_6.py [--no-plots]
Writes:  v8_6_results_macros.tex, fig1_testimony_saturation_v8_6.png, fig2_sheaf_robustness_v8_6.png,
         fig3_spacetime_sheaf_v8_6.png
"""

import argparse
from itertools import product
import numpy as np
from scipy import optimize
from scipy.special import gammaln, betaln, comb, softmax
from scipy.stats import norm, chi2, ncx2


def generate_set_partitions(n):
    """Yield all Bell(n) set partitions of {0, ..., n-1} (used for small K in Exp. 4)."""
    if n == 0:
        yield []
        return
    for part in generate_set_partitions(n - 1):
        for i in range(len(part)):
            yield part[:i] + [part[i] + [n - 1]] + part[i + 1:]
        yield part + [[n - 1]]


def exchangeable_lex_prob(w_block, V_sig=50, a_mu=1.0, b_mu=19.0):
    """Normalized exchangeable Beta-categorical probability P(w_b | block b)."""
    s = len(w_block)
    if s == 1:
        return 1.0 / V_sig
    vals, counts = np.unique(w_block, return_counts=True)
    total = 0.0
    if V_sig > len(vals):
        total += (V_sig - len(vals)) * (V_sig ** (-s)) * np.exp(
            betaln(a_mu + s, b_mu) - betaln(a_mu, b_mu))
    for kv in counts:
        j = np.arange(kv + 1)
        log_t = (np.log(comb(kv, j)) - (s - j) * np.log(V_sig)
                 + betaln(a_mu + s - j, b_mu + j) - betaln(a_mu, b_mu))
        total += float(np.sum(np.exp(log_t)))
    return float(total) / V_sig


def exchangeable_deriv_prob(w_block, V_sig=50, p_unv=0.5, mu_drop=0.10, eps_ind=1e-3):
    """Normalized exchangeable costly-derivation likelihood over {0, 1, ..., V_sig}^s."""
    s = len(w_block)
    n0 = int(np.sum(w_block == 0))
    n_pos = s - n0
    p_branch0 = p_unv * ((1.0 - eps_ind) ** n0) * ((eps_ind / V_sig) ** n_pos)
    p_same = 1.0 - mu_drop - eps_ind
    p_other = eps_ind / (V_sig - 1.0)
    pos_vals, pos_counts = np.unique(w_block[w_block > 0], return_counts=True)
    p_branch_pos = 0.0
    if V_sig > len(pos_vals):
        p_branch_pos += (V_sig - len(pos_vals)) * (mu_drop ** n0) * (p_other ** n_pos)
    for kv in pos_counts:
        p_branch_pos += (mu_drop ** n0) * (p_same ** kv) * (p_other ** (n_pos - kv))
    return float(p_branch0 + (1.0 - p_unv) * (p_branch_pos / V_sig))


def exact_crp_dp(stances, sigs, mode="none", V_sig=50, delta_copy=0.001,
                 alpha_crp=1.5, eps=0.10, eta=0.10, xi=0.50, **pd):
    """Exact Bell(n) CRP marginalization via O(3^(n-1)) subset dynamic programming."""
    A = (1.0 - eps) * (1.0 - eta) + eps * (1.0 - xi)
    B = 1.0 - A
    C = (1.0 - eps) * eta + eps * (1.0 - xi)
    D = 1.0 - C

    n = len(stances)
    N_mask = 1 << n
    log_w1 = np.empty(N_mask)
    log_w0 = np.empty(N_mask)
    log_alpha = np.log(alpha_crp)
    log_gamma = [0.0] + [float(gammaln(k)) for k in range(1, n + 2)]
    cache = {}

    for mask in range(1, N_mask):
        idx = [i for i in range(n) if (mask >> i) & 1]
        s = len(idx)
        n1 = int(np.sum(stances[idx] == 1))
        w_tup = () if mode == "none" else tuple(sorted(sigs[idx].tolist()))
        key = (s, n1, w_tup)
        res = cache.get(key)
        if res is None:
            n0 = s - n1
            pz1 = ((1.0 - delta_copy) ** n1) * (delta_copy ** n0)
            pz0 = (delta_copy ** n1) * ((1.0 - delta_copy) ** n0)
            if mode == "none":
                pm1, pm0, pw = D * pz1 + C * pz0, B * pz1 + A * pz0, 1.0
            elif mode == "lexical":
                pm1, pm0 = D * pz1 + C * pz0, B * pz1 + A * pz0
                pw = exchangeable_lex_prob(
                    np.array(w_tup), V_sig=V_sig,
                    a_mu=pd.get("a_mu", 1.0), b_mu=pd.get("b_mu", 19.0))
            elif mode == "verified_deriv":
                pm1, pm0 = D * pz1 + C * pz0, B * pz1 + A * pz0
                pw = exchangeable_deriv_prob(
                    np.array(w_tup), V_sig=V_sig, eps_ind=pd.get("eps_ind", 1e-3))
            elif mode == "coupled_deriv":
                w_arr = np.array(w_tup)
                nz_w = int(np.sum(w_arr == 0))
                np_w = s - nz_w
                eps_ind = pd.get("eps_ind", 1e-3)
                mu_drop, p_unv = 0.10, 0.50
                pw_g0 = ((1.0 - eps_ind) ** nz_w) * ((eps_ind / V_sig) ** np_w)
                p_same = 1.0 - mu_drop - eps_ind
                p_oth = eps_ind / (V_sig - 1.0)
                pv, pc = np.unique(w_arr[w_arr > 0], return_counts=True)
                pw_gp = (V_sig - len(pv)) * (mu_drop ** nz_w) * (p_oth ** np_w)
                for kv in pc:
                    pw_gp += (mu_drop ** nz_w) * (p_same ** kv) * (p_oth ** (np_w - kv))
                pw_gp /= V_sig
                eps_u, eta_u = pd.get("eps_u", 0.20), pd.get("eta_u", 0.25)
                eps_v, eta_v = pd.get("eps_v", 0.02), pd.get("eta_v", 0.02)
                Au = (1.0 - eps_u) * (1.0 - eta_u) + eps_u * 0.5
                Cu = (1.0 - eps_u) * eta_u + eps_u * 0.5
                Du, Bu = 1.0 - Cu, 1.0 - Au
                Av = (1.0 - eps_v) * (1.0 - eta_v) + eps_v * 0.5
                Cv = (1.0 - eps_v) * eta_v + eps_v * 0.5
                Dv, Bv = 1.0 - Cv, 1.0 - Av
                pm1 = p_unv * pw_g0 * (Du * pz1 + Cu * pz0) + (1.0 - p_unv) * pw_gp * (Dv * pz1 + Cv * pz0)
                pm0 = p_unv * pw_g0 * (Bu * pz1 + Au * pz0) + (1.0 - p_unv) * pw_gp * (Bv * pz1 + Av * pz0)
                pw = 1.0
            else:
                raise ValueError(f"Unknown provenance mode: {mode}")
            base_c = log_alpha + log_gamma[s]
            res = (base_c + float(np.log(pm1 * pw + 1e-300)),
                   base_c + float(np.log(pm0 * pw + 1e-300)))
            cache[key] = res
        log_w1[mask], log_w0[mask] = res

    dp1 = np.full(N_mask, -np.inf)
    dp0 = np.full(N_mask, -np.inf)
    dp1[0] = dp0[0] = 0.0
    for mask in range(1, N_mask):
        lsb = mask & -mask
        rem = mask ^ lsb
        sub = rem
        while True:
            blk = sub | lsb
            dp1[mask] = np.logaddexp(dp1[mask], log_w1[blk] + dp1[mask ^ blk])
            dp0[mask] = np.logaddexp(dp0[mask], log_w0[blk] + dp0[mask ^ blk])
            if sub == 0:
                break
            sub = (sub - 1) & rem
    return float(1.0 / (1.0 + np.exp(dp0[N_mask - 1] - dp1[N_mask - 1])))


def run_experiment_1_testimony(n_mc_seeds=20, ch_per_seed=250_000, seed=42):
    """Exp 1(a-c): Utterance MC, K=4 unanimity peak, and Bell(m+4) provenance ablations."""
    rng = np.random.default_rng(seed)
    eps, eta, xi, delta = 0.10, 0.10, 0.50, 0.01
    A = (1.0 - eps) * (1.0 - eta) + eps * (1.0 - xi)
    B, C = 1.0 - A, (1.0 - eps) * eta + eps * (1.0 - xi)
    D = 1.0 - C
    alpha_crp = 1.5

    rows_1a = []
    for N in [1, 3, 10, 100]:
        mc_ests = []
        for _ in range(n_mc_seeds):
            cap0 = rng.random(ch_per_seed) < eps
            z0 = np.where(cap0, rng.random(ch_per_seed) < xi, rng.random(ch_per_seed) < eta)
            cap1 = rng.random(ch_per_seed) < eps
            z1 = np.where(cap1, rng.random(ch_per_seed) < xi, rng.random(ch_per_seed) < (1.0 - eta))
            c0 = np.sum(rng.binomial(N, np.where(z0 == 0, delta, 1.0 - delta)) == 0)
            c1 = np.sum(rng.binomial(N, np.where(z1 == 0, delta, 1.0 - delta)) == 0)
            mc_ests.append(c0 / max(c1, 1))
        r_N = (delta / (1.0 - delta)) ** N
        bf_ex = float((A + B * r_N) / (C + D * r_N))
        bf_n_std = float(((1 - eps)*((1 - eta)**N) + eps*(1 - xi)*((1 - delta)**N)) /
                         ((1 - eps)*(eta**N)       + eps*(1 - xi)*((1 - delta)**N)))
        bf_n_div = float(((1 - eps)*(0.999**N) + eps*(1 - xi)*((1 - delta)**N)) /
                         ((1 - eps)*(0.001**N) + eps*(1 - xi)*((1 - delta)**N)))
        rows_1a.append((N, float(np.mean(mc_ests)), float(np.std(mc_ests)), bf_ex, bf_n_std, bf_n_div))

    N1k = 1000
    r_1k = (delta / (1.0 - delta)) ** N1k
    rows_1a.append((
        N1k, None, None,
        float((A + B * r_1k) / (C + D * r_1k)),
        float(((1 - eps)*((1 - eta)**N1k) + eps*(1 - xi)*((1 - delta)**N1k)) /
              ((1 - eps)*(eta**N1k)       + eps*(1 - xi)*((1 - delta)**N1k))),
        float(((1 - eps)*(0.999**N1k) + eps*(1 - xi)*((1 - delta)**N1k)) /
              ((1 - eps)*(0.001**N1k) + eps*(1 - xi)*((1 - delta)**N1k))),
    ))

    p0 = 0.10
    unanimity_dict = {}
    for K_val in [1, 3, 4, 5, 10, 20, 40, 100, 1000]:
        idx = np.arange(K_val, dtype=float)
        bf_crp = float(np.prod((alpha_crp * D + idx) / (alpha_crp * B + idx)))
        log_num = np.logaddexp(np.log(p0) + np.log(D), np.log(1.0 - p0) + K_val * np.log(D))
        log_den = np.logaddexp(np.log(p0) + np.log(B), np.log(1.0 - p0) + K_val * np.log(B))
        bf_fl = float(np.exp(log_num - log_den))
        ind_val = float((D / B) ** K_val) if K_val <= 10 else float(K_val * np.log10(D / B))
        unanimity_dict[K_val] = (bf_fl, bf_crp, ind_val)

    peaks_p0 = {}
    Ks_scan = np.arange(1, 30)
    for p_test in [0.5, 0.1, 0.01, 0.001]:
        ln_n = np.logaddexp(np.log(p_test) + np.log(D), np.log(1.0 - p_test) + Ks_scan * np.log(D))
        ln_d = np.logaddexp(np.log(p_test) + np.log(B), np.log(1.0 - p_test) + Ks_scan * np.log(B))
        bfs = np.exp(ln_n - ln_d)
        peaks_p0[p_test] = (int(Ks_scan[np.argmax(bfs)]), float(np.max(bfs)))

    rows_1b = []
    coupled_ablation = []
    for m in [1, 2, 4, 6]:
        st = np.array([0]*m + [1]*4)
        w_verb = np.array([10]*m + [1, 2, 3, 4])
        w_para = np.array(list(range(20, 20 + m)) + [1, 2, 3, 4])
        w_ver_s = np.array([0]*m + [1, 2, 3, 4])
        w_ver_f = np.array(list(range(20, 20 + m)) + [1, 2, 3, 4])
        rows_1b.append((
            m,
            exact_crp_dp(st, w_verb, "none"),
            exact_crp_dp(st, w_verb, "lexical"),
            exact_crp_dp(st, w_para, "lexical"),
            exact_crp_dp(st, w_ver_s, "verified_deriv"),
            exact_crp_dp(st, w_ver_f, "verified_deriv"),
            exact_crp_dp(st, w_ver_f, "coupled_deriv", eps_u=0.20, eta_u=0.25),
        ))
        st_mir = np.array([1]*m + [0]*4)
        w_mir = np.array([1]*m + [0, 0, 0, 0])
        w_mir_lex = np.array([1]*m + [2, 3, 4, 5])
        coupled_ablation.append((
            m,
            exact_crp_dp(st_mir, w_mir_lex, "lexical"),
            exact_crp_dp(st_mir, w_mir, "verified_deriv"),
            exact_crp_dp(st_mir, w_mir, "coupled_deriv", eps_u=0.10, eta_u=0.10, eps_v=0.02, eta_v=0.02),
            exact_crp_dp(st_mir, w_mir, "coupled_deriv", eps_u=0.20, eta_u=0.25, eps_v=0.02, eta_v=0.02),
        ))

    Av, Cv = 0.98 * 0.98 + 0.02 * 0.5, 0.98 * 0.02 + 0.02 * 0.5
    Dv, Bv = 1.0 - Cv, 1.0 - Av
    Au, Cu = 0.80 * 0.75 + 0.20 * 0.5, 0.80 * 0.25 + 0.20 * 0.5
    ind_rumors = {r: float(((Dv/Bv) * ((Cu/Au)**r)) / (1.0 + (Dv/Bv) * ((Cu/Au)**r))) for r in [3, 4, 5]}

    st6 = np.array([0]*6 + [1]*4)
    sens6 = (
        exact_crp_dp(st6, np.array(list(range(20, 26)) + [1, 2, 3, 4]), "lexical", a_mu=2.0, b_mu=2.0),
        exact_crp_dp(st6, np.array(list(range(20, 26)) + [1, 2, 3, 4]), "lexical", a_mu=9.0, b_mu=1.0),
        exact_crp_dp(st6, np.array([0]*6 + [1, 2, 3, 4]), "verified_deriv", eps_ind=0.05),
    )
    # Growth of corroboration with K unanimous sources asserting M = 1, under four provenance models:
    # known-independent, verified with distinct proof IDs, lexical with distinct signatures, unverified
    def odds(p_v):
        return float(p_v / (1.0 - p_v))

    growth = {}
    for K_g in [1, 2, 4, 8, 10, 12]:
        st_g, ids_g = np.ones(K_g, dtype=int), np.arange(1, K_g + 1)
        growth[K_g] = (float((D / B) ** K_g),
                       odds(exact_crp_dp(st_g, ids_g, "verified_deriv")),
                       odds(exact_crp_dp(st_g, ids_g, "lexical")),
                       odds(exact_crp_dp(st_g, np.zeros(K_g, dtype=int), "verified_deriv")))

    return {
        "rows_1a": rows_1a, "unanimity": unanimity_dict, "peaks_p0": peaks_p0,
        "rows_1b": rows_1b, "coupled_ablation": coupled_ablation,
        "ind_rumors": ind_rumors, "sens6": sens6, "ceiling": float(A / C),
        "growth": growth,
    }


def infer_l1_inverse_cluster_sizes(W_sigs, V_sig=50, alpha_crp=1.5, p_unv=0.5,
                                   mu_drop=0.10, eps_ind=1e-3):
    """Exact posterior expected inverse block size E_{Pi | W}[1 / |b(v)|] under CRP(alpha_crp) and the
    verified-derivation block likelihood (label eq:deriv_sig_lik in the paper).  It sees the signatures
    only: no claims, no vertex indices.

    The block likelihood is symmetric in the signature IDs, so sources fall into exchangeable groups:
    the zeros, the copies of each repeated positive ID, and the positive IDs that occur once.  A
    log-space dynamic programme over the group counts sums over all set partitions without sampling.
    (Count dynamic programme contributed by Gemini; brute_force_inverse_cluster_sizes() checks it.)"""
    W_arr = np.asarray(W_sigs, dtype=int)
    N = len(W_arr)
    if N == 0:
        return np.array([], dtype=float)

    m0 = int(np.sum(W_arr == 0))
    pos_vals, pos_counts = np.unique(W_arr[W_arr > 0], return_counts=True)
    if len(pos_vals) > V_sig:
        raise ValueError("Number of distinct positive signatures exceeds V_sig.")

    rep_vals = [int(v) for v, c in zip(pos_vals, pos_counts) if c >= 2]
    rep_mults = [int(c) for c in pos_counts if c >= 2]
    u_mult = int(np.sum(pos_counts == 1))
    R = len(rep_vals)

    # Group layout: 0 = zeros, 1..R = repeated positive IDs, R+1 = singleton positive IDs
    c_full = (m0, *rep_mults, u_mult)
    G = len(c_full)
    log_alpha = float(np.log(alpha_crp))
    max_c = max(c_full)
    log_comb = np.zeros((max_c + 1, max_c + 1))
    for n_i in range(max_c + 1):
        for k_i in range(n_i + 1):
            log_comb[n_i, k_i] = float(np.log(comb(n_i, k_i)))

    phi_cache = {}

    def log_phi(b_tup):
        res = phi_cache.get(b_tup)
        if res is not None:
            return res
        s = sum(b_tup)
        w_rep = ([0] * b_tup[0]
                 + [r + 1 for r in range(R) for _ in range(b_tup[r + 1])]
                 + list(range(R + 1, R + 1 + b_tup[R + 1])))
        pw = exchangeable_deriv_prob(
            np.array(w_rep, dtype=int), V_sig=V_sig,
            p_unv=p_unv, mu_drop=mu_drop, eps_ind=eps_ind)
        val = log_alpha + float(gammaln(s)) + float(np.log(pw + 1e-300))
        phi_cache[b_tup] = val
        return val

    all_states = sorted(product(*(range(m + 1) for m in c_full)), key=sum)
    log_Z = {all_states[0]: 0.0}
    for state in all_states[1:]:
        g_anchor = next(g for g in range(G) if state[g] > 0)
        c_rem = list(state)
        c_rem[g_anchor] -= 1
        terms = []
        for k_tup in product(*(range(c_rem[g] + 1) for g in range(G))):
            b_tup = tuple(k_tup[g] + (1 if g == g_anchor else 0) for g in range(G))
            rem_tup = tuple(c_rem[g] - k_tup[g] for g in range(G))
            lc = sum(log_comb[c_rem[g], k_tup[g]] for g in range(G))
            terms.append(lc + log_phi(b_tup) + log_Z[rem_tup])
        t_arr = np.array(terms)
        mx = float(np.max(t_arr))
        log_Z[state] = mx + float(np.log(np.sum(np.exp(t_arr - mx))))

    group_s_hat = np.zeros(G)
    log_Z_full = log_Z[c_full]
    for g_target in range(G):
        if c_full[g_target] == 0:
            continue
        c_rem = list(c_full)
        c_rem[g_target] -= 1
        prob_terms, weighted_terms = [], []
        for k_tup in product(*(range(c_rem[g] + 1) for g in range(G))):
            b_tup = tuple(k_tup[g] + (1 if g == g_target else 0) for g in range(G))
            rem_tup = tuple(c_rem[g] - k_tup[g] for g in range(G))
            lc = sum(log_comb[c_rem[g], k_tup[g]] for g in range(G))
            lp = lc + log_phi(b_tup) + log_Z[rem_tup] - log_Z_full
            prob_terms.append(lp)
            weighted_terms.append(lp - float(np.log(sum(b_tup))))
        p_arr = np.array(prob_terms)
        mp = float(np.max(p_arr))
        assert abs(float(np.exp(mp) * np.sum(np.exp(p_arr - mp))) - 1.0) < 1e-11
        w_arr = np.array(weighted_terms)
        mw = float(np.max(w_arr))
        group_s_hat[g_target] = float(np.exp(mw) * np.sum(np.exp(w_arr - mw)))

    rep_map = {v: idx + 1 for idx, v in enumerate(rep_vals)}
    out = np.empty(N, dtype=float)
    for idx_v, w_v in enumerate(W_arr):
        if w_v == 0:
            out[idx_v] = group_s_hat[0]
        elif int(w_v) in rep_map:
            out[idx_v] = group_s_hat[rep_map[int(w_v)]]
        else:
            out[idx_v] = group_s_hat[R + 1]
    return out

def brute_force_inverse_cluster_sizes(W_sigs, V_sig=50, alpha_crp=1.5, p_unv=0.5, mu_drop=0.10, eps_ind=1e-3):
    """The same expectation by exhaustive enumeration of all Bell(N) set partitions (small N only)."""
    W_arr = np.asarray(W_sigs, dtype=int)
    acc, z_tot = np.zeros(len(W_arr)), 0.0
    for part in generate_set_partitions(len(W_arr)):
        wgt = 1.0
        for b in part:
            wgt *= alpha_crp * float(np.exp(gammaln(len(b)))) * exchangeable_deriv_prob(
                W_arr[b], V_sig=V_sig, p_unv=p_unv, mu_drop=mu_drop, eps_ind=eps_ind)
        z_tot += wgt
        for b in part:
            acc[b] += wgt / len(b)
    return acc / z_tot


def robust_centre_sections(Y_b, B_b, d, n_iter=100, w_b=None, n_newton=60, tol=1e-12):
    """Stage 1: (weighted) spatial-median fit of a global section on H^0_tau.

    B_b[s] is an orthonormal basis (columns) of the approximate harmonic subspace for seed s, i.e. the
    lowest eigenvectors of the estimated sheaf Laplacian.  Minimises sum_v w_v ||y_v - (B c)_v||_2 over c
    and returns the fitted section B c.  With w_b = None every vertex has weight 1; a weight of 0 removes
    a vertex from the fit.

    The objective is convex in c.  Weiszfeld iterations, started at the (weighted) least-squares fit,
    bring c near the minimiser, and damped Newton steps then converge to it.  The second phase matters
    when two groups of sources carry nearly equal total weight: the objective is then almost flat between
    the groups, and Weiszfeld iterations alone are still far from the minimiser after 2000 steps.  A
    Newton step is accepted only if it does not increase the objective, so the result is never worse
    than the Weiszfeld point.

    Both phases have fixed caps (n_iter, n_newton), and where the fitted section passes through one
    vertex's data the objective is not differentiable, so the iteration can stop short of the minimiser.
    Every returned fit is therefore audited with stage1_duality_gap(), and the outcome is accumulated
    in STAGE1_AUDIT.  The audit does not change the fit.
    """
    S, n = Y_b.shape
    N = n // d
    m = B_b.shape[2]
    Bv = B_b.reshape(S, N, d, m)
    Yv = Y_b.reshape(S, N, d)
    w0 = np.ones((S, N)) if w_b is None else w_b

    def wls(w):
        A = np.einsum('sv,svdi,svdj->sij', w, Bv, Bv)
        b = np.einsum('sv,svdi,svd->si', w, Bv, Yv)
        return np.linalg.solve(A, b[:, :, None])[:, :, 0]

    def cost(c_x):
        res_x = Yv - np.einsum('svdj,sj->svd', Bv, c_x)
        return np.sum(w0 * np.linalg.norm(res_x, axis=2), axis=1), res_x

    # Phase 1: Weiszfeld iterations
    c = np.einsum('sij,si->sj', B_b, Y_b) if w_b is None else wls(w0)
    for _ in range(n_iter):
        res = Yv - np.einsum('svdj,sj->svd', Bv, c)
        c = wls(w0 / np.maximum(np.linalg.norm(res, axis=2), 1e-9))

    # Phase 2: damped Newton steps, each seed until its step falls below the tolerance
    f_c, res = cost(c)
    active = np.ones(S, dtype=bool)
    for _ in range(n_newton):
        r_n = np.maximum(np.linalg.norm(res, axis=2), 1e-9)
        Bu = np.einsum('svdi,svd->svi', Bv, res / r_n[:, :, None])
        grad = -np.einsum('sv,svi->si', w0, Bu)
        a_w = w0 / r_n
        H = np.einsum('sv,svdi,svdj->sij', a_w, Bv, Bv) - np.einsum('sv,svi,svj->sij', a_w, Bu, Bu)
        tr_H = np.trace(H, axis1=1, axis2=2)
        # The ridge scales with the trace.  With scalar stalks (d = 1) the Hessian vanishes identically, so a
        # zero or negative trace falls back to an absolute ridge; a positive trace is treated exactly as before.
        H = H + np.where(tr_H > 0.0, 1e-12 * tr_H, 1e-12)[:, None, None] * np.eye(m)[None]
        step = -np.linalg.solve(H, grad[:, :, None])[:, :, 0] * active[:, None]
        t_s = np.ones(S)
        for _ in range(40):
            f_new, res_new = cost(c + t_s[:, None] * step)
            worse = f_new > f_c
            if not worse.any():
                break
            t_s[worse] *= 0.5
        ok = f_new <= f_c
        c = np.where(ok[:, None], c + t_s[:, None] * step, c)
        f_c = np.where(ok, f_new, f_c)
        res = np.where(ok[:, None, None], res_new, res)
        moved = np.where(ok, t_s * np.linalg.norm(step, axis=1), 0.0)
        active &= moved > tol * (1.0 + np.linalg.norm(c, axis=1))
        if not active.any():
            break
    C_b = np.einsum('sij,sj->si', B_b, c)
    gap_b = stage1_duality_gap(Y_b, B_b, d, C_b, w_b)
    loose_b = gap_b > STAGE1_TOL
    STAGE1_AUDIT["fits"] += S
    STAGE1_AUDIT["loose"] += int(np.sum(loose_b))
    if np.any(~loose_b):
        STAGE1_AUDIT["gap_certified"] = max(STAGE1_AUDIT["gap_certified"], float(np.max(gap_b[~loose_b])))
    if np.any(loose_b):
        STAGE1_AUDIT["gap_loose"] = max(STAGE1_AUDIT["gap_loose"], float(np.max(gap_b[loose_b])))
    return C_b


STAGE1_TOL = 1e-8
STAGE1_AUDIT = {"fits": 0, "loose": 0, "gap_certified": 0.0, "gap_loose": 0.0}


def stage1_duality_gap(Y_b, B_b, d, C_b, w_b=None):
    """Relative duality gap of Stage 1 fits: an upper bound on (objective at the fit - minimum) / objective.

    Stage 1 minimises f(c) = sum_v w_v ||y_v - (B c)_v||_2, which is convex.  Its dual is
    max { z'y : ||z_v||_2 <= w_v for every vertex, B'z = 0 }, and every feasible z gives z'y <= min f.
    Two candidates are tried and the better bound is kept.  The first gives every vertex w_v times its
    unit residual.  Where the fitted section passes through one vertex's data, f has a kink and that
    vertex's direction is free, so the second gives the vertex with the smallest residual the vector
    that balances the others, clipped to its ball.  Either candidate is moved onto B'z = 0 by the
    least-norm change that is tangential at the other vertices, which lengthens their vectors only to
    second order, and is then scaled into the balls.  The rounding-level remainder of B'z is charged
    against the bound at the scale of the fit.  B_b must have orthonormal columns.

    A gap near zero certifies the fit.  A gap that is not small means only that this bound is not tight
    there, which is what happens when the iteration has stopped short of the minimiser."""
    S, n = Y_b.shape
    N = n // d
    m = B_b.shape[2]
    w = np.ones((S, N)) if w_b is None else w_b
    Bv = B_b.reshape(S, N, d, m)
    res = (Y_b - C_b).reshape(S, N, d)
    nr = np.linalg.norm(res, axis=2)
    f = np.sum(w * nr, axis=1)
    u = res / np.maximum(nr, 1e-300)[:, :, None]
    rows = np.arange(S)
    v_min = np.argmin(np.where(w > 0, nr, np.inf), axis=1)
    c_norm = np.linalg.norm(C_b, axis=1)

    def lower_bound(kink):
        z = w[:, :, None] * u
        free = w > 0
        if kink:
            z[rows, v_min] = 0.0
            g_rest = np.einsum('svdi,svd->si', Bv, z)
            z_k = -np.einsum('sij,sj->si', np.linalg.pinv(np.transpose(Bv[rows, v_min], (0, 2, 1))), g_rest)
            z_k *= np.minimum(1.0, w[rows, v_min] / np.maximum(np.linalg.norm(z_k, axis=1), 1e-300))[:, None]
            z[rows, v_min] = z_k
            free = free.copy()
            free[rows, v_min] = False
        g = np.einsum('svdi,svd->si', Bv, z)
        P = (np.eye(d)[None, None] - u[:, :, :, None] * u[:, :, None, :]) * (w * free)[:, :, None, None]
        BtP = np.einsum('svdi,svde->svie', Bv, P)
        A = np.einsum('svie,svej->sij', BtP, Bv)
        lam = -np.einsum('sij,sj->si', np.linalg.pinv(A), g)
        z = z + np.einsum('svie,si->sve', BtP, lam)
        zn = np.linalg.norm(z, axis=2)
        scale = np.maximum(1.0, np.max(np.where(w > 0, zn / np.where(w > 0, w, 1.0), 0.0), axis=1))
        z = z / scale[:, None, None]
        rem = np.linalg.norm(np.einsum('svdi,svd->si', Bv, z), axis=1)
        return np.sum(z.reshape(S, n) * Y_b, axis=1) - rem * (1.0 + 2.0 * c_norm)

    low = np.maximum(lower_bound(False), lower_bound(True))
    return (f - low) / np.maximum(f, 1e-300)


def run_experiment_2_sheaf_robustness(n_seeds=100, seed=42):
    """SO(2) sheaf filtering: baselines, placements, probit cap, two-stage filter, three-stage variant, Layer 1."""
    N_verts, d = 20, 2
    edges = [(i, (i + 1) % N_verts) for i in range(N_verts)]
    edges += [(i, (i + 5) % N_verts) for i in range(0, N_verts, 2)]
    gamma0_eq, gamma0_legacy, gamma_sh, nu = 0.0005, 0.05, 50.0, 3.0

    omega_base = 160.0
    omega_cap = float(omega_base * (norm.ppf(0.9999) ** 2))
    omega_unv = float(omega_base * (norm.ppf(0.72) ** 2))
    pi_hats = np.array([0.94]*4 + [0.84]*10 + [0.72]*6)
    omega_het = omega_base * (norm.ppf(np.maximum(pi_hats, 0.5)) ** 2)

    def pregenerate(miscal=False):
        rng_s = np.random.default_rng(seed)
        f_all = np.zeros((n_seeds, N_verts, d, d))
        f_ex_all = np.zeros((n_seeds, N_verts, d, d))
        L_all = np.zeros((n_seeds, N_verts * d, N_verts * d))
        L_ex_all = np.zeros((n_seeds, N_verts * d, N_verts * d))
        d0_all = np.zeros((n_seeds, len(edges) * d, N_verts * d))
        e_norms, mu_gaps, lam2_p, sin_th, lam3_p = [], [], [], [], []
        noise_all = np.zeros((n_seeds, N_verts * d))
        w_true = omega_het.copy()
        if miscal:
            w_true[-4:] = 25.0
        for s in range(n_seeds):
            thetas = rng_s.uniform(0, 2 * np.pi, size=N_verts)
            frames = np.array([[[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]] for t in thetas])
            d0_p = np.zeros((len(edges) * d, N_verts * d))
            d0_e = np.zeros((len(edges) * d, N_verts * d))
            for e_idx, (u, v) in enumerate(edges):
                ang = rng_s.normal(0.0, 0.01)
                R_p = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                d0_p[e_idx*d:(e_idx+1)*d, u*d:(u+1)*d] = R_p @ frames[u]
                d0_p[e_idx*d:(e_idx+1)*d, v*d:(v+1)*d] = -frames[v]
                d0_e[e_idx*d:(e_idx+1)*d, u*d:(u+1)*d] = frames[u]
                d0_e[e_idx*d:(e_idx+1)*d, v*d:(v+1)*d] = -frames[v]
            f_all[s], f_ex_all[s] = frames, frames
            L_all[s], L_ex_all[s] = d0_p.T @ d0_p, d0_e.T @ d0_e
            d0_all[s] = d0_p
            if not miscal:
                ev_e, U_e = np.linalg.eigh(L_ex_all[s])
                ev_p, U_p = np.linalg.eigh(L_all[s])
                en = float(np.linalg.norm(d0_p - d0_e, ord=2))
                svals = np.linalg.svd(U_e[:, :2].T @ U_p[:, :2], compute_uv=False)
                e_norms.append(en)
                mu_gaps.append(float(ev_e[2]))
                lam2_p.append(float(ev_p[1]))
                lam3_p.append(float(ev_p[2]))
                sin_th.append(float(np.sqrt(max(0.0, 1.0 - np.min(svals) ** 2))))
            noise_all[s] = rng_s.normal(0.0, np.repeat(1.0 / np.sqrt(w_true), d))
        return f_all, L_all, noise_all, (e_norms, mu_gaps, lam2_p, sin_th, lam3_p), L_ex_all, d0_all

    frames_std, L_std, noise_std, spec_stats, L_ex_std, d0_std = pregenerate(False)
    _, L_mis, noise_mis, _, _, _ = pregenerate(True)

    def batched_solve(Y_b, L_b, W_nom_b, gamma0=gamma0_eq, mode="warm_het_t", W_init_b=None, n_iter=35,
                      coupling=None, S_exp_b=None, X_init_b=None):
        # `coupling` overrides gamma_sh for one call; `S_exp_b` multiplies each vertex's EM weight by a
        # likelihood exponent; `X_init_b` starts the EM of mode "warm_het_t" from a given state instead
        # of the fixed-scale Gaussian warm start.
        g_sh = gamma_sh if coupling is None else coupling
        S = Y_b.shape[0]
        I40 = np.eye(N_verts * d)[None, :, :]
        if W_init_b is None:
            W_init_b = np.full((S, N_verts), omega_base)
        if mode == "gauss_het":
            A = g_sh * L_b + np.repeat(W_nom_b + gamma0, d, axis=1)[:, :, None] * I40
            return np.linalg.solve(A, (np.repeat(W_nom_b, d, axis=1) * Y_b)[:, :, None])[:, :, 0]
        if mode == "nom_seeded_t":
            A0 = g_sh * L_b + np.repeat(W_nom_b + gamma0, d, axis=1)[:, :, None] * I40
            X = np.linalg.solve(A0, (np.repeat(W_nom_b, d, axis=1) * Y_b)[:, :, None])[:, :, 0]
            W_em = W_nom_b
        elif mode == "equal_t":
            A0 = g_sh * L_b + np.repeat(W_init_b + gamma0, d, axis=1)[:, :, None] * I40
            X = np.linalg.solve(A0, (np.repeat(W_init_b, d, axis=1) * Y_b)[:, :, None])[:, :, 0]
            W_em = np.full((S, N_verts), omega_base)
        elif X_init_b is not None:
            X = X_init_b
            W_em = W_nom_b
        else:
            A0 = g_sh * L_b + np.repeat(W_init_b + gamma0, d, axis=1)[:, :, None] * I40
            X = np.linalg.solve(A0, (np.repeat(W_init_b, d, axis=1) * Y_b)[:, :, None])[:, :, 0]
            W_em = W_nom_b
        for _ in range(n_iter):
            diff = Y_b.reshape(S, N_verts, d) - X.reshape(S, N_verts, d)
            res_sq = np.sum(diff ** 2, axis=2)
            eff_w = ((nu + d) / (nu + W_em * res_sq)) * W_em
            if S_exp_b is not None:
                eff_w = S_exp_b * eff_w
            A_m = g_sh * L_b + np.repeat(eff_w + gamma0, d, axis=1)[:, :, None] * I40
            X_n = np.linalg.solve(A_m, (np.repeat(eff_w, d, axis=1) * Y_b)[:, :, None])[:, :, 0]
            if np.max(np.abs(X_n - X)) < 1e-8:
                X = X_n
                break
            X = X_n
        return X

    def log_post_t(X_b, Y_b, L_b, W_nom_b, gamma0):
        S = X_b.shape[0]
        pq = 0.5 * gamma0 * np.sum(X_b ** 2, axis=1) + 0.5 * gamma_sh * np.einsum('si,sij,sj->s', X_b, L_b, X_b)
        rsq = np.sum((Y_b.reshape(S, N_verts, d) - X_b.reshape(S, N_verts, d)) ** 2, axis=2)
        return -0.5 * (nu + d) * np.sum(np.log1p(W_nom_b * rsq / nu), axis=1) - pq

    X_t0 = np.einsum('svji,j->svi', frames_std, np.array([1.0, 0.0])).reshape(n_seeds, 40)
    off10 = 10.0 * np.array([1.0, -1.0]) / np.sqrt(2.0)

    def eval_cfg(mult, kc, dm, miscal=False, gamma0=gamma0_eq):
        Lb, nb = (L_mis, noise_mis) if miscal else (L_std, noise_std)
        W_b = np.tile(omega_het, (n_seeds, 1))
        if miscal:
            W_b[:, -4:] = omega_base * (norm.ppf(0.99) ** 2)
        Y_b = X_t0 + nb
        if kc > 0:
            off_v = dm * np.array([1.0, -1.0]) / np.sqrt(2.0)
            X_rk = np.einsum('svji,j->svi', frames_std, np.array([1.0, 0.0]) + off_v).reshape(n_seeds, 40)
            Y_b = Y_b.copy()
            Y_b[:, :kc*d] = X_rk[:, :kc*d]
            W_b[:, :kc] = mult * omega_base
        out = {}
        out = {}
        for m in ["gauss_het", "nom_seeded_t", "equal_t", "warm_het_t"]:
            Xh = batched_solve(Y_b, Lb, W_b, gamma0=gamma0, mode=m)
            rm = np.sqrt(np.mean((Xh - X_t0) ** 2, axis=1))
            out[m] = (float(np.mean(rm)), float(np.std(rm)))
        return out["gauss_het"][0], out["nom_seeded_t"][0], out["equal_t"][0], out["warm_het_t"][0], out["warm_het_t"][1]

    cap_mult = omega_cap / omega_base
    main_cfgs = [
        ("Clean calibrated ($k=0$)",                 1.0,      0,  0.0, False),
        ("Miscalibrated ($4$ noisy)",                1.0,      0,  0.0, True),
        ("Trusted $1000\\times$ ($k=1, \\Delta=10$)", 1000.0,   1, 10.0, False),
        ("Capped $13.8\\times$ ($k=8$)", cap_mult, 8, 10.0, False),
        ("Colluders $k=9$ ($500\\times$, $45\\%$)",  500.0,    9, 10.0, False),
        ("Colluders $k=10$ ($500\\times$, $50\\%$)", 500.0,   10, 10.0, False),
        ("Colluders $k=11$ ($500\\times$, maj.)",    500.0,   11, 10.0, False),
    ]
    rows_main = [(c[0], *eval_cfg(*c[1:], gamma0=gamma0_eq)) for c in main_cfgs]

    # 8-Row Placement & Prior-Scale Table (gamma0 in {0.05, 0.0005} x 4 placements x k in {8,9,10,11})
    placements = [
        ("Truth near $0$",          np.array([1.0, 0.0]),          np.array([1.0, 0.0]) + off10),
        ("False claim near $0$",    np.array([1.0, 0.0]) + off10,  np.array([1.0, 0.0])),
        ("Shifted $(20, 20)$",      np.array([21.0, 20.0]),        np.array([21.0, 20.0]) + off10),
        ("Symmetric about $0$",     -0.5 * off10,                  +0.5 * off10),
    ]
    rows_place = []
    for g0 in [gamma0_legacy, gamma0_eq]:
        for p_lbl, g_tr, g_rg in placements:
            X_tr = np.einsum('svji,j->svi', frames_std, g_tr).reshape(n_seeds, 40)
            X_rg = np.einsum('svji,j->svi', frames_std, g_rg).reshape(n_seeds, 40)
            cells = []
            for kc in [8, 9, 10, 11]:
                Y_b = X_tr + noise_std
                W_b = np.tile(omega_het, (n_seeds, 1))
                Y_b[:, :kc*d] = X_rg[:, :kc*d]
                W_b[:, :kc] = 500.0 * omega_base
                Xh = batched_solve(Y_b, L_std, W_b, gamma0=g0, mode="warm_het_t")
                rm = np.sqrt(np.mean((Xh - X_tr) ** 2, axis=1))
                cells.append((float(np.mean(rm)), int(round(100.0 * np.mean(rm > 1.0)))))
            rows_place.append((g0, p_lbl, cells))

    # Posterior mode log-density gaps: (gamma0, mult_label, k) -> gap
    X_r0 = np.einsum('svji,j->svi', frames_std, np.array([1.0, 0.0]) + off10).reshape(n_seeds, 40)
    mode_gaps, mode_gap_seeds = {}, {}
    for g0 in [gamma0_legacy, gamma0_eq]:
        for m_key, mult in [("uncap", 500.0), ("cap", cap_mult)]:
            for kc in [8, 9]:
                Y_b = X_t0 + noise_std
                W_b = np.tile(omega_het, (n_seeds, 1))
                Y_b[:, :kc*d] = X_r0[:, :kc*d]
                W_b[:, :kc] = mult * omega_base
                Xw = batched_solve(Y_b, L_std, W_b, gamma0=g0, mode="warm_het_t")
                Xr = batched_solve(Y_b, L_std, W_b, gamma0=g0, mode="nom_seeded_t")
                gap_s = log_post_t(Xw, Y_b, L_std, W_b, g0) - log_post_t(Xr, Y_b, L_std, W_b, g0)
                mode_gaps[(g0, m_key, kc)] = float(np.mean(gap_s))
                # smallest and largest gap over the seeds, and how many seeds share the sign of the mean
                mode_gap_seeds[(g0, m_key, kc)] = (float(np.min(gap_s)), float(np.max(gap_s)),
                                                   int(np.sum(np.sign(gap_s) == np.sign(np.mean(gap_s)))))

    # Proposition 4: warm-start residuals for k = 9 with the estimated maps and with the exact maps
    prop4_res, prop4_exact = {}, {}
    for gsh in [50.0, 5e5, 5e7]:
        Y_b = X_t0 + noise_std
        Y_b[:, :9*d] = X_r0[:, :9*d]
        for L_use, store in [(L_std, prop4_res), (L_ex_std, prop4_exact)]:
            A0 = gsh * L_use + (omega_base + gamma0_eq) * np.eye(40)[None, :, :]
            X0 = np.linalg.solve(A0, (omega_base * Y_b)[:, :, None])[:, :, 0]
            res = np.sqrt(np.sum((Y_b.reshape(n_seeds, 20, 2) - X0.reshape(n_seeds, 20, 2)) ** 2, axis=2))
            store[gsh] = (float(np.mean(res[:, :9])), float(np.mean(res[:, 9:])))

    # Proposition 4: the EM iteration against the scalar map in the stated limit (exact maps, noiseless
    # honest reports, gamma_sh = 1e5, gamma_0 = 1e-10, Delta = 1e3)
    def limit_trajectory(kc, s_vec, steps=5, dl=1.0e3, gsh=1.0e5, g0=1.0e-10):
        s_vec = np.asarray(s_vec, float)
        dir_l = np.array([1.0, -1.0]) / np.sqrt(2.0)
        X_fl = np.einsum('svji,j->svi', frames_std, np.array([1.0, 0.0]) + dl * dir_l).reshape(n_seeds, 40)
        Y_l = X_t0.copy()
        Y_l[:, :kc*d] = X_fl[:, :kc*d]
        W_l = np.tile(omega_het, (n_seeds, 1))
        W_l[:, :kc] = 500.0 * omega_base
        S_l = np.tile(s_vec, (n_seeds, 1))
        I40 = np.eye(40)[None, :, :]

        def step(w_b):
            A_l = gsh * L_ex_std + np.repeat(w_b + g0, d, axis=1)[:, :, None] * I40
            return np.linalg.solve(A_l, (np.repeat(w_b, d, axis=1) * Y_l)[:, :, None])[:, :, 0]

        def frac(X_l):
            err = np.einsum('svij,svj->svi', frames_std, (X_l - X_t0).reshape(n_seeds, N_verts, d))
            return float(np.mean(err @ dir_l) / dl)

        X_l = step(omega_base * S_l)
        em_f = [frac(X_l)]
        for _ in range(steps):
            r2 = np.sum((Y_l - X_l).reshape(n_seeds, N_verts, d) ** 2, axis=2)
            X_l = step(S_l * ((nu + d) / (nu + W_l * r2)) * W_l)
            em_f.append(frac(X_l))
        s_f, s_h = float(np.sum(s_vec[:kc])), float(np.sum(s_vec[kc:]))
        f_t = s_f / (s_f + s_h)
        map_f = [f_t]
        for _ in range(steps):
            f_t = s_f * f_t ** 2 / (s_f * f_t ** 2 + s_h * (1.0 - f_t) ** 2)
            map_f.append(f_t)
        return em_f, map_f

    prop4_map = {"uniform": limit_trajectory(9, np.ones(N_verts))}

    # Fault-size sweep
    sweep_cfgs = [
        ("Single $1000\\times, \\Delta = 0.2$", 1000.0, 1, 0.2, False),
        ("Single $1000\\times, \\Delta = 0.4$", 1000.0, 1, 0.4, False),
        ("Single $1000\\times, \\Delta = 0.8$", 1000.0, 1, 0.8, False),
        ("Single $1000\\times, \\Delta = 2.0$", 1000.0, 1, 2.0, False),
    ]
    rows_sweep = [(c[0], *eval_cfg(*c[1:], gamma0=gamma0_eq)) for c in sweep_cfgs]

    # Proposition 3 & Theorem 2 asymptotic decay summary    # Proposition 3 & Theorem 2 asymptotic decay summary
    en_l, mg_l, l2_l, st_l, l3_l = spec_stats
    weyl_l = [e ** 2 for e in en_l]
    wed_l = [e / (np.sqrt(m) - e) for e, m in zip(en_l, mg_l)]
    n_pass3 = int(sum((e < np.sqrt(m) / 2.0) and (l2 <= e ** 2) and (l3 >= (np.sqrt(m) - e) ** 2)
                      and (st <= e / (np.sqrt(m) - e))
                      for e, m, l2, l3, st in zip(en_l, mg_l, l2_l, l3_l, st_l)))
    L_t2 = L_std[-1]
    y_cl = X_t0[-1] + noise_std[-1]
    w_t2 = omega_het.copy()
    w_t2[0] = omega_cap
    w_om = w_t2.copy()
    w_om[0] = 0.0
    x_om = batched_solve(y_cl[None, :], L_t2[None, :, :], w_om[None, :], gamma0=gamma0_eq, n_iter=60)[0]
    t2_errs = {}
    for Dt2 in [10.0, 1000.0]:
        yo = y_cl.copy()
        yo[:2] += np.array([Dt2, 0.0])
        xo = batched_solve(yo[None, :], L_t2[None, :, :], w_t2[None, :], gamma0=gamma0_eq, n_iter=60)[0]
        t2_errs[int(Dt2)] = float(np.linalg.norm(xo - x_om))

    # ------------------------------------------------------------------------------------------
    # The paper's two-stage filter, the three-stage variant, and the sweeps that test them.
    # ------------------------------------------------------------------------------------------
    B_std = np.linalg.eigh(L_std)[1][:, :, :d]      # per-seed basis of H^0_tau (lowest d eigenvectors)
    g_base = np.array([1.0, 0.0])
    dir_f = np.array([1.0, -1.0]) / np.sqrt(2.0)
    sigma_map = 0.01                                # s.d. of the map-angle error drawn in pregenerate()
    edge_src = np.array([u_e for u_e, _ in edges])

    def two_stage(Yc_b, Wc_b, S_b=None, weighted=True, **kw):
        """The paper's filter.  Stage 1: spatial-median fit of a global section on H^0_tau,
        weighted by the likelihood exponents (weighted=False gives the unweighted fit of Version 7).
        Stage 2: exponent-weighted Student-t EM on the centred data from the fixed-scale warm start."""
        C_b = robust_centre_sections(Yc_b, B_std, d, w_b=(S_b if weighted else None))
        if S_b is None:
            return batched_solve(Yc_b - C_b, L_std, Wc_b, mode="warm_het_t", **kw) + C_b
        return batched_solve(Yc_b - C_b, L_std, Wc_b, mode="warm_het_t",
                             W_init_b=omega_base * S_b, S_exp_b=S_b, **kw) + C_b

    def three_stage(Yc_b, Wc_b, S_b=None, coupling=None, fit_start=True, map_aware=True, sigma=None,
                    info=None, **kw):
        """Variant with a scale stage.  Stage 1 as in two_stage().  Stage 2: every edge coupling admits
        the map error, gamma_e = 1 / (1/gamma_sh + sigma_map^2 |c_u|^2), with c the Stage 1 fit at the
        edge's source vertex.  Stage 3: exponent-weighted Student-t EM on the centred data, started
        from the fit instead of the pooled warm start.

        The switches serve the ingredient experiment only.  fit_start=False keeps the pooled warm
        start, map_aware=False keeps the uniform coupling, and sigma replaces the generating sigma_map
        by an assumed value; info, if given, receives the edge couplings.  With both changes switched
        off the function is two_stage()."""
        g_sh = gamma_sh if coupling is None else coupling
        sg = sigma_map if sigma is None else sigma
        C_b = robust_centre_sections(Yc_b, B_std, d, w_b=S_b)
        if map_aware:
            nrm2 = np.sum(C_b.reshape(n_seeds, N_verts, d) ** 2, axis=2)[:, edge_src]
            g_e = 1.0 / (1.0 / g_sh + sg ** 2 * nrm2)
            if info is not None:
                info["g_e"] = g_e
            L_w = np.einsum('sei,se,sej->sij', d0_std, np.repeat(g_e, d, axis=1), d0_std)
            kw_s = dict(coupling=1.0)
        else:
            L_w = L_std
            kw_s = dict(coupling=g_sh)
        if fit_start:
            kw_s["X_init_b"] = np.zeros_like(Yc_b)
        elif S_b is not None:
            kw_s["W_init_b"] = omega_base * S_b
        return batched_solve(Yc_b - C_b, L_w, Wc_b, mode="warm_het_t", S_exp_b=S_b, **kw_s, **kw) + C_b

    def colluder_data(g_true, g_false, idx_f, mult_f=500.0):
        """Honest vertices report the truth plus noise; vertices in idx_f report g_false exactly."""
        X_true = np.einsum('svji,j->svi', frames_std, g_true).reshape(n_seeds, N_verts * d)
        X_false = np.einsum('svji,j->svi', frames_std, g_false).reshape(n_seeds, N_verts * d)
        Yc_b = X_true + noise_std
        Wc_b = np.tile(omega_het, (n_seeds, 1))
        for c_i in idx_f:
            Yc_b[:, c_i*d:(c_i+1)*d] = X_false[:, c_i*d:(c_i+1)*d]
            Wc_b[:, c_i] = mult_f * omega_base
        return X_true, Yc_b, Wc_b

    def rm_cell(X_hat, X_true, thr=1.0):
        rm_v = np.sqrt(np.mean((X_hat - X_true) ** 2, axis=1))
        return float(np.mean(rm_v)), int(round(100.0 * np.mean(rm_v > thr)))

    def honest_drift(X_hat, X_true, idx_h, g_true, g_false):
        """Per-seed mean displacement of the honest vertices toward the false claim, as a fraction of
        the offset (0 = fault rejected, 1 = consensus fully captured)."""
        off_v = np.asarray(g_false, float) - np.asarray(g_true, float)
        dl_v = float(np.linalg.norm(off_v))
        err = np.einsum('svij,svj->svi', frames_std, (X_hat - X_true).reshape(n_seeds, N_verts, d))
        return np.mean(err[:, idx_h, :] @ (off_v / dl_v), axis=1) / dl_v

    def capt_pct(X_hat, X_true, idx_h, g_true, g_false, thr=0.25):
        """Percent of seeds whose honest vertices move more than `thr` of the way to the false claim."""
        return int(round(100.0 * np.mean(honest_drift(X_hat, X_true, idx_h, g_true, g_false) > thr)))

    def part_rmse(X_hat, X_true, idx_p):
        e_p = (X_hat - X_true).reshape(n_seeds, N_verts, d)[:, idx_p, :]
        return float(np.mean(np.sqrt(np.mean(e_p ** 2, axis=(1, 2)))))

    # Why distance matters.  The truth obeys the exact maps, so part of it lies outside H^0_tau of the
    # estimated sheaf; that part, not the shrinkage of the harmonic modes, sets the warm-start residual.
    g_300 = g_base + 300.0 * np.array([1.0, 1.0])
    X_300 = np.einsum('svji,j->svi', frames_std, g_300).reshape(n_seeds, N_verts * d)
    P_h = np.einsum('sij,skj->sik', B_std, B_std)

    def vnorm(V_b):
        return float(np.mean(np.linalg.norm(V_b.reshape(n_seeds, N_verts, d), axis=2)))

    A_w = gamma_sh * L_std + (omega_base + gamma0_eq) * np.eye(N_verts * d)[None, :, :]
    res_w = X_300 - np.linalg.solve(A_w, (omega_base * X_300)[:, :, None])[:, :, 0]
    res_w_h = np.einsum('sij,sj->si', P_h, res_w)
    mech = {
        "nonharm_coef": vnorm(X_300 - np.einsum('sij,sj->si', P_h, X_300)) / (sigma_map * float(np.linalg.norm(g_300))),
        "warm_res_harm": vnorm(res_w_h), "warm_res_nonharm": vnorm(res_w - res_w_h),
        "core_min": float(np.sqrt(nu / np.max(omega_het))), "core_max": float(np.sqrt(nu / np.min(omega_het))),
    }

    # Sweep 1: distance of the truth from the origin (truth = (1, 0) + shift * (1, 1), false claim 10 away)
    hon9 = list(range(9, N_verts))
    S_oracle9 = np.ones((n_seeds, N_verts))
    S_oracle9[:, :9] = 0.0
    rows_dist = []
    for sh_v in [0.0, 20.0, 100.0, 300.0, 1000.0]:
        g_far = g_base + sh_v * np.array([1.0, 1.0])
        g_fl = g_far + off10
        cells_d = {}
        X_true, Yc_b, Wc_b = colluder_data(g_far, g_fl, [])
        cells_d[(0, "raw")] = rm_cell(Yc_b, X_true)
        cells_d[(0, "gauss")] = rm_cell(
            batched_solve(Yc_b, L_std, Wc_b, gamma0=gamma0_eq, mode="gauss_het"), X_true)
        cells_d[(0, "warm")] = rm_cell(
            batched_solve(Yc_b, L_std, Wc_b, gamma0=gamma0_eq, mode="warm_het_t"), X_true)
        cells_d[(0, "centred")] = rm_cell(two_stage(Yc_b, Wc_b), X_true)
        cells_d[(0, "three")] = rm_cell(three_stage(Yc_b, Wc_b), X_true)
        X_true, Yc_b, Wc_b = colluder_data(g_far, g_fl, range(9))
        cells_d[(9, "warm")] = rm_cell(
            batched_solve(Yc_b, L_std, Wc_b, gamma0=gamma0_eq, mode="warm_het_t"), X_true)
        X_c = two_stage(Yc_b, Wc_b)
        cells_d[(9, "centred")] = rm_cell(X_c, X_true)
        cells_d[(9, "centred_capt")] = capt_pct(X_c, X_true, hon9, g_far, g_fl)
        cells_d[(9, "centred_drift")] = float(100.0 * np.mean(honest_drift(X_c, X_true, hon9, g_far, g_fl)))
        cells_d[(9, "oracle")] = rm_cell(two_stage(Yc_b, Wc_b, S_b=S_oracle9), X_true)
        X_3 = three_stage(Yc_b, Wc_b)
        cells_d[(9, "three")] = rm_cell(X_3, X_true)
        cells_d[(9, "three_hon")] = part_rmse(X_3, X_true, hon9)
        cells_d[(9, "three_fault")] = part_rmse(X_3, X_true, list(range(9)))
        cells_d[(9, "three_capt")] = capt_pct(X_3, X_true, hon9, g_far, g_fl)
        rows_dist.append((sh_v, float(np.linalg.norm(g_far)), cells_d))

    # Sweep 1b: offset as a multiple of the map discrepancy sigma_map * |x|, at two distances
    ratio_grid = [1.5, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
    lay_adj = lambda k_l: list(range(k_l))
    lay_sca = lambda k_l: [(2 * i_l) % N_verts for i_l in range(k_l)]
    rows_ratio = []
    for f_lbl, f_fn in [("Two-stage", two_stage), ("Three-stage", three_stage)]:
        for lay_lbl, lay_fn in [("Adjacent", lay_adj), ("Scattered", lay_sca)]:
            idx_f = lay_fn(9)
            idx_h = [v_i for v_i in range(N_verts) if v_i not in idx_f]
            for sh_v in [100.0, 300.0]:
                g_far = g_base + sh_v * np.array([1.0, 1.0])
                sc_v = sigma_map * float(np.linalg.norm(g_far))
                cells_r = []
                for rt_v in ratio_grid:
                    g_fl = g_far + rt_v * sc_v * dir_f
                    X_true, Yc_b, Wc_b = colluder_data(g_far, g_fl, idx_f)
                    cells_r.append(capt_pct(f_fn(Yc_b, Wc_b), X_true, idx_h, g_far, g_fl))
                rows_ratio.append((f_lbl, lay_lbl, sh_v, cells_r))

    # Sweep 2: coupling strength across the four placements (uncentred, two-stage, three-stage)
    rows_coup = []
    for gsh_v in [5.0, 50.0, 500.0, 5000.0]:
        for pl_lbl, pl_true, pl_false in placements:
            cells_plain, cells_cent, cells_three = [], [], []
            for kc_v in [8, 9, 10, 11]:
                X_true, Yc_b, Wc_b = colluder_data(pl_true, pl_false, range(kc_v))
                cells_plain.append(rm_cell(batched_solve(
                    Yc_b, L_std, Wc_b, gamma0=gamma0_eq, mode="warm_het_t", coupling=gsh_v), X_true))
                cells_cent.append(rm_cell(two_stage(Yc_b, Wc_b, coupling=gsh_v), X_true))
                cells_three.append(rm_cell(three_stage(Yc_b, Wc_b, coupling=gsh_v), X_true))
            rows_coup.append((gsh_v, pl_lbl, cells_plain, cells_cent, cells_three))

    # Sweep 2b: what each change in the three-stage variant does, and what a wrong sigma_map costs.
    # Rows: the two filters, each change alone, and the variant given half and twice the true sigma_map.
    # Cells: near the origin (four placements, k = 8 and 9), the percent of seeds failing and of seeds
    # whose honest vertices are captured; the no-fault RMSE at (20,20) and (1000,1000); and the worst
    # percent captured over the cells of the offset-ratio sweep, for ratios below five and from five up.
    ingr_variants = [
        ("Warm start, uniform (\\textbf{two-stage})",        dict(fit_start=False, map_aware=False)),
        ("Fit start, map-aware (three-stage)",                dict()),
        ("Fit start, uniform",                                dict(map_aware=False)),
        ("Warm start, map-aware",                             dict(fit_start=False)),
        ("Three-stage given $\\sigma_{\\mathrm{map}}/2$",    dict(sigma=0.5 * sigma_map)),
        ("Three-stage given $2\\sigma_{\\mathrm{map}}$",     dict(sigma=2.0 * sigma_map)),
    ]
    ingr_shifts = [20.0, 1000.0]
    ingr_cells = [(p_i, kc_v) for p_i in range(len(placements)) for kc_v in [8, 9]]
    rows_ingr = []
    for v_lbl, v_kw in ingr_variants:
        cell_i = {}
        for gsh_v in [5.0, 50.0]:
            for p_i, kc_v in ingr_cells:
                _pl_lbl, pl_true, pl_false = placements[p_i]
                X_true, Yc_b, Wc_b = colluder_data(pl_true, pl_false, range(kc_v))
                X_h = three_stage(Yc_b, Wc_b, coupling=gsh_v, **v_kw)
                cell_i[("fail", gsh_v, p_i, kc_v)] = rm_cell(X_h, X_true)[1]
                cell_i[("capt", gsh_v, p_i, kc_v)] = capt_pct(
                    X_h, X_true, list(range(kc_v, N_verts)), pl_true, pl_false)
            for key_q in ["fail", "capt"]:
                vals_q = [cell_i[(key_q, gsh_v, p_i, kc_v)] for p_i, kc_v in ingr_cells]
                cell_i[(key_q, gsh_v)] = (min(vals_q), max(vals_q))
        for sh_v in ingr_shifts:
            g_far = g_base + sh_v * np.array([1.0, 1.0])
            X_true, Yc_b, Wc_b = colluder_data(g_far, g_far, [])
            cell_i[("nofault", sh_v)] = rm_cell(three_stage(Yc_b, Wc_b, **v_kw), X_true)[0]
        low_i, high_i = [], []
        for lay_fn in [lay_adj, lay_sca]:
            idx_f = lay_fn(9)
            idx_h = [v_i for v_i in range(N_verts) if v_i not in idx_f]
            for sh_v in [100.0, 300.0]:
                g_far = g_base + sh_v * np.array([1.0, 1.0])
                sc_v = sigma_map * float(np.linalg.norm(g_far))
                for rt_v in ratio_grid:
                    g_fl = g_far + rt_v * sc_v * dir_f
                    X_true, Yc_b, Wc_b = colluder_data(g_far, g_fl, idx_f)
                    c_v = capt_pct(three_stage(Yc_b, Wc_b, **v_kw), X_true, idx_h, g_far, g_fl)
                    (low_i if rt_v < 5.0 else high_i).append(c_v)
        cell_i["capt_low"], cell_i["capt_high"] = max(low_i), max(high_i)
        rows_ingr.append((v_lbl, cell_i))

    # Is it the weakened edge that breaks the warm start?  At the shifted placement the map-aware formula
    # lowers the couplings; give the two-stage filter one uniform coupling of the same mean size.
    ingr_shifted = [p_i for p_i, pl in enumerate(placements) if pl[0].startswith("Shifted")][0]
    ge_info, ge_sh = {}, []
    for kc_v in [8, 9]:
        X_true, Yc_b, Wc_b = colluder_data(placements[ingr_shifted][1], placements[ingr_shifted][2], range(kc_v))
        three_stage(Yc_b, Wc_b, fit_start=False, info=ge_info)
        ge_sh.append(float(np.mean(ge_info["g_e"])))
    ingr_gamma_shifted = float(np.mean(ge_sh))
    ingr_unif_fail = {}
    for kc_v in [8, 9]:
        X_true, Yc_b, Wc_b = colluder_data(placements[ingr_shifted][1], placements[ingr_shifted][2], range(kc_v))
        ingr_unif_fail[kc_v] = rm_cell(two_stage(Yc_b, Wc_b, coupling=ingr_gamma_shifted), X_true)[1]

    # The first two rows repeat cells of the distance, offset-ratio and coupling sweeps; they must agree.
    for gsh_v in [5.0, 50.0]:
        for row_i, col_c in [(0, 3), (1, 4)]:
            ref_c = [c[1] for r in rows_coup if r[0] == gsh_v for c in r[col_c][:2]]
            assert rows_ingr[row_i][1][("fail", gsh_v)] == (min(ref_c), max(ref_c))
    for row_i, f_lbl in [(0, "Two-stage"), (1, "Three-stage")]:
        ref_r = [r[3] for r in rows_ratio if r[0] == f_lbl]
        assert rows_ingr[row_i][1]["capt_low"] == max(c for cells_r in ref_r for c in cells_r[:2])
        assert rows_ingr[row_i][1]["capt_high"] == max(c for cells_r in ref_r for c in cells_r[2:])
    for row_i, key_d in [(0, "centred"), (1, "three")]:
        for sh_v in ingr_shifts:
            ref_d = [cd[(0, key_d)][0] for s_d, _, cd in rows_dist if s_d == sh_v][0]
            assert abs(rows_ingr[row_i][1][("nofault", sh_v)] - ref_d) < 1e-12

    # Sweep 3: capture map.  Cell = mean displacement of the HONEST vertices toward the false claim,
    # as a percentage of the offset Delta (0 = fault rejected, 100 = consensus fully captured).
    capt_deltas = [0.4, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 5.0, 10.0]
    rows_capt = []
    for f_lbl, f_fn, combos in [
        ("Two-stage", two_stage, [("Adjacent", lay_adj, [1, 3, 5, 7, 9]), ("Scattered", lay_sca, [3, 5, 7, 9])]),
        ("Three-stage", three_stage, [("Adjacent", lay_adj, [5, 7, 9]), ("Scattered", lay_sca, [9])]),
    ]:
        for lay_lbl, lay_fn, ks_lay in combos:
            for kc_v in ks_lay:
                idx_f = lay_fn(kc_v)
                idx_h = [v_i for v_i in range(N_verts) if v_i not in idx_f]
                cells_c = []
                for dm_v in capt_deltas:
                    g_fl = g_base + dm_v * dir_f
                    X_true, Yc_b, Wc_b = colluder_data(g_base, g_fl, idx_f)
                    cells_c.append(float(100.0 * np.mean(
                        honest_drift(f_fn(Yc_b, Wc_b), X_true, idx_h, g_base, g_fl))))
                rows_capt.append((f_lbl, lay_lbl, kc_v, cells_c))

    # Layer 1 sees signatures only; s_hat is the exact posterior expected inverse block size.
    s_hat_cache = {}

    def s_hat_for(sig_list):
        key = tuple(int(w_s) for w_s in sig_list)
        if key not in s_hat_cache:
            s_hat_cache[key] = infer_l1_inverse_cluster_sizes(np.array(key))
        return s_hat_cache[key]

    # Self-test: the dynamic programme against all Bell(9) = 21147 set partitions of a pattern that has
    # zeros, a repeated positive ID and positive IDs that occur once.
    l1_test_sig = np.array([0, 0, 0, 0, 7, 7, 1, 2, 3])
    l1_test_bell = sum(1 for _ in generate_set_partitions(len(l1_test_sig)))
    assert np.max(np.abs(infer_l1_inverse_cluster_sizes(l1_test_sig)
                         - brute_force_inverse_cluster_sizes(l1_test_sig))) < 1e-12

    def sig_patterns(k_s):
        n_h = N_verts - k_s
        return [
            ("Clones unverified, honest IDs unique",     [0]*k_s + list(range(1, n_h + 1))),
            ("Clones and six honest sources unverified", [0]*k_s + list(range(1, n_h - 5)) + [0]*6),
            ("Clones share one verified ID",             [40]*k_s + list(range(1, n_h + 1))),
            ("Clones forge distinct IDs",                list(range(20, 20 + k_s)) + list(range(1, n_h + 1))),
            ("Everyone unverified",                      [0]*N_verts),
        ]

    def tiered(W_raw_b, sig_arr):
        """Tier capping: an unverified source is assigned the lowest calibrated precision tier."""
        W_t = W_raw_b.copy()
        W_t[:, np.asarray(sig_arr) == 0] = omega_unv
        return W_t

    # Signature scenarios: precision division against exponents.  Both arms use the same weighted Stage 1
    # and the same warm start, so they differ only in how s_hat enters the EM weights.  The uncentred
    # precision division of Version 6 is kept as a third cell for one sentence of the paper.
    l1_cells = {}
    for kc_v, dm_v in [(12, 10.0), (3, 0.4)]:
        thr_v = 1.0 if dm_v > 2.0 else 0.5 * dm_v / np.sqrt(2.0)
        X_true, Yc_b, W_raw_s = colluder_data(g_base, g_base + dm_v * dir_f, range(kc_v))
        for p_i, (p_lbl_s, sig_list) in enumerate(sig_patterns(kc_v)):
            sig_arr = np.array(sig_list)
            s_hat_s = s_hat_for(sig_list)
            S_s = np.tile(s_hat_s, (n_seeds, 1))
            W_tier_s = tiered(W_raw_s, sig_arr)
            C_s = robust_centre_sections(Yc_b, B_std, d, w_b=S_s)
            cell_div = rm_cell(batched_solve(Yc_b - C_s, L_std, W_tier_s * S_s, gamma0=gamma0_eq,
                                             mode="warm_het_t", W_init_b=omega_base * S_s) + C_s, X_true, thr_v)
            cell_div_unc = rm_cell(batched_solve(Yc_b, L_std, W_tier_s * S_s, gamma0=gamma0_eq,
                                                 mode="warm_het_t", W_init_b=omega_base * S_s), X_true, thr_v)
            cell_exp = rm_cell(two_stage(Yc_b, W_tier_s, S_b=S_s), X_true, thr_v)
            hon_unv = [v_i for v_i in range(kc_v, N_verts) if sig_arr[v_i] == 0]
            hon_ver = [v_i for v_i in range(kc_v, N_verts) if sig_arr[v_i] != 0]
            l1_cells[(p_i, kc_v)] = (p_lbl_s, cell_div, cell_exp,
                                     float(np.mean(s_hat_s[:kc_v])), float(np.mean(s_hat_s[kc_v:])),
                                     float(np.sum(s_hat_s[:kc_v])), float(np.sum(s_hat_s[kc_v:])),
                                     float(np.mean(s_hat_s[hon_unv])) if hon_unv else None,
                                     float(np.mean(s_hat_s[hon_ver])) if hon_ver else None,
                                     cell_div_unc)
    rows_l1_scen = [(l1_cells[(p_i, 12)][0], l1_cells[(p_i, 12)][1], l1_cells[(p_i, 12)][2],
                     l1_cells[(p_i, 3)][1], l1_cells[(p_i, 3)][2],
                     l1_cells[(p_i, 12)][3], l1_cells[(p_i, 12)][4],
                     l1_cells[(p_i, 12)][5], l1_cells[(p_i, 12)][6],
                     l1_cells[(p_i, 12)][7], l1_cells[(p_i, 12)][8],
                     l1_cells[(p_i, 12)][9]) for p_i in range(5)]
    # Exchangeability: in the mixed pattern an unverified clone and an unverified honest source get one value
    assert abs(rows_l1_scen[1][5] - rows_l1_scen[1][9]) < 1e-12

    # Verified-source boundary: k unverified clones, h verified honest sources, the rest unverified
    rows_bound = []
    g_fl10 = g_base + 10.0 * dir_f
    for kc_v in [12, 14, 16]:
        X_true, Yc_b, W_raw_s = colluder_data(g_base, g_fl10, range(kc_v))
        idx_h = list(range(kc_v, N_verts))
        cells_b = []
        for h_v in [0, 1, 2, 3, 4]:
            sig_list = [0]*kc_v + list(range(1, h_v + 1)) + [0]*(N_verts - kc_v - h_v)
            s_hat_s = s_hat_for(sig_list)
            X_h = two_stage(Yc_b, tiered(W_raw_s, sig_list), S_b=np.tile(s_hat_s, (n_seeds, 1)))
            cells_b.append((float(np.sum(s_hat_s[:kc_v])), float(np.sum(s_hat_s[kc_v:])),
                            capt_pct(X_h, X_true, idx_h, g_base, g_fl10)))
        rows_bound.append((kc_v, cells_b))

    # The cell nearest parity, (k, h) = (16, 3), has S_fault / S_hon = 1.0013.  Repeat it at ten times the
    # coupling and under the three-stage variant: the sign of S_fault - S_hon should decide all three.
    X_true, Yc_b, W_raw_s = colluder_data(g_base, g_fl10, range(16))
    sig_list = [0]*16 + [1, 2, 3] + [0]
    S_par = np.tile(s_hat_for(sig_list), (n_seeds, 1))
    W_par = tiered(W_raw_s, sig_list)
    idx_h = list(range(16, N_verts))
    bound_near = {
        "two_strong": capt_pct(two_stage(Yc_b, W_par, S_b=S_par, coupling=10.0 * gamma_sh), X_true, idx_h, g_base, g_fl10),
        "three": capt_pct(three_stage(Yc_b, W_par, S_b=S_par), X_true, idx_h, g_base, g_fl10),
    }

    # Combined filter under a clone majority: 12 unverified clones, 6 unverified honest, 2 verified honest
    comb_deltas = [10.0, 20.0, 40.0, 80.0, 160.0]
    sig_mix = sig_patterns(12)[1][1]
    S_mix = np.tile(s_hat_for(sig_mix), (n_seeds, 1))
    hon12 = list(range(12, N_verts))
    comb_filters = [
        ("Exponents only (no centring)",
         lambda Y_c, W_c: batched_solve(Y_c, L_std, W_c, gamma0=gamma0_eq, mode="warm_het_t",
                                        W_init_b=omega_base * S_mix, S_exp_b=S_mix)),
        ("Unweighted Stage 1, then exponents", lambda Y_c, W_c: two_stage(Y_c, W_c, S_b=S_mix, weighted=False)),
        ("\\textbf{Two-stage (weighted Stage 1)}", lambda Y_c, W_c: two_stage(Y_c, W_c, S_b=S_mix)),
        ("Three-stage variant",                lambda Y_c, W_c: three_stage(Y_c, W_c, S_b=S_mix)),
    ]
    rows_comb = []
    for pl_lbl, pl_fn in [("Truth near $0$",       lambda dl_v: (g_base, g_base + dl_v * dir_f)),
                          ("False claim near $0$", lambda dl_v: (g_base + dl_v * dir_f, g_base))]:
        for f_lbl, f_fn in comb_filters:
            cells_m = []
            for dl_v in comb_deltas:
                g_tr, g_fl = pl_fn(dl_v)
                X_true, Yc_b, W_raw_s = colluder_data(g_tr, g_fl, range(12))
                cells_m.append(capt_pct(f_fn(Yc_b, tiered(W_raw_s, sig_mix)), X_true, hon12, g_tr, g_fl))
            rows_comb.append((pl_lbl, f_lbl, cells_m))

    prop4_map["exponent"] = limit_trajectory(12, s_hat_for(sig_mix))

    # ------------------------------------------------------------------------------------------
    # Checks added in Version 8.6.  They reuse the data above and alter none of its results.
    # ------------------------------------------------------------------------------------------
    stage1_audit = dict(STAGE1_AUDIT)            # every Stage 1 fit behind the tables, before the reruns below

    # (a) EM iteration cap.  batched_solve() runs at most 35 iterations and stops early only when every
    #     seed of a batch has converged, so slowly converging cells are reported at the cap.  Rerun the
    #     filtered cells of the distance sweep and every cell of the offset-ratio sweep with a cap of 500.
    cap_long = 500
    cap_rmse, cap_capt = [], []
    for sh_v, _dist_v, cd in rows_dist:
        g_far = g_base + sh_v * np.array([1.0, 1.0])
        g_fl = g_far + off10
        for kc_v in (0, 9):
            X_true, Yc_b, Wc_b = colluder_data(g_far, g_fl, range(kc_v))
            runs_l = {
                "warm": batched_solve(Yc_b, L_std, Wc_b, gamma0=gamma0_eq, mode="warm_het_t", n_iter=cap_long),
                "centred": two_stage(Yc_b, Wc_b, n_iter=cap_long),
                "three": three_stage(Yc_b, Wc_b, n_iter=cap_long),
            }
            for f_key, X_l in runs_l.items():
                cap_rmse.append((sh_v, kc_v, f_key, cd[(kc_v, f_key)][0], rm_cell(X_l, X_true)[0]))
            if kc_v == 9:
                cap_capt.append((cd[(9, "centred_capt")], capt_pct(runs_l["centred"], X_true, hon9, g_far, g_fl)))
                cap_capt.append((cd[(9, "three_capt")], capt_pct(runs_l["three"], X_true, hon9, g_far, g_fl)))
    for f_lbl, lay_lbl, sh_v, cells_r in rows_ratio:
        f_fn = two_stage if f_lbl == "Two-stage" else three_stage
        idx_f = (lay_adj if lay_lbl == "Adjacent" else lay_sca)(9)
        idx_h = [v_i for v_i in range(N_verts) if v_i not in idx_f]
        g_far = g_base + sh_v * np.array([1.0, 1.0])
        sc_v = sigma_map * float(np.linalg.norm(g_far))
        for rt_v, c_35 in zip(ratio_grid, cells_r):
            g_fl = g_far + rt_v * sc_v * dir_f
            X_true, Yc_b, Wc_b = colluder_data(g_far, g_fl, idx_f)
            cap_capt.append((c_35, capt_pct(f_fn(Yc_b, Wc_b, n_iter=cap_long), X_true, idx_h, g_far, g_fl)))

    # (b) Verified-source boundary with the precision advantage of verified sources removed: every source
    #     is given the lowest calibrated tier, so a verified honest source differs by its exponent alone.
    bound_flat_same = 0
    for kc_v, cells_b in rows_bound:
        X_true, Yc_b, W_raw_s = colluder_data(g_base, g_fl10, range(kc_v))
        idx_h = list(range(kc_v, N_verts))
        for h_v, cell_b in enumerate(cells_b):
            sig_list = [0]*kc_v + list(range(1, h_v + 1)) + [0]*(N_verts - kc_v - h_v)
            X_h = two_stage(Yc_b, np.full_like(W_raw_s, omega_unv), S_b=np.tile(s_hat_for(sig_list), (n_seeds, 1)))
            bound_flat_same += int(capt_pct(X_h, X_true, idx_h, g_base, g_fl10) == cell_b[2])

    # (c) Capture map for nine colluders with every honest precision set to omega_base, so that the
    #     contiguous and the scattered layout differ only in where the colluders sit.
    capt_equal = {}
    for lay_lbl, lay_fn in [("Adjacent", lay_adj), ("Scattered", lay_sca)]:
        idx_f = lay_fn(9)
        idx_h = [v_i for v_i in range(N_verts) if v_i not in idx_f]
        cells_e = []
        for dm_v in capt_deltas:
            g_fl = g_base + dm_v * dir_f
            X_true, Yc_b, Wc_b = colluder_data(g_base, g_fl, idx_f)
            Wc_b[:, idx_h] = omega_base
            cells_e.append(float(100.0 * np.mean(
                honest_drift(two_stage(Yc_b, Wc_b), X_true, idx_h, g_base, g_fl))))
        capt_equal[lay_lbl] = cells_e

    return {
        "mode_gap_seeds": mode_gap_seeds, "stage1_audit": stage1_audit,
        "cap_long": cap_long, "cap_rmse": cap_rmse, "cap_capt": cap_capt,
        "bound_flat_same": bound_flat_same, "capt_equal": capt_equal,
        "main": rows_main, "placement": rows_place, "mode_gaps": mode_gaps,
        "prop4_res": prop4_res, "prop4_exact": prop4_exact, "prop4_map": prop4_map, "sweep": rows_sweep,
        "prop3_thm2": {
            "lam2_mean": float(np.mean(l2_l)), "weyl_mean": float(np.mean(weyl_l)),
            "sin_mean": float(np.mean(st_l)), "wedin_mean": float(np.mean(wed_l)),
            "t2_10": t2_errs[10], "t2_1000": t2_errs[1000],
            "n_pass": n_pass3, "n_seeds": n_seeds,
        },
        "mechanism": mech, "sigma_map": sigma_map,
        "dist_sweep": rows_dist, "ratio_sweep": rows_ratio, "ratio_grid": ratio_grid,
        "coupling_sweep": rows_coup, "capture_map": rows_capt, "capture_deltas": capt_deltas,
        "ingredients": rows_ingr, "ingr_shifted": ingr_shifted, "ingr_gamma_shifted": ingr_gamma_shifted,
        "ingr_unif_fail": ingr_unif_fail,
        "l1_scenarios": rows_l1_scen, "l1_boundary": rows_bound, "bound_near": bound_near,
        "l1_test_bell": l1_test_bell,
        "combined": rows_comb, "comb_deltas": comb_deltas,
    }


def run_experiment_3_spacetime_and_annulus(n_seeds=100, M_ev=20, seed=42):
    """Dial-free 2x2 boost/shear sheaves, low-speed simultaneity check, and exact annulus decision rates."""
    rng = np.random.default_rng(seed)
    edges_tri = [(0, 1), (1, 2), (0, 2)]
    d = 2
    sigma_y = 0.01
    omega_y = 1.0 / (sigma_y ** 2)
    gamma0_def, gamma_sh_fixed = 0.10, 50.0

    def boost_mat(beta):
        g = 1.0 / np.sqrt(1.0 - beta ** 2)
        return g * np.array([[1.0, -beta], [-beta, 1.0]])

    def gal_mat(beta):
        return np.array([[1.0, 0.0], [-beta, 1.0]])

    def build_sys(s_rap, world="lorentz"):
        phis_lor = np.array([0.0, 0.5 * s_rap, 1.1 * s_rap])
        beta_max = float(np.tanh(phis_lor[-1] - phis_lor[0]))
        v_gal = np.array([0.0, (0.5 / 1.1) * beta_max, 1.0 * beta_max])
        d0_g, d0_l = np.zeros((6, 6)), np.zeros((6, 6))
        for e_i, (u, v) in enumerate(edges_tri):
            b_uv = float(np.tanh(phis_lor[v] - phis_lor[u])) if world == "lorentz" else float(v_gal[v] - v_gal[u])
            d0_g[e_i*d:(e_i+1)*d, u*d:(u+1)*d] = gal_mat(b_uv)
            d0_g[e_i*d:(e_i+1)*d, v*d:(v+1)*d] = -np.eye(d)
            d0_l[e_i*d:(e_i+1)*d, u*d:(u+1)*d] = boost_mat(b_uv)
            d0_l[e_i*d:(e_i+1)*d, v*d:(v+1)*d] = -np.eye(d)
        J_t = np.vstack([boost_mat(np.tanh(phis_lor[v])) for v in range(3)]) if world == "lorentz" else \
              np.vstack([gal_mat(v_gal[v]) for v in range(3)])
        return beta_max, d0_g.T @ d0_g, d0_l.T @ d0_l, J_t, J_t @ J_t.T + (sigma_y ** 2) * np.eye(6)

    def exact_ev(Sig_t, ev_L, U_L, g0, gsh):
        tau = omega_y * (g0 + gsh * ev_L) / (omega_y + g0 + gsh * ev_L)
        S_r = np.diag(U_L.T @ Sig_t @ U_L)
        return -0.5 * (6.0 * np.log(2.0 * np.pi) - float(np.sum(np.log(tau))) + float(np.sum(tau * S_r)))

    def batch_ev(Y_sq, ev_L, g0, gsh, oy=omega_y):
        tau = oy * (g0 + gsh * ev_L) / (oy + g0 + gsh * ev_L)
        return 0.5 * M_ev * (float(np.sum(np.log(tau))) - 6.0 * np.log(2.0 * np.pi)) - 0.5 * float(np.sum(tau * Y_sq))

    gl_x, gl_w = np.polynomial.legendre.leggauss(48)
    th_q = 0.25 * np.pi * (gl_x + 1.0)
    w_q = (2.0 / np.pi) * (0.25 * np.pi * gl_w)
    gsh_q = 50.0 * np.tan(th_q)

    def marg_hc(Y_sq, ev_L, g0=gamma0_def, oy=omega_y):
        taus = oy * (g0 + gsh_q[:, None] * ev_L[None, :]) / (oy + g0 + gsh_q[:, None] * ev_L[None, :])
        lls = 0.5 * M_ev * (np.sum(np.log(taus), axis=1) - 6.0 * np.log(2.0 * np.pi)) - \
              0.5 * np.sum(taus * Y_sq[None, :], axis=1)
        mx = np.max(lls)
        return float(mx + np.log(np.sum(w_q * np.exp(lls - mx))))

    def prof_ev(Y_sq, ev_L, g0=gamma0_def):
        r = optimize.minimize_scalar(
            lambda lg: -batch_ev(Y_sq, ev_L, g0, np.exp(lg)), bounds=(-6.0, 16.0), method='bounded')
        return float(-r.fun)

    rows_3a = []
    event_forms = []
    for s_rap in [0.05, 0.30, 0.60, 0.85]:
        b_max, L_gal, L_lor, J_lor, Sig_lor = build_sys(s_rap, "lorentz")
        _, L_gc, L_lc, J_gal, _ = build_sys(s_rap, "galilean")
        ev_g, U_g = np.linalg.eigh(L_gal)
        ev_l, U_l = np.linalg.eigh(L_lor)
        ev_gc, U_gc = np.linalg.eigh(L_gc)
        ev_lc, U_lc = np.linalg.eigh(L_lc)

        ex_fix = exact_ev(Sig_lor, ev_l, U_l, gamma0_def, gamma_sh_fixed) - \
                 exact_ev(Sig_lor, ev_g, U_g, gamma0_def, gamma_sh_fixed)
        ol = optimize.minimize_scalar(
            lambda lg: -exact_ev(Sig_lor, ev_l, U_l, gamma0_def, np.exp(lg)), bounds=(-6, 18), method='bounded')
        og = optimize.minimize_scalar(
            lambda lg: -exact_ev(Sig_lor, ev_g, U_g, gamma0_def, np.exp(lg)), bounds=(-6, 18), method='bounded')
        ex_prof = float(-ol.fun - (-og.fun))

        hc_l_list, hc_gc_list, prof_list, single_bfs = [], [], [], []
        for _ in range(n_seeds):
            evs = rng.normal(0.0, 1.0, size=(2, M_ev))
            Y_l = J_lor @ evs + rng.normal(0.0, sigma_y, size=(6, M_ev))
            pl_sq = np.sum((U_l.T @ Y_l) ** 2, axis=1)
            pg_sq = np.sum((U_g.T @ Y_l) ** 2, axis=1)

            tl = omega_y * (gamma0_def + gamma_sh_fixed * ev_l) / (omega_y + gamma0_def + gamma_sh_fixed * ev_l)
            tg = omega_y * (gamma0_def + gamma_sh_fixed * ev_g) / (omega_y + gamma0_def + gamma_sh_fixed * ev_g)
            c50 = 0.5 * float(np.sum(np.log(tl)) - np.sum(np.log(tg)))
            ebfs = c50 + 0.5 * (np.sum(tg[:, None] * ((U_g.T @ Y_l) ** 2), axis=0) -
                                np.sum(tl[:, None] * ((U_l.T @ Y_l) ** 2), axis=0))
            single_bfs.extend(ebfs.tolist())
            prof_list.append((prof_ev(pl_sq, ev_l) - prof_ev(pg_sq, ev_g)) / M_ev)
            hc_l_list.append((marg_hc(pl_sq, ev_l) - marg_hc(pg_sq, ev_g)) / M_ev)

            Y_g = J_gal @ evs + rng.normal(0.0, sigma_y, size=(6, M_ev))
            plc_sq = np.sum((U_lc.T @ Y_g) ** 2, axis=1)
            pgc_sq = np.sum((U_gc.T @ Y_g) ** 2, axis=1)
            hc_gc_list.append((marg_hc(plc_sq, ev_lc) - marg_hc(pgc_sq, ev_gc)) / M_ev)

        # Without noise the single-event log BF at fixed gamma_sh is a constant plus a quadratic form
        # ev' Q ev in the event's base-frame coordinates ev = (ct, x).  Record Q's two eigenvalues and
        # the size of the spatial component of its dominant eigenvector.
        tl_f = omega_y * (gamma0_def + gamma_sh_fixed * ev_l) / (omega_y + gamma0_def + gamma_sh_fixed * ev_l)
        tg_f = omega_y * (gamma0_def + gamma_sh_fixed * ev_g) / (omega_y + gamma0_def + gamma_sh_fixed * ev_g)
        Q_ev = 0.5 * J_lor.T @ ((U_g * tg_f[None, :]) @ U_g.T - (U_l * tl_f[None, :]) @ U_l.T) @ J_lor
        q_val, q_vec = np.linalg.eigh(Q_ev)
        event_forms.append((float(q_val[0]), float(q_val[1]), float(abs(q_vec[1, 1]))))

        s_arr = np.array(single_bfs)
        rows_3a.append((
            s_rap, b_max, float(ev_g[1]), float(max(ev_l[1], 1e-16)),
            ex_fix, float(np.median(s_arr)), float(np.percentile(s_arr, 75) - np.percentile(s_arr, 25)),
            100.0 * float(np.mean(np.abs(s_arr) < 1.0)),
            ex_prof, float(np.mean(prof_list)), float(np.std(prof_list)),
            float(np.mean(hc_l_list)), float(np.std(hc_l_list)),
            float(np.mean(hc_gc_list)), float(np.std(hc_gc_list)),
        ))

    # Low-speed simultaneity check at s_rap = 0.05 with coarser sensor noise sigma_y = 0.10 > beta_max
    _, L_g05, L_l05, J_l05, _ = build_sys(0.05, "lorentz")
    ev_g05, U_g05 = np.linalg.eigh(L_g05)
    ev_l05, U_l05 = np.linalg.eigh(L_l05)
    rng_sy = np.random.default_rng(seed + 7)
    oy_coarse = 1.0 / (0.10 ** 2)
    hc_coarse = []
    for _ in range(n_seeds):
        evs = rng_sy.normal(0.0, 1.0, size=(2, M_ev))
        Yl = J_l05 @ evs + rng_sy.normal(0.0, 0.10, size=(6, M_ev))
        pl_sq = np.sum((U_l05.T @ Yl) ** 2, axis=1)
        pg_sq = np.sum((U_g05.T @ Yl) ** 2, axis=1)
        hc_coarse.append((marg_hc(pl_sq, ev_l05, oy=oy_coarse) - marg_hc(pg_sq, ev_g05, oy=oy_coarse)) / M_ev)
    low_speed_coarse = (float(np.mean(hc_coarse)), float(np.std(hc_coarse)))

    # Closed-form (gamma_0, gamma_sh) sensitivity & modal share at s_rap = 0.85
    _, L_g85, L_l85, _, Sig_l85 = build_sys(0.85, "lorentz")
    ev_g85, U_g85 = np.linalg.eigh(L_g85)
    ev_l85, U_l85 = np.linalg.eigh(L_l85)
    S_r85 = np.diag(U_g85.T @ Sig_l85 @ U_g85)
    exact_sweep = {}
    for g0 in [0.10, 1e-3, 1e-6]:
        for gsh in [5.0, 50.0, 500.0]:
            ex_bf = exact_ev(Sig_l85, ev_l85, U_l85, g0, gsh) - exact_ev(Sig_l85, ev_g85, U_g85, g0, gsh)
            tl = omega_y * (g0 + gsh * ev_l85) / (omega_y + g0 + gsh * ev_l85)
            tg = omega_y * (g0 + gsh * ev_g85) / (omega_y + g0 + gsh * ev_g85)
            occ = 0.5 * float(np.sum(np.log(tl)) - np.sum(np.log(tg)))
            qm = 0.5 * tg * S_r85
            exact_sweep[(g0, gsh)] = (ex_bf, occ, 100.0 * float(qm[1] / np.sum(qm[1:])))

    # Triangulated annulus 2-complex with moderate (Delta=0.20) & unit (Delta=1.0) raw edge faults
    edges_ann = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
                 (0, 3), (1, 4), (2, 5), (0, 4), (1, 5), (2, 3)]
    e_map = {e: i for i, e in enumerate(edges_ann)}
    faces_ann = [(0, 1, 4), (0, 3, 4), (1, 2, 5), (1, 4, 5), (0, 2, 3), (2, 3, 5)]
    d0_a, d1_a = np.zeros((12, 6)), np.zeros((6, 12))
    for ei, (u, v) in enumerate(edges_ann):
        d0_a[ei, u], d0_a[ei, v] = -1.0, 1.0
    for fi, tri in enumerate(faces_ann):
        u, v, w = sorted(tri)
        d1_a[fi, e_map[(u, v)]], d1_a[fi, e_map[(v, w)]], d1_a[fi, e_map[(u, w)]] = 1.0, 1.0, -1.0
    Delta1 = d0_a @ d0_a.T + d1_a.T @ d1_a
    ev_D1, U_D1 = np.linalg.eigh(Delta1)
    Q_g, _ = np.linalg.qr(d0_a)
    P_grad = Q_g[:, :5] @ Q_g[:, :5].T
    Q_c, _ = np.linalg.qr(d1_a.T)
    P_curl = Q_c[:, :6] @ Q_c[:, :6].T
    h1_v = U_D1[:, np.argmin(np.abs(ev_D1))]
    P_harm = np.outer(h1_v, h1_v)

    y_base = d0_a @ np.array([0.0, 0.4, -0.3, 0.5, -0.2, 0.1])
    e_single, e_cut, e_spoke = np.zeros(12), np.zeros(12), np.zeros(12)
    e_single[e_map[(0, 1)]] = 1.0
    for ce in [(0, 1), (0, 4), (3, 4)]:
        e_cut[e_map[ce]] = 1.0
    e_spoke[e_map[(1, 4)]] = 1.0

    sigma_e = 0.05
    q_c, q_h = float(chi2.ppf(0.99, df=6)), float(chi2.ppf(0.99, df=1))

    def reject_prob(q_v, df_v, lam_v):
        """P(a chi-square statistic with non-centrality lam_v exceeds q_v), exact."""
        return float(chi2.sf(q_v, df_v)) if lam_v < 1e-12 else float(ncx2.sf(q_v, df_v, lam_v))

    # Two-step rule: (i) the curl test rejects -> 'local'; (ii) otherwise the hole test rejects -> 'global'.
    # P_curl and P_harm are orthogonal projectors, so the two statistics are independent and every rate
    # below is exact.  No simulation is needed.
    ann_rows = []
    for name, truth_v, sig_v in [
        ("Consistent potential (null)",           "none",   y_base),
        ("Mod.\\ faulty edge $(0,1)$ ($+0.2$)",   "local",  y_base + 0.20 * e_single),
        ("Mod.\\ branch cut ($+0.2$ on cut)",     "global", y_base + 0.20 * e_cut),
        ("Mod.\\ cut $+$ spoke $(1,4)$ ($+0.2$)", "both",   y_base + 0.20 * (e_cut + e_spoke)),
        ("Unit faulty edge $(0,1)$ ($+1.0$)",     "local",  y_base + 1.00 * e_single),
        ("Unit branch cut ($+1.0$ on cut)",       "global", y_base + 1.00 * e_cut),
        ("Unit cut $+$ spoke $(1,4)$ ($+1.0$)",   "both",   y_base + 1.00 * (e_cut + e_spoke)),
    ]:
        c2 = max(0.0, float(sig_v @ P_curl @ sig_v))
        h2 = max(0.0, float(sig_v @ P_harm @ sig_v))
        p_c = reject_prob(q_c, 6, c2 / sigma_e ** 2)
        p_h = reject_prob(q_h, 1, h2 / sigma_e ** 2)
        ann_rows.append((name, truth_v, float(np.sqrt(c2)), float(np.sqrt(h2)), 100.0 * p_c, 100.0 * p_h,
                         100.0 * p_c, 100.0 * (1.0 - p_c) * p_h, 100.0 * (1.0 - p_c) * (1.0 - p_h)))
    return {
        "event_forms": event_forms, "M_ev": M_ev,
        "rows_3a": rows_3a, "low_speed_coarse": low_speed_coarse,
        "exact_sweep": exact_sweep, "annulus": ann_rows,
    }


def run_experiment_4_gflownet(n_seeds=50, n_steps=900, batch_size=64, seed=42):
    """Sampled mini-batch TB GFlowNet vs. On-Policy RKL, Mean-Field VI, Tabular VI & Forged Angles."""
    n_phi, n_w, n_a = 3, 4, 5

    def eval_phi(p_idx, z, u):
        if p_idx == 0:
            return z % 5
        if p_idx == 1:
            return u % 5
        return (z + u) % 5

    def eval_w(w_idx, s):
        if w_idx == 0:
            return (2 * s + 1) % 5
        if w_idx == 1:
            return (2 * (s ** 2) + 1) % 5
        if w_idx == 2:
            return (3 * s + 1) % 5
        return (4 * s + 2) % 5

    code_len = np.array([[2.0, 2.3, 2.0, 2.2],
                         [2.0, 2.3, 2.0, 2.2],
                         [2.8, 3.1, 2.8, 3.0]])
    prior_M = 2.0 ** (-code_len)
    kraft_sum = float(np.sum(prior_M))          # the nominal lengths are relative: their Kraft sum is not 1
    prior_M /= np.sum(prior_M)

    eps_data = 0.03
    lik_D = np.zeros((n_phi, n_w))
    for pi in range(n_phi):
        for wi in range(n_w):
            p1 = (1.0 - eps_data) if eval_w(wi, eval_phi(pi, 0, 0)) == 1 else eps_data / 4.0
            p2 = (1.0 - eps_data) if eval_w(wi, eval_phi(pi, 1, 1)) == 3 else eps_data / 4.0
            lik_D[pi, wi] = p1 * p2

    def angle_valid_prob(ang_type, pi, wi, a_val):
        pred = eval_w(wi, eval_phi(pi, 2, 0))
        if a_val != pred:
            return 0.02 if ang_type == "A" else 0.0
        if ang_type == "A":
            return 0.84 if (eval_phi(pi, 2, 0) in (0, 1)) else 0.62
        if ang_type == "B":
            return 0.92 if (eval_phi(pi, 2, 0) == eval_phi(pi, 2, 3)) else 0.04
        if ang_type == "C_lin":
            return 0.92 if ((eval_w(wi, 2) - eval_w(wi, 1)) % 5 == (eval_w(wi, 1) - eval_w(wi, 0)) % 5) else 0.04
        if ang_type == "C_quad":
            return 0.92 if ((eval_w(wi, 2) - 2 * eval_w(wi, 1) + eval_w(wi, 0)) % 5 == 4) else 0.04
        return 0.0

    eps_ang = 0.02
    alpha_crp = 1.5
    log_gamma = [0.0] + [float(gammaln(k)) for k in range(1, 12)]

    def compute_reward(angle_types, sigs, sig_mode="verified_deriv", scope_gate=True):
        P_list, sig_list = [], []
        for at, sg in zip(angle_types, sigs):
            mat = np.zeros((n_phi, n_w, n_a))
            for pi in range(n_phi):
                for wi in range(n_w):
                    for av in range(n_a):
                        mat[pi, wi, av] = angle_valid_prob(at, pi, wi, av)
            if scope_gate and np.sum(lik_D[:, :, None] * mat) == 0.0:
                continue
            P_list.append(mat)
            sig_list.append(sg)
        if not P_list:
            # No angle is in scope: the partition sum of the reward defaults to the uniform P_0 over answers
            return prior_M[:, :, None] * lik_D[:, :, None] * np.full((n_phi, n_w, n_a), 1.0 / n_a)
        P_ang = np.stack(P_list, axis=0)
        K_q = len(P_list)
        parts = list(generate_set_partitions(K_q))
        weights = []
        for part in parts:
            lp = len(part) * np.log(alpha_crp) + sum(log_gamma[len(b)] for b in part)
            for b in part:
                w_sub = np.array([sig_list[i] for i in b])
                pw = exchangeable_lex_prob(w_sub, V_sig=25) if sig_mode == "lexical" else \
                     exchangeable_deriv_prob(w_sub, V_sig=25)
                lp += np.log(pw + 1e-300)
            weights.append(lp)
        w_post = softmax(np.array(weights))
        R_tot = np.zeros((n_phi, n_w, n_a))
        for wp, part in zip(w_post, parts):
            prod_part = np.ones((n_phi, n_w, n_a))
            for b in part:
                p_bar = np.mean(P_ang[b], axis=0)
                prod_part *= ((1.0 - eps_ang) * p_bar + eps_ang * (1.0 / n_a))
            R_tot += wp * prod_part
        return prior_M[:, :, None] * lik_D[:, :, None] * R_tot

    def train_all_methods(R_target, run_seed, init_sd=2.5):
        r_gen = np.random.default_rng(run_seed)
        P_true = R_target / np.sum(R_target)
        log_R = np.log(R_target + 1e-30)
        log_PB = -np.log(2.0)

        init_p = [
            r_gen.normal(0, init_sd, 7),
            r_gen.normal(0, init_sd, (n_phi, n_w)),
            r_gen.normal(0, init_sd, (n_w, n_phi)),
            r_gen.normal(0, init_sd, (n_phi, n_w, n_a)),
        ]

        # 1. Off-Policy Trajectory Balance GFlowNet (fresh draws, half of the mass uniform over the 120 trajectories; B=64)
        params = [x.copy() for x in init_p] + [np.array([float(np.log(np.sum(R_target))) + r_gen.normal(0, 1.5)])]
        m_a = [np.zeros_like(p) for p in params]
        v_a = [np.zeros_like(p) for p in params]
        for step in range(1, n_steps + 1):
            l_s0, l_fp, l_fw, l_a, lz = params
            p_s0, p_fp, p_fw, p_a = softmax(l_s0), softmax(l_fp, 1), softmax(l_fw, 1), softmax(l_a, 2)
            pf0 = p_s0[:n_phi, None, None] * p_fp[:, :, None] * p_a
            pf1 = p_s0[n_phi:][None, :, None] * p_fw.T[:, :, None] * p_a
            bf = np.concatenate([(0.5 * pf0 + 0.5 / 120.0).ravel(), (0.5 * pf1 + 0.5 / 120.0).ravel()])
            cnt = r_gen.multinomial(batch_size, bf / np.sum(bf)) / float(batch_size)
            w0, w1 = cnt[:60].reshape(n_phi, n_w, n_a), cnt[60:].reshape(n_phi, n_w, n_a)
            res0 = lz[0] + np.log(pf0 + 1e-30) - log_R - log_PB
            res1 = lz[0] + np.log(pf1 + 1e-30) - log_R - log_PB
            c_s0 = np.zeros(7)
            c_s0[:n_phi] = np.sum(w0 * res0, (1, 2))
            c_s0[n_phi:] = np.sum(w1 * res1, (0, 2))
            cfp = np.sum(w0 * res0, 2)
            cfw = np.sum(w1 * res1, 2).T
            ca = w0 * res0 + w1 * res1
            grads = [
                c_s0 - p_s0 * np.sum(c_s0),
                cfp - p_fp * np.sum(cfp, 1, keepdims=True),
                cfw - p_fw * np.sum(cfw, 1, keepdims=True),
                ca - p_a * np.sum(ca, 2, keepdims=True),
                np.array([float(np.sum(ca))]),
            ]
            for i, gr in enumerate(grads):
                m_a[i] = 0.9 * m_a[i] + 0.1 * gr
                v_a[i] = 0.999 * v_a[i] + 0.001 * (gr ** 2)
                params[i] -= 0.05 * (m_a[i] / (1 - 0.9**step)) / (np.sqrt(v_a[i] / (1 - 0.999**step)) + 1e-8)

        p_s0, p_fp, p_fw, p_a = softmax(params[0]), softmax(params[1], 1), softmax(params[2], 1), softmax(params[3], 2)
        P_gf = (p_s0[:n_phi, None] * p_fp + p_s0[n_phi:][None, :] * p_fw.T)[:, :, None] * p_a
        tv_gf = 0.5 * float(np.sum(np.abs(P_gf - P_true)))
        lz_err = abs(float(params[4][0] - np.log(np.sum(R_target))))

        # 2. On-Policy Trajectory Reverse-KL on the SAME 120-trajectory DAG (B=64, on-policy draws)
        ps_r = [x.copy() for x in init_p]
        mr = [np.zeros_like(x) for x in ps_r]
        vr = [np.zeros_like(x) for x in ps_r]
        for step in range(1, n_steps + 1):
            l_s0, l_fp, l_fw, l_a = ps_r
            p_s0, p_fp, p_fw, p_a = softmax(l_s0), softmax(l_fp, 1), softmax(l_fw, 1), softmax(l_a, 2)
            pf0 = p_s0[:n_phi, None, None] * p_fp[:, :, None] * p_a
            pf1 = p_s0[n_phi:][None, :, None] * p_fw.T[:, :, None] * p_a
            on_f = np.concatenate([pf0.ravel(), pf1.ravel()])
            cnt = r_gen.multinomial(batch_size, on_f / np.sum(on_f)) / float(batch_size)
            w0, w1 = cnt[:60].reshape(n_phi, n_w, n_a), cnt[60:].reshape(n_phi, n_w, n_a)
            adv0 = np.log(pf0 + 1e-30) - (log_R + log_PB)
            adv1 = np.log(pf1 + 1e-30) - (log_R + log_PB)
            bm = float(np.sum(w0 * adv0 + w1 * adv1))
            a0, a1 = w0 * (adv0 - bm), w1 * (adv1 - bm)
            c_s0 = np.zeros(7)
            c_s0[:n_phi] = np.sum(a0, (1, 2))
            c_s0[n_phi:] = np.sum(a1, (0, 2))
            cfp = np.sum(a0, 2)
            cfw = np.sum(a1, 2).T
            ca = a0 + a1
            grads_r = [
                c_s0 - p_s0 * np.sum(c_s0),
                cfp - p_fp * np.sum(cfp, 1, keepdims=True),
                cfw - p_fw * np.sum(cfw, 1, keepdims=True),
                ca - p_a * np.sum(ca, 2, keepdims=True),
            ]
            for i, gr in enumerate(grads_r):
                mr[i] = 0.9 * mr[i] + 0.1 * gr
                vr[i] = 0.999 * vr[i] + 0.001 * (gr ** 2)
                ps_r[i] -= 0.05 * (mr[i] / (1 - 0.9**step)) / (np.sqrt(vr[i] / (1 - 0.999**step)) + 1e-8)
        p_s0, p_fp, p_fw, p_a = softmax(ps_r[0]), softmax(ps_r[1], 1), softmax(ps_r[2], 1), softmax(ps_r[3], 2)
        P_onrkl = (p_s0[:n_phi, None] * p_fp + p_s0[n_phi:][None, :] * p_fw.T)[:, :, None] * p_a
        tv_onrkl = 0.5 * float(np.sum(np.abs(P_onrkl - P_true)))

        # 3. Factorized Mean-Field Reverse-KL VI (CAVI)
        qp = softmax(r_gen.normal(0, 0.5, n_phi))
        qw = softmax(r_gen.normal(0, 0.5, n_w))
        qa = softmax(r_gen.normal(0, 0.5, n_a))
        for _ in range(80):
            qp = softmax(np.sum(qw[None, :, None] * qa[None, None, :] * log_R, axis=(1, 2)))
            qw = softmax(np.sum(qp[:, None, None] * qa[None, None, :] * log_R, axis=(0, 2)))
            qa = softmax(np.sum(qp[:, None, None] * qw[None, :, None] * log_R, axis=(0, 1)))
        P_mfvi = qp[:, None, None] * qw[None, :, None] * qa[None, None, :]
        tv_mfvi = 0.5 * float(np.sum(np.abs(P_mfvi - P_true)))

        # 4. Exact Full-Support 60-Cell Tabular VI (exhaustive state-space gradient)
        ql = r_gen.normal(0, init_sd, (n_phi, n_w, n_a))
        mq, vq = np.zeros_like(ql), np.zeros_like(ql)
        for step in range(1, n_steps + 1):
            q_tab = softmax(ql)
            gr = q_tab * (np.log(q_tab + 1e-30) - log_R + 1.0)
            g_l = gr - q_tab * np.sum(gr)
            mq = 0.9 * mq + 0.1 * g_l
            vq = 0.999 * vq + 0.001 * (g_l ** 2)
            ql -= 0.08 * (mq / (1 - 0.9**step)) / (np.sqrt(vq / (1 - 0.999**step)) + 1e-8)
        tv_tab = 0.5 * float(np.sum(np.abs(softmax(ql) - P_true)))
        return P_gf, tv_gf, tv_onrkl, tv_mfvi, tv_tab, lz_err

    R_single     = compute_reward(["A"], [0], sig_mode="verified_deriv")
    R_para6_lex  = compute_reward(["A"]*6, [10, 11, 12, 13, 14, 15], sig_mode="lexical")
    R_para6_ver  = compute_reward(["A"]*6, [0]*6, sig_mode="verified_deriv")
    R_forged4    = compute_reward(["A"]*4, [10, 11, 12, 13], sig_mode="verified_deriv")
    R_multimodal = compute_reward(["A", "B"], [0, 1], sig_mode="verified_deriv")
    R_cons_lin   = compute_reward(["A", "B", "C_lin"], [0, 1, 2], sig_mode="verified_deriv")
    R_cons_quad  = compute_reward(["A", "B", "C_quad"], [0, 1, 3], sig_mode="verified_deriv")
    R_zero_unscr = compute_reward(["A", "B", "C_lin", "D"], [0, 1, 2, 4], sig_mode="verified_deriv", scope_gate=False)
    R_zero_gated = compute_reward(["A", "B", "C_lin", "D"], [0, 1, 2, 4], sig_mode="verified_deriv", scope_gate=True)
    # Self-test of the empty-scope fallback: the out-of-scope angle alone leaves no active angle
    R_empty = compute_reward(["D"], [4], sig_mode="verified_deriv", scope_gate=True)
    assert R_empty.shape == R_single.shape
    assert np.allclose(R_empty, prior_M[:, :, None] * lik_D[:, :, None] / n_a)

    mm_gf, mm_on, mm_mf, mm_tab = [], [], [], []
    cl_gf, cl_on, cl_mf, lz_cl, p0_cl = [], [], [], [], []
    for s_i in range(n_seeds):
        _, tg, ton, tmf, ttb, _ = train_all_methods(R_multimodal, seed + s_i)
        mm_gf.append(tg); mm_on.append(ton); mm_mf.append(tmf); mm_tab.append(ttb)
        P_c, tgc, tonc, tmfc, _, lzc = train_all_methods(R_cons_lin, seed + 1000 + s_i)
        cl_gf.append(tgc); cl_on.append(tonc); cl_mf.append(tmfc); lz_cl.append(lzc)
        p0_cl.append(float(np.sum(P_c[:, :, 0])))

    # ------------------------------------------------------------------------------------------
    # The two remaining cells of the objective-by-sampling comparison.
    # train_variant() mirrors train_all_methods() step for step: for a given run_seed it draws the
    # same initial logits (and, for trajectory balance, the same initial log Z), and it uses the same
    # Adam settings, batch size and number of steps.  Only the objective and the sampling change.
    # It draws from its own generator, so no number above is affected.
    # ------------------------------------------------------------------------------------------
    def train_variant(R_target, run_seed, objective, explore, init_sd=2.5, blind_z=False, full=False):
        # blind_z=True starts log Z at 0 instead of the exact value plus noise (the noise is still drawn,
        # so the random stream is unchanged); full=True also returns the final |log Z - log Z*|.
        r_v = np.random.default_rng(run_seed)
        P_true = R_target / np.sum(R_target)
        log_R = np.log(R_target + 1e-30)
        log_PB = -np.log(2.0)
        ps_v = [
            r_v.normal(0, init_sd, 7),
            r_v.normal(0, init_sd, (n_phi, n_w)),
            r_v.normal(0, init_sd, (n_w, n_phi)),
            r_v.normal(0, init_sd, (n_phi, n_w, n_a)),
        ]
        if objective == "tb":
            lz_start = float(np.log(np.sum(R_target))) + r_v.normal(0, 1.5)
            ps_v.append(np.array([0.0 if blind_z else lz_start]))
        m_v = [np.zeros_like(p) for p in ps_v]
        v_v = [np.zeros_like(p) for p in ps_v]
        for step in range(1, n_steps + 1):
            p_s0, p_fp, p_fw, p_a = softmax(ps_v[0]), softmax(ps_v[1], 1), softmax(ps_v[2], 1), softmax(ps_v[3], 2)
            pf0 = p_s0[:n_phi, None, None] * p_fp[:, :, None] * p_a
            pf1 = p_s0[n_phi:][None, :, None] * p_fw.T[:, :, None] * p_a
            on_f = np.concatenate([pf0.ravel(), pf1.ravel()])
            beh = (0.5 * on_f + 0.5 / 120.0) if explore else on_f
            beh = beh / np.sum(beh)
            cnt = r_v.multinomial(batch_size, beh) / float(batch_size)
            if objective == "rkl" and explore:
                cnt = cnt * on_f / beh          # importance weights back to the on-policy expectation
            w0, w1 = cnt[:60].reshape(n_phi, n_w, n_a), cnt[60:].reshape(n_phi, n_w, n_a)
            if objective == "tb":
                c0 = w0 * (ps_v[4][0] + np.log(pf0 + 1e-30) - log_R - log_PB)
                c1 = w1 * (ps_v[4][0] + np.log(pf1 + 1e-30) - log_R - log_PB)
            else:
                adv0 = np.log(pf0 + 1e-30) - (log_R + log_PB)
                adv1 = np.log(pf1 + 1e-30) - (log_R + log_PB)
                bm = float(np.sum(w0 * adv0 + w1 * adv1))
                if explore:
                    bm = bm / max(float(np.sum(w0 + w1)), 1e-30)     # self-normalised baseline
                c0, c1 = w0 * (adv0 - bm), w1 * (adv1 - bm)
            c_s0 = np.zeros(7)
            c_s0[:n_phi] = np.sum(c0, (1, 2))
            c_s0[n_phi:] = np.sum(c1, (0, 2))
            cfp = np.sum(c0, 2)
            cfw = np.sum(c1, 2).T
            ca = c0 + c1
            grads_v = [
                c_s0 - p_s0 * np.sum(c_s0),
                cfp - p_fp * np.sum(cfp, 1, keepdims=True),
                cfw - p_fw * np.sum(cfw, 1, keepdims=True),
                ca - p_a * np.sum(ca, 2, keepdims=True),
            ]
            if objective == "tb":
                grads_v.append(np.array([float(np.sum(ca))]))
            for i, gr in enumerate(grads_v):
                m_v[i] = 0.9 * m_v[i] + 0.1 * gr
                v_v[i] = 0.999 * v_v[i] + 0.001 * (gr ** 2)
                ps_v[i] -= 0.05 * (m_v[i] / (1 - 0.9**step)) / (np.sqrt(v_v[i] / (1 - 0.999**step)) + 1e-8)
        p_s0, p_fp, p_fw, p_a = softmax(ps_v[0]), softmax(ps_v[1], 1), softmax(ps_v[2], 1), softmax(ps_v[3], 2)
        P_v = (p_s0[:n_phi, None] * p_fp + p_s0[n_phi:][None, :] * p_fw.T)[:, :, None] * p_a
        tv_v = 0.5 * float(np.sum(np.abs(P_v - P_true)))
        if full:
            return tv_v, abs(float(ps_v[4][0] - np.log(np.sum(R_target))))
        return tv_v

    def tv_stats(tv_list):
        tv_a = np.array(tv_list)
        return float(np.mean(tv_a)), float(np.std(tv_a)), int(round(100.0 * np.mean(tv_a > 0.1)))

    tb_on = ([train_variant(R_multimodal, seed + s_i, "tb", False) for s_i in range(n_seeds)],
             [train_variant(R_cons_lin, seed + 1000 + s_i, "tb", False) for s_i in range(n_seeds)])
    rkl_off = ([train_variant(R_multimodal, seed + s_i, "rkl", True) for s_i in range(n_seeds)],
               [train_variant(R_cons_lin, seed + 1000 + s_i, "rkl", True) for s_i in range(n_seeds)])
    # train_variant() with trajectory balance and exploratory sampling repeats train_all_methods() exactly
    assert abs(train_variant(R_multimodal, seed, "tb", True) - mm_gf[0]) < 1e-12
    assert abs(train_variant(R_cons_lin, seed + 1000, "tb", True) - cl_gf[0]) < 1e-12
    # The same arm with a blind start for log Z: both published trajectory-balance arms start log Z at the
    # exact value plus N(0, 1.5^2) noise, so their final error alone does not show that Z is estimated.
    tb_blind = ([train_variant(R_multimodal, seed + s_i, "tb", True, blind_z=True, full=True) for s_i in range(n_seeds)],
                [train_variant(R_cons_lin, seed + 1000 + s_i, "tb", True, blind_z=True, full=True) for s_i in range(n_seeds)])
    rows_objsamp = [
        ("Trajectory balance", "Exploratory ($50\\%$ uniform)", tv_stats(mm_gf), tv_stats(cl_gf)),
        ("Trajectory balance", "On-policy", tv_stats(tb_on[0]), tv_stats(tb_on[1])),
        ("Reverse KL", "Exploratory, importance-weighted", tv_stats(rkl_off[0]), tv_stats(rkl_off[1])),
        ("Reverse KL", "On-policy", tv_stats(mm_on), tv_stats(cl_on)),
    ]

    def sum_R(R_t):
        P_t = R_t / np.sum(R_t)
        am = np.sum(P_t, axis=(0, 1))
        p0, p1, p4 = float(am[0]), float(am[1]), float(am[4])
        pm = float(np.max(am))
        ba = int(np.argmax(am))
        return p0, p1, p4, pm, (f"Commit $a={ba}$" if pm >= 0.75 else "\\texttt{ABSTAIN}")

    framing_log_priors = np.log(softmax(np.array([
        [+1.1, -0.6, -0.8],
        [-0.8, +1.4, -0.8],
        [-0.6, +1.2, -0.8],
        [+0.9, -0.4, -0.8],
    ]), axis=1))
    log_lik_a0 = np.log(np.array([0.90, 0.25, 0.85]))

    def solve_nuis_exact(lam_sym):
        logits_q = framing_log_priors.copy()
        m_a, v_a = np.zeros_like(logits_q), np.zeros_like(logits_q)
        for st in range(1, 2501):
            qs = softmax(logits_q, axis=1)
            q_bar = np.mean(qs, axis=0, keepdims=True)
            d_neg = qs * (-log_lik_a0[None, :] + (np.log(qs + 1e-30) - framing_log_priors + 1.0)
                          + lam_sym * (np.log(qs + 1e-30) - np.log(q_bar + 1e-30)))
            g_l = d_neg - qs * np.sum(d_neg, axis=1, keepdims=True)
            m_a = 0.9 * m_a + 0.1 * g_l
            v_a = 0.999 * v_a + 0.001 * (g_l ** 2)
            logits_q -= 0.05 * (m_a / (1 - 0.9**st)) / (np.sqrt(v_a / (1 - 0.999**st)) + 1e-8)
        qs = softmax(logits_q, axis=1)
        q_bar = np.mean(qs, axis=0, keepdims=True)
        js_val = float(np.mean(np.sum(qs * (np.log(qs + 1e-30) - np.log(q_bar + 1e-30)), axis=1)))
        raw_elbo = float(np.mean(np.sum(qs * (log_lik_a0[None, :] - np.log(qs + 1e-30) + framing_log_priors), axis=1)))
        return js_val, raw_elbo

    return {
        "kraft_sum": kraft_sum,
        "log_z_true": (float(np.log(np.sum(R_multimodal))), float(np.log(np.sum(R_cons_lin)))),
        "tb_blind": tuple((float(np.mean([r[0] for r in arm])), float(np.std([r[0] for r in arm])),
                           float(np.mean([r[1] for r in arm]))) for arm in tb_blind),
        "rows_cons": [
            ("Single angle $[A]$",                        *sum_R(R_single),     False),
            ("$6\\times$ Paraphr.\\ $[A]^6$ (lexical)",    *sum_R(R_para6_lex),  False),
            ("$6\\times$ Paraphr.\\ $[A]^6$ (verified)",   *sum_R(R_para6_ver),  False),
            ("$4\\times$ Cloned $[A]^4$ (forged IDs)",     *sum_R(R_forged4),    False),
            ("Bimodal invariant $[A, B]$",                *sum_R(R_multimodal), False),
            ("Linear $[A, B, C_{\\mathrm{lin}}]$",         *sum_R(R_cons_lin),   True),
            ("Quadratic $[A, B, C_{\\mathrm{quad}}]$",     *sum_R(R_cons_quad),  True),
            ("$[A,B,C_{\\mathrm{lin}}]+D$ (unscreened)",   *sum_R(R_zero_unscr), False),
            ("$[A,B,C_{\\mathrm{lin}}]+D$ (gated)",        *sum_R(R_zero_gated), True),
        ],
        "mm_gf":  (float(np.mean(mm_gf)),  float(np.std(mm_gf))),
        "mm_on":  (float(np.mean(mm_on)),  float(np.std(mm_on))),
        "mm_mf":  (float(np.mean(mm_mf)),  float(np.std(mm_mf))),
        "mm_tab": (float(np.mean(mm_tab)), float(np.std(mm_tab))),
        "cl_gf":  (float(np.mean(cl_gf)),  float(np.std(cl_gf))),
        "cl_on":  (float(np.mean(cl_on)),  float(np.std(cl_on))),
        "cl_mf":  (float(np.mean(cl_mf)),  float(np.std(cl_mf))),
        "cl_lz":  float(np.mean(lz_cl)),
        "cl_p0":  float(np.mean(p0_cl)),
        "nuis0":  solve_nuis_exact(0.0),
        "nuis2":  solve_nuis_exact(2.0),
        "obj_samp": rows_objsamp,
    }


def export_latex_macros(res1, res2, res3, res4, path="v8_6_results_macros.tex"):
    """Export all table bodies, row macros and inline numbers to v8_6_results_macros.tex."""
    def mac(name, val):
        return "\\newcommand{\\" + name + "}{" + str(val) + "}"

    def sci_tex(x):
        exp = int(np.floor(np.log10(abs(x))))
        mant = x / (10.0 ** exp)
        return f"{mant:.2f} \\times 10^{{{exp}}}"

    lines = ["% Auto-generated by simulate_sheaf_bayes_v8_6.py; do not edit by hand"]

    # --- Scalar Macros for Section 6 Prose ---
    u = res1["unanimity"]
    pk = res1["peaks_p0"]
    s6 = res1["sens6"]
    ir = res1["ind_rumors"]
    lines += [
        mac("resCeil", f"{res1['ceiling']:.4f}"),
        mac("resMCOne", f"{res1['rows_1a'][0][1]:.4f}"),
        mac("resMCSDOne", f"{res1['rows_1a'][0][2]:.4f}"),
        mac("resDipKOne", f"{u[1][0]:.2f}"),
        mac("resDipKThree", f"{u[3][0]:.2f}"),
        mac("resDipKFour", f"{u[4][0]:.2f}"),
        mac("resDipKFive", f"{u[5][0]:.2f}"),
        mac("resDipKTen", f"{u[10][0]:.2f}"),
        mac("resDipKForty", f"{u[40][0]:.2f}"),
        mac("resDipKHund", f"{u[100][0]:.2f}"),
        mac("resCRPKOne", f"{u[1][1]:.2f}"),
        mac("resCRPKThree", f"{u[3][1]:.2f}"),
        mac("resCRPKFour", f"{u[4][1]:.2f}"),
        mac("resCRPKFive", f"{u[5][1]:.2f}"),
        mac("resCRPKTen", f"{u[10][1]:.2f}"),
        mac("resCRPKForty", f"{u[40][1]:.2f}"),
        mac("resCRPKHund", f"{u[100][1]:.2f}"),
        mac("resCRPKThou", f"{u[1000][1]:.2f}"),
        mac("resIndKFour", f"${sci_tex(u[4][2])}$"),
        mac("resIndKFive", f"${sci_tex(u[5][2])}$"),
        mac("resIndKTen", f"${sci_tex(u[10][2])}$"),
        mac("resIndLogForty", f"{u[40][2]:.1f}"),
        mac("resIndLogHund", f"{u[100][2]:.1f}"),
        mac("resIndLogThou", f"{u[1000][2]:.1f}"),
        mac("resPeakHalfK", f"{pk[0.5][0]}"),
        mac("resPeakHalfBF", f"{pk[0.5][1]:.2f}"),
        mac("resPeakTenthK", f"{pk[0.1][0]}"),
        mac("resPeakTenthBF", f"{pk[0.1][1]:.2f}"),
        mac("resPeakHundK", f"{pk[0.01][0]}"),
        mac("resPeakHundBF", f"{pk[0.01][1]:.2f}"),
        mac("resPeakThouK", f"{pk[0.001][0]}"),
        mac("resPeakThouBF", f"{pk[0.001][1]:.2f}"),
        mac("resSensBetaTwo", f"{s6[0]:.4f}"),
        mac("resSensBetaNine", f"{s6[1]:.4f}"),
        mac("resSensVerHigh", f"{s6[2]:.4f}"),
        mac("resIndRumorThree", f"{ir[3]:.4f}"),
        mac("resIndRumorFour", f"{ir[4]:.4f}"),
        mac("resIndRumorFive", f"{ir[5]:.4f}"),
    ]

    mg = res2["mode_gaps"]
    p4 = res2["prop4_res"]
    p3t2 = res2["prop3_thm2"]
    lines += [
        mac("resWarmNine", f"{res2['main'][4][4]:.4f}"),
        mac("resWarmNineSD", f"{res2['main'][4][5]:.4f}"),
        mac("resWarmTen", f"{res2['main'][5][4]:.4f}"),
        mac("resWarmEleven", f"{res2['main'][6][4]:.4f}"),
        mac("resGapLegacyUncapNine", f"{mg[(0.05, 'uncap', 9)]:+.2f}"),
        mac("resGapLegacyCapNine", f"{mg[(0.05, 'cap', 9)]:+.2f}"),
        mac("resGapEqUncapEight", f"{mg[(0.0005, 'uncap', 8)]:+.2f}"),
        mac("resGapEqCapEight", f"{mg[(0.0005, 'cap', 8)]:+.2f}"),
        mac("resGapEqCapNine", f"{mg[(0.0005, 'cap', 9)]:+.2f}"),
        mac("resPropFourFaultMod", f"{p4[50.0][0]:.3f}"),
        mac("resPropFourHonMod", f"{p4[50.0][1]:.3f}"),
        mac("resPropFourFaultStr", f"{p4[5e5][0]:.3f}"),
        mac("resPropFourHonStr", f"{p4[5e5][1]:.3f}"),
        mac("resPropThreeLamTwo", f"${sci_tex(p3t2['lam2_mean'])}$"),
        mac("resPropThreeWeyl", f"${sci_tex(p3t2['weyl_mean'])}$"),
        mac("resPropThreeSin", f"{p3t2['sin_mean']:.5f}"),
        mac("resPropThreeWedin", f"{p3t2['wedin_mean']:.5f}"),
        mac("resThmTwoTen", f"${sci_tex(p3t2['t2_10'])}$"),
        mac("resThmTwoThou", f"${sci_tex(p3t2['t2_1000'])}$"),
    ]

    r3a = res3["rows_3a"]
    esw = res3["exact_sweep"]
    lsc = res3["low_speed_coarse"]
    lines += [
        mac("resSpaceHCmin", f"{r3a[0][11]:+.2f}"),
        mac("resSpaceHCmax", f"{r3a[-1][11]:+.2f}"),
        mac("resSpaceNoisyCtrl", f"{lsc[0]:+.2f}"),
        mac("resSpaceNoisyCtrlSD", f"{lsc[1]:.2f}"),
        mac("resSpaceMedA", f"{r3a[0][5]:+.2f}"),
        mac("resSpaceMedB", f"{r3a[1][5]:+.2f}"),
        mac("resSpaceMedC", f"{r3a[2][5]:+.2f}"),
        mac("resSpaceMedD", f"{r3a[3][5]:+.2f}"),
        mac("resSpaceIQRA", f"{r3a[0][6]:.2f}"),
        mac("resSpaceIQRB", f"{r3a[1][6]:.2f}"),
        mac("resSpaceIQRC", f"{r3a[2][6]:.2f}"),
        mac("resSpaceIQRD", f"{r3a[3][6]:.2f}"),
        mac("resSpaceUninfA", f"{r3a[0][7]:.1f}"),
        mac("resSpaceUninfB", f"{r3a[1][7]:.1f}"),
        mac("resSpaceUninfC", f"{r3a[2][7]:.1f}"),
        mac("resSpaceUninfD", f"{r3a[3][7]:.1f}"),
        mac("resSwGshFiveBF", f"{esw[(0.10, 5.0)][0]:+.2f}"),
        mac("resSwGshFiftyBF", f"{esw[(0.10, 50.0)][0]:+.2f}"),
        mac("resSwGshFiveHundBF", f"{esw[(0.10, 500.0)][0]:+.2f}"),
        mac("resSwGshFiveLam", f"{esw[(0.10, 5.0)][2]:.2f}"),
        mac("resSwGshFiftyLam", f"{esw[(0.10, 50.0)][2]:.2f}"),
        mac("resSwGshFiveHundLam", f"{esw[(0.10, 500.0)][2]:.2f}"),
        mac("resSwGzeroTenthOcc", f"{esw[(0.10, 50.0)][1]:+.2f}"),
        mac("resSwGzeroMilliOcc", f"{esw[(1e-3, 50.0)][1]:+.2f}"),
        mac("resSwGzeroMicroOcc", f"{esw[(1e-6, 50.0)][1]:+.2f}"),
        mac("resSwGzeroTenthBF", f"{esw[(0.10, 50.0)][0]:+.2f}"),
        mac("resSwGzeroMilliBF", f"{esw[(1e-3, 50.0)][0]:+.2f}"),
        mac("resSwGzeroMicroBF", f"{esw[(1e-6, 50.0)][0]:+.2f}"),
    ]

    lines += [
        mac("resGFTVmm", f"{res4['mm_gf'][0]:.4f}"),
        mac("resGFTVmmSD", f"{res4['mm_gf'][1]:.4f}"),
        mac("resOnRKLTVmm", f"{res4['mm_on'][0]:.4f}"),
        mac("resOnRKLTVmmSD", f"{res4['mm_on'][1]:.4f}"),
        mac("resMFVITVmm", f"{res4['mm_mf'][0]:.4f}"),
        mac("resMFVITVmmSD", f"{res4['mm_mf'][1]:.4f}"),
        mac("resTabVITVmm", f"{res4['mm_tab'][0]:.4f}"),
        mac("resTabVITVmmSD", f"{res4['mm_tab'][1]:.4f}"),
        mac("resGFTVcl", f"{res4['cl_gf'][0]:.4f}"),
        mac("resGFTVclSD", f"{res4['cl_gf'][1]:.4f}"),
        mac("resOnRKLTVcl", f"{res4['cl_on'][0]:.4f}"),
        mac("resOnRKLTVclSD", f"{res4['cl_on'][1]:.4f}"),
        mac("resMFVITVcl", f"{res4['cl_mf'][0]:.4f}"),
        mac("resMFVITVclSD", f"{res4['cl_mf'][1]:.4f}"),
        mac("resGFLZerr", f"{res4['cl_lz']:.4f}"),
        mac("resGFPzero", f"{res4['cl_p0']:.4f}"),
        mac("resJSzero", f"{res4['nuis0'][0]:.4f}"),
        mac("resELBOzero", f"{res4['nuis0'][1]:.4f}"),
        mac("resJStwo", f"{res4['nuis2'][0]:.4f}"),
        mac("resELBOtwo", f"{res4['nuis2'][1]:.4f}"),
    ]

    # --- Table Body Macros ---
    def add_table(prefix, row_strs):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        refs = []
        for idx, rs in enumerate(row_strs):
            rname = f"{prefix}{letters[idx]}"
            lines.append(mac(rname, rs))
            refs.append("\\" + rname)
        lines.append(mac(f"{prefix}Body", "\n".join(refs)))

    # Table 2: Exp 1(a)
    sat_rows = []
    for N, mc_m, mc_s, bf_ex, bf_ns, bf_nd in res1["rows_1a"]:
        mc_c = f"${mc_m:.4f} \\pm {mc_s:.4f}$" if mc_m is not None else "---"
        sat_rows.append(
            f"${N}$ & {mc_c} & $\\mathbf{{{bf_ex:.4f}}}$ & ${bf_ns:.4f}$ & ${sci_tex(bf_nd)}$ \\\\")
    add_table("resTabSat", sat_rows)

    # Table 3(a): Exp 1(b) Main Provenance & Forgery
    prov_rows = []
    for m, ns, lc, lp, vs, vfu, vfc in res1["rows_1b"]:
        prov_rows.append(
            f"${m}$ & ${ns:.4f}$ & ${lc:.4f}$ & ${lp:.4f}$ & $\\mathbf{{{vs:.4f}}}$ & ${vfu:.4f}$ & ${vfc:.4f}$ \\\\")
    add_table("resTabProv", prov_rows)

    # Table 3(b): Exp 1(c) Fair Coupled-Mirror Ablation on Identical Data
    mir_rows = []
    for m, m_lex, m_unc, m_co, m_cn in res1["coupled_ablation"]:
        mir_rows.append(
            f"${m}$ & ${m_lex:.4f}$ & ${m_unc:.4f}$ & ${m_co:.4f}$ & $\\mathbf{{{m_cn:.4f}}}$ \\\\")
    add_table("resTabMir", mir_rows)

    # Exp 1(d): growth of the Bayes factor with K unanimous sources
    def big_txt(v):
        return f"{v:.2f}" if v < 1.0e3 else sci_tex(v)

    gr_rows = []
    for K_g in [1, 2, 4, 8, 12]:
        ind_v, ver_v, lex_v, unv_v = res1["growth"][K_g]
        gr_rows.append(f"${K_g}$ & ${big_txt(ind_v)}$ & $\\mathbf{{{big_txt(ver_v)}}}$ & ${big_txt(lex_v)}$ "
                       f"& ${big_txt(unv_v)}$ \\\\")
    add_table("resTabGrowth", gr_rows)

    # Table 4: Exp 2(a) Main Equivariant Sheaf Breakdown (gamma_0 = 0.0005)
    sheaf_rows = []
    for lbl, g_m, n_m, e_m, w_m, w_s in res2["main"]:
        w_cell = f"$\\mathbf{{{w_m:.4f} \\pm {w_s:.4f}}}$" if w_m < 0.06 else f"${w_m:.4f} \\pm {w_s:.4f}$"
        sheaf_rows.append(f"{lbl} & ${g_m:.4f}$ & ${n_m:.4f}$ & ${e_m:.4f}$ & {w_cell} \\\\")
    add_table("resTabSheaf", sheaf_rows)

    # Table 5: Exp 2(b) 8-Row Placement & Prior-Scale Grid
    place_rows = []
    for g0, plbl, cells in res2["placement"]:
        c_strs = []
        for rm_v, fp_v in cells:
            txt = f"{rm_v:.4f}\\,({fp_v}\\%)"
            c_strs.append(f"$\\mathbf{{{txt}}}$" if fp_v == 0 else f"${txt}$")
        place_rows.append(f"${g0}$ & {plbl} & " + " & ".join(c_strs) + " \\\\")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx_p, rs in enumerate(place_rows):
        lines.append(mac(f"resTabPlace{letters[idx_p]}", rs))
    lines.append(mac(
        "resTabPlaceBody",
        "\n".join([f"\\resTabPlace{letters[i]}" for i in range(4)]
                  + ["\\midrule"]
                  + [f"\\resTabPlace{letters[i]}" for i in range(4, 8)])
    ))

    # Fault-size sweep
    sw_rows = []
    for lbl, g_m, n_m, e_m, w_m, w_s in res2["sweep"]:
        w_cell = f"$\\mathbf{{{w_m:.4f} \\pm {w_s:.4f}}}$" if w_m < 0.055 else f"${w_m:.4f} \\pm {w_s:.4f}$"
        sw_rows.append(f"{lbl} & ${g_m:.4f}$ & ${n_m:.4f}$ & ${e_m:.4f}$ & {w_cell} \\\\")
    add_table("resTabSweep", sw_rows)

    # Table 7: Exp 3(a) Spacetime Sheaves
    sp_rows = []
    for r in r3a:
        sp_rows.append(
            f"${r[1]:.2f}c$ & ${sci_tex(r[2])}$ & ${r[4]:+.2f}$ & ${r[9]:+.2f} \\pm {r[10]:.2f}$ "
            f"& $\\mathbf{{{r[11]:+.2f} \\pm {r[12]:.2f}}}$ & ${r[13]:+.2f} \\pm {r[14]:.2f}$ \\\\")
    add_table("resTabSpace", sp_rows)

    # Exp 3(b): annulus, exact rates of the two tests and of the two-step rule's three decisions
    def dec_cell(v, hit):
        return f"$\\mathbf{{{v:.1f}}}$" if hit else f"${v:.1f}$"

    ann_rows = []
    for nm, truth_v, cn, hn, pc, ph, p_loc, p_glo, p_non in res3["annulus"]:
        ann_rows.append(f"{nm} & ${cn:.3f}$ & ${hn:.3f}$ & ${pc:.1f}$ & ${ph:.1f}$ & "
                        f"{dec_cell(p_loc, truth_v == 'local')} & {dec_cell(p_glo, truth_v == 'global')} "
                        f"& {dec_cell(p_non, truth_v == 'none')} \\\\")
    for idx_a, rs in enumerate(ann_rows):
        lines.append(mac(f"resTabAnn{letters[idx_a]}", rs))
    lines.append(mac(
        "resTabAnnBody",
        "\n".join(["\\resTabAnnA", "\\midrule",
                   "\\resTabAnnB", "\\resTabAnnC", "\\resTabAnnD", "\\midrule",
                   "\\resTabAnnE", "\\resTabAnnF", "\\resTabAnnG"])
    ))

    # Table 9: Exp 4 Symbolic 60-State DAG Consilience
    cons_rows = []
    for lbl, p0, p1, p4, pm, act, bold_flag in res4["rows_cons"]:
        p0_s = f"$\\mathbf{{{p0:.4f}}}$" if (bold_flag and p0 > 0.75) else f"${p0:.4f}$"
        p4_s = f"$\\mathbf{{{p4:.4f}}}$" if (bold_flag and p4 > 0.75) else f"${p4:.4f}$"
        act_s = f"\\textbf{{{act}}}" if bold_flag else act
        cons_rows.append(f"{lbl} & {p0_s} & ${p1:.4f}$ & {p4_s} & ${pm:.4f}$ & {act_s} \\\\")
    add_table("resTabCons", cons_rows)

    # --- Experiment 2 sweep tables ---
    def rm_txt(v):
        return f"{v:.1f}" if v >= 100.0 else (f"{v:.2f}" if v >= 10.0 else f"{v:.4f}")

    def cmp_txt(v):
        return f"{v:.1f}" if v >= 100.0 else (f"{v:.2f}" if v >= 10.0 else f"{v:.3f}")

    def pct_cell(c):
        return f"$\\mathbf{{{c[1]}}}$" if c[1] == 0 else f"${c[1]}$"

    def int_cell(n_v):
        return f"$\\mathbf{{{n_v}}}$" if n_v == 0 else f"${n_v}$"

    # Distance sweep: no faulty source (five estimators), then k = 9 colluders at offset 10
    dist_rows = []
    for sh_v, dist_v, cd in res2["dist_sweep"]:
        dist_rows.append(
            f"$({sh_v:g}, {sh_v:g})$ & ${dist_v:.1f}$ & "
            + " & ".join(f"${cmp_txt(cd[(0, kq)][0])}$" for kq in ["raw", "gauss", "warm", "centred", "three"])
            + f" & ${cmp_txt(cd[(9, 'centred')][0])}$ & {int_cell(cd[(9, 'centred_capt')])}"
            + f" & ${cmp_txt(cd[(9, 'oracle')][0])}$ & ${cmp_txt(cd[(9, 'three_hon')])}$"
            + f" & {int_cell(cd[(9, 'three_capt')])} \\\\")
    add_table("resTabDist", dist_rows)

    # Offset-ratio sweep: percent of seeds captured, by offset / (sigma_map |x|)
    ratio_body = []
    for idx_r, (f_lbl, lay_lbl, sh_v, cells_r) in enumerate(res2["ratio_sweep"]):
        lines.append(mac(f"resTabRatio{letters[idx_r]}",
                         f"{f_lbl} & {lay_lbl} & $({sh_v:g}, {sh_v:g})$ & "
                         + " & ".join(int_cell(c) for c in cells_r) + " \\\\"))
        if idx_r == 4:
            ratio_body.append("\\midrule")
        ratio_body.append(f"\\resTabRatio{letters[idx_r]}")
    lines.append(mac("resTabRatioBody", "\n".join(ratio_body)))

    # Coupling sweep (share of seeds with RMSE > 1, in percent): uncentred against two-stage
    coup_rows = []
    for gsh_v, pl_lbl, cells_plain, cells_cent, _cells_three in res2["coupling_sweep"]:
        coup_rows.append(f"${gsh_v:g}$ & {pl_lbl} & " + " & ".join(pct_cell(c) for c in cells_plain)
                         + " & " + " & ".join(pct_cell(c) for c in cells_cent) + " \\\\")
    coup_body = []
    for idx_c, rs in enumerate(coup_rows):
        lines.append(mac(f"resTabCoup{letters[idx_c]}", rs))
        if idx_c > 0 and idx_c % 4 == 0:
            coup_body.append("\\midrule")
        coup_body.append(f"\\resTabCoup{letters[idx_c]}")
    lines.append(mac("resTabCoupBody", "\n".join(coup_body)))

    # Ingredients of the three-stage variant
    def rng_cell(lo_hi):
        lo_v, hi_v = lo_hi
        return int_cell(lo_v) if lo_v == hi_v else f"${lo_v}$--${hi_v}$"

    ingr_body = []
    for idx_i, (v_lbl, ci) in enumerate(res2["ingredients"]):
        lines.append(mac(f"resTabIngr{letters[idx_i]}",
                         f"{v_lbl} & {rng_cell(ci[('fail', 5.0)])} & {rng_cell(ci[('capt', 5.0)])} & "
                         f"{rng_cell(ci[('fail', 50.0)])} & ${cmp_txt(ci[('nofault', 20.0)])}$ & "
                         f"${cmp_txt(ci[('nofault', 1000.0)])}$ & "
                         f"{int_cell(ci['capt_low'])} & {int_cell(ci['capt_high'])} \\\\"))
        if idx_i == 4:
            ingr_body.append("\\midrule")
        ingr_body.append(f"\\resTabIngr{letters[idx_i]}")
    lines.append(mac("resTabIngrBody", "\n".join(ingr_body)))

    # Capture map (honest-vertex displacement toward the false claim, percent of Delta)
    capt_body, prev_key = [], None
    for idx_c, (f_lbl, lay_lbl, kc_v, cells_c) in enumerate(res2["capture_map"]):
        c_strs = []
        for c in cells_c:
            txt = f"{c:.0f}"
            txt = "0" if txt == "-0" else txt
            c_strs.append(f"$\\mathbf{{{txt}}}$" if c < 5.0 else f"${txt}$")
        lines.append(mac(f"resTabCapt{letters[idx_c]}",
                         f"{f_lbl} & {lay_lbl} & ${kc_v}$ & " + " & ".join(c_strs) + " \\\\"))
        if prev_key is not None and (f_lbl, lay_lbl) != prev_key:
            capt_body.append("\\midrule")
        capt_body.append(f"\\resTabCapt{letters[idx_c]}")
        prev_key = (f_lbl, lay_lbl)
    lines.append(mac("resTabCaptBody", "\n".join(capt_body)))

    # Layer-1 signature scenarios (precision division vs. two-stage filter with exponents); mean RMSE
    def l1_cell(c):
        txt = f"{c[0]:.4f}"
        return f"$\\mathbf{{{txt}}}$" if c[0] < 0.07 else f"${txt}$"

    l1e_rows = []
    for row in res2["l1_scenarios"]:
        p_lbl_s, big_div, big_exp, small_div, small_exp, s_cl, _s_hn, sum_cl, sum_hn = row[:9]
        l1e_rows.append(f"{p_lbl_s} & ${s_cl:.3f}$ & ${sum_cl:.2f}$ & ${sum_hn:.2f}$ & "
                        + " & ".join(l1_cell(c) for c in [big_div, big_exp, small_div, small_exp]) + " \\\\")
    add_table("resTabLOneExp", l1e_rows)

    # Verified-source boundary: S_fault / S_hon and percent of seeds captured
    bound_rows = []
    for kc_v, cells_b in res2["l1_boundary"]:
        cs = []
        for s_f, s_h, cp_v in cells_b:
            txt = f"{s_f:.2f}/{s_h:.2f}\\,({cp_v}\\%)"
            cs.append(f"$\\mathbf{{{txt}}}$" if cp_v == 0 else f"${txt}$")
        bound_rows.append(f"$k = {kc_v}$ & " + " & ".join(cs) + " \\\\")
    add_table("resTabBound", bound_rows)

    # Combined filter under a clone majority: percent of seeds captured, by offset
    comb_body = []
    for idx_m, (pl_lbl, f_lbl, cells_m) in enumerate(res2["combined"]):
        lines.append(mac(f"resTabComb{letters[idx_m]}",
                         f"{pl_lbl} & {f_lbl} & " + " & ".join(int_cell(c) for c in cells_m) + " \\\\"))
        if idx_m == 4:
            comb_body.append("\\midrule")
        comb_body.append(f"\\resTabComb{letters[idx_m]}")
    lines.append(mac("resTabCombBody", "\n".join(comb_body)))

    # Experiment 4, objective by sampling (TV mean +- SD and share of seeds with TV > 0.1)
    def tv_cell(st):
        txt = f"{st[0]:.4f} \\pm {st[1]:.4f}"
        return f"$\\mathbf{{{txt}}}$" if st[2] == 0 else f"${txt}$"

    os_rows = []
    for o_lbl, s_lbl, st_mm, st_cl in res4["obj_samp"]:
        os_rows.append(f"{o_lbl} & {s_lbl} & {tv_cell(st_mm)} & ${st_mm[2]}\\%$ & {tv_cell(st_cl)} & ${st_cl[2]}\\%$ \\\\")
    add_table("resTabObjSamp", os_rows)


    # --- Scalar macros added in Version 7 ---
    omega_base_x = 160.0
    omega_cap_x = float(omega_base_x * (norm.ppf(0.9999) ** 2))
    omega_tiers_x = [float(omega_base_x * (norm.ppf(p) ** 2)) for p in [0.94, 0.84, 0.72]]
    ds_map = {int(round(r[0])): r for r in res2["dist_sweep"]}
    l1s = res2["l1_scenarios"]
    os_m = res4["obj_samp"]
    lines += [
        mac("resExBFOne", f"{res1['rows_1a'][0][3]:.4f}"),
        mac("resNaivePeakThree", f"{res1['rows_1a'][1][4]:.4f}"),
        mac("resNaiveOneThou", f"{res1['rows_1a'][4][4]:.4f}"),
        mac("resNaiveDivOneThou", f"${sci_tex(res1['rows_1a'][4][5])}$"),
        mac("resParaSix", f"{res1['rows_1b'][3][3]:.4f}"),
        mac("resSoundSix", f"{res1['rows_1b'][3][4]:.4f}"),
        mac("resForgedUncoupSix", f"{res1['rows_1b'][3][5]:.4f}"),
        mac("resForgedCoupSix", f"{res1['rows_1b'][3][6]:.4f}"),
        mac("resMirUncMin", f"{res1['coupled_ablation'][0][2]:.4f}"),
        mac("resMirUncMax", f"{res1['coupled_ablation'][3][2]:.4f}"),
        mac("resMirCoupMin", f"{res1['coupled_ablation'][0][3]:.4f}"),
        mac("resMirCoupMax", f"{res1['coupled_ablation'][3][3]:.4f}"),
        mac("resMirNoisyMin", f"{res1['coupled_ablation'][0][4]:.4f}"),
        mac("resMirNoisyMax", f"{res1['coupled_ablation'][3][4]:.4f}"),
        mac("resOmegaCap", f"{omega_cap_x:.1f}"),
        mac("resOmegaCapMult", f"{omega_cap_x / omega_base_x:.2f}"),
        mac("resOmegaTierHigh", f"{omega_tiers_x[0]:.1f}"),
        mac("resOmegaTierMid", f"{omega_tiers_x[1]:.1f}"),
        mac("resOmegaTierLow", f"{omega_tiers_x[2]:.1f}"),
        mac("resDistWarmThreeHund", rm_txt(ds_map[300][2][(0, 'warm')][0])),
        mac("resDistWarmThou", rm_txt(ds_map[1000][2][(0, 'warm')][0])),
        mac("resDistCentThreeHund", rm_txt(ds_map[300][2][(0, 'centred')][0])),
        mac("resDistCentThou", rm_txt(ds_map[1000][2][(0, 'centred')][0])),
        mac("resDistNineCentThreeHund", rm_txt(ds_map[300][2][(9, 'centred')][0])),
        mac("resLOneMixDiv", f"{l1s[1][1][0]:.4f}"),
        mac("resLOneMixDivUnc", f"{l1s[1][11][0]:.4f}"),
        mac("resLOneExactTestBell", f"{res2['l1_test_bell']}"),
        mac("resLOneMixExp", f"{l1s[1][2][0]:.4f}"),
        mac("resTBOnTVmm", f"{os_m[1][2][0]:.4f}"),
        mac("resTBOnSDmm", f"{os_m[1][2][1]:.4f}"),
        mac("resTBOnFailmm", f"{os_m[1][2][2]}"),
        mac("resTBOnTVcl", f"{os_m[1][3][0]:.4f}"),
        mac("resTBOnSDcl", f"{os_m[1][3][1]:.4f}"),
        mac("resTBOnFailcl", f"{os_m[1][3][2]}"),
        mac("resRKLOffTVmm", f"{os_m[2][2][0]:.4f}"),
        mac("resRKLOffSDmm", f"{os_m[2][2][1]:.4f}"),
        mac("resRKLOffFailmm", f"{os_m[2][2][2]}"),
        mac("resRKLOffTVcl", f"{os_m[2][3][0]:.4f}"),
        mac("resRKLOffSDcl", f"{os_m[2][3][1]:.4f}"),
        mac("resRKLOffFailcl", f"{os_m[2][3][2]}"),
        mac("resRKLOnFailmm", f"{os_m[3][2][2]}"),
        mac("resRKLOnFailcl", f"{os_m[3][3][2]}"),
    ]

    # --- Scalar macros added in Version 8 ---
    p4, p4x, p4m = res2["prop4_res"], res2["prop4_exact"], res2["prop4_map"]
    mech, sm_v = res2["mechanism"], res2["sigma_map"]
    d0c, d100, d300, d1000 = ds_map[0][2], ds_map[100][2], ds_map[300][2], ds_map[1000][2]
    sc300 = sm_v * ds_map[300][1]

    def traj_txt(seq):
        return ", ".join(f"{v:.4f}" for v in seq[1:5])

    gap_map = max(abs(a_v - b_v) for key in ["uniform", "exponent"] for a_v, b_v in zip(*p4m[key]))
    rs_all, grid = res2["ratio_sweep"], res2["ratio_grid"]

    def ratio_rows(f_lbl, lay_lbl):
        return [r for r in rs_all if r[0] == f_lbl and r[1] == lay_lbl]

    def clear_from(rows_r):
        for i_g, g_v in enumerate(grid):
            if all(c == 0 for r in rows_r for c in r[3][i_g:]):
                return g_v
        return float("nan")

    cp_s = [r for r in res2["coupling_sweep"] if r[0] >= 50.0]
    cp_w = [r for r in res2["coupling_sweep"] if r[0] < 50.0]
    cm = {(r[0], r[1], r[2]): r[3] for r in res2["capture_map"]}
    cdl = res2["capture_deltas"]
    adj9, sca9 = cm[("Two-stage", "Adjacent", 9)], cm[("Two-stage", "Scattered", 9)]
    adj9t = cm[("Three-stage", "Adjacent", 9)]
    low9 = [adj9[i_d] for i_d, dv in enumerate(cdl) if dv <= 2.0]
    cb = res2["combined"]
    ig = [r[1] for r in res2["ingredients"]]
    ig_sh = res2["ingr_shifted"]
    bd = res2["l1_boundary"]
    bd_first = [min(h_v for h_v, c_b in enumerate(cells_b) if c_b[2] == 0) for _k, cells_b in bd]
    bd_ratio = {(k_b, h_v): c_b[0] / c_b[1] for k_b, cells_b in bd for h_v, c_b in enumerate(cells_b)}
    bd_near = min(bd_ratio, key=lambda kh: abs(bd_ratio[kh] - 1.0))
    assert bd_near == (16, 3)
    # every cell of the boundary table is decided by the sign of S_fault - S_hon
    bd_rule = all(c_b[2] == (100 if c_b[0] > c_b[1] else 0) for _k, cells_b in bd for c_b in cells_b)
    an = res3["annulus"]
    gw = res1["growth"]
    lex_k = max(gw, key=lambda k_g: gw[k_g][2])
    lines += [
        mac("resPropThreePass", f"{p3t2['n_pass']}"),
        mac("resPropThreeSeeds", f"{p3t2['n_seeds']}"),
        mac("resPropFourFaultExact", f"{p4x[5e5][0]:.3f}"),
        mac("resPropFourHonExact", f"{p4x[5e5][1]:.3f}"),
        mac("resPropFourFaultCollapse", f"{p4[5e7][0]:.2f}"),
        mac("resPropFourHonCollapse", f"{p4[5e7][1]:.2f}"),
        mac("resPropFourTrajEM", traj_txt(p4m["uniform"][0])),
        mac("resPropFourTrajMap", traj_txt(p4m["uniform"][1])),
        mac("resPropFourTrajEMExp", traj_txt(p4m["exponent"][0])),
        mac("resPropFourTrajMapExp", traj_txt(p4m["exponent"][1])),
        mac("resPropFourMapGap", f"${sci_tex(gap_map)}$"),
        mac("resMechNonHarmCoef", f"{mech['nonharm_coef']:.2f}"),
        mac("resMechWarmHarm", f"{mech['warm_res_harm']:.4f}"),
        mac("resMechWarmNonHarm", f"{mech['warm_res_nonharm']:.4f}"),
        mac("resMechCoreMin", f"{mech['core_min']:.2f}"),
        mac("resMechCoreMax", f"{mech['core_max']:.2f}"),
        mac("resRawRMSE", f"{d0c[(0, 'raw')][0]:.4f}"),
        mac("resFloorCentCoef", f"{d300[(0, 'centred')][0] / sc300:.2f}"),
        mac("resFloorGaussCoef", f"{d300[(0, 'gauss')][0] / sc300:.2f}"),
        mac("resDistGaussHund", rm_txt(d100[(0, 'gauss')][0])),
        mac("resDistGaussThreeHund", rm_txt(d300[(0, 'gauss')][0])),
        mac("resDistThreeHund", rm_txt(d100[(0, 'three')][0])),
        mac("resDistThreeThreeHund", rm_txt(d300[(0, 'three')][0])),
        mac("resDistThreeThou", rm_txt(d1000[(0, 'three')][0])),
        mac("resDistNineCaptHund", f"{d100[(9, 'centred_capt')]}"),
        mac("resDistNineCaptThreeHund", f"{d300[(9, 'centred_capt')]}"),
        mac("resDistNineDriftThreeHund", f"{d300[(9, 'centred_drift')]:.0f}"),
        mac("resDistNineOracleThreeHund", rm_txt(d300[(9, 'oracle')][0])),
        mac("resDistNineThreeHonThreeHund", rm_txt(d300[(9, 'three_hon')])),
        mac("resDistNineThreeAllThreeHund", rm_txt(d300[(9, 'three')][0])),
        mac("resDistNineThreeFaultThreeHund", rm_txt(d300[(9, 'three_fault')])),
        mac("resDistRatioThreeHund", f"{10.0 / sc300:.1f}"),
        mac("resRatioAdjLowMin", f"{min(r[3][0] for r in ratio_rows('Two-stage', 'Adjacent'))}"),
        mac("resRatioAdjLowMax", f"{max(r[3][0] for r in ratio_rows('Two-stage', 'Adjacent'))}"),
        mac("resRatioClearAdj", f"{clear_from(ratio_rows('Two-stage', 'Adjacent')):g}"),
        mac("resRatioClearSca", f"{clear_from(ratio_rows('Two-stage', 'Scattered')):g}"),
        mac("resRatioThreeAdjMax", f"{max(c for r in ratio_rows('Three-stage', 'Adjacent') for c in r[3])}"),
        mac("resRatioThreeScaMax", f"{max(c for r in ratio_rows('Three-stage', 'Scattered') for c in r[3])}"),
        mac("resRatioThreeScaClear", f"{clear_from(ratio_rows('Three-stage', 'Scattered')):g}"),
        mac("resCoupCentMaxEightNine", f"{max(c[1] for r in cp_s for c in r[3][:2])}"),
        mac("resCoupCentTenMin", f"{min(r[3][2][1] for r in cp_s)}"),
        mac("resCoupCentTenMax", f"{max(r[3][2][1] for r in cp_s)}"),
        mac("resCoupThreeMaxEightNine", f"{max(c[1] for r in cp_s for c in r[4][:2])}"),
        mac("resCoupCentWeakMinEightNine", f"{min(c[1] for r in cp_w for cells_w in (r[2], r[3]) for c in cells_w)}"),
        mac("resCoupThreeWeakMaxEightNine", f"{max(c[1] for r in cp_w for c in r[4][:2])}"),
        mac("resCaptAdjNineLowMin", f"{min(low9):.0f}"),
        mac("resCaptAdjNineLowMax", f"{max(low9):.0f}"),
        mac("resCaptAdjNineAtTwoFive", f"{adj9[cdl.index(2.5)]:.0f}"),
        mac("resCaptScaNineAtOne", f"{sca9[cdl.index(1.0)]:.0f}"),
        mac("resCaptThreeAdjNineAtOne", f"{adj9t[cdl.index(1.0)]:.0f}"),
        mac("resCaptThreeAdjNineAtOneTwoFive", f"{adj9t[cdl.index(1.25)]:.0f}"),
        mac("resLOneCloneShat", f"{l1s[0][5]:.3f}"),
        mac("resLOneCloneSum", f"{l1s[0][7]:.2f}"),
        mac("resLOneSharedSum", f"{l1s[2][7]:.2f}"),
        mac("resLOneMixClone", f"{l1s[1][5]:.3f}"),
        mac("resLOneMixHonVer", f"{l1s[1][10]:.3f}"),
        mac("resLOneMixSf", f"{l1s[1][7]:.2f}"),
        mac("resLOneMixSh", f"{l1s[1][8]:.2f}"),
        mac("resLOneAllSf", f"{l1s[4][7]:.2f}"),
        mac("resLOneAllSh", f"{l1s[4][8]:.2f}"),
        mac("resLOneSmallSharedDiv", f"{l1s[2][3][0]:.4f}"),
        mac("resLOneSmallSharedExp", f"{l1s[2][4][0]:.4f}"),
        mac("resBoundFirstHTwelve", f"{bd_first[0]}"),
        mac("resBoundFirstHFourteen", f"{bd_first[1]}"),
        mac("resBoundFirstHSixteen", f"{bd_first[2]}"),
        mac("resBoundFirstRatioTwelve", f"{bd_ratio[(12, bd_first[0])]:.2f}"),
        mac("resBoundFirstRatioFourteen", f"{bd_ratio[(14, bd_first[1])]:.2f}"),
        mac("resBoundFirstRatioSixteen", f"{bd_ratio[(16, bd_first[2])]:.2f}"),
        mac("resBoundNearRatio", f"{bd_ratio[bd_near]:.4f}"),
        mac("resBoundNearCapt", f"{bd[2][1][3][2]}"),
        mac("resBoundNearCaptStrong", f"{res2['bound_near']['two_strong']}"),
        mac("resBoundNearCaptThree", f"{res2['bound_near']['three']}"),
        mac("resBoundRuleCells", f"{sum(len(cells_b) for _k, cells_b in bd) if bd_rule else 0}"),
        mac("resIngrTwoWeakCaptMin", f"{ig[0][('capt', 5.0)][0]}"),
        mac("resIngrFitWeakFailMax", f"{max(ig[1][('fail', 5.0)][1], ig[2][('fail', 5.0)][1])}"),
        mac("resIngrWarmWeakFailMin", f"{min(ig[0][('fail', 5.0)][0], ig[3][('fail', 5.0)][0])}"),
        mac("resIngrWarmWeakCaptMin", f"{min(ig[0][('capt', 5.0)][0], ig[3][('capt', 5.0)][0])}"),
        mac("resIngrMapShiftedFailEight", f"{ig[3][('fail', 50.0, ig_sh, 8)]}"),
        mac("resIngrMapShiftedFailNine", f"{ig[3][('fail', 50.0, ig_sh, 9)]}"),
        mac("resIngrMapElseFailMax", f"{max(v for k_q, v in ig[3].items() if len(k_q) == 4 and k_q[:2] == ('fail', 50.0) and k_q[2] != ig_sh)}"),
        mac("resIngrGammaShifted", f"{res2['ingr_gamma_shifted']:.1f}"),
        mac("resIngrUnifShiftedFailEight", f"{res2['ingr_unif_fail'][8]}"),
        mac("resIngrUnifShiftedFailNine", f"{res2['ingr_unif_fail'][9]}"),
        mac("resIngrMapHighCaptMax", f"{ig[3]['capt_high']}"),
        mac("resIngrMapNoFaultThou", f"{ig[3][('nofault', 1000.0)]:.4f}"),
        mac("resIngrFitNoFaultThou", f"{ig[2][('nofault', 1000.0)]:.4f}"),
        mac("resIngrFitLowCaptMax", f"{ig[2]['capt_low']}"),
        mac("resIngrThreeLowCaptMax", f"{ig[1]['capt_low']}"),
        mac("resIngrThreeNoFaultTwenty", f"{ig[1][('nofault', 20.0)]:.4f}"),
        mac("resIngrHalfNoFaultThou", f"{ig[4][('nofault', 1000.0)]:.4f}"),
        mac("resIngrHalfLowCaptMax", f"{ig[4]['capt_low']}"),
        mac("resIngrDoubleNoFaultTwenty", f"{ig[5][('nofault', 20.0)]:.4f}"),
        mac("resIngrDoubleNoFaultThou", f"{ig[5][('nofault', 1000.0)]:.4f}"),
        mac("resIngrDoubleCaptMax", f"{max(ig[5]['capt_low'], ig[5]['capt_high'])}"),
        mac("resCombUnwTruthTwenty", f"{cb[1][2][1]}"),
        mac("resCombUnwTruthForty", f"{cb[1][2][2]}"),
        mac("resCombExpMirrorForty", f"{cb[4][2][2]}"),
        mac("resCombTwoMax", f"{max(cb[2][2] + cb[6][2])}"),
        mac("resCombThreeMax", f"{max(cb[3][2] + cb[7][2])}"),
        mac("resTBOffFailmm", f"{os_m[0][2][2]}"),
        mac("resTBOffFailcl", f"{os_m[0][3][2]}"),
        mac("resGrowthVerRate", f"{np.sqrt(gw[12][1] / gw[10][1]):.2f}"),
        mac("resGrowthIndRate", f"{gw[1][0]:.2f}"),
        mac("resGrowthLexPeakK", f"{lex_k}"),
        mac("resGrowthLexPeak", f"{gw[lex_k][2]:.0f}"),
        mac("resGrowthLexTwelve", f"{gw[12][2]:.0f}"),
        mac("resAnnNullAlarm", f"{an[0][6] + an[0][7]:.1f}"),
        mac("resAnnEdgeModLocal", f"{an[1][6]:.1f}"),
        mac("resAnnEdgeModGlobal", f"{an[1][7]:.1f}"),
        mac("resAnnEdgeModNone", f"{an[1][8]:.1f}"),
        mac("resAnnCutModLocal", f"{an[2][6]:.1f}"),
        mac("resAnnCutModGlobal", f"{an[2][7]:.1f}"),
        mac("resAnnMixModLocal", f"{an[3][6]:.1f}"),
        mac("resAnnMixModGlobal", f"{an[3][7]:.1f}"),
        mac("resAnnEdgeUnitLocal", f"{an[4][6]:.1f}"),
        mac("resAnnCutUnitGlobal", f"{an[5][7]:.1f}"),
        mac("resAnnMixUnitLocal", f"{an[6][6]:.1f}"),
    ]

    # --- Macros added in Version 8.6 (appended, so the earlier macros keep their order) ---
    mgs = res2["mode_gap_seeds"]
    gap_cited = [(0.05, "uncap", 9), (0.05, "cap", 9), (0.0005, "uncap", 8), (0.0005, "cap", 8), (0.0005, "cap", 9)]
    sa = res2["stage1_audit"]
    cap_rel = [abs(r[4] - r[3]) / r[3] for r in res2["cap_rmse"]]
    cap_dif = [abs(c[1] - c[0]) for c in res2["cap_capt"]]
    ceq, cdl6 = res2["capt_equal"], res2["capture_deltas"]
    eq_low = [ceq["Adjacent"][cdl6.index(dv)] for dv in cdl6 if dv <= 1.5]
    efm = res3["event_forms"]
    tbb, lzt = res4["tb_blind"], res4["log_z_true"]
    lines += [
        mac("resDipKThou", f"{u[1000][0]:.2f}"),
        mac("resAlphaInf", f"{gw[1][0]:.2f}"),
        mac("resGapEqCapEightMin", f"{mgs[(0.0005, 'cap', 8)][0]:+.2f}"),
        mac("resGapSignSeeds", f"{min(mgs[key][2] for key in gap_cited)}"),
        mac("resStageOneFits", f"{sa['fits']:,}".replace(",", "{,}")),
        mac("resStageOneLoose", f"{sa['loose']}"),
        # the largest certified gap is a rounding-level number, so only its power-of-ten ceiling is exported
        mac("resStageOneGapCert", f"$10^{{{int(np.ceil(np.log10(max(sa['gap_certified'], 1e-300))))}}}$"),
        mac("resCapRmseCells", f"{len(cap_rel)}"),
        mac("resCapRmseMaxRel", f"{100.0 * max(cap_rel):.2f}"),
        mac("resCapCaptCells", f"{len(cap_dif)}"),
        mac("resCapCaptChanged", f"{sum(1 for dv in cap_dif if dv > 0)}"),
        mac("resCapCaptMaxDiff", f"{max(cap_dif)}"),
        mac("resBoundFlatSame", f"{res2['bound_flat_same']}"),
        mac("resCaptEqAdjNineLowMin", f"{min(eq_low):.0f}"),
        mac("resCaptEqAdjNineLowMax", f"{max(eq_low):.0f}"),
        mac("resCaptEqAdjNineAtTwo", f"{ceq['Adjacent'][cdl6.index(2.0)]:.0f}"),
        mac("resCaptEqScaNineAtOne", f"{ceq['Scattered'][cdl6.index(1.0)]:.0f}"),
        mac("resZeroCaptUpper", f"{100.0 * (1.0 - 0.05 ** (1.0 / p3t2['n_seeds'])):.2f}"),
        mac("resSpaceNoisyCtrlTotal", f"{res3['M_ev'] * lsc[0]:+.2f}"),
        mac("resSpaceFormRatioMax", f"{100.0 * max(abs(q[0]) / q[1] for q in efm):.1f}"),
        mac("resSpaceFormXLow", f"{efm[0][2]:.2f}"),
        mac("resSpaceFormXHigh", f"{efm[-1][2]:.2f}"),
        mac("resKraftSum", f"{res4['kraft_sum']:.2f}"),
        mac("resKraftBits", f"{np.log2(res4['kraft_sum']):.3f}"),
        mac("resELBOtwoPen", f"{res4['nuis2'][1] - 2.0 * res4['nuis2'][0]:.4f}"),
        mac("resLogZmm", f"{lzt[0]:.2f}"),
        mac("resLogZcl", f"{lzt[1]:.2f}"),
        mac("resTBBlindTVmm", f"{tbb[0][0]:.4f}"),
        mac("resTBBlindTVcl", f"{tbb[1][0]:.4f}"),
        mac("resTBBlindLZerr", f"{tbb[1][2]:.4f}"),
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_figures(res1, res2, res3):
    # 300-DPI PNG figures
    import matplotlib.pyplot as plt
    r1a = [r for r in res1["rows_1a"] if r[1] is not None]
    r1b = res1["rows_1b"]
    ceil = res1["ceiling"]
    r2 = res2["main"]
    r_dist = res2["dist_sweep"]
    r3 = res3["rows_3a"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    Ns = [r[0] for r in r1a]
    axes[0].plot(Ns, [r[4] for r in r1a], 'o--', label=r'Naive i.i.d. Utterance BF ($\delta < \eta$)')
    axes[0].plot(Ns, [r[3] for r in r1a], 's-', linewidth=2.2, label=r'3-Level Latent-Belief BF (Exact)')
    axes[0].errorbar(Ns, [r[1] for r in r1a], yerr=[r[2] for r in r1a], fmt='x', color='black',
                     markersize=7, capsize=3, label=r'Utterance MC ($5\times 10^6$ ch.)')
    axes[0].axhline(ceil, color='crimson', linestyle=':', label=rf'Ceiling $A/C = {ceil:.2f}$')
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'Repetitions $N$ within Single Channel')
    axes[0].set_ylabel(r'Bayes Factor $\mathrm{BF}_{0:1}(N)$')
    axes[0].set_title(r'(a) Utterance-Level Testimony Saturation')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    ms = [r[0] for r in r1b]
    axes[1].plot(ms, [r[1] for r in r1b], '^-.', color='purple', label=r'Exchangeable CRP (No Sig.)')
    axes[1].plot(ms, [r[3] for r in r1b], 'v--', color='crimson', label=r'Lexical Sig. (Paraphrase Attack)')
    axes[1].plot(ms, [r[5] for r in r1b], 'x:', color='darkorange', label=r'Unsound Verifier (Forged IDs)')
    axes[1].plot(ms, [r[4] for r in r1b], 's-', color='teal', linewidth=2.2, label=r'Sound Verified Derivations')
    axes[1].set_xlabel(r'Cloned False Outlets $m$ (vs. $K=4$ Independent True)')
    axes[1].set_ylabel(r'Posterior $P(M=1 \mid \mathcal{D})$')
    axes[1].set_title(r'(b) Paraphrase & Forgery vs. Sound Derivations')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=7.8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig1_testimony_saturation_v8_6.png', dpi=300)
    plt.close(fig)

    fig, axes2 = plt.subplots(1, 2, figsize=(11.6, 4.4))
    short_lbls = [r'Clean', r'Miscal.', r'$10^3\times$', r'Cap $k=8$', r'$k=9$', r'$k=10$', r'$k=11$']
    x_pos = np.arange(len(r2))
    w = 0.2
    axes2[0].bar(x_pos - 1.5 * w, [r[1] for r in r2], w, label=r'Het. Gaussian MAP', color='#d95f02')
    axes2[0].bar(x_pos - 0.5 * w, [r[2] for r in r2], w, label=r'Nominal-Seeded-$t$', color='#7570b3')
    axes2[0].bar(x_pos + 0.5 * w, [r[3] for r in r2], w, label=r'Equal-$t$', color='#e6ab02')
    axes2[0].bar(x_pos + 1.5 * w, [r[4] for r in r2], w, label=r'Fixed-Warm Het.-$t$', color='#1b9e77')
    axes2[0].set_yscale('log')
    axes2[0].set_ylim(2e-2, 90.0)
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels(short_lbls, rotation=18, fontsize=8.2)
    axes2[0].set_ylabel(r'Global Consensus RMSE (log scale)')
    axes2[0].set_title(r'(a) Near-Origin Breakdown ($\|x_{*,v}\|_2 = 1, \gamma_0 = 5\times 10^{-4}$)')
    axes2[0].legend(fontsize=7.4, loc='upper left', ncol=2, framealpha=0.92)
    axes2[0].grid(True, axis='y', alpha=0.3)

    d_norms = [r[1] for r in r_dist]
    axes2[1].plot(d_norms, [r[2][(0, 'raw')][0] for r in r_dist], ':', color='gray', linewidth=1.6,
                  label=r'Each source read directly')
    axes2[1].plot(d_norms, [r[2][(0, 'gauss')][0] for r in r_dist], 'D-', color='#d95f02', markersize=5,
                  label=r'Het. Gaussian MAP')
    axes2[1].plot(d_norms, [r[2][(0, 'warm')][0] for r in r_dist], 'o--', color='crimson',
                  label=r'Uncentred Fixed-Warm Het.-$t$')
    axes2[1].plot(d_norms, [r[2][(0, 'centred')][0] for r in r_dist], 's-.', color='#7570b3',
                  label=r'Two-stage filter')
    axes2[1].plot(d_norms, [r[2][(0, 'three')][0] for r in r_dist], '^-', color='#1b9e77', linewidth=2.1,
                  label=r'Three-stage variant')
    axes2[1].set_xscale('log')
    axes2[1].set_yscale('log')
    axes2[1].set_ylim(2e-2, 2.5e3)
    axes2[1].set_xlabel(r'Per-Stalk Section Norm $\|x_{*,v}\|_2$ (log scale)')
    axes2[1].set_ylabel(r'Global Consensus RMSE (log scale)')
    axes2[1].set_title(r'(b) No Faulty Source: Error vs. Distance ($\sigma_{\mathrm{map}} = 0.01$)')
    axes2[1].legend(fontsize=7.4, loc='upper left', framealpha=0.92)
    axes2[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig2_sheaf_robustness_v8_6.png', dpi=300)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    b_vals = [r[1] for r in r3]
    ax1.plot(b_vals, [r[2] for r in r3], 'o-', color='firebrick', linewidth=2, label=r'Galilean Shear $\lambda_2(L_{\mathcal{F}})$')
    ax1.plot(b_vals, [r[3] for r in r3], 's--', color='navy', linewidth=2, label=r'Lorentz Boost $\lambda_2$ (num. zero)')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-17, 1e2)
    ax1.set_xlabel(r'Maximum Pairwise Speed $\beta_{\max} = v_{\max}/c$')
    ax1.set_ylabel(r'Second Eigenvalue $\lambda_2(L_{\mathcal{F}})$ (log scale)')
    ax2 = ax1.twinx()
    ax2.errorbar(b_vals, [r[11] for r in r3], yerr=[r[12] for r in r3], fmt='^-.', color='darkgreen',
                 linewidth=2, capsize=3, label=r'Dial-Free $\log\mathrm{BF}$ (Lorentz World)')
    ax2.errorbar(b_vals, [r[13] for r in r3], yerr=[r[14] for r in r3], fmt='v:', color='darkorange',
                 linewidth=1.8, capsize=3, label=r'Dial-Free $\log\mathrm{BF}$ (Galilean Control)')
    ax2.set_ylim(-18, 30)
    ax2.set_ylabel(r'Half-Cauchy Marginalized Per-Event $\log\mathrm{BF}$', color='darkgreen')
    ax1.set_title(r'Dial-Free 2D Spacetime Sheaves ($\gamma_{\mathrm{sh}} \sim \mathrm{HalfCauchy}(50), \gamma_0=0.10$)')
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='upper left', ncol=2, fontsize=7.4, framealpha=0.92)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig3_spacetime_sheaf_v8_6.png', dpi=300)
    plt.close(fig)


def print_summary(res1, res2, res3, res4):
    print("=== Experiment 1: Testimony Saturation & Subset-DP Provenance ===")
    for r in res1["rows_1b"]:
        print(f"  m={r[0]} | NoSig={r[1]:.4f} | Para={r[3]:.4f} | SoundVer={r[4]:.4f} | Forged={r[5]:.4f}->{r[6]:.4f}")
    print("\n=== Experiment 2: Equivariant SO(2) Sheaf Breakdown (gamma_0=0.0005) ===")
    for r in res2["main"]:
        print(f"  {r[0]:42s} | Gauss={r[1]:.4f} | WarmHet-t={r[4]:.4f}+-{r[5]:.4f}")
    print("\n=== Experiment 2, two-stage filter and three-stage variant ===")
    for sh_v, dist_v, cd in res2["dist_sweep"]:
        print(f"  shift=({sh_v:g},{sh_v:g}) |x|={dist_v:7.1f} | k=0: Raw={cd[(0, 'raw')][0]:.4f} Gauss={cd[(0, 'gauss')][0]:.4f} "
              f"Uncentred={cd[(0, 'warm')][0]:.4f} Two-stage={cd[(0, 'centred')][0]:.4f} Three-stage={cd[(0, 'three')][0]:.4f} "
              f"| k=9: Two-stage={cd[(9, 'centred')][0]:.4f} (captured {cd[(9, 'centred_capt')]}%) "
              f"Oracle={cd[(9, 'oracle')][0]:.4f} Three-stage honest={cd[(9, 'three_hon')]:.4f} "
              f"(captured {cd[(9, 'three_capt')]}%)")
    print("  offset-ratio sweep, percent of seeds captured at Delta / (sigma_map |x|) = "
          + " ".join(f"{g_v:g}" for g_v in res2["ratio_grid"]))
    for f_lbl, lay_lbl, sh_v, cells_r in res2["ratio_sweep"]:
        print(f"  {f_lbl:11s} {lay_lbl:9s} shift {sh_v:5g} | " + " ".join(f"{c:3d}" for c in cells_r))
    for gsh_v, pl_lbl, cells_plain, cells_cent, cells_three in res2["coupling_sweep"]:
        print(f"  gamma_sh={gsh_v:6g} {pl_lbl:22s} | fail% k=8..11: Uncentred={[c[1] for c in cells_plain]} "
              f"Two-stage={[c[1] for c in cells_cent]} Three-stage={[c[1] for c in cells_three]}")
    print("  ingredients: fail% and honest capt% at gamma_sh=5 | fail% at gamma_sh=50 | no-fault RMSE at (20,20), (1000,1000) "
          "| worst capt% at ratios <5 and >=5")
    for v_lbl, ci in res2["ingredients"]:
        print(f"  {v_lbl[:46]:46s} | {ci[('fail', 5.0)][0]:3d}-{ci[('fail', 5.0)][1]:3d} {ci[('capt', 5.0)][0]:3d}-{ci[('capt', 5.0)][1]:3d} "
              f"| {ci[('fail', 50.0)][0]:3d}-{ci[('fail', 50.0)][1]:3d} | {ci[('nofault', 20.0)]:7.4f} {ci[('nofault', 1000.0)]:7.4f} "
              f"| {ci['capt_low']:3d} {ci['capt_high']:3d}")
    print(f"  shifted placement, gamma_sh = 50: mean map-aware coupling {res2['ingr_gamma_shifted']:.3f}; two-stage filter with that "
          f"uniform coupling fails in {res2['ingr_unif_fail'][8]}% (k=8) and {res2['ingr_unif_fail'][9]}% (k=9) of seeds")
    print("  capture map, Delta = " + " ".join(f"{dv:5g}" for dv in res2["capture_deltas"]))
    for f_lbl, lay_lbl, kc_v, cells_c in res2["capture_map"]:
        print(f"  {f_lbl:11s} {lay_lbl:9s} k={kc_v} | " + " ".join(f"{c:5.0f}" for c in cells_c))
    for row in res2["l1_scenarios"]:
        print(f"  L1 {row[0]:42s} | s_hat clones={row[5]:.3f} S_fault={row[7]:.2f} S_hon={row[8]:.2f} "
              f"| k=12: Div={row[1][0]:.4f} (uncentred {row[11][0]:.4f}) Two-stage={row[2][0]:.4f} "
              f"| k=3: Div={row[3][0]:.4f} Two-stage={row[4][0]:.4f}")
    for kc_v, cells_b in res2["l1_boundary"]:
        print(f"  boundary k={kc_v} | " + " | ".join(f"h={h_v}: {c[0]:.2f}/{c[1]:.2f} ({c[2]:3d}%)"
                                                   for h_v, c in enumerate(cells_b)))
    print(f"  boundary k=16, h=3 (nearest parity) | two-stage at 10 x gamma_sh: captured {res2['bound_near']['two_strong']}% "
          f"| three-stage: captured {res2['bound_near']['three']}%")
    print("  combined filter, percent captured at Delta = " + " ".join(f"{dv:g}" for dv in res2["comb_deltas"]))
    for pl_lbl, f_lbl, cells_m in res2["combined"]:
        print(f"  {pl_lbl:22s} {f_lbl:40s} | " + " ".join(f"{c:3d}" for c in cells_m))
    sa = res2["stage1_audit"]
    print(f"  Stage 1 audit: {sa['fits']} fits, {sa['loose']} with relative duality gap above {STAGE1_TOL:g} "
          f"(largest certified gap {sa['gap_certified']:.2e}, largest uncertified gap {sa['gap_loose']:.2e})")
    for key, (g_min, g_max, n_sign) in res2["mode_gap_seeds"].items():
        print(f"  log-posterior gap gamma0={key[0]:g} {key[1]:5s} k={key[2]}: mean {res2['mode_gaps'][key]:+8.2f} "
              f"min {g_min:+8.2f} max {g_max:+8.2f} | seeds with the sign of the mean: {n_sign}")
    print(f"  EM cap 35 -> {res2['cap_long']}: mean RMSE of the distance sweep")
    for sh_v, kc_v, f_key, r_35, r_long in res2["cap_rmse"]:
        print(f"    shift={sh_v:6g} k={kc_v} {f_key:8s} {r_35:10.5f} -> {r_long:10.5f} ({100.0 * (r_long - r_35) / r_35:+.3f}%)")
    print("    capture cells (distance and offset-ratio sweeps) that change: "
          + str(sum(1 for c in res2["cap_capt"] if c[0] != c[1])) + f" of {len(res2['cap_capt'])}: "
          + " ".join(f"{c[0]}->{c[1]}" for c in res2["cap_capt"] if c[0] != c[1]))
    print(f"  boundary table with every precision at the lowest tier: {res2['bound_flat_same']} cells unchanged")
    for lay_lbl, cells_e in res2["capt_equal"].items():
        print(f"  capture map, equal honest precisions, {lay_lbl:9s} k=9 | " + " ".join(f"{c:5.0f}" for c in cells_e))
    p4m = res2["prop4_map"]
    print("  Proposition 4 limit, k=9: EM  f_t = " + " ".join(f"{v:.4f}" for v in p4m["uniform"][0]))
    print("                            map f_t = " + " ".join(f"{v:.4f}" for v in p4m["uniform"][1]))
    print("\n=== Experiment 3: Spacetime Half-Cauchy & exact two-step annulus decisions ===")
    for r in res3["rows_3a"]:
        print(f"  beta={r[1]:.2f}c | Fixed50={r[4]:+6.2f} | HalfCauchy={r[11]:+6.2f}+-{r[12]:.2f} | GalCtrl={r[13]:+6.2f}")
    for r3, q3 in zip(res3["rows_3a"], res3["event_forms"]):
        print(f"  beta={r3[1]:.2f}c | single-event quadratic form: eigenvalues {q3[0]:+.4f}, {q3[1]:.4f}; "
              f"|x-component| of the dominant direction {q3[2]:.3f}")
    for r in res3["annulus"]:
        print(f"  {r[0]:40s} | curl {r[4]:5.1f}% hole {r[5]:5.1f}% | local {r[6]:5.1f}% global {r[7]:5.1f}% none {r[8]:5.1f}%")
    print("\n=== Experiment 4: Symbolic DAG GFlowNet & Fair Reverse-KL Baselines ===")
    print(f"  Bimodal [A, B] TV : OffPol-TB={res4['mm_gf'][0]:.4f} | OnPol-RKL={res4['mm_on'][0]:.4f} "
          f"| MF-VI={res4['mm_mf'][0]:.4f} | FullTab-VI={res4['mm_tab'][0]:.4f}")
    print(f"  Kraft sum of the nominal code lengths: {res4['kraft_sum']:.6f}; log Z* = {res4['log_z_true'][0]:.4f}, "
          f"{res4['log_z_true'][1]:.4f}")
    print(f"  Trajectory balance, exploratory, blind start log Z = 0: TV {res4['tb_blind'][0][0]:.4f}+-{res4['tb_blind'][0][1]:.4f} "
          f"| {res4['tb_blind'][1][0]:.4f}+-{res4['tb_blind'][1][1]:.4f}; |log Z - log Z*| = {res4['tb_blind'][0][2]:.4f} | {res4['tb_blind'][1][2]:.4f}")
    print("  Objective x sampling: TV on [A, B] | on [A, B, C_lin] (share of seeds with TV > 0.1)")
    for o_lbl, s_lbl, st_mm, st_cl in res4["obj_samp"]:
        print(f"    {o_lbl:18s} | {s_lbl:32s} | {st_mm[0]:.4f}+-{st_mm[1]:.4f} ({st_mm[2]:3d}%) "
              f"| {st_cl[0]:.4f}+-{st_cl[1]:.4f} ({st_cl[2]:3d}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    res1 = run_experiment_1_testimony()
    res2 = run_experiment_2_sheaf_robustness()
    res3 = run_experiment_3_spacetime_and_annulus()
    res4 = run_experiment_4_gflownet()
    export_latex_macros(res1, res2, res3, res4)
    print_summary(res1, res2, res3, res4)
    if not args.no_plots:
        generate_figures(res1, res2, res3)
    print("\nSaved v8_6_results_macros.tex" + ("" if args.no_plots else " and fig1..3_v8_6.png") + ".")
