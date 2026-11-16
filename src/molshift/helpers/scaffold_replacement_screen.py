"""
screen_scaffolds.py -- scaffold replacement screen on the parquet fragment library.

Inputs (all from the pipeline):
  fragment_features.parquet      main table (descriptors, n_sites, site_sym, murcko, core_smiles_mapped, ...)
  fragment_site_extras.parquet   heavy JSON columns (site_dist, site_pairs, site_env)
  usrcat_cache.parquet           60-d USRCAT vectors (optional)

Flow
  1. Filter : same number of attachment points
              heavy-atom window (stricter on the low side than the high side)
              bond-distance between attachment points (>=2 sites), stricter on the low side
              [optional] minimum USRCAT similarity
  2. Rank   : physchem similarity (HBA, HBD, rotatable bonds, rings, aromatic rings) blended with
              USRCAT similarity (usrcat_weight; 0 = physchem only)
  3. Map    : attachment mapping by ECFP4 environment similarity of the attachment atoms.
              Kept completely separate from the score.

Mapping values are SITE NUMBERS of the library fragment (1, 2, ...). They are the atom-map numbers in
`core_smiles_mapped`, and the order in which the '*' dummies appear in `frag_smiles`.
"""
import os
import json
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from molshift.helpers.frag_features import load_usrcat, featurize_scaffold, env_tanimoto,usrcat_scores, usrcat_vector
from molshift.helpers.frag_ecfp import (ecfp4_settings, fingerprint_bits,
                                        scaffold_to_dummy_smiles, tanimoto_column)

PHYSCHEM_PROPS = ["hba", "hbd", "rot_bonds", "rings", "arom_rings"]


def _prop_sim(a, b):
    """1 - |a-b| / max(a, b, 1): safe for zeros, 1 = identical."""
    return 1.0 - abs(a - b) / max(a, b, 1)


def _clean(o):
    """Make nested results JSON-safe (numpy types -> python, NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if o != o else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ───────────────────────── library handle ─────────────────────────
@dataclass
class Library:
    main_path: str
    extras_path: str
    usrcat_index: pd.Index = None      # frag_id -> row in usrcat_M
    usrcat_M: np.ndarray = None        # (N, 60) float32


def load_library(main_path, extras_path, usrcat_path=None):
    """Create once, reuse for every screen. The big tables are read per call with filters; only the
    USRCAT matrix is held in memory."""
    lib = Library(main_path, extras_path)
    if usrcat_path:
        ids, M = load_usrcat(usrcat_path)
        lib.usrcat_index = pd.Index(ids)
        lib.usrcat_M = M
        print(f"USRCAT cache loaded: {len(ids):,} vectors")
    return lib

def ref_ecfp4_bits(scaff_smi, selected_atts, settings):
    """ECFP4 ON-bits of the reference scaffold, in the same format as the library.
    The scaffold has no dummies: if the library keeps the '*' as an atom, one is added on each
    selected attachment atom; if the library is H-capped, the plain scaffold is used as is."""
    q = fingerprint_bits(scaff_smi, settings["keep_dummy"], settings["radius"], settings["n_bits"])
    if q is None:
        raise ValueError(f"Could not fingerprint reference: {scaff_smi}")
    return q


def ecfp4_similarity(df, q, n_bits):
    """Tanimoto of the reference bits `q` against df['ecfp4'] (empty fingerprints give 0)."""
    return tanimoto_column(df["ecfp4"], q, n_bits)


# ───────────────────────── screen ─────────────────────────
def screen_scaffolds(
    scaff_smi,
    selected_atts,
    lib,                          # from load_library(...)
    sim_mode="desc",              # "desc" = physchem | "ecfp4" = ECFP4 Tanimoto (both can be blended with USRCAT)
    # --- filters (low = candidate smaller/shorter than reference, high = larger/longer)
    ha_low=0.20, ha_high=0.50,    # heavy atoms: -20% / +50%
    dist_low=1, dist_high=3,      # bond distance between attachment atoms: -1 / +3 bonds
    # --- physchem ranking (desc mode only)
    props=PHYSCHEM_PROPS,
    # --- USRCAT (both modes)
    usrcat_weight=0.5,            # 0 = off (base similarity only), 1 = USRCAT only
    usrcat_min=None,              # optional hard cut-off on the raw USRCAT score
    usrcat_ref_seeds=(42, 1, 2, 3, 4),
    missing_usrcat="drop",        # "drop" or "keep" (neutral USRCAT part)
    combine="rank",               # "rank": blend percentile ranks; "raw": blend raw scores
    # --- optional diversity: keep at most N per ring system (None = off)
    max_per_ring_system=None,
    top_n=None,
    save_dir=None,
    save_name="ranked_scaffolds",
):
    """
    Returns (ranked_dict, df)
      ranked_dict (best first):
        {frag_smiles: {"rank", "mapping": {scaffold_atom_idx: fragment_site_number},
                       "final_score", "score" (physchem), "ecfp4_sim", "usrcat_sim",
                       "mapping_sim", "mapping_tied", "alt_mappings", ...}}
    sim_mode="desc"  : final_score = physchem score, blended with USRCAT if available
    sim_mode="ecfp4" : final_score = whole-fragment ECFP4 Tanimoto, blended with USRCAT if available
    usrcat_weight    : 0 = base similarity only, 1 = USRCAT only (applies to both modes)
    Fallback         : if USRCAT is unavailable (no cache, no reference conformer, or no candidate
                       has a vector), final_score = the base similarity alone.
    """
    if sim_mode not in ("desc", "ecfp4"):
        raise ValueError("sim_mode must be 'desc' or 'ecfp4'")

    selected_atts = [int(a) for a in selected_atts]
    n = len(selected_atts)

    # similarity that USRCAT is blended with / falls back to
    base_col = "ecfp4_sim" if sim_mode == "ecfp4" else "score"

    # ── reference scaffold ────────────────────────────────────────────────
    mol = Chem.MolFromSmiles(scaff_smi)
    if mol is None:
        raise ValueError("Invalid scaffold SMILES")
    if len(set(selected_atts)) != n or any(a < 0 or a >= mol.GetNumAtoms() for a in selected_atts):
        raise ValueError("selected_atts must be unique, valid atom indices of scaff_smi")
    for a in selected_atts:
        if mol.GetAtomWithIdx(a).GetTotalNumHs() == 0:
            print(f"[WARN] scaffold atom {a} has no H: library anchors always carry the H they would lose, "
                  f"so its environment may match poorly")
    ref = featurize_scaffold(scaff_smi, selected_atts)

    # ── 1. filters on the main table (parquet predicate pushdown: only survivors are loaded) ──
    ha = ref["heavy_atoms"]
    lo, hi = ha * (1 - ha_low), ha * (1 + ha_high)
    df = pq.read_table(lib.main_path, filters=[("n_sites", "=", n),
                                               ("heavy_atoms", ">=", lo),
                                               ("heavy_atoms", "<=", hi)]).to_pandas()
    print(f"{len(df):,} pass attachment-count + heavy-atom window ({lo:.1f}-{hi:.1f} heavy atoms)")
    if df.empty:
        return {}, pd.DataFrame()

    # ── ECFP4 similarity of the whole fragment (ecfp4 mode only) ──
    if sim_mode not in ("desc", "ecfp4"):
        raise ValueError("sim_mode must be 'desc' or 'ecfp4'")

    if sim_mode == "ecfp4":
        if "ecfp4" not in df.columns:
            raise ValueError("lib.main_path has no 'ecfp4' column: use the table made by add_ecfp4_column")
        st = ecfp4_settings(lib.main_path)
        q = ref_ecfp4_bits(scaff_smi, selected_atts, st)
        df["ecfp4_sim"] = ecfp4_similarity(df, q, st["n_bits"])
        df = df.drop(columns="ecfp4")          # big list column no longer needed

    # heavy JSON columns only for these survivors
    ex = pq.read_table(lib.extras_path, filters=[("frag_id", "in", df["frag_id"].tolist())]).to_pandas()
    df = df.merge(ex, on="frag_id", how="inner")

    perms = list(itertools.permutations(range(n)))
    S = np.array(ref["site_dist"])
    ref_pairs = {(p["i"], p["j"]): p for p in ref["site_pairs"]}

    rows = []
    for r in df.itertuples(index=False):
        F = np.array(json.loads(r.site_dist))

        # bond-distance check: which pairings (scaffold site i -> fragment site perm[i]) are acceptable
        ok_perms, devs = [], []
        for p in perms:
            if n == 1:
                ok_perms.append(p)
                devs.append(0.0)
                continue
            d = [F[p[i]][p[j]] - S[i][j] for i in range(n) for j in range(i + 1, n)]
            if all(-dist_low <= x <= dist_high for x in d):
                ok_perms.append(p)
                devs.append(float(np.mean(np.abs(d))))
        if not ok_perms:
            continue

        # decode the heavy columns only now (candidate passed the distance filter)
        site_env = [{int(k): v for k, v in e.items()} for e in json.loads(r.site_env)]
        site_pairs = json.loads(r.site_pairs)

        # ── physchem score (desc mode only) ──
        if sim_mode == "desc":
            score = float(np.mean([_prop_sim(getattr(r, k), ref[k]) for k in props]))
        else:
            score = np.nan

        # ── mapping by ECFP4 environment similarity (separate from score) ──
        E = np.array([[env_tanimoto(ref["site_env"][i], site_env[j]) for j in range(n)] for i in range(n)])
        tot = [float(sum(E[i][p[i]] for i in range(n))) for p in ok_perms]
        best = max(tot)
        tied, seen = [], set()
        for p, t in zip(ok_perms, tot):
            if t >= best - 1e-9:
                key = tuple(int(r.site_sym[p[i]]) for i in range(n))     # symmetry-equivalent pairings collapse
                if key not in seen:
                    seen.add(key)
                    tied.append(p)
        bp = tied[0]
        to_map = lambda p: {selected_atts[i]: int(p[i]) + 1 for i in range(n)}   # fragment site number (1-based)

        # informational flags: how the span between sites is built, vs the reference (not in score)
        flex_diff = arom_diff = np.nan
        if n > 1:
            fp = {(q_["i"], q_["j"]): q_ for q_ in site_pairs}
            flex_diff = arom_diff = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = sorted((bp[i], bp[j]))
                    q_ = fp[(a, b)]
                    flex_diff += q_["path_rot"] - ref_pairs[(i, j)]["path_rot"]
                    arom_diff += q_["path_arom_frac"] - ref_pairs[(i, j)]["path_arom_frac"]
            arom_diff /= (n * (n - 1) / 2)

        rows.append(dict(
            frag_id=r.frag_id, frag_smiles=r.frag_smiles, core_smiles_mapped=r.core_smiles_mapped,
            n_molecules=r.n_molecules, score=round(score, 4),
            ecfp4_sim=getattr(r, "ecfp4_sim", np.nan),
            span_dev=round(min(devs), 3), ha_diff=abs(r.heavy_atoms - ha),
            mapping=to_map(bp), mapping_sim=round(best / n, 4), mapping_tied=len(tied) > 1,
            alt_mappings=[to_map(p) for p in tied[1:]],
            path_flex_diff=flex_diff, path_arom_diff=arom_diff,
            murcko=r.murcko, heavy_atoms=r.heavy_atoms,
            **{k: getattr(r, k) for k in props},
        ))
    print(f"{len(rows):,} pass the bond-distance filter")
    if not rows:
        return {}, pd.DataFrame()
    out = pd.DataFrame(rows)

    # ── 2. USRCAT (both modes) ─────────────────────────────────────────────
    use_usr = usrcat_weight > 0
    if use_usr and lib.usrcat_M is None:
        print(f"[WARN] no USRCAT cache loaded in the library: ranking by {base_col} only")
        use_usr = False
    if use_usr:
        ref_vecs = [v for v in (usrcat_vector(scaff_smi, seed=s)[0] for s in usrcat_ref_seeds) if v is not None]
        if not ref_vecs:
            print(f"[WARN] could not build a 3D conformer for the reference scaffold: ranking by {base_col} only")
            use_usr = False

    # how many of the screened-in candidates actually have a USRCAT vector?
    if use_usr:
        ridx = lib.usrcat_index.get_indexer(out["frag_id"].to_numpy())
        has = ridx >= 0
        print(f"{int(has.sum()):,} of {len(out):,} candidates have a USRCAT vector "
              f"(reference conformers used: {len(ref_vecs)})")
        if not has.any():
            print(f"[WARN] none of the screened-in candidates has a USRCAT vector -> "
                  f"ranking and shortlisting by {base_col} only "
                  f"(missing_usrcat / usrcat_min ignored)")
            use_usr = False

    if use_usr:
        usr = np.full(len(out), np.nan)
        Msub = lib.usrcat_M[ridx[has]]
        usr[has] = np.max([usrcat_scores(q, Msub) for q in ref_vecs], axis=0)   # best reference conformer
        out["usrcat_sim"] = np.round(usr, 4)

        if missing_usrcat == "drop":
            out = out[has].reset_index(drop=True)
        if usrcat_min is not None:
            out = out[out["usrcat_sim"] >= usrcat_min].reset_index(drop=True)      # NaN fails the cut-off
            print(f"{len(out):,} remain after USRCAT >= {usrcat_min}")
        if out.empty:
            return {}, pd.DataFrame()

        if combine == "rank":
            ph = out[base_col].rank(pct=True)
            us = out["usrcat_sim"].rank(pct=True).fillna(0.5)                      # no vector -> neutral
        else:
            ph = out[base_col]
            fill = out["usrcat_sim"].median() if out["usrcat_sim"].notna().any() else 0.0
            us = out["usrcat_sim"].fillna(fill)
        out["final_score"] = ((1 - usrcat_weight) * ph + usrcat_weight * us).round(4)
    else:
        out["usrcat_sim"] = np.nan
        out["final_score"] = out["ecfp4_sim"].round(4) if sim_mode == "ecfp4" else out["score"]

    # ── rank ──
    out = (out.sort_values(["final_score", base_col, "span_dev", "ha_diff"],
                           ascending=[False, False, True, True])
           .reset_index(drop=True))

    if max_per_ring_system:
        keep = out["murcko"].eq("") | (out.groupby("murcko").cumcount() < max_per_ring_system)
        out = out[keep].reset_index(drop=True)
    if top_n:
        out = out.head(top_n)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    ranked = {
        r.frag_smiles: dict(rank=r.rank, mapping=r.mapping, final_score=r.final_score, score=r.score,
                            ecfp4_sim=r.ecfp4_sim, usrcat_sim=r.usrcat_sim,
                            mapping_sim=r.mapping_sim, mapping_tied=r.mapping_tied,
                            alt_mappings=r.alt_mappings, span_dev=r.span_dev, n_molecules=r.n_molecules,
                            path_flex_diff=r.path_flex_diff, path_arom_diff=r.path_arom_diff)
        for r in out.itertuples()
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv = out.copy()
        csv["mapping"] = csv["mapping"].apply(lambda m: json.dumps(_clean(m)))
        csv["alt_mappings"] = csv["alt_mappings"].apply(lambda m: json.dumps(_clean(m)))
        csv.to_csv(os.path.join(save_dir, save_name + ".csv"), index=False)
        with open(os.path.join(save_dir, save_name + ".json"), "w") as fh:
            json.dump(_clean(ranked), fh, indent=2)
        print(f"✓ {len(out)} ranked -> {save_dir}/{save_name}.csv / .json")
    return ranked, out