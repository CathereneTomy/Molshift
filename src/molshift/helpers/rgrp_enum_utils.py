"""
Pharmacophore Similarity Pipeline
Converted from pharmacophore_pipeline_stepwise.ipynb

This module contains all functions needed for pharmacophore-based molecular
similarity computation.
"""

import os
import warnings
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,Lipinski, rdMolDescriptors,

)
import pandas as pd
import re
from rdkit.Chem import Draw
from molshift.helpers.frag_ecfp import ecfp4_settings, fingerprint_bits, tanimoto_column, RADIUS, N_BITS


from multiprocessing import Pool, cpu_count




warnings.filterwarnings("ignore")



def mol_idx_image(mol):
    """
    Setting the Atom Indexes as Map Numbers to the molecules
    
    Args:
    mol: rdkit.Chem.Mol 
    
    Returns:
    mol with its indexes set as AtomMapNumbers"""

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom.SetAtomMapNum(idx)
    return mol

def find_frag_atts(whole_mol, scaffold_mol, scaffold_smiles, show=True):
    if not whole_mol.HasSubstructMatch(scaffold_mol):
        raise ValueError("Scaffold is not a substructure of the whole molecule.")

    # 1. match[scaffold_idx] = whole_idx
    match = whole_mol.GetSubstructMatches(scaffold_mol)[0]
    whole_to_scaff = {w: s for s, w in enumerate(match)}
    scaffold_set = set(match)

    # 2. Bonds between scaffold and non-scaffold atoms
    cut_bonds = []
    nbr_to_atts = {}   # outside atom idx (whole_mol) -> [scaffold att idx, ...]
    for w in match:
        for nbr in whole_mol.GetAtomWithIdx(w).GetNeighbors():
            n = nbr.GetIdx()
            if n not in scaffold_set:
                cut_bonds.append(whole_mol.GetBondBetweenAtoms(w, n).GetIdx())
                nbr_to_atts.setdefault(n, []).append(whole_to_scaff[w])

    attachment_points = sorted({a for lst in nbr_to_atts.values() for a in lst})

    # 3. Label scaffold atoms
    labelled_scaff = Chem.MolFromSmiles(scaffold_smiles)
    for att in attachment_points:
        labelled_scaff.GetAtomWithIdx(att).SetProp("atomNote", str(att))

    # 4. Show image
    # if show:
    #     img = Draw.MolToImage(
    #         labelled_scaff, size=(500, 500),
    #         highlightAtoms=attachment_points, highlightColor=(0, 0, 1)
    #     )
    #     display(img)

    # 5. Cut bonds (plain dummies, no labels), then tag the dummies ourselves.
    #    Original atoms keep their indices; new dummies are appended at the end.
    fragmented = Chem.RWMol(Chem.FragmentOnBonds(whole_mol, cut_bonds, addDummies=True))
    n_orig = whole_mol.GetNumAtoms()
    remaining = {n: list(v) for n, v in nbr_to_atts.items()}

    for d in fragmented.GetAtoms():
        if d.GetIdx() < n_orig:
            continue                          # original atom, not a dummy
        nb = d.GetNeighbors()[0].GetIdx()
        if nb in scaffold_set:
            continue                          # dummy on the scaffold side
        att = remaining[nb].pop(0)            # dummy on the substituent side
        d.SetIsotope(att)                     # shows as [att*] (plain * if att == 0)
        d.SetIntProp("scaff_att", att)        # reliable key, works for att == 0 too

    fragmented = fragmented.GetMol()

    # 6. Split and collect substituent fragments
    mapping = []
    frags = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True,
                             fragsMolAtomMapping=mapping)

    frag_repl = {}
    for f, idxs in zip(frags, mapping):
        if scaffold_set & set(idxs):
            continue                          # this is the scaffold piece
        smi = Chem.MolToSmiles(f)
        for a in f.GetAtoms():
            if a.HasProp("scaff_att"):
                frag_repl[a.GetIntProp("scaff_att")] = smi

    # # 7. User selection
    # chosen = input(f"Select attachment points from {attachment_points}, comma-separated: ")
    # selected_attachments = []
    # for sel in chosen.split(","):
    #     sel = sel.strip()
    #     if sel.isdigit():
    #         val = int(sel)
    #         if val in attachment_points:
    #             selected_attachments.append(val)
    #         else:
    #             print(f"{val} not present in the presented attachment points")

    selected_attachments = attachment_points

    frag_repl = {a: s for a, s in frag_repl.items() if a in selected_attachments}

    return {
        "attachment_points": attachment_points,
        "frag_repl": frag_repl,
        "selected_attachments": selected_attachments,
        "labelled_scaff": labelled_scaff,
    }


def _calc_props_worker(args):
    """Worker function for multiprocessing — must be top-level."""
    idx, smi = args
    try:
        if pd.isna(smi) or smi is None:
            return {"_idx": idx}
        props = calc_props(smi)  # call directly, not via func argument
        if props is None:
            return {"_idx": idx}
        return {"_idx": idx, **props}
    except Exception:
        return {"_idx": idx}


def cap_with_h(smi):
    """Dummy atom -> H at the atom level, sanitize, remove Hs. Returns canonical SMILES or None."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:          # any dummy: [8*], [*], *
            a.SetAtomicNum(1)
            a.SetIsotope(0)
            a.SetNoImplicit(True)
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(Chem.RemoveHs(m))
    except Exception:
        return None
    
def process_fragments(fragment_df, calc_props_func):
    pattern = r"\[[^\]]*\*[^\]]*\]"
    fragment_df = fragment_df.copy()

    print(f"Total fragments incoming: {len(fragment_df):,}")  # ← check this

    fragment_df["core_smiles_mapped"] = fragment_df["frag_smiles"].map(cap_with_h)

    null_count = fragment_df["core_smiles_mapped"].isna().sum()
    print(f"Fragments with null att_repl: {null_count:,} / {len(fragment_df):,}")
    
    args = list(fragment_df["core_smiles_mapped"].items())
    total = len(args)
    results = []

    with Pool(cpu_count()) as pool:
        for i in range(0, total, 1000):
            batch = args[i:i+1000]
            batch_results = pool.map(_calc_props_worker, batch)
            results.extend(batch_results)
            print(f"Processed {min(i+1000, total):,} / {total:,} SMILES")

    desc_df = pd.DataFrame(results).set_index("_idx")
    fragment_df = fragment_df.join(desc_df, how="left")
    return fragment_df


def calc_diameter(mol):
    """Longest shortest-path (in bonds) between any two heavy atoms."""
    dm = Chem.GetDistanceMatrix(mol)
    return int(dm.max()) if dm.size else 0


def calc_n_branch(mol):
    """Heavy atoms with >=3 connections, counting each attachment bond as a connection
    (same definition as the library, so isopropyl != n-propyl)."""
    n_att = {}
   
    return sum(1 for at in mol.GetAtoms() if at.GetDegree() + n_att.get(at.GetIdx(), 0) >= 3)


def calc_props(smi, anchors=None):
    """
    smi     : reference fragment.
              (a) WITH dummies, e.g. '[8*]CCO' or '*CCO'  -> capped with H, anchors found automatically; or
              (b) plain SMILES (already capped) + `anchors` = atom indices of the attachment atoms in that SMILES.
    Returns the same descriptors as before plus 'diameter' and 'n_branch'.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None


    return {
        'mol_wt':      Descriptors.MolWt(mol),
        'logp':        Descriptors.MolLogP(mol),
        'tpsa':        Descriptors.TPSA(mol),
        'hba':         Lipinski.NumHAcceptors(mol),     # was rdMolDescriptors.CalcNumHBA: now matches the library
        'hbd':         Lipinski.NumHDonors(mol),        # was rdMolDescriptors.CalcNumHBD: now matches the library
        'rot_bonds':   rdMolDescriptors.CalcNumRotatableBonds(mol),
        'fsp3':        rdMolDescriptors.CalcFractionCSP3(mol),
        'rings':       rdMolDescriptors.CalcNumRings(mol),
        'arom_rings':  rdMolDescriptors.CalcNumAromaticRings(mol),
        'heavy_atoms':  mol.GetNumHeavyAtoms(),
        'heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'kappa1':       Descriptors.Kappa1(mol),
        'kappa2':       Descriptors.Kappa2(mol),
        'kappa3':       Descriptors.Kappa3(mol),
        'bertzct':     Descriptors.BertzCT(mol),
        'diameter':    calc_diameter(mol),              # NEW
        'n_branch':    calc_n_branch(mol),     # NEW
    }


GATES = {
    "heavy_atoms": (-3, 2),          # asymmetric: bigger than the reference risks clashes, smaller only loses contacts
    "rings":       (0, 0),
    "hbd":         (0, 0),           # use (0, np.inf) for "at least as many donors as the reference"
    "hba":         (-1, 1),
    "tpsa":        (-15.0, 15.0),    # loose on purpose: largely overlaps with hba/hbd
    "fsp3":        (-0.25, 0.25),
    "n_branch":    (-1, 2),
}

RANK_FEATURES = {
    "logp":        (1.0,  2.0),
    "tpsa":        (15.0, 1.5),
    "fsp3":        (0.2,  1.0),
    "arom_rings":  (1.0,  1.0),      # aromatic vs aliphatic character (total rings is already gated)
    "rot_bonds":   (2.0,  1.0),      # flexibility
    "kappa2":      (1.5,  1.0),      # linear vs compact
    "kappa3":      (1.0,  1.0),      # branching / bulk
    "diameter":    (2.0,  1.0),      # overall elongation
    "heavy_atoms": (2.0,  1.0),      # closeness inside the allowed size window
}


def apply_gates(fd, ref, gates=GATES, verbose=True):
    """Returns the rows of `fd` that pass every gate. Prints a funnel so a too-strict gate is easy to spot."""
    missing = [p for p in gates if p not in fd.columns or p not in ref]
    if missing:
        raise KeyError(f"gate properties missing from the library or the reference: {missing}")

    eps = 1e-9
    masks = {}
    for p, (lo, hi) in gates.items():
        v = fd[p].to_numpy(dtype=float)
        masks[p] = (v >= ref[p] + lo - eps) & (v <= ref[p] + hi + eps)     # NaN fails the gate

    keep = np.ones(len(fd), dtype=bool)
    if verbose:
        print(f"{'gate':<12}{'window':<22}{'fail this gate alone':>22}{'left after (cumulative)':>26}")
        print(f"{'start':<12}{'':<22}{'':>22}{len(fd):>26,}")
    for p, (lo, hi) in gates.items():
        keep &= masks[p]
        if verbose:
            window = f"[{ref[p] + lo:g}, {ref[p] + hi:g}]"
            print(f"{p:<12}{window:<22}{(~masks[p]).sum():>22,}{keep.sum():>26,}")
    return fd[keep].copy()


def physchem_similarity(fd, ref, features=RANK_FEATURES):
    """Weighted mean of per-feature closeness, 0-1, vectorised. A NaN value counts as 0 for that feature."""
    total, wsum = np.zeros(len(fd)), 0.0
    for p, (tol, w) in features.items():
        d = np.abs(fd[p].to_numpy(dtype=float) - float(ref[p]))
        s = np.exp(-(d / tol) ** 2)
        total += w * np.nan_to_num(s, nan=0.0)
        wsum += w
    return total / wsum

def shortlist_ecfp4(fd, ref_smi, n_cutoff, ha_window=(-3, 2), factor=5,
                    drop_identical=False, settings=None):
    """
    fd        : DataFrame loaded from the table made by add_ecfp4_column (needs 'heavy_atoms' and 'ecfp4')
    ref_smi   : reference WITH a dummy, e.g. '[8*]c1ccc(O)cc1'
    settings  : dict(n_bits, radius, keep_dummy); None = the frag_ecfp defaults
    Returns the rows within the heavy-atom window, top factor * n_cutoff by ECFP4 Tanimoto,
    with an 'ecfp4_sim' column, sorted descending.
    """
    s = settings or dict(n_bits=N_BITS, radius=RADIUS, keep_dummy=True)

    mol = Chem.MolFromSmiles(ref_smi)
    if mol is None:
        raise ValueError(f"Invalid reference SMILES: {ref_smi}")
    if s["keep_dummy"] and "*" not in ref_smi:
        raise ValueError("The library keeps the dummy atom, so the reference needs a '*'.")

    q = fingerprint_bits(ref_smi, s["keep_dummy"], s["radius"], s["n_bits"])
    if q is None:
        raise ValueError(f"Could not fingerprint: {ref_smi}")

    n_ha = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)   # '*' not counted
    lo, hi = n_ha + ha_window[0], n_ha + ha_window[1]
    df = fd[(fd["heavy_atoms"] >= lo) & (fd["heavy_atoms"] <= hi)]
    df = df[df["ecfp4"].map(len) > 0]                               # drop empty fingerprints

    df = df.assign(ecfp4_sim=tanimoto_column(df["ecfp4"], q, s["n_bits"]))
    if drop_identical:
        df = df[df["ecfp4_sim"] < 1 - 1e-6]
    return (df.sort_values("ecfp4_sim", ascending=False, kind="stable")
              .head(int(factor * n_cutoff)).reset_index(drop=True))

def att_to_c_conversion(mol, keep_idx, att_indices):
    """Convert specified atoms to carbon, except the designated keep_idx."""
    rw = Chem.RWMol(mol)
    for idx in att_indices:
        if idx != keep_idx:            
            rw.GetAtomWithIdx(idx).SetAtomicNum(6)
    return rw.GetMol()


def prepare_fragments(fragment_df):
    """Process fragment DataFrame to generate molecules with attachment points."""
    fragment_df = fragment_df.dropna(subset=["frag_smiles"]).reset_index(drop=True)
    fragment_df["mol"] = fragment_df["frag_smiles"].apply(lambda x: Chem.MolFromSmiles(x))
    fragment_df['dupli_count'] = fragment_df['mol'].apply(
        lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "*")
    )
    fragment_df = fragment_df[fragment_df['dupli_count'] > 0].reset_index(drop=True)

    fragment_df['att_points'] = fragment_df['mol'].apply(
        lambda mol: [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
    )

    duplicated = []
    for i, m in enumerate(fragment_df['dupli_count']):
        row = fragment_df.loc[[i]]
        duplicated.append(pd.concat([row] * m, ignore_index=True))
    expanded = pd.concat(duplicated, ignore_index=True)

    new_mols, frag_atts,new_smis = [], [], []
    for i in range(len(expanded)):
        mol = expanded.loc[i, 'mol']
        att_indices = expanded.loc[i, 'att_points']
        dupli_idx = expanded.loc[i, 'dupli_count']
        keep_idx = att_indices[i % dupli_idx]
        new_mol = att_to_c_conversion(mol, keep_idx, att_indices)
        new_mols.append(new_mol)
        new_smis.append(Chem.MolToSmiles(new_mol))
        frag_atts.append(keep_idx)

    expanded['new_mol'] = new_mols
    expanded['new_mol_smi'] = new_smis
    expanded['frag_att'] = frag_atts
    return expanded


from rdkit import RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
 
RDLogger.DisableLog("rdApp.*")
 
_FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
 
# family -> weight in the final average (starting guesses: tune on known replacements)
FAMILY_WEIGHTS = {
    "Donor": 2.0, "Acceptor": 2.0, "PosIonizable": 2.0, "NegIonizable": 2.0,
    "Aromatic": 1.5, "LumpedHydrophobe": 1.0, "Hydrophobe": 1.0,
}
MAX_SHELL = 10                  # features further than this are counted in the last shell
KERNEL = (0.25, 0.5, 0.25)      # a feature one bond off still earns partial credit; (1,) = exact distances only
HALOGENS = {9, 17, 35, 53}
 
 
# ───────────────────────── fragment -> capped mol + anchor ─────────────────────────
def capped_with_anchor(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise ValueError(f"Need exactly one attachment point (*), found {len(dummies)}: {smi}")
    rw = Chem.RWMol(mol)
    d = rw.GetAtomWithIdx(dummies[0])
    nbrs = d.GetNeighbors()
    if len(nbrs) != 1:
        raise ValueError(f"Attachment atom must have exactly one neighbour: {smi}")
    nbrs[0].SetIntProp("anchor", 1)
    d.SetAtomicNum(1)
    d.SetIsotope(0)
    d.SetNoImplicit(True)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    m = Chem.RemoveHs(m)
    anchor = next(a.GetIdx() for a in m.GetAtoms() if a.HasProp("anchor"))
    return m, anchor
 
 
# ───────────────────────── profile ─────────────────────────
def _features(smi, halogens_as_acceptors=False):
    """[(family, shell, atom_ids)] for every pharmacophore feature, shell = bonds from the attachment atom."""
    m, anchor = capped_with_anchor(smi)
    dm = Chem.GetDistanceMatrix(m)
    out = []
    for ft in _FACTORY.GetFeaturesForMol(m):
        fam, ids = ft.GetFamily(), ft.GetAtomIds()
        if fam not in FAMILY_WEIGHTS:
            continue
        if (fam == "Acceptor" and not halogens_as_acceptors
                and all(m.GetAtomWithIdx(i).GetAtomicNum() in HALOGENS for i in ids)):
            continue                                    # organic F/Cl/Br/I are weak acceptors
        out.append((fam, int(min(dm[anchor][i] for i in ids)), tuple(ids)))
    return out
 
 
def profile(smi, halogens_as_acceptors=False, max_shell=MAX_SHELL):
    H = {f: np.zeros(max_shell + 1) for f in FAMILY_WEIGHTS}
    for fam, shell, _ in _features(smi, halogens_as_acceptors):
        H[fam][min(shell, max_shell)] += 1
    return H
 
 
def describe(smi, halogens_as_acceptors=False):
    """Readable view: {family: [shell, shell, ...]} (shell = bonds from the attachment atom)."""
    d = {}
    for fam, shell, _ in _features(smi, halogens_as_acceptors):
        d.setdefault(fam, []).append(shell)
    return {k: sorted(v) for k, v in d.items()}
 
 
# ───────────────────────── similarity ─────────────────────────
def _smooth(h, kernel):
    s = np.convolve(h, kernel, mode="same")
    return s * (h.sum() / s.sum()) if s.sum() > 0 else s          # keep the feature count unchanged at the edges
 
 
def family_similarities(ref_prof, cand_prof, alpha=1.0, beta=1.5, kernel=KERNEL):
    """
    Tversky-style similarity per family on the smoothed shell histograms:
        overlap / (overlap + alpha * missing + beta * extra)
    missing = reference features the candidate does not reproduce at that distance
    extra   = candidate features the reference does not have there
    beta > alpha penalises 'too many groups' more than 'a group missing'. alpha = beta = 1 is plain Jaccard.
    NaN when neither fragment has the family (not rewarded, not penalised).
    """
    out = {}
    for f in ref_prof:
        x, y = _smooth(ref_prof[f], kernel), _smooth(cand_prof[f], kernel)
        if x.sum() == 0 and y.sum() == 0:
            out[f] = np.nan
            continue
        ov = np.minimum(x, y).sum()
        miss = np.maximum(x - y, 0).sum()
        extra = np.maximum(y - x, 0).sum()
        out[f] = ov / (ov + alpha * miss + beta * extra)
    return out
 
 
def profile_similarity(ref_prof, cand_prof, weights=FAMILY_WEIGHTS, alpha=1.0, beta=1.5, kernel=KERNEL):
    """
    Weighted mean of the per-family similarities. NaN if neither fragment has any pharmacophore feature.
    Whole families are weighted asymmetrically as well:
      candidate has a family the reference lacks (extra group)   -> that family's weight x beta
      candidate lacks a family the reference has (missing group) -> that family's weight x alpha
    so with beta > alpha, adding groups costs more than leaving one out.
    """
    sims = family_similarities(ref_prof, cand_prof, alpha, beta, kernel)
    num = den = 0.0
    for f, s in sims.items():
        if np.isnan(s):
            continue
        ref_has, cand_has = ref_prof[f].sum() > 0, cand_prof[f].sum() > 0
        w = weights[f] * (beta if (cand_has and not ref_has) else alpha if (ref_has and not cand_has) else 1.0)
        num += w * s
        den += w
    return num / den if den else np.nan
 
 
# ───────────────────────── rank a shortlist ─────────────────────────
def rank_rgroups(shortlist, ref_smi, smiles_col="frag_smiles", weights=FAMILY_WEIGHTS, alpha=1.0, beta=1.5,
                 kernel=KERNEL, halogens_as_acceptors=False, top_n=None):
    """
    shortlist : DataFrame with a SMILES column (single-attachment fragments), or a list of SMILES
    Returns the shortlist sorted by profile_sim (best first) with per-family columns sim_<Family>.
    Fragments that cannot be profiled (not exactly one attachment point, invalid SMILES) get NaN and are
    reported once.
    """
    df = pd.DataFrame({smiles_col: list(shortlist)}) if not isinstance(shortlist, pd.DataFrame) else shortlist.copy()
    ref_prof = profile(ref_smi, halogens_as_acceptors)
 
    cache, scores, fam_cols, bad = {}, [], {f: [] for f in weights}, []
    for smi in df[smiles_col]:
        if smi not in cache:
            try:
                p = profile(smi, halogens_as_acceptors)
                fs = family_similarities(ref_prof, p, alpha, beta, kernel)
                cache[smi] = (profile_similarity(ref_prof, p, weights, alpha, beta, kernel), fs)
            except ValueError as e:
                bad.append(str(e))
                cache[smi] = (np.nan, {f: np.nan for f in weights})
        s, fs = cache[smi]
        scores.append(s)
        for f in weights:
            fam_cols[f].append(fs.get(f, np.nan))
 
    df["profile_sim"] = np.round(scores, 4)
    for f in weights:
        df[f"sim_{f}"] = np.round(fam_cols[f], 3)
    if bad:
        print(f"[WARN] {len(bad)} fragments could not be profiled, e.g.: {bad[0]}")
    df = df.sort_values("profile_sim", ascending=False, na_position="last").reset_index(drop=True)
    return df.head(top_n) if top_n else df
 

#=========1 stage screening (physchem 2D only) =============
def process_and_save_similar_fragments(
    frag_repl,
    fragment_desc,
    cutoff,
    sim_mode = "ecfp4",
    save_dir="Similar_frag",
):

    if sim_mode not in ("desc", "ecfp4"):
        raise ValueError("sim_mode must be 'desc' or 'ecfp4'")
    os.makedirs(save_dir, exist_ok=True)
    similar_dfs = {}
    
    # ── De-duplicate ONCE outside the loop ────────────────────────────────
    smiles_col = 'frag_smiles' if 'frag_smiles' in fragment_desc.columns else 'SMILES'

    save_dir = os.path.join(save_dir, "Similar_fragments")
    os.makedirs(save_dir, exist_ok=True)
    
    for att_point, frag_smi in frag_repl.items():

        # ── Fresh copy of full library for every att_point ─────────────────
        fd = fragment_desc.copy()

        if sim_mode == "ecfp4":

            cand = shortlist_ecfp4(fd, ref_smi=frag_smi, n_cutoff=cutoff, drop_identical=True)

        elif sim_mode == "desc":
            frag_props = calc_props(frag_smi)
            if frag_props is None:
                print(f"[SKIP] att={att_point}: could not compute 2D props")
                continue

            cand = apply_gates(fd, frag_props, gates = GATES, verbose = True)

        

        expanded_df = prepare_fragments(cand)
        

        ranked = rank_rgroups(expanded_df, frag_smi, smiles_col='new_mol_smi')

        filtered_df = (
        ranked
        .nlargest(len(ranked) if len(ranked) < cutoff else cutoff, "profile_sim")
        .dropna(subset=["frag_smiles"])
        .reset_index(drop=True)
        )


        similar_dfs[att_point] = filtered_df
        csv_path = os.path.join(save_dir, f"att_{att_point}.csv")
        filtered_df.to_csv(csv_path, index=False)
        print(f"att={att_point}: ✓ saved {len(filtered_df)} fragments → {csv_path}")

    return similar_dfs


def remove_dummy_atoms(mol):
    """Remove all dummy atoms (*) from a molecule."""
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if not dummy_indices:
        return mol

    edit = Chem.RWMol(mol)
    for idx in sorted(dummy_indices, reverse=True):
        edit.RemoveAtom(idx)

    cleaned = edit.GetMol()
    Chem.SanitizeMol(cleaned)
    return cleaned


def generate_final_molecules(combinations, scaffold_mol, similar_dfs, att_to_c_conversion):
    """Generate combined molecules from scaffold and fragments.

    Returns a DataFrame with columns:
        smiles          - generated molecule
        rgrp_fragments  - dict {attachment_point: fragment SMILES used}
    """

    def has_free_valence(mol, idx):
        atom = mol.GetAtomWithIdx(idx)
        try:
            atom.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        try:
            if atom.GetNumImplicitHs() > 0:
                return True, False
        except RuntimeError:
            pass
        if atom.GetNumExplicitHs() > 0:
            return True, True
        return False, False

    def remove_h_if_needed(rwmol, idx):
        _, needs_h = has_free_valence(rwmol, idx)
        if needs_h:
            a = rwmol.GetAtomWithIdx(idx)
            a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() - 1))

    def clean_bracketed_carbons(smiles):
        fixed = re.sub(r"\[\d*C\]", "C", smiles)
        mol = Chem.MolFromSmiles(fixed)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    att_pts = list(similar_dfs.keys())
    records = []

    for c in combinations:
        try:
            rgrp_fragments = {}
            current_mol = scaffold_mol

            for i, mol_i in enumerate(c):
                att_pt = att_pts[i]
                row = similar_dfs[att_pt].loc[mol_i]
                frag_att_idx = int(row["frag_att"])

                # record the fragment as it was in similar_dfs
                rgrp_fragments[att_pt] = Chem.MolToSmiles(row["new_mol"])

                frag_i = att_to_c_conversion(row["new_mol"], None, [frag_att_idx])

                offset = current_mol.GetNumAtoms()
                frag_att = offset + frag_att_idx

                combo_rw = Chem.RWMol(Chem.CombineMols(current_mol, frag_i))
                remove_h_if_needed(combo_rw, int(att_pt))
                remove_h_if_needed(combo_rw, int(frag_att))
                combo_rw.AddBond(int(att_pt), int(frag_att),
                                 Chem.rdchem.BondType.SINGLE)
                current_mol = combo_rw.GetMol()

            combined_mol = remove_dummy_atoms(current_mol)
            Chem.SanitizeMol(combined_mol)
            for atom in combined_mol.GetAtoms():
                atom.SetAtomMapNum(0)

            cleaned = clean_bracketed_carbons(Chem.MolToSmiles(combined_mol))
            if cleaned is None:
                continue

            records.append({"smiles": cleaned, "rgrp_fragments": rgrp_fragments})

        except Exception:
            continue

    return pd.DataFrame(records, columns=["smiles", "rgrp_fragments"])


