"""
frag_features.py -- per-fragment featurisation (dummy atoms -> H, labels ignored).

`featurize_core` is shared by the fragment library and the reference scaffold
(`featurize_scaffold`) so both sides are computed identically.

Changes vs. the original build_fragment_library.py:
  * BRICS labels are ignored (no site_labels, no label-based dedup).
  * Kappa1-3 and BertzCT added (previously in the separate descriptor script).
  * No dedup / CSV / pkl here -- dedup happens upstream in DuckDB, storage is Parquet.
"""
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

FEATURE_VERSION = "2.1"
MORGAN_RADIUS = 2  # ECFP4

# columns that live in the "extras" table (heavy, only needed at screening time)
EXTRA_COLS = ["site_dist", "site_pairs", "site_env"]

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS)


def _safe(fn, mol, nd=3):
    try:
        v = fn(mol)
        return round(float(v), nd)
    except Exception:
        return float("nan")


def _is_rot(bond):
    """Acyclic single bond between non-terminal atoms (used only for path flags)."""
    return (bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()
            and bond.GetBeginAtom().GetDegree() > 1 and bond.GetEndAtom().GetDegree() > 1)


def featurize_core(core, anchors):
    """
    core    : sanitized heavy-atom mol (attachment positions carry implicit Hs)
    anchors : atom indices of the attachment atoms, in site order
    """
    f = dict(
        n_sites=len(anchors),
        heavy_atoms=core.GetNumHeavyAtoms(),
        hba=Lipinski.NumHAcceptors(core),
        hbd=Lipinski.NumHDonors(core),
        rot_bonds=rdMolDescriptors.CalcNumRotatableBonds(core),
        rings=rdMolDescriptors.CalcNumRings(core),
        arom_rings=rdMolDescriptors.CalcNumAromaticRings(core),
        heteroatoms=rdMolDescriptors.CalcNumHeteroatoms(core),
        mol_wt=round(Descriptors.MolWt(core), 3),
        logp=round(Crippen.MolLogP(core), 3),
        tpsa=round(rdMolDescriptors.CalcTPSA(core), 3),
        fsp3=round(rdMolDescriptors.CalcFractionCSP3(core), 3),
        kappa1=_safe(Descriptors.Kappa1, core),
        kappa2=_safe(Descriptors.Kappa2, core),
        kappa3=_safe(Descriptors.Kappa3, core),
        bertzct=_safe(Descriptors.BertzCT, core),
    )
    # branching: heavy atoms with >=3 connections, counting each attachment bond as a connection
    # (otherwise isopropyl and n-propyl would look identical once the dummy is capped with H)
    n_att = {}
    for a in anchors:
        n_att[int(a)] = n_att.get(int(a), 0) + 1
    f["n_branch"] = sum(1 for at in core.GetAtoms() if at.GetDegree() + n_att.get(at.GetIdx(), 0) >= 3)

    dm = Chem.GetDistanceMatrix(core)
    f["diameter"] = int(dm.max()) if dm.size else 0

    ranks = list(Chem.CanonicalRankAtoms(core, breakTies=False))
    f["site_anchor_idx"] = [int(a) for a in anchors]
    f["site_sym"] = [int(ranks[a]) for a in anchors]
    f["site_dist"] = [[int(dm[a][b]) for b in anchors] for a in anchors]

    pairs = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if anchors[i] == anchors[j]:          # both sites on one atom (e.g. *C*, *O*)
                path = (anchors[i],)
            else:
                path = Chem.GetShortestPath(core, anchors[i], anchors[j])
            n_rot = sum(_is_rot(core.GetBondBetweenAtoms(path[k], path[k + 1]))
                        for k in range(len(path) - 1))
            atoms = [core.GetAtomWithIdx(p) for p in path]
            pairs.append(dict(
                i=i, j=j, dist=int(dm[anchors[i]][anchors[j]]), path_rot=int(n_rot),
                path_arom_frac=round(sum(a.GetIsAromatic() for a in atoms) / len(atoms), 3),
                path_ring_frac=round(sum(a.IsInRing() for a in atoms) / len(atoms), 3)))
    f["site_pairs"] = pairs

    f["site_env"] = [
        {int(k): int(v) for k, v in
         _GEN.GetSparseCountFingerprint(core, fromAtoms=[int(a)]).GetNonzeroElements().items()}
        for a in anchors
    ]
    return f


def featurize_scaffold(scaff_smi, selected_atts):
    """Same features for the reference scaffold (selected_atts = atom indices in scaff_smi)."""
    core = Chem.MolFromSmiles(scaff_smi)
    if core is None:
        raise ValueError("Invalid scaffold SMILES")
    return featurize_core(core, [int(a) for a in selected_atts])


def env_tanimoto(a, b):
    """Min/max (count) Tanimoto between two site_env dicts."""
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    den = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return num / den if den else 0.0


def featurize_fragment(smi):
    """
    smi: fragment SMILES with '*' dummies (labels, if any, are ignored).
    Returns (features dict, None) or (None, error string).
    Anchor order = order of '*' atoms in the parsed SMILES (deterministic for canonical input).
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None, "invalid SMILES"

        rw = Chem.RWMol(mol)
        tags, n_sites = {}, 0
        for a in rw.GetAtoms():
            if a.GetAtomicNum() == 0:
                nbs = a.GetNeighbors()
                if len(nbs) != 1:
                    return None, "attachment atom must have exactly one neighbour"
                tags.setdefault(nbs[0].GetIdx(), []).append(n_sites)
                n_sites += 1
                a.SetAtomicNum(1)      # dummy -> H
                a.SetIsotope(0)
                a.SetNoImplicit(True)
        if n_sites == 0:
            return None, "no attachment points"
        for idx, ks in tags.items():
            rw.GetAtomWithIdx(idx).SetProp("site_ids", ",".join(map(str, ks)))

        m = rw.GetMol()
        Chem.SanitizeMol(m)
        core = Chem.RemoveHs(m)

        anchors = [None] * n_sites
        for a in core.GetAtoms():
            if a.HasProp("site_ids"):
                for k in a.GetProp("site_ids").split(","):
                    anchors[int(k)] = a.GetIdx()
        if any(x is None for x in anchors):
            return None, "could not locate attachment atoms"

        f = featurize_core(core, anchors)
        mapped = Chem.Mol(core)
        for k, a in enumerate(anchors):
            mapped.GetAtomWithIdx(a).SetAtomMapNum(k + 1)
        f["core_smiles_mapped"] = Chem.MolToSmiles(mapped)
        try:
            f["murcko"] = MurckoScaffold.MurckoScaffoldSmiles(mol=core)
        except Exception:
            f["murcko"] = ""
        return f, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ───────────────────────── USRCAT (3D shape + pharmacophore) ─────────────────────────
USRCAT_DIM = 60     # 5 atom subsets x 12 moments


def _core_from_smiles(smi):
    """Fragment (with '*') or scaffold (without) SMILES -> sanitized H-capped heavy-atom mol.
    Same capping as featurize_fragment, so library and reference scaffolds go through one path."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(1)
            a.SetIsotope(0)
            a.SetNoImplicit(True)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return Chem.RemoveHs(m)


def get_3d_mol(smi, seed=42):
    """One seeded ETKDGv3 conformer (+ MMFF clean-up), returned WITHOUT explicit Hs. None on failure."""
    core = _core_from_smiles(smi)
    if core is None:
        return None
    mh = Chem.AddHs(core)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.maxIterations = 100
    if AllChem.EmbedMolecule(mh, ps) != 0:
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(mh, ps) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
    except Exception:
        pass
    return Chem.RemoveHs(mh)


def usrcat_vector(smi, seed=42):
    """Returns (float32 array of length 60, None) or (None, error string)."""
    try:
        mol = get_3d_mol(smi, seed)
        if mol is None:
            return None, "embedding failed"
        v = rdMolDescriptors.GetUSRCAT(mol)
        return np.asarray(v, dtype=np.float32), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:80]}"


def usrcat_scores(q, M):
    """USRCAT similarity of one query vector (60,) against a matrix (N, 60).
    Equivalent to rdMolDescriptors.GetUSRScore with default (equal) subset weights."""
    d = np.abs(M - q).reshape(len(M), 5, 12).mean(axis=2)     # mean abs diff per subset, (N, 5)
    return 1.0 / (1.0 + d.sum(axis=1))                        # RDKit default: equal weights, summed


def load_usrcat(path):
    """usrcat_cache.parquet -> (frag_ids int64 array (N,), matrix float32 (N, 60))."""
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    ids = t["frag_id"].to_numpy()
    M = np.asarray(t["usrcat"].combine_chunks().values, dtype=np.float32).reshape(len(ids), USRCAT_DIM)
    return ids, M


def load_fragment_library(path, extras_path=None):
    """
    Load fragment_features.parquet. If extras_path is given, join the heavy columns
    (site_dist, site_pairs, site_env) and decode them to Python objects.
    """
    df = pd.read_parquet(path)
    for c in ("site_anchor_idx", "site_sym"):
        df[c] = df[c].apply(list)
    if extras_path:
        ex = pd.read_parquet(extras_path)
        df = df.merge(ex, on="frag_id", how="left")
        df["site_dist"] = df["site_dist"].apply(json.loads)
        df["site_pairs"] = df["site_pairs"].apply(json.loads)
        df["site_env"] = df["site_env"].apply(
            lambda s: [{int(k): v for k, v in e.items()} for e in json.loads(s)])
    return df
