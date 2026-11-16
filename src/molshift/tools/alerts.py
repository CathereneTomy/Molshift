"""
alert_filter.py -- remove molecules / fragments that carry PAINS or BRENK structural alerts.

    clean, flagged = filter_alerts(df, smiles_col="frag_smiles")

  clean   : rows with no alerts (and a valid SMILES)
  flagged : the removed rows, with the alert names that fired (pains_alerts, brenk_alerts, n_alerts)

Options
  catalogs          which RDKit catalogs to use, default ("PAINS", "BRENK"); e.g. add "NIH"
  ignore_alerts     alert names to disregard (a noisy BRENK alert you do not care about)
  reference_smiles  alerts already present in the reference molecule are not held against candidates
  cap_dummies       fragments written with '*' / '[8*]' are capped with H before checking (default True)
  n_jobs            >1 uses multiprocessing (worth it for hundreds of thousands of rows)

An alert is a warning, not proof of a problem: many marketed drugs trigger PAINS/BRENK patterns.
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

_CATS = {}


def _catalog(name):
    if name not in _CATS:                                   # built once per process
        p = FilterCatalogParams()
        p.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
        _CATS[name] = FilterCatalog(p)
    return _CATS[name]


def _cap(mol):
    """Dummy atoms (attachment points) -> H, so alerts are judged on the fragment as a real molecule."""
    if not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return mol
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(1)
            a.SetIsotope(0)
            a.SetNoImplicit(True)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return Chem.RemoveHs(m)


def alerts_for(smi, catalogs=("PAINS", "BRENK"), cap_dummies=True):
    """{catalog: sorted alert names that fired}, or None if the SMILES cannot be parsed."""
    mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
    if mol is None:
        return None
    try:
        if cap_dummies:
            mol = _cap(mol)
    except Exception:
        return None
    return {c: sorted({e.GetDescription() for e in _catalog(c).GetMatches(mol)}) for c in catalogs}


def _worker(args):
    smi, catalogs, cap = args
    return alerts_for(smi, catalogs, cap)


def filter_alerts(data, smiles_col="smiles", catalogs=("PAINS", "BRENK"), ignore_alerts=(),
                  reference_smiles=None, cap_dummies=True, n_jobs=1, verbose=True):
    df = pd.DataFrame({smiles_col: list(data)}) if not isinstance(data, pd.DataFrame) else data.copy()
    smiles = df[smiles_col].tolist()
    ignore = set(ignore_alerts)

    ref_alerts = set()
    if reference_smiles:
        r = alerts_for(reference_smiles, catalogs, cap_dummies)
        if r is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles}")
        ref_alerts = {(c, a) for c, names in r.items() for a in names}
        if verbose and ref_alerts:
            print(f"reference carries {len(ref_alerts)} alert(s), not held against candidates: "
                  f"{sorted(a for _, a in ref_alerts)}")

    jobs = [(s, tuple(catalogs), cap_dummies) for s in smiles]
    if n_jobs and n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = pool.map(_worker, jobs, chunksize=500)
    else:
        results = [_worker(j) for j in jobs]

    valid, n_alerts = [], []
    cols = {f"{c.lower()}_alerts": [] for c in catalogs}
    for res in results:
        if res is None:
            valid.append(False)
            n_alerts.append(0)
            for k in cols:
                cols[k].append("")
            continue
        valid.append(True)
        total = 0
        for c in catalogs:
            kept = [a for a in res[c] if a not in ignore and (c, a) not in ref_alerts]
            cols[f"{c.lower()}_alerts"].append(";".join(kept))
            total += len(kept)
        n_alerts.append(total)

    df["valid_smiles"] = valid
    for k, v in cols.items():
        df[k] = v
    df["n_alerts"] = n_alerts
    ok = df["valid_smiles"] & (df["n_alerts"] == 0)

    if verbose:
        n = len(df)
        print(f"{n:,} molecules | invalid SMILES: {(~df['valid_smiles']).sum():,} | "
              f"removed for alerts: {(df['valid_smiles'] & (df['n_alerts'] > 0)).sum():,} | kept: {ok.sum():,}")
        for c in catalogs:
            col = f"{c.lower()}_alerts"
            print(f"  {c:6} alone flags {(df[col] != '').sum():,}")
        top = pd.Series([a for s in sum((df[f'{c.lower()}_alerts'].tolist() for c in catalogs), []) for a in s.split(";") if a])
        if len(top):
            print("  most frequent alerts:", {k: int(v) for k, v in top.value_counts().head(8).items()})
    return df[ok].drop(columns=["valid_smiles"]).reset_index(drop=True), df[~ok].reset_index(drop=True)