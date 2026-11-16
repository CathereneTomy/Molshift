"""
Filter a set of SMILES against a reference SMILES using:
  - Lipinski-type properties (MW, logP, HBD, HBA, TPSA, RotB): window around the reference value
  - SA score: upper cutoff  (keep SA  <= ref_SA  + buffer)
  - QED:      lower cutoff  (keep QED >= ref_QED - buffer)

Usage (notebook):
    from ref_filter import filter_by_reference, filter_by_reference_parallel

    result, report = filter_by_reference(df, "smiles", ref_smiles)             # single process
    result, report = filter_by_reference_parallel(df, "smiles", ref_smiles)    # multi-core

Adaptive relaxation
-------------------
If fewer than `min_fraction` (default 0.20) of the parseable molecules pass, the
filter that is mainly responsible for the losses is widened step by step until
the target is reached (or every filter hit its max expansion). Pass
min_fraction=0 to switch relaxation off (default windows only).

Output
------
result : DataFrame of passing molecules, with property columns plus
           relaxation_step         0 = passed with default windows, k = first passed at relaxation step k
           passed_default          True / False
           failed_default_filters  filters the molecule failed under the default windows
         sorted: default passers first, then step 1, step 2, ...; within a tier by
         Tanimoto_to_ref (descending).
report : dict of the filters used in this run (reference value, initial/final window,
         whether relaxed, relaxation log, counts). report["windows"] holds the final
         {name: (low, high)} windows, in the same format the old second return value had.

SA score needs RDKit's Contrib sascorer. If the import fails (common with pip
installs of rdkit), download sascorer.py and fpscores.pkl.gz from
https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
and put them in the same folder as this script.
"""
import os
import sys
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors, AllChem
from rdkit.SimDivFilters import rdSimDivPickers

# ---- SA scorer import -------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
except ImportError as e:
    raise ImportError(
        "Could not import sascorer. Place sascorer.py and fpscores.pkl.gz "
        "(from RDKit Contrib/SA_Score) next to this script."
    ) from e

# ---- Default buffers --------------------------------------------------------
# ("rel", x) -> ref * (1 +/- x)      ("abs", x) -> ref +/- x
LIPINSKI_BUFFERS = {
    "MW":   ("rel", 0.20),   # +/- 20 %
    "logP": ("abs", 1.0),    # +/- 1 log unit
    "HBD":  ("abs", 1),      # +/- 1
    "HBA":  ("abs", 2),      # +/- 2
    "TPSA": ("rel", 0.20),
    "RotB": ("abs", 2),
}
SA_BUFFER = 1.0     # keep SA <= ref_SA + 1.0   (lower SA = easier to make)
QED_BUFFER = 0.10   # keep QED >= ref_QED - 0.10 (higher QED = more drug-like)

_NON_NEGATIVE = {"HBD", "HBA", "RotB", "MW", "TPSA"}   # lower bound floored at 0


# =============================================================================
# Properties
# =============================================================================
def calc_properties(mol):
    """Return dict of all properties for an RDKit mol."""
    return {
        "MW": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "SA": sascorer.calculateScore(mol),
        "QED": QED.qed(mol),
    }


def _fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def _props_chunk(args):
    """Worker: properties + Tanimoto for one chunk of (position, smiles) pairs."""
    chunk, ref_smiles = args
    ref_fp = _fp(Chem.MolFromSmiles(ref_smiles))
    out = []
    for pos, smi in chunk:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        p = calc_properties(mol)
        p["Tanimoto_to_ref"] = DataStructs.TanimotoSimilarity(ref_fp, _fp(mol))
        p["_pos"] = pos
        out.append(p)
    return out


def _compute_all_properties(df, smiles_col, ref_smiles, n_jobs=1,
                            chunksize=5000, progress=False):
    """
    Properties for every parseable molecule, computed once. Returns a DataFrame
    indexed by row position (iloc) in df. n_jobs=1 runs in the current process.
    """
    smiles = df[smiles_col].tolist()
    tasks = [(list(enumerate(smiles[i:i + chunksize], start=i)), ref_smiles)
             for i in range(0, len(smiles), chunksize)]
    rows = []
    if n_jobs == 1:
        it = tasks
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(tasks, desc="Properties")
            except ImportError:
                pass
        for t in it:
            rows.extend(_props_chunk(t))
    else:
        with Pool(n_jobs) as pool:
            it = pool.imap_unordered(_props_chunk, tasks)
            if progress:
                try:
                    from tqdm import tqdm
                    it = tqdm(it, total=len(tasks), desc=f"Filtering ({n_jobs} procs)")
                except ImportError:
                    pass
            for res in it:
                rows.extend(res)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_pos").set_index("_pos")


# =============================================================================
# Windows
# =============================================================================
def _make_windows(ref_props, buffers):
    """buffers: {name: (kind, value)} incl. 'SA' and 'QED' -> {name: (low, high)}."""
    windows = {}
    for name, (kind, b) in buffers.items():
        r = ref_props[name]
        if name == "SA":
            lo, hi = -np.inf, r + b                      # upper limit only
        elif name == "QED":
            lo, hi = max(r - b, 0.0), np.inf             # lower limit only
        else:
            lo, hi = (r * (1 - b), r * (1 + b)) if kind == "rel" else (r - b, r + b)
            if name in _NON_NEGATIVE:
                lo = max(lo, 0)
        windows[name] = (lo, hi)
    return windows


def make_windows(ref_props, lipinski_buffers=LIPINSKI_BUFFERS,
                 sa_buffer=SA_BUFFER, qed_buffer=QED_BUFFER):
    """Build (low, high) windows from the reference values."""
    buffers = dict(lipinski_buffers)
    buffers["SA"] = ("abs", sa_buffer)
    buffers["QED"] = ("abs", qed_buffer)
    return _make_windows(ref_props, buffers)


def _pass_table(props, windows):
    return pd.DataFrame(
        {n: (props[n] >= lo) & (props[n] <= hi) for n, (lo, hi) in windows.items()},
        index=props.index,
    )


def _clean(x):
    """JSON-friendly number (inf -> None)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return float(x)


def _win(w):
    return [_clean(w[0]), _clean(w[1])]


def sort_by_tier(result):
    """Default passers first, then relaxation step 1, 2, ...; within a tier,
    most similar to the reference first."""
    return result.sort_values(["relaxation_step", "Tanimoto_to_ref"],
                              ascending=[True, False], kind="stable")


# =============================================================================
# Shared core: relaxation + annotation + report
# =============================================================================
def _select(df, ref_smiles, ref_props, props, n_unparsed, buffers,
            min_fraction, expand_factor, max_expand, max_iter, verbose):
    n_eval = len(props)
    initial_windows = _make_windows(ref_props, buffers)

    if n_eval == 0:
        report = {
            "reference_smiles": ref_smiles, "n_input": int(len(df)),
            "n_unparseable": int(n_unparsed), "n_evaluated": 0, "n_passed": 0,
            "n_passed_default": 0, "n_passed_only_after_relaxation": 0,
            "fraction_passed": 0.0, "target_met": False, "relaxation_applied": False,
            "windows": initial_windows, "filters": {}, "relaxation_steps": [],
        }
        return df.iloc[0:0].copy(), report

    n_target = math.ceil(min_fraction * n_eval) if min_fraction else 0
    scale = {n: 1.0 for n in buffers}        # multiplier on each initial buffer
    windows = dict(initial_windows)
    steps = []
    pass_step = pd.Series(np.nan, index=props.index)
    ok_default = None

    def current_buffers():
        return {n: (buffers[n][0], buffers[n][1] * scale[n]) for n in buffers}

    for it in range(max_iter + 1):
        ok = _pass_table(props, windows)
        if ok_default is None:
            ok_default = ok.copy()
        n_fail = (~ok).sum(axis=1)
        n_pass = int((n_fail == 0).sum())
        # windows only widen, so a molecule that passes keeps passing:
        # tier 0 = default windows, tier k = first passed after relaxation step k
        pass_step[(n_fail == 0) & pass_step.isna()] = len(steps)
        if steps:
            steps[-1]["n_passing_after"] = n_pass
        if n_pass >= n_target or it == max_iter:
            break

        # --- which filter is mainly responsible? ---
        # 1) molecules that fail ONLY that filter (relaxing it alone rescues them)
        # 2) if no filter has such molecules, the filter with lowest standalone pass rate
        candidates = [n for n in buffers if scale[n] * expand_factor <= max_expand + 1e-9]
        if not candidates:
            break
        unique = {n: int(((~ok[n]) & (n_fail == 1)).sum()) for n in candidates}
        pass_rate = {n: float(ok[n].mean()) for n in candidates}
        if max(unique.values()) > 0:
            chosen = max(candidates, key=lambda n: (unique[n], -pass_rate[n]))
        else:
            chosen = min(candidates, key=lambda n: pass_rate[n])

        old_w = windows[chosen]
        scale[chosen] *= expand_factor
        windows = _make_windows(ref_props, current_buffers())
        steps.append({
            "step": it + 1,
            "filter": chosen,
            "uniquely_blocked_by_filter": unique[chosen],
            "standalone_pass_rate_before": round(pass_rate[chosen], 4),
            "old_window": _win(old_w),
            "new_window": _win(windows[chosen]),
            "n_passing_after": None,         # filled in on the next loop pass
        })

    ok = _pass_table(props, windows)
    keep = ok.all(axis=1)
    n_pass = int(keep.sum())
    target_met = n_pass >= n_target

    # --- result frame --------------------------------------------------------
    pos = props.index[keep.values].to_numpy()
    result = df.iloc[pos].copy()
    for col in props.columns:
        result[col] = props.loc[pos, col].values
    result["relaxation_step"] = pass_step.loc[pos].astype(int).values   # 0 = default
    result["passed_default"] = result["relaxation_step"] == 0
    result["failed_default_filters"] = [
        ", ".join(n for n in buffers if not ok_default.loc[p, n]) for p in pos
    ]
    result = sort_by_tier(result)
    n_pass_default = int(result["passed_default"].sum())

    # --- report dictionary ---------------------------------------------------
    final_buffers = current_buffers()
    filters = {}
    for n in buffers:
        filters[n] = {
            "reference_value": float(ref_props[n]),
            "buffer_type": buffers[n][0],
            "side": "upper_limit_only" if n == "SA"
                    else "lower_limit_only" if n == "QED" else "two_sided",
            "initial_buffer": float(buffers[n][1]),
            "final_buffer": float(final_buffers[n][1]),
            "expansion_factor": round(scale[n], 4),
            "relaxed": scale[n] > 1.0,
            "initial_window": _win(initial_windows[n]),
            "final_window": _win(windows[n]),
            "n_pass_standalone": int(ok[n].sum()),
        }
    report = {
        "reference_smiles": ref_smiles,
        "n_input": int(len(df)),
        "n_unparseable": int(n_unparsed),
        "n_evaluated": int(n_eval),
        "min_fraction_target": min_fraction,
        "n_required": n_target,
        "n_passed": n_pass,
        "n_passed_default": n_pass_default,
        "n_passed_only_after_relaxation": n_pass - n_pass_default,
        "fraction_passed": round(n_pass / n_eval, 4),
        "target_met": bool(target_met),
        "relaxation_applied": len(steps) > 0,
        "expand_factor": expand_factor,
        "max_expand": max_expand,
        "windows": windows,                 # final {name: (low, high)}, old return format
        "filters": filters,
        "relaxation_steps": steps,
    }

    if verbose:
        print_report(report)
    if not target_met:
        print(f"WARNING: only {n_pass}/{n_eval} ({n_pass / n_eval:.1%}) passed even after "
              f"relaxing to the max allowed expansion ({max_expand}x). "
              f"Raise max_expand or lower min_fraction.")
    return result, report


def _prepare(ref_smiles, lipinski_buffers, sa_buffer, qed_buffer):
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if ref_mol is None:
        raise ValueError("Reference SMILES could not be parsed.")
    ref_props = calc_properties(ref_mol)
    buffers = dict(lipinski_buffers)
    buffers["SA"] = ("abs", sa_buffer)
    buffers["QED"] = ("abs", qed_buffer)
    return ref_props, buffers


# =============================================================================
# Public entry points
# =============================================================================
def filter_by_reference(df, smiles_col, ref_smiles,
                        lipinski_buffers=LIPINSKI_BUFFERS,
                        sa_buffer=SA_BUFFER, qed_buffer=QED_BUFFER,
                        min_fraction=0.20, expand_factor=1.25, max_expand=5.0,
                        max_iter=100, verbose=True):
    """
    Single-process version.

    Returns
    -------
    result : DataFrame of molecules passing all (possibly relaxed) filters, with
             relaxation_step / passed_default / failed_default_filters columns,
             sorted default-first
    report : dict of the filters used (see module docstring)
    """
    ref_props, buffers = _prepare(ref_smiles, lipinski_buffers, sa_buffer, qed_buffer)
    props = _compute_all_properties(df, smiles_col, ref_smiles, n_jobs=1)
    n_unparsed = len(df) - len(props)
    return _select(df, ref_smiles, ref_props, props, n_unparsed, buffers,
                   min_fraction, expand_factor, max_expand, max_iter, verbose)


def filter_by_reference_parallel(df, smiles_col, ref_smiles, n_jobs=None,
                                 chunksize=5000,
                                 lipinski_buffers=LIPINSKI_BUFFERS,
                                 sa_buffer=SA_BUFFER, qed_buffer=QED_BUFFER,
                                 min_fraction=0.20, expand_factor=1.25, max_expand=5.0,
                                 max_iter=100, progress=True, verbose=False):
    """
    Multi-core version of filter_by_reference. Same windows, relaxation, columns
    and report; only the property calculation is spread over processes.

    n_jobs    : number of processes (default: all cores minus one)
    chunksize : molecules per task; 2000-10000 works well
    verbose   : print the report (off by default, as before)

    Note: all properties are computed for every molecule (no early exit), so the
    windows can be relaxed afterwards without recomputing anything.
    """
    ref_props, buffers = _prepare(ref_smiles, lipinski_buffers, sa_buffer, qed_buffer)
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    props = _compute_all_properties(df, smiles_col, ref_smiles, n_jobs=n_jobs,
                                    chunksize=chunksize, progress=progress)
    n_unparsed = len(df) - len(props)
    return _select(df, ref_smiles, ref_props, props, n_unparsed, buffers,
                   min_fraction, expand_factor, max_expand, max_iter, verbose)


# =============================================================================
# Utilities
# =============================================================================
def pick_diverse(result, smiles_col, n, seed=42):
    """MaxMin diversity pick of n molecules from the filtered set (tier order kept)."""
    if len(result) <= n:
        return result
    mols = [Chem.MolFromSmiles(s) for s in result[smiles_col]]
    fps = [_fp(m) for m in mols]
    picker = rdSimDivPickers.MaxMinPicker()
    picks = picker.LazyBitVectorPick(fps, len(fps), n, seed=seed)
    sub = result.iloc[list(picks)]
    if "relaxation_step" in sub.columns:
        sub = sort_by_tier(sub)
    return sub


def print_windows(ref_props, windows):
    print("Reference properties and applied windows")
    print("-" * 50)
    for name, (lo, hi) in windows.items():
        lo_s = "-inf" if np.isneginf(lo) else f"{lo:.2f}"
        hi_s = "+inf" if np.isposinf(hi) else f"{hi:.2f}"
        print(f"{name:5s} ref = {ref_props[name]:8.2f}   keep in [{lo_s}, {hi_s}]")
    print("-" * 50)


def print_report(report):
    print(f"Input: {report['n_input']}  unparseable: {report['n_unparseable']}  "
          f"evaluated: {report['n_evaluated']}")
    print(f"  of which with default windows: {report['n_passed_default']}, "
          f"only after relaxation: {report['n_passed_only_after_relaxation']}")
    print(f"Passed: {report['n_passed']} ({report['fraction_passed']:.1%}) | "
          f"target >= {report['min_fraction_target']:.0%} "
          f"({'met' if report['target_met'] else 'NOT met'})")
    print("-" * 78)
    print(f"{'filter':6s} {'ref':>8s}  {'initial window':>22s}  {'final window':>22s}  relaxed")

    def rng(w):
        lo = "-inf" if w[0] is None else f"{w[0]:.2f}"
        hi = "+inf" if w[1] is None else f"{w[1]:.2f}"
        return f"[{lo}, {hi}]"

    for n, f in report["filters"].items():
        print(f"{n:6s} {f['reference_value']:8.2f}  "
              f"{rng(f['initial_window']):>22s}  {rng(f['final_window']):>22s}  "
              f"{'x%.2f' % f['expansion_factor'] if f['relaxed'] else '-'}")
    print("-" * 78)
    if report["relaxation_steps"]:
        print(f"{len(report['relaxation_steps'])} relaxation step(s):")
        for s in report["relaxation_steps"]:
            print(f"  step {s['step']:2d}: widened {s['filter']:5s} "
                  f"(sole blocker for {s['uniquely_blocked_by_filter']} mols) "
                  f"-> {s['n_passing_after']} passing")