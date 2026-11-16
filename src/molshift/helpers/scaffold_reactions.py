"""
Turn the ranked scaffolds from scaffold_replacement_screen.py into reaction SMARTS that swap the
reference scaffold for each candidate inside a full molecule, keeping the R groups on the mapped sites.

Design (so nothing about the old scaffold leaks into the product):
  reactant : the WHOLE reference scaffold as a query (element / aromaticity / degree / H-count per atom),
             plus one mapped R-group atom  [!#1:10k]  on every selected attachment atom.
             Scaffold atoms are unmapped -> deleted.  R-group atoms are mapped -> carried over with everything
             hanging off them.
  product  : the candidate fragment, with each dummy replaced by the mapped R-group atom  [*:10k], where k is the
             scaffold site the mapping assigns to that dummy.  All candidate atoms are created fresh from the template.
"""
import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

R_MAP_BASE = 100          # R group on selected_atts[k] gets atom-map number 101 + k
_VALIDATION_R = ["F", "Cl", "Br", "I"]   # distinct monovalent atoms: also checks the mapping direction


# ───────────────────────── reactant / product SMARTS ─────────────────────────
def _atom_query(atom, delta_h=0, strict=True):
    s = f"#{atom.GetAtomicNum()};{'a' if atom.GetIsAromatic() else 'A'}"
    ch = atom.GetFormalCharge()
    s += f";{'+' if ch >= 0 else '-'}{abs(ch)}"
    if strict:
        s += f";D{atom.GetDegree() + (1 if delta_h else 0)};H{atom.GetTotalNumHs() - delta_h}"
    return f"[{s}]"


def build_reactant_smarts(scaff_smi, selected_atts, strict=False):
    mol = Chem.MolFromSmiles(scaff_smi)
    if mol is None:
        raise ValueError("Invalid scaffold SMILES")
    for a in selected_atts:
        if mol.GetAtomWithIdx(int(a)).GetTotalNumHs() < 1:
            raise ValueError(f"Scaffold atom {a} has no H to replace with an R group")

    rw = Chem.RWMol(mol)
    r_idx = {}
    for k, a in enumerate(selected_atts):
        r = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(int(a), r, Chem.BondType.SINGLE)
        r_idx[k] = r

    sel = {int(a) for a in selected_atts}
    for atom in list(rw.GetAtoms()):
        i = atom.GetIdx()
        if i in r_idx.values():
            k = [kk for kk, v in r_idx.items() if v == i][0]
            q = Chem.AtomFromSmarts(f"[!#1:{R_MAP_BASE + 1 + k}]")
        elif i in sel:
            q = Chem.AtomFromSmarts(_atom_query(mol.GetAtomWithIdx(i), delta_h=1, strict=strict))
        else:
            q = Chem.AtomFromSmarts(_atom_query(mol.GetAtomWithIdx(i), strict=strict))
        rw.ReplaceAtom(i, q)
    return Chem.MolToSmarts(rw)


def build_product_smarts(fragment_smi, mapping, selected_atts):
    """
    fragment_smi : the library's `frag_smiles` string, unchanged (e.g. '*c1ccnc(*)c1')
    mapping      : {scaffold_atom_idx: fragment SITE NUMBER}  (1-based; k-th '*' in fragment_smi)
    """
    frag = Chem.MolFromSmiles(fragment_smi)
    if frag is None:
        raise ValueError("Invalid fragment SMILES")

    site_to_k = {}
    for k, a in enumerate(selected_atts):
        site = int(mapping[int(a)])
        if site in site_to_k:
            raise ValueError("duplicate fragment sites in mapping")
        site_to_k[site] = k

    rw = Chem.RWMol(frag)
    dummies = [a for a in rw.GetAtoms() if a.GetAtomicNum() == 0]    # atom order == order of '*' in the SMILES
    if len(dummies) != len(selected_atts):
        raise ValueError("number of dummies != number of selected attachment atoms")

    for site, atom in enumerate(dummies, start=1):
        if site not in site_to_k:
            raise ValueError(f"fragment site {site} not in mapping")
        atom.SetIsotope(0)
        atom.SetAtomMapNum(R_MAP_BASE + 1 + site_to_k[site])
    return Chem.MolToSmiles(rw)

# ───────────────────────── apply + validate ─────────────────────────
def apply_replacement(main_smi, rxn_smarts, max_products=100):
    """Run one reaction on a molecule; returns unique, sanitized product SMILES."""
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    m = Chem.MolFromSmiles(main_smi)
    if m is None:
        raise ValueError("Invalid main molecule SMILES")
    out = []
    for prods in rxn.RunReactants((m,), maxProducts=max_products):
        p = prods[0]
        try:
            Chem.SanitizeMol(p)
            smi = Chem.MolToSmiles(p)
        except Exception:
            continue
        if smi not in out:
            out.append(smi)
    return out


def _validate(scaff_smi, selected_atts, fragment_smi, mapping, rxn_smarts):
    """Decorate the scaffold with F/Cl/Br/I on the selected atoms, run the reaction, and check that the
    expected product (right candidate, right halogen on the right attachment point) is produced."""
    if len(selected_atts) > len(_VALIDATION_R):
        return None
    mol = Chem.MolFromSmiles(scaff_smi)
    rw = Chem.RWMol(mol)
    for k, a in enumerate(selected_atts):
        r = rw.AddAtom(Chem.Atom(_VALIDATION_R[k]))
        rw.AddBond(int(a), r, Chem.BondType.SINGLE)
    test = rw.GetMol()
    Chem.SanitizeMol(test)

    frag = Chem.RWMol(Chem.MolFromSmiles(fragment_smi))
    lab_to_k = {int(mapping[int(a)]): k for k, a in enumerate(selected_atts)}
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            k = lab_to_k[atom.GetIsotope()]
            atom.SetIsotope(0)
            atom.SetAtomicNum(Chem.Atom(_VALIDATION_R[k]).GetAtomicNum())
    expected = frag.GetMol()
    Chem.SanitizeMol(expected)
    exp_smi = Chem.MolToSmiles(expected)
    return exp_smi in apply_replacement(Chem.MolToSmiles(test), rxn_smarts)


# ───────────────────────── one reaction at a time ─────────────────────────
def build_reaction(scaff_smi, selected_atts, fragment_smi, mapping, strict=True, validate=True):
    """
    Build ONE reaction for one fragment + one mapping ({scaffold_atom_idx: fragment_attachment_label}).
    Returns dict(reaction_smarts=..., valid=True/False/None).
    """
    selected_atts = [int(a) for a in selected_atts]
    mapping = {int(k): int(v) for k, v in mapping.items()}
    smarts = (f"{build_reactant_smarts(scaff_smi, selected_atts, strict=strict)}>>"
              f"{build_product_smarts(fragment_smi, mapping, selected_atts)}")
    rdChemReactions.ReactionFromSmarts(smarts)            # must parse
    valid = _validate(scaff_smi, selected_atts, fragment_smi, mapping, smarts) if validate else None
    return dict(reaction_smarts=smarts, valid=valid)


# ───────────────────────── run the top-N ranked reactions on a molecule ─────────────────────────
def generate_products(
    main_smiles,
    scaff_smi,
    selected_atts,
    ranked,                       # dict or DataFrame from screen_scaffolds
    top_n=500,                    # take the top N ranked scaffolds (fewer if fewer exist)
    strict=True,
    include_alt_mappings=False,
    drop_duplicate_products=True, # keep the best-ranked occurrence of each product
    out_dir="Similar_frag",
    out_name="scaff_replaced_products",
):
    """
    Builds the reaction for each of the top_n ranked scaffolds, runs it on main_smiles, and saves one CSV row
    per product:  rank, name, fragment_smiles, score, mapping, product_smiles.
    Returns the DataFrame.
    """
    selected_atts = [int(a) for a in selected_atts]
    reactant = build_reactant_smarts(scaff_smi, selected_atts, strict=strict)

    if isinstance(ranked, pd.DataFrame):
        items = [(r.fragment_smiles, dict(rank=r.rank, score=r.score, mapping=r.mapping,
                                          alt_mappings=r.alt_mappings)) for r in ranked.itertuples()]
    else:
        items = list(ranked.items())
    items = items[:top_n]

    # quick check that the scaffold is in the molecule at all
    probe = Chem.MolFromSmarts(reactant.split(">>")[0] if ">>" in reactant else reactant)
    main = Chem.MolFromSmiles(main_smiles)
    if main is None:
        raise ValueError("Invalid main molecule SMILES")
    if not main.HasSubstructMatch(probe):
        print("[WARN] the scaffold pattern was not found in the main molecule "
              "(check selected_atts / R-group positions, or try strict=False)")
        diagnose_match(main_smiles, scaff_smi, selected_atts)

    rows, seen, n_hit, n_fail, errors = [], set(), 0, 0, {}
    for frag_smi, info in items:
        maps = [info["mapping"]] + (list(info.get("alt_mappings", [])) if include_alt_mappings else [])
        for m_i, mp in enumerate(maps):
            mp = {int(k): int(v) for k, v in mp.items()}
            try:
                smarts = f"{reactant}>>{build_product_smarts(frag_smi, mp, selected_atts)}"
                products = apply_replacement(main_smiles, smarts)
            except Exception as e:
                n_fail += 1
                errors[str(e)[:90]] = errors.get(str(e)[:90], 0) + 1
                continue
            if products:
                n_hit += 1
            for p in products:
                if drop_duplicate_products:
                    if p in seen:
                        continue
                    seen.add(p)
                rows.append(dict(rank=info["rank"], name=f"{info['rank']}" + (f"_alt{m_i}" if m_i else ""),
                                 scaff_fragment=frag_smi, score=info.get("score"),
                                 mapping=json.dumps(mp), product_smiles=p))

    df = pd.DataFrame(rows, columns=["rank", "name", "scaff_fragment", "score", "mapping", "product_smiles"])
    path = None
    if out_dir and out_name:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_name + ".csv")
        df.to_csv(path, index=False)
    print(f"{len(items)} ranked scaffolds used (top_n={top_n}) → {n_hit} gave products, "
          f"{len(items) - n_hit - n_fail} didn't match, {n_fail} failed to build")
    for msg, c in sorted(errors.items(), key=lambda kv: -kv[1])[:3]:
        print(f"   build error x{c}: {msg}")
    print(f"✓ {len(df)} products" + (f" saved → {path}" if path else " (not saved: out_dir/out_name is None)"))
    return df


# ───────────────────────── why doesn't the scaffold match? ─────────────────────────
def diagnose_match(main_smiles, scaff_smi, selected_atts, verbose=True):
    """
    Finds the scaffold in the main molecule by element/aromaticity/bond pattern only, then compares, atom by atom,
    how many substituents each scaffold atom carries in the main molecule versus what the strict reaction expects
    (1 extra heavy neighbour on every selected_atts atom, 0 on every other atom).
    """
    main = Chem.MolFromSmiles(main_smiles)
    sc = Chem.MolFromSmiles(scaff_smi)
    sel = {int(a) for a in selected_atts}
    loose = Chem.MolFromSmarts(Chem.MolToSmarts(sc))
    matches = main.GetSubstructMatches(loose, uniquify=False, maxMatches=2000)
    if not matches:
        if verbose:
            print("[diagnose] scaffold NOT found even by element/bond pattern -> the scaffold SMILES is not a "
                  "substructure of the main molecule (aromaticity/kekulisation, charges or dummy atoms in scaff_smi?)")
        return None

    best = None
    for m in matches:
        issues = []
        for i, mi in enumerate(m):
            extra = main.GetAtomWithIdx(mi).GetDegree() - sc.GetAtomWithIdx(i).GetDegree()
            want = 1 if i in sel else 0
            hs_ok = (main.GetAtomWithIdx(mi).GetTotalNumHs()
                     == sc.GetAtomWithIdx(i).GetTotalNumHs() - want)
            if extra != want or not hs_ok:
                issues.append(dict(scaffold_atom=i, selected=i in sel, extra_substituents=extra,
                                   expected=want, h_count_ok=hs_ok, main_atom=mi))
        if best is None or len(issues) < len(best[1]):
            best = (m, issues)
        if not issues:
            break
    if verbose:
        if not best[1]:
            print("[diagnose] a match with exactly the expected substituent pattern exists; "
                  "the problem is elsewhere (charge/valence, or the pattern text)")
        else:
            print(f"[diagnose] scaffold found ({len(matches)} loose matches). Closest match differs at "
                  f"{len(best[1])} atom(s):")
            for d in best[1]:
                role = "selected" if d["selected"] else "NOT in selected_atts"
                print(f"   scaffold atom {d['scaffold_atom']} ({role}): has {d['extra_substituents']} extra "
                      f"substituent(s) in the main molecule, strict mode expects {d['expected']}"
                      + ("" if d["h_count_ok"] else "  [H count also differs]"))
            print("   -> fix selected_atts to list every substituted scaffold atom, or use strict=False "
                  "(substituents on the other atoms are then dropped with the old scaffold)")
    return dict(match=best[0], issues=best[1])


# ───────────────────────── many molecules, reactions built once, multiple CPUs ─────────────────────────
_RX, _PROBE = [], None


def _init_worker(specs, reactant):
    """Runs once per process: parse the reactions a single time."""
    global _RX, _PROBE
    _PROBE = Chem.MolFromSmarts(reactant)
    _RX = [rdChemReactions.ReactionFromSmarts(s["smarts"]) for s in specs]


def _run_one(args):
    idx, smi = args
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return idx, "invalid", []
    if not m.HasSubstructMatch(_PROBE):          # all reactions share this reactant: skip all of them at once
        return idx, "no_scaffold", []
    seen, hits = set(), []
    for j, rxn in enumerate(_RX):
        for prods in rxn.RunReactants((m,), maxProducts=100):
            p = prods[0]
            try:
                Chem.SanitizeMol(p)
                ps = Chem.MolToSmiles(p)
            except Exception:
                continue
            if ps not in seen:
                seen.add(ps)
                hits.append((j, ps))
    return idx, "ok", hits


def generate_products_batch(
    molecule_df,
    smiles_col,                                        # name of the SMILES column in molecule_df
    scaff_smi,
    selected_atts,
    ranked,
    top_n=500,
    strict=False,
    include_alt_mappings=False,
    drop_duplicate_products=True, # per input molecule: keep the best-ranked occurrence of each product
    n_jobs=None,                  # None = all CPUs but one; 1 = no multiprocessing
    out_dir="Similar_frag",
    out_name="scaff_replaced_products_batch",
):
    """
    Same as generate_products, for many molecules at once. Reactions are built and parsed once (per process),
    molecules without the scaffold are skipped immediately, and molecules are spread over n_jobs processes.
    Returns one DataFrame: input_mol, input_smile, rank, name, fragment_smiles, score, mapping, product_smiles,
    plus every other column of molecule_df (same column names, values copied from the input molecule's row).
    """
    import multiprocessing as _mp
    selected_atts = [int(a) for a in selected_atts]
    molecules = molecule_df[smiles_col].tolist()
    ids = [f"rgrp_{i}" for i in molecule_df.index]
    if len(ids) != len(molecules):
        raise ValueError("ids and molecules must have the same length")

    # NEW: the extra columns to carry over (everything except the SMILES column)
    extra_cols = [c for c in molecule_df.columns if c != smiles_col]
    # NEW: positional records so each row can be looked up by the task index (robust to non-unique index)
    extra_records = molecule_df[extra_cols].to_dict("records") if extra_cols else [{} for _ in molecules]

    # ── build the reactions ONCE ──
    reactant = build_reactant_smarts(scaff_smi, selected_atts, strict=strict)
    if isinstance(ranked, pd.DataFrame):
        items = [(r.fragment_smiles, dict(rank=r.rank, score=r.score, mapping=r.mapping,
                                          alt_mappings=r.alt_mappings)) for r in ranked.itertuples()]

    else:
        items = list(ranked.items())


    specs, errors = [], {}
    for frag_smi, info in items[:top_n]:

        maps = [info["mapping"]] + (list(info.get("alt_mappings", [])) if include_alt_mappings else [])
        for m_i, mp in enumerate(maps):
            mp = {int(k): int(v) for k, v in mp.items()}
            try:
                smarts = f"{reactant}>>{build_product_smarts(frag_smi, mp, selected_atts)}"
               
                rdChemReactions.ReactionFromSmarts(smarts)
            except Exception as e:
                errors[str(e)[:90]] = errors.get(str(e)[:90], 0) + 1
                continue
            specs.append(dict(rank=info["rank"], name=f"{info['rank']}" + (f"_alt{m_i}" if m_i else ""),
                              fragment_smiles=frag_smi, score=info.get("score"),
                              mapping=json.dumps(mp), smarts=smarts))
    print(f"{len(specs)} reactions built from top_n={top_n} ({sum(errors.values())} failed to build)")
    for msg, c in sorted(errors.items(), key=lambda kv: -kv[1])[:3]:
        print(f"   build error x{c}: {msg}")

    # ── run ──
    n_jobs = max(1, (_mp.cpu_count() - 1) if n_jobs is None else int(n_jobs))
    tasks = list(enumerate(molecules))
    if n_jobs == 1 or len(tasks) < 2:
        _init_worker(specs, reactant)
        results = [_run_one(t) for t in tasks]
    else:
        with _mp.Pool(n_jobs, initializer=_init_worker, initargs=(specs, reactant)) as pool:
            results = list(pool.imap(_run_one, tasks, chunksize=max(1, len(tasks) // (n_jobs * 8))))

    # ── assemble ──
    rows, status = [], {"ok": 0, "no_scaffold": 0, "invalid": 0}
    seen_by_mol = {}
    for idx, st, hits in results:
        status[st] += 1
        for j, ps in hits:
            if drop_duplicate_products:
                s = seen_by_mol.setdefault(idx, set())
                if ps in s:
                    continue
                s.add(ps)
            sp = specs[j]
            row = dict(input_mol=ids[idx], input_smile=molecules[idx], rank=sp["rank"], name=sp["name"],
                       scaff_fragment=sp["fragment_smiles"], score=sp["score"],
                       mapping=sp["mapping"], product_smiles=ps)
            # NEW: copy over the other columns from the input molecule's row.
            # Core output columns take priority if a name collides (see note below).
            for c in extra_cols:
                if c not in row:
                    row[c] = extra_records[idx][c]
            rows.append(row)

        core_cols = ["input_mol", "input_smile", "rank", "name", "scaff_fragment",
                 "score", "mapping", "product_smiles"]
    # NEW: core columns first, then the carried-over columns in their original order
    out_cols = core_cols + [c for c in extra_cols if c not in core_cols]
    df = pd.DataFrame(rows, columns=out_cols)

    print(f"{len(molecules)} molecules: {status['ok']} processed, {status['no_scaffold']} without the scaffold "
          f"pattern, {status['invalid']} invalid SMILES")
    if out_dir and out_name:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_name + ".csv")
        df.to_csv(path, index=False)
        print(f"✓ {len(df)} products saved → {path}")
    else:
        print(f"✓ {len(df)} products (not saved: out_dir/out_name is None)")
    return df