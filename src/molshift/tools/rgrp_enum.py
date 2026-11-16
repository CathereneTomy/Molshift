from rdkit import Chem
from rdkit.Chem import Draw
from molshift.helpers.rgrp_enum_utils import (find_frag_atts,process_and_save_similar_fragments,
                           generate_final_molecules,att_to_c_conversion)
import matplotlib.pyplot as plt
import os
import pandas as pd
import itertools
from pathlib import Path
import math

# pip install rdkit
from rdkit import Chem
from rdkit.Chem import BRICS, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import BRICS, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_brics_aware_scaffold(smiles: str, coverage_limit: float = 0.70,
                             show: bool = True, size=(550, 450),
                             min_rgroup: int = 2, max_rgroup: int = 12,
                             min_scaffold_frac: float = 0.25,
                             max_candidates: int = 20000):
    """
    Return the SMILES of a scaffold built from BRICS fragments, chosen so that
    the pieces it leaves out (the R-groups) are neither huge nor tiny.
    Displays the molecule with the scaffold highlighted (nothing is saved).

    How it works
    ------------
    1. The molecule is cut at all BRICS bonds. The fragments form a graph
       (fragments = nodes, BRICS bonds = edges).
    2. Every connected group of fragments is a candidate scaffold. Candidates must
       contain a ring, cover between `min_scaffold_frac` and `coverage_limit`
       of the heavy atoms, and be a single connected piece.
    3. For each candidate the left-out pieces (connected components of the
       remaining atoms = R-groups) are measured in heavy atoms. A candidate is
       penalised for every R-group larger than `max_rgroup` or smaller than
       `min_rgroup` (penalty = number of atoms outside the allowed range).
    4. The winner is the candidate with
           (a) the lowest penalty,
           (b) then the LARGEST scaffold,
           (c) then the best overlap with the Murcko scaffold.
       If no candidate has penalty 0, the best compromise is used and the title
       of the plot says so.

    Parameters
    ----------
    coverage_limit    : scaffold may cover at most this fraction of heavy atoms
    min_rgroup        : smallest acceptable R-group (heavy atoms); None = no lower limit
    max_rgroup        : largest acceptable R-group (heavy atoms);  None = no upper limit
    min_scaffold_frac : scaffold must cover at least this fraction of heavy atoms
    max_candidates    : safety cap on enumerated candidate scaffolds (big molecules)

    Returns None if the molecule has no rings (no Murcko scaffold).
    Raises ValueError for an invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    murcko = MurckoScaffold.GetScaffoldForMol(mol)
    if murcko is None or murcko.GetNumAtoms() == 0:
        print("No Murcko scaffold (molecule has no rings).")
        if show:
            Draw.MolToImage(mol, size=size).show()
        return None

    n = mol.GetNumAtoms()
    nbrs = [[x.GetIdx() for x in a.GetNeighbors()] for a in mol.GetAtoms()]
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}

    # --- BRICS fragments as sets of original atom indices ---
    bond_idxs = [mol.GetBondBetweenAtoms(a, b).GetIdx()
                 for (a, b), _ in BRICS.FindBRICSBonds(mol)]
    if bond_idxs:
        frag_mol = Chem.FragmentOnBonds(mol, bond_idxs, addDummies=True)
        raw = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
        frags = [frozenset(i for i in f if i < n) for f in raw]  # drop dummies
    else:
        frags = [frozenset(range(n))]
    nf = len(frags)

    # fragment adjacency graph
    frag_of = {a: i for i, f in enumerate(frags) for a in f}
    adj = {i: set() for i in range(nf)}
    for b in mol.GetBonds():
        fi, fj = frag_of[b.GetBeginAtomIdx()], frag_of[b.GetEndAtomIdx()]
        if fi != fj:
            adj[fi].add(fj)
            adj[fj].add(fi)

    # Murcko scaffold locations (used only as a tie-breaker)
    matches = mol.GetSubstructMatches(murcko, uniquify=True, maxMatches=1000)
    murcko_sets = list({frozenset(m) for m in matches})
    if not murcko_sets:
        return None

    # --- enumerate connected groups of fragments (pruned by max size) ---
    max_atoms = coverage_limit * n
    fsize = [len(f) for f in frags]
    seen, stack = set(), []
    for i in range(nf):
        if fsize[i] <= max_atoms:
            s = frozenset([i])
            seen.add(s)
            stack.append(s)
    while stack and len(seen) < max_candidates:
        s = stack.pop()
        frontier = set().union(*(adj[i] for i in s)) - s
        tot = sum(fsize[i] for i in s)
        for j in frontier:
            if tot + fsize[j] > max_atoms:
                continue
            t = s | {j}
            if t not in seen:
                seen.add(t)
                stack.append(t)

    def rgroup_sizes(scaf_atoms):
        """Heavy-atom size of each connected piece left after removing the scaffold."""
        visited, sizes = set(scaf_atoms), []
        for start in range(n):
            if start in visited:
                continue
            visited.add(start)
            st, cnt = [start], 0
            while st:
                a = st.pop()
                cnt += 1
                for b in nbrs[a]:
                    if b not in visited:
                        visited.add(b)
                        st.append(b)
            sizes.append(cnt)
        return sizes

    def penalty(sizes):
        p = 0
        for s in sizes:
            if max_rgroup is not None and s > max_rgroup:
                p += s - max_rgroup
            elif min_rgroup is not None and s < min_rgroup:
                p += min_rgroup - s
        return p

    # --- score candidates ---
    chosen, best_key, best_sizes = None, None, None
    for s in seen:
        atoms = frozenset().union(*(frags[i] for i in s))
        if len(atoms) / n < min_scaffold_frac or not (atoms & ring_atoms):
            continue
        sizes = rgroup_sizes(atoms)
        if not sizes:
            continue
        overlap = max(len(atoms & m) / len(atoms | m) for m in murcko_sets)
        key = (penalty(sizes), -len(atoms), -overlap)
        if best_key is None or key < best_key:
            chosen, best_key, best_sizes = atoms, key, sizes

    if chosen is None:
        # Fallback (e.g. no BRICS bonds): plain Murcko scaffold
        chosen = murcko_sets[0]
        best_sizes = rgroup_sizes(chosen)
        message = "No valid BRICS-fragment scaffold; fell back to Murcko scaffold"
    else:
        message = (f"Scaffold = {len(chosen)}/{n} atoms ({len(chosen) / n:.0%}); "
                   f"R-group sizes: {sorted(best_sizes, reverse=True)}")
        if best_key[0] > 0:
            message += f"\n(no scaffold met R-group limits {min_rgroup}-{max_rgroup}; best compromise)"

    # --- Atom/bond sets -> SMILES (no dummy atoms) ---
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in chosen and b.GetEndAtomIdx() in chosen]

    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    result = Chem.MolFragmentToSmiles(kek, atomsToUse=sorted(chosen),
                                      bondsToUse=bonds, kekuleSmiles=True,
                                      canonical=True)

    # --- Guarantee: result parses and is a substructure of the input ---
    result_mol = Chem.MolFromSmiles(result)
    if result_mol is None or not mol.HasSubstructMatch(result_mol):
        raise ValueError(f"Scaffold '{result}' is not a valid substructure of {smiles}")
    result = Chem.MolToSmiles(result_mol)   # canonical, re-aromatized form

    # --- Display only (no saving) ---
    if show:
        img = Draw.MolToImage(mol, size=size,
                              highlightAtoms=sorted(chosen),
                              highlightBonds=bonds,
                              highlightColor=(0.4, 0.8, 1.0))
        plt.figure(figsize=(7, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{message}\nScaffold: {result}", fontsize=8, wrap=True)
        plt.show()

    return result



def run_rgrp(
        whole_smiles,
        scaffold_smiles,
        no_rgrp_mols,
        output_dir,
        fragment_desc,
        sim_mode = "ecfp4"
    ):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    whole_mol = Chem.MolFromSmiles(whole_smiles)
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)

    result = find_frag_atts(whole_mol, scaffold_mol, scaffold_smiles)

    frag_repl = result["frag_repl"]
    sites = len(result["selected_attachments"])

    cutoff = math.ceil(round(no_rgrp_mols ** (1 / sites), 10))  # fragments per site
    while cutoff ** sites < no_rgrp_mols:                       # guard against float error
        cutoff += 1

    print(f"\nTotal R-groups: {sites}, Cutoff per site: {cutoff}, Total mols: {cutoff ** sites}")

    similar_dfs = process_and_save_similar_fragments(frag_repl, fragment_desc, cutoff, sim_mode, output_dir)
    index_dict = {}
    for k, v in similar_dfs.items():
        index_dict[k] = similar_dfs[k].index.to_list()

    # Generate all combinations
    combinations = list(itertools.product(*index_dict.values()))
    # Generate final molecules
    final_mol_df = generate_final_molecules(combinations, Chem.MolFromSmiles(scaffold_smiles), similar_dfs,att_to_c_conversion)
    final_mol_df.to_csv(f"{output_dir}/rgrp_all.csv", index=False)
    print("\n Total Generated mols: ", len(final_mol_df))
    print(f"Output saved to: {output_dir}/rgrp_all.csv")

    return {
        "output_dir": str(output_dir),
        "selected_attachments": result['selected_attachments'],
        "Total Generated mols": len(final_mol_df),
        "Output_path": f"{output_dir}/rgrp_all.csv",
        }



