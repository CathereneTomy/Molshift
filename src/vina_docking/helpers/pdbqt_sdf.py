import os
import re
import subprocess
import multiprocessing as mp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# ══════════════════════════════════════════════════════════════════════════════
# 1. Convert a folder of PDBQTs to a folder of PDBs (via OpenBabel, parallel)
# ══════════════════════════════════════════════════════════════════════════════

def _convert_one(args: tuple) -> dict:
    """Worker: converts ONE (possibly multi-pose) PDBQT into a single
    multi-MODEL PDB. Top-level function so it's picklable on Windows."""
    in_path, out_path, obabel_exe = args
    try:
        result = subprocess.run(
            [obabel_exe, in_path, "-O", out_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return {"file": os.path.basename(in_path), "status": "failed",
                     "error": result.stderr.strip()}
        return {"file": os.path.basename(in_path), "status": "ok"}
    except Exception as exc:
        return {"file": os.path.basename(in_path), "status": "failed", "error": str(exc)}


def pdbqt_folder_to_pdb(input_dir: str, output_dir: str, obabel_exe: str, n_workers: int = None):
    """
    Convert every *.pdbqt in input_dir to a *.pdb in output_dir.
    Multi-pose PDBQTs (multiple MODEL/ENDMDL blocks) become a single
    multi-MODEL PDB file with all poses preserved.
    """
    os.makedirs(output_dir, exist_ok=True)

    job_args = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".pdbqt"):
            continue
        stem = os.path.splitext(fname)[0]
        in_path = os.path.abspath(os.path.join(input_dir, fname))
        out_path = os.path.abspath(os.path.join(output_dir, stem + ".pdb"))
        job_args.append((in_path, out_path, obabel_exe))

    if not job_args:
        print("No PDBQT files found.")
        return

    n_workers = n_workers or min(len(job_args), os.cpu_count() or 1)
    print(f"Converting {len(job_args)} file(s) using {n_workers} worker(s)...")

    if n_workers == 1:
        outcomes = [_convert_one(a) for a in job_args]
    else:
        with mp.Pool(processes=n_workers) as pool:
            outcomes = pool.map(_convert_one, job_args)

    n_ok = n_failed = 0
    for o in outcomes:
        if o["status"] == "ok":
            n_ok += 1
        else:
            n_failed += 1
            print(f"  FAILED: {o['file']} -> {o['error']}")

    print(f"PDBQT -> PDB done: {n_ok} converted, {n_failed} failed.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Match pose files (by index) back to their SMILES in the CSV
# ══════════════════════════════════════════════════════════════════════════════

def load_index_to_smiles(csv_file: str, smiles_column: str = "smiles",
                          index_column: str = "ligand_index") -> dict:
    df = pd.read_csv(csv_file)
    for col in (smiles_column, index_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_file}")
    return {
        int(row[index_column]): str(row[smiles_column]).strip()
        for _, row in df.iterrows()
    }


def ligand_index_from_filename(fname: str) -> int:
    """'0_out.pdb' -> 0, '12_out.pdb' -> 12"""
    m = re.match(r"^(\d+)_out", os.path.splitext(fname)[0])
    if not m:
        raise ValueError(f"Could not extract ligand index from filename: {fname}")
    return int(m.group(1))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Load the docked poses purely as 3D coordinate targets
# ══════════════════════════════════════════════════════════════════════════════

def load_docked_poses(pdb_path: str):
    """
    Reads a multi-MODEL PDB. RDKit stacks each MODEL as an additional
    conformer on ONE Mol (same atoms/order across models). proximityBonding
    stays on so AssignBondOrdersFromTemplate has a connectivity graph to
    match against; the guessed bonds are just scaffolding for that match -
    the final bond orders come from the SMILES template, not from here.
    """
    ref_mol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False,
                                   proximityBonding=True)
    if ref_mol is None:
        raise RuntimeError(f"RDKit could not parse {pdb_path}")
    Chem.SanitizeMol(ref_mol, catchErrors=True)
    return ref_mol


def _single_conformer_copy(multi_conf_mol, conf_id):
    """Isolate ONE conformer into its own Mol copy."""
    m = Chem.Mol(multi_conf_mol)
    for c in list(m.GetConformers()):
        if c.GetId() != conf_id:
            m.RemoveConformer(c.GetId())
    return m

def parse_pdbqt_model_remarks(pdbqt_path: str) -> list[dict]:
    """
    Returns one dict per MODEL (same order as the poses) with the REMARK
    fields: smiles, ligand, index, vina. Missing fields are simply absent.
    """
    models, current = [], None
    with open(pdbqt_path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                current = {}
            elif line.startswith("ENDMDL"):
                if current is not None:
                    models.append(current)
                current = None
            elif line.startswith("REMARK"):
                if current is None:          # single-pose file with no MODEL line
                    current = {}
                text = line[6:].strip()
                if text.startswith("SMILES:"):
                    current["smiles"] = text.split(":", 1)[1].strip()
                elif text.startswith("Ligand:"):
                    current["ligand"] = text.split(":", 1)[1].strip()
                elif text.startswith("Index:"):
                    current["index"] = text.split(":", 1)[1].strip()
                elif text.startswith("VINA RESULT:"):
                    current["vina"] = " ".join(text.split(":", 1)[1].split())
    if current:
        models.append(current)
    return models

# ══════════════════════════════════════════════════════════════════════════════
# 4. For each docked pose: assign the SMILES-derived bond orders directly
#    onto the pose's own (unmodified) coordinates.
# ══════════════════════════════════════════════════════════════════════════════

def assign_bonds_to_poses(smiles: str, pdb_path: str, output_sdf_path: str,
                          pdbqt_path: str = None) -> int:
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    ref_mol = load_docked_poses(pdb_path)
    stem = os.path.splitext(os.path.basename(pdb_path))[0]

    # Per-pose REMARK info from the original PDBQT
    model_remarks = parse_pdbqt_model_remarks(pdbqt_path) if pdbqt_path else []

    writer = Chem.SDWriter(output_sdf_path)
    n_written = 0
    try:
        for pose_idx, ref_conf in enumerate(ref_mol.GetConformers()):
            pose_mol = _single_conformer_copy(ref_mol, ref_conf.GetId())
            pose_mol_noh = Chem.RemoveHs(pose_mol, sanitize=False)

            try:
                fixed_mol = AllChem.AssignBondOrdersFromTemplate(template, pose_mol_noh)
            except (ValueError, RuntimeError) as exc:
                print(f"    pose {pose_idx + 1}: bond-order assignment failed ({exc})")
                continue

            fixed_mol.SetProp("_Name", f"pose{pose_idx + 1}")

            # ---- carry the PDBQT remarks into the SDF ----
            info = model_remarks[pose_idx] if pose_idx < len(model_remarks) else {}
            remark_smiles = info.get("smiles") or smiles      # fall back to CSV SMILES
            remark_ligand = info.get("ligand") or stem        # PDBQT 'Ligand:' can be empty
            remark_lines = [f"SMILES: {remark_smiles}", f"Ligand: {remark_ligand}"]
            if "index" in info:
                remark_lines.append(f"Index: {info['index']}")
            if "vina" in info:
                remark_lines.append(f"VINA RESULT: {info['vina']}")
            fixed_mol.SetProp("REMARK", "\n".join(remark_lines))

            writer.write(fixed_mol)
            n_written += 1
    finally:
        writer.close()
    return n_written

# ══════════════════════════════════════════════════════════════════════════════
# 5. Batch driver: PDBQT folder -> PDB folder -> SDF folder
# ══════════════════════════════════════════════════════════════════════════════

def batch_pdbqt_to_sdf(pdbqt_dir: str, pdb_dir: str, output_dir: str, obabel_exe: str,csv_file: str,
                        smiles_column: str = "smiles",
                        index_column: str = "ligand_index",
                        n_workers: int = None):
    # Step 1: PDBQT -> PDB
    pdbqt_folder_to_pdb(pdbqt_dir, pdb_dir, obabel_exe, n_workers=n_workers)

    # Step 2: PDB -> SDF (bond orders assigned from SMILES)
    os.makedirs(output_dir, exist_ok=True)
    index_to_smiles = load_index_to_smiles(csv_file, smiles_column, index_column)

    pdb_files = [f for f in os.listdir(pdb_dir) if f.lower().endswith(".pdb")]
    if not pdb_files:
        print("No PDB files found.")
        return

    print(f"\nAssigning bond orders for {len(pdb_files)} file(s)...")
    for fname in sorted(pdb_files):
        idx = ligand_index_from_filename(fname)
        smiles = index_to_smiles.get(idx)
        if smiles is None:
            print(f"  SKIPPED (no SMILES match): {fname}")
            continue

        pdb_path = os.path.join(pdb_dir, fname)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, stem + ".sdf")
        pdbqt_path = os.path.join(pdbqt_dir, stem + ".pdbqt")   # <-- new

        try:
            n = assign_bonds_to_poses(smiles, pdb_path, out_path, pdbqt_path=pdbqt_path)
            print(f"  OK ({n} pose(s)): {fname}")
        except Exception as exc:
            print(f"  FAILED: {fname} -> {exc}")


if __name__ == "__main__":
    batch_pdbqt_to_sdf(
        pdbqt_dir=r"Validation\TYK2\diverse_5\vina_batch_output\poses",
        pdb_dir=r"Validation\TYK2\diverse_5\vina_batch_output\pdb_folder",
        output_dir=r"Validation\TYK2\diverse_5\vina_batch_output\sdf_folder",
        csv_file=r"Validation\TYK2\diverse_5\vina_batch_output\vina_results.csv",
        smiles_column="smiles",
        index_column="ligand_index",
    )