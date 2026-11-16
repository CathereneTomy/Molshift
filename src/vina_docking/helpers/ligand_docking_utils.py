"""
Molecular Docking Pipeline (multiprocessing-enabled)
=====================================================
Modular pipeline for virtual screening:
  1. Clustering & shortlisting
  2. Ligand preparation (SMILES -> PDBQT)                       [PARALLEL]
  3. AutoDock Vina batch docking                                [PARALLEL]
  4. Energy-component re-scoring                                [PARALLEL]
  5. PDBQT -> SDF conversion (Open Babel)                       [PARALLEL]
  6. Protein-ligand interaction profiling (ProLIF)               [PARALLEL]
  7. Residue-number remapping (fixed PDB -> original PDB)
  8. Optional shape + interaction similarity scoring vs. a reference pose  [PARALLEL]
  9. Filtering & deduplication

Steps marked [PARALLEL] use multiprocessing.Pool with a shared dynamic
core-allocation function (_concurrency_plan) instead of a fixed/hardcoded
core count, so the available CPU budget is used efficiently regardless
of how many items (ligands/poses/files) are being processed in a given
run. Steps 1, 7, and 9 are NOT parallelized because they are single
vectorized/global operations (or interactive), not per-item loops --
adding multiprocessing there would add overhead with no benefit.

WINDOWS NOTE: multiprocessing on Windows uses the 'spawn' start method,
which re-imports this script in every child process. This means:
  - the `if __name__ == "__main__":` guard at the bottom is MANDATORY,
    not optional -- without it you will get a fork bomb / RuntimeError.
  - every function passed to a Pool must be a top-level module function
    (not a nested closure/lambda), since spawn needs to pickle it.
Both are already handled correctly below.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import ast
import glob
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import warnings
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import prolif as plf
import MDAnalysis as mda
from MDAnalysis.converters.RDKitInferring import MDAnalysisInferrer

from Bio import pairwise2
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, rdShapeHelpers
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import argparse
from vina_docking.helpers.pdbqt_sdf import batch_pdbqt_to_sdf
warnings.filterwarnings("ignore")



W_TANIMOTO    = 0.50   # shape overlap    - higher is better
W_PROTRUDE    = 0.20   # shape protrusion - lower is better (inverted)
W_INTERACTION = 0.50   # interaction overlap - higher is better



# ══════════════════════════════════════════════════════════════════════════════
# 0. SHARED MULTIPROCESSING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _concurrency_plan(n_items: int, total_cores: int = 12, max_cpu_per_job: int = 1) -> tuple[int, int]:
    """
    Dynamic core allocator used by every parallel step in this pipeline.

    Returns (pool_size, cpu_per_job):
      - pool_size    -> how many worker PROCESSES to run concurrently
      - cpu_per_job  -> how many threads EACH job should use internally
                        (only meaningful for tools that accept a --cpu
                        flag, i.e. Vina; for single-threaded subprocess
                        steps this is always 1)

    Logic: give each concurrent job as many cores as is still useful
    (capped at max_cpu_per_job, since e.g. Vina's internal speedup
    flattens out past ~4 threads), then run as many jobs in parallel as
    the remaining budget allows. Never spawns more concurrent workers
    than there are items to process, and never exceeds total_cores.

    This is a FUNCTION, not a lookup table, so it generalizes correctly
    to any n_items without hardcoded rows. Example with total_cores=12,
    max_cpu_per_job=4:
        n_items=1  -> pool_size=1,  cpu_per_job=4   (4 cores used)
        n_items=2  -> pool_size=2,  cpu_per_job=4   (8 cores used)
        n_items=3  -> pool_size=3,  cpu_per_job=4   (12 cores used)
        n_items=4  -> pool_size=4,  cpu_per_job=3   (12 cores used)
        n_items=6  -> pool_size=6,  cpu_per_job=2   (12 cores used)
        n_items=12+-> pool_size=12, cpu_per_job=1   (12 cores used)
    """
    n_items = max(1, n_items)

    cpu_per_job = max(1, min(max_cpu_per_job, total_cores // min(n_items, total_cores)))
    pool_size = max(1, total_cores // cpu_per_job)
    pool_size = min(pool_size, n_items)
    return pool_size, cpu_per_job


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLUSTERING & SHORTLISTING   (NOT parallelized - single vectorized op)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_shortlist(csv_file: str, n_clusters: int, k_selection: int, smiles_column: str) -> pd.DataFrame:
    """
    Read a CSV, compute Morgan fingerprints, cluster with MiniBatchKMeans,
    and return up to *k_selection* representative molecules.
    """
    df = pd.read_csv(csv_file).dropna(subset=[smiles_column]).reset_index(drop=True)

    # Fingerprints
    fps, valid_indices = [], []
    for idx, smi in enumerate(df[smiles_column]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        arr = np.zeros((2048,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048), arr
        )
        fps.append(arr)
        valid_indices.append(idx)

    fps = np.array(fps)
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    print(f"Valid molecules: {len(df_valid)}")

    # Cluster
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    clusters = kmeans.fit_predict(fps)
    df_valid["cluster"] = clusters

    # Pick representatives
    mols_per_cluster = max(1, k_selection // n_clusters)
    selected_rows = []

    for cid in range(n_clusters):
        mask = df_valid["cluster"] == cid
        cluster_df = df_valid[mask]
        if cluster_df.empty:
            continue

        centroid = kmeans.cluster_centers_[cid].reshape(1, -1)
        nearest, _ = pairwise_distances_argmin_min(centroid, fps[mask])
        rep_idx = cluster_df.index[nearest[0]]
        selected_rows.append(rep_idx)

        remaining = cluster_df.drop(rep_idx)
        extra = min(mols_per_cluster - 1, len(remaining))
        if extra > 0:
            selected_rows.extend(remaining.sample(n=extra, random_state=42).index.tolist())

    selected_rows = list(dict.fromkeys(selected_rows))
    selected_df = df_valid.loc[selected_rows]

    # Top-up if still short
    if len(selected_df) < k_selection:
        add_n = min(k_selection - len(selected_df), len(df_valid.drop(selected_rows)))
        extra = df_valid.drop(selected_rows).sample(n=add_n, random_state=42)
        selected_df = pd.concat([selected_df, extra], ignore_index=True)

    return selected_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. LIGAND PREPARATION  (SMILES → PDB → PDBQT)                      [PARALLEL]
# ══════════════════════════════════════════════════════════════════════════════

def _embed_molecule(smiles: str):
    """Return an RDKit mol with a 3-D conformer, or raise on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    mol = Chem.AddHs(mol)

    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise ValueError("3D embedding failed")

    status = AllChem.MMFFOptimizeMolecule(mol)
    if status == -1:
        print("WARNING: MMFF optimisation failed - using unoptimised geometry")

    return mol


def _write_pdb(mol, pdb_path: str, smiles: str, ligand_name: str, idx: int):
    remarks = [
        f"REMARK SMILES: {smiles}",
        f"REMARK Ligand: {ligand_name}",
        f"REMARK Index: {idx}",
    ]
    with open(pdb_path, "w") as fh:
        fh.write("\n".join(remarks) + "\n" + rdmolfiles.MolToPDBBlock(mol))


def _pdb_to_pdbqt(pdb_filename: str, pdbqt_filename: str, mgl_python: str, vina_prepare_lig: str, cwd: str):

    cmd = [
        mgl_python, vina_prepare_lig,
        "-l", pdb_filename, "-o", pdbqt_filename, "-A", "hydrogens",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"prepare_ligand4.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")


def _prepend_remarks_to_pdbqt(pdbqt_path: str, smiles: str, ligand_name: str, idx: int):
    remarks = [
        f"REMARK SMILES: {smiles}",
        f"REMARK Ligand: {ligand_name}",
        f"REMARK Index: {idx}",
    ]
    with open(pdbqt_path, "r") as fh:
        content = fh.read()
    with open(pdbqt_path, "w") as fh:
        fh.write("\n".join(remarks) + "\n" + content)


def _prepare_one_ligand(args: tuple) -> dict:
    """
    Worker (top-level function, required for picklability under Windows
    'spawn'): embed, write PDB, convert to PDBQT for ONE molecule.
    """
    idx, smiles, ligand_name, output_dir,mgl_python,vina_prepare_lig = args
    print(idx, smiles, ligand_name, output_dir)
    pdb_name   = f"{idx}.pdb"
    pdbqt_name = f"{idx}.pdbqt"
    pdb_path   = os.path.normpath(os.path.abspath(os.path.join(output_dir, pdb_name)))
    pdbqt_path = os.path.normpath(os.path.abspath(os.path.join(output_dir, pdbqt_name)))

    try:
        mol = _embed_molecule(smiles)

        _write_pdb(mol, pdb_path, smiles, ligand_name, idx)

        _pdb_to_pdbqt(pdb_name, pdbqt_name, mgl_python, vina_prepare_lig, cwd=output_dir)

        if not os.path.exists(pdbqt_path):
            raise RuntimeError("PDBQT file not created")
        _prepend_remarks_to_pdbqt(pdbqt_path, smiles, ligand_name, idx)
        return {"idx": idx, "status": "ok", "path": pdbqt_path}
    except Exception as exc:
        return {"idx": idx, "status": "failed", "error": str(exc)}


def prepare_ligands_from_csv(
    csv_file: str,
    output_dir: str,
    smiles_column: str,
    total_cores: int,
    mgl_python: str,
    vina_prepare_lig: str,
    top_k: int = None,
) -> None:
    """Convert each SMILES in *csv_file* to a PDBQT file ready for Vina (parallel across molecules)."""


    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    if top_k is not None:
        df = df.head(top_k)
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in CSV.")

    job_args = [
        (idx, str(row[smiles_column]).strip(),
         str(row.get("Ligand", "")).strip() if pd.notna(row.get("Ligand")) else "",
         output_dir,mgl_python,vina_prepare_lig)
        for idx, row in df.iterrows()
    ]

    pool_size, _ = _concurrency_plan(len(job_args), total_cores=total_cores, max_cpu_per_job=1)
    print(f"\n[Ligand prep] {len(job_args)} molecule(s) -> {pool_size} parallel worker(s)\n")

    if pool_size == 1:
        outcomes = [_prepare_one_ligand(a) for a in job_args]
    else:
        with mp.Pool(processes=pool_size) as pool:
            outcomes = pool.map(_prepare_one_ligand, job_args)

    success = sum(1 for o in outcomes if o["status"] == "ok")
    fail = len(outcomes) - success
    for o in outcomes:
        if o["status"] == "ok":
            print(f"  [{o['idx']}] Saved: {o['path']}")
        else:
            print(f"  [{o['idx']}] FAILED - {o['error']}")

    print(f"\nSummary: {success} succeeded, {fail} failed -> {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. AUTODOCK VINA – 12 CORES FOR ONE MOL  [SEQUENTIAL]
# ══════════════════════════════════════════════════════════════════════════════

def dock_single_ligand(
    ligand_path,
    config_file,
    poses_dir,
    logs_dir,
    smiles_lookup,
    vina_executable,
    num_cores
):
    """
    Dock a single ligand using Vina with multiple CPU cores.
    """
    ligand_name  = os.path.basename(ligand_path)
    ligand_index = os.path.splitext(ligand_name)[0]

    print(f"\nDocking ligand: {ligand_name}")

    out_pose = os.path.join(poses_dir, f"{ligand_index}_out.pdbqt")
    log_file = os.path.join(logs_dir,  f"{ligand_index}.txt")

    command = [
        vina_executable,
        "--config",  config_file,
        "--ligand",  ligand_path,
        "--out",     out_pose,
        "--cpu",     str(num_cores)
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        vina_output = result.stdout

        with open(log_file, "w") as f:
            f.write(vina_output)

        mode_pattern = re.compile(
            r"^\s*(\d+)\s+(-?\d+\.\d+)\s+"
            r"(\d+\.\d+)\s+(\d+\.\d+)",
            re.MULTILINE
        )

        matches = mode_pattern.findall(vina_output)

        if matches:
            return [
                {
                    "ligand_index":      ligand_index,
                    "smiles":            smiles_lookup.get(str(ligand_index), None),
                    "mode":              int(match[0]),
                    "affinity_kcal_mol": float(match[1]),
                    "rmsd_lb":           float(match[2]),
                    "rmsd_ub":           float(match[3]),
                    "pose_file":         out_pose,
                    "log_file":          log_file
                }
                for match in matches
            ]
        else:
            return [{
                "ligand_index":      ligand_index,
                "smiles":            smiles_lookup.get(str(ligand_index), None),
                "mode":              None,
                "affinity_kcal_mol": None,
                "rmsd_lb":           None,
                "rmsd_ub":           None,
                "pose_file":         out_pose,
                "log_file":          log_file
            }]

    except subprocess.CalledProcessError as e:
        print(f"Docking failed for {ligand_name}")

        with open(log_file, "w") as f:
            f.write(e.stdout or "")
            f.write("\n\nSTDERR:\n")
            f.write(e.stderr or "")

        return [{
            "ligand_index":      ligand_index,
            "smiles":            smiles_lookup.get(ligand_index, None),
            "mode":              "FAILED",
            "affinity_kcal_mol": None,
            "rmsd_lb":           None,
            "rmsd_ub":           None,
            "pose_file":         None,
            "log_file":          log_file
        }]


def run_vina_batch(
    ligand_dir,
    config_file,
    vina_exe_path,
    total_cores,
    output_dir="vina_results",
    csv_with_smiles=None,
    smiles_column="smiles",

):
    """
    Run AutoDock Vina sequentially for all ligand PDBQT files,
    assigning all cores to each individual docking run.

    Parameters
    ----------
    ligand_dir : str
        Directory containing ligand .pdbqt files
    config_file : str
        Path to Vina config file
    output_dir : str
        Directory to save docking outputs
    csv_with_smiles : str or None
        Optional CSV containing SMILES
    smiles_column : str
        Column name for SMILES
    vina_executable : str
        Path to vina executable if not in PATH
    num_cores : int
        Number of CPU cores Vina uses per docking run
    """

    vina_executable=vina_exe_path
    num_cores=total_cores

    # ---------------------------------------
    # Create output directories
    # ---------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    poses_dir = os.path.join(output_dir, "poses")
    logs_dir  = os.path.join(output_dir, "logs")

    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(logs_dir,  exist_ok=True)

    # ---------------------------------------
    # Load SMILES if available
    # ---------------------------------------
    smiles_lookup = {}

    if csv_with_smiles:
        df_smiles = pd.read_csv(csv_with_smiles)

        if smiles_column not in df_smiles.columns:
            raise ValueError(f"Column '{smiles_column}' not found")

        for idx, row in df_smiles.iterrows():
            smiles_lookup[str(idx)] = str(row[smiles_column])

    print(f"SMILES lookup table loaded with {len(smiles_lookup)} entries.")
    print(smiles_lookup)

    # ---------------------------------------
    # Find ligands
    # ---------------------------------------
    ligand_files = sorted(
        glob.glob(os.path.join(ligand_dir, "*.pdbqt"))
    )

    if not ligand_files:
        print("No ligand PDBQT files found.")
        return

    print(f"Found {len(ligand_files)} ligands. "
          f"Docking each with {num_cores} cores...\n")

    # ---------------------------------------
    # Run docking sequentially
    # ---------------------------------------
    results = []

    for ligand_path in ligand_files:
        ligand_results = dock_single_ligand(
            ligand_path=ligand_path,
            config_file=config_file,
            poses_dir=poses_dir,
            logs_dir=logs_dir,
            smiles_lookup=smiles_lookup,
            vina_executable=vina_executable,
            num_cores=num_cores
        )
        results.extend(ligand_results)

    # ---------------------------------------
    # Save results table
    # ---------------------------------------
    results_df = pd.DataFrame(results)

    csv_output = os.path.join(output_dir, "vina_results.csv")
    results_df.to_csv(csv_output, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENERGY-COMPONENT RE-SCORING                                     [PARALLEL]
# ══════════════════════════════════════════════════════════════════════════════

def _split_vina_poses(pdbqt_file: str) -> list[str]:
    """Return a list of pose text blocks from a multi-model PDBQT."""
    poses, current = [], []
    inside = False
    with open(pdbqt_file) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                inside, current = True, []
            elif line.startswith("ENDMDL"):
                poses.append("".join(current))
                inside = False
            elif inside:
                current.append(line)
    return poses


def _score_single_pose(vina_exe: str, receptor_pdbqt: str, pose_path: str) -> dict:
    cmd = [vina_exe, "--receptor", receptor_pdbqt, "--ligand", pose_path, "--score_only"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Vina scoring failed:\n{res.stdout}\n{res.stderr}")

    out = res.stdout + "\n" + res.stderr

    def _grab(pattern):
        m = re.search(pattern, out, re.IGNORECASE)
        return float(m.group(1)) if m else None

    return {
        "affinity_kcal_mol": _grab(r"Affinity:\s*([-0-9.]+)"),
        "gauss1":            _grab(r"gauss\s*1\s*:\s*([-0-9.]+)"),
        "gauss2":            _grab(r"gauss\s*2\s*:\s*([-0-9.]+)"),
        "repulsion":         _grab(r"repulsion\s*:\s*([-0-9.]+)"),
        "hydrophobic":       _grab(r"hydrophobic\s*:\s*([-0-9.]+)"),
        "hydrogen":          _grab(r"Hydrogen\s*:\s*([-0-9.]+)"),
    }


def _score_one_pose(args: tuple) -> dict:
    """Worker (top-level function): scores ONE pose with Vina --score_only."""
    vina_exe, receptor_pdbqt, stem, pose_num, pose_text = args
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False, mode="w") as tmp:
        tmp.write(pose_text)
        tmp_path = tmp.name
    try:
        scores = _score_single_pose(vina_exe, receptor_pdbqt, tmp_path)
        return {"Name": stem, "Pose": pose_num, **scores}
    except Exception as exc:
        return {"Name": stem, "Pose": pose_num, "affinity_kcal_mol": None,
                "gauss1": None, "gauss2": None, "repulsion": None,
                "hydrophobic": None, "hydrogen": None, "error": str(exc)}
    finally:
        os.remove(tmp_path)


def batch_score_vina_outputs(
    poses_dir: str,
    receptor_pdbqt: str,
    output_csv: str,
    vina_exe_path: str,
    total_cores: int,
) -> pd.DataFrame:
    """Re-score every pose in *poses_dir* with Vina --score_only (parallel across poses)."""

    vina_executable = vina_exe_path

    if not os.path.exists(vina_executable):
        raise FileNotFoundError(f"Vina not found: {vina_executable}")

    job_args = []
    for pdbqt_path in glob.glob(os.path.join(poses_dir, "*.pdbqt")):
        stem = os.path.splitext(os.path.basename(pdbqt_path))[0]
        poses = _split_vina_poses(pdbqt_path)
        print(f"{os.path.basename(pdbqt_path)}: {len(poses)} pose(s) queued")
        for pose_num, pose_text in enumerate(poses, 1):
            job_args.append((vina_executable, receptor_pdbqt, stem, pose_num, pose_text))

    pool_size, _ = _concurrency_plan(len(job_args), total_cores=total_cores, max_cpu_per_job=1)
    print(f"\n[Re-scoring] {len(job_args)} pose(s) -> {pool_size} parallel worker(s)\n")

    if pool_size == 1:
        rows = [_score_one_pose(a) for a in job_args]
    else:
        with mp.Pool(processes=pool_size) as pool:
            rows = pool.map(_score_one_pose, job_args)

    for r in rows:
        if r.get("error"):
            print(f"  {r['Name']} pose {r['Pose']} FAILED - {r['error']}")
        else:
            print(f"  {r['Name']} pose {r['Pose']}: {r['affinity_kcal_mol']}")

    df = pd.DataFrame(rows).drop(columns=["error"], errors="ignore")
    df.to_csv(output_csv, index=False)
    print(f"\nScores saved -> {output_csv}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. PROTEIN–LIGAND INTERACTION PROFILING  (ProLIF)                  [PARALLEL]
# ══════════════════════════════════════════════════════════════════════════════

_PROLIF_PARAMS = {
    "HBDonor":    {"distance": 4.0},
    "HBAcceptor": {"distance": 4.0},
    "Hydrophobic": {"distance": 5.0},
    "CationPi":   {"distance": 5.0},
    "PiCation":   {"distance": 5.0},
}


def _extract_interaction_summary(name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame_idx, row in df.iterrows():
        interactions = [
            (residue, bondtype)
            for (_, residue, bondtype), value in row.items()
            if value is True or value == "True"
        ]
        rows.append({"Name": name, "Pose": frame_idx, "Interactions": interactions})
    return pd.DataFrame(rows)


def _parse_sdf_remarks(mol) -> tuple[str, str]:
    remark = mol.GetProp("REMARK") if mol.HasProp("REMARK") else ""
    smiles_match = re.search(r"SMILES:\s*(.+)", remark)
    ligand_match = re.search(r"Ligand:\s*(.+)", remark)
    smiles = smiles_match.group(1).strip() if smiles_match else ""
    ligand = ligand_match.group(1).strip() if ligand_match else ""
    return smiles, ligand


def _profile_one_ligand(args: tuple):
    """
    Worker (top-level function): rebuilds the protein Molecule from disk
    (once per worker process call -- cheap relative to fingerprinting)
    and fingerprints ONE ligand's SDF file against it.

    n_jobs=1 is passed to ProLIF's own run_from_iterable deliberately:
    parallelism is already happening at THIS (outer, per-file) level via
    our own Pool. Letting ProLIF ALSO spawn its down internal workers here
    would oversubscribe cores (nested multiprocessing).
    """
    sdf_path, protein_path, name = args
    try:
        u = mda.Universe(protein_path)
        
        # Sanitize=False bypasses the valence check entirely
        inferrer = MDAnalysisInferrer(sanitize=False)
        protein_mol = plf.Molecule.from_mda(u, inferrer=inferrer)
        fp = plf.Fingerprint(parameters=_PROLIF_PARAMS)

        rdkit_mols = [m for m in Chem.SDMolSupplier(sdf_path, removeHs=False) if m]
        smiles, ligand_name = _parse_sdf_remarks(rdkit_mols[0]) if rdkit_mols else ("", "")

        fp.run_from_iterable(plf.sdf_supplier(sdf_path), protein_mol, n_jobs=1, progress=False)

        summary = _extract_interaction_summary(name, fp.to_dataframe())
        summary["smiles"] = smiles
        summary["Ligand"] = ligand_name
        return summary
    except Exception as exc:
        print(f"Error processing {name}: {exc}")
        return None


def docked_mols_interaction(sdf_dir: str, protein_path: str, total_cores: int,) -> pd.DataFrame:
    """
    Run ProLIF fingerprinting on all SDF poses against the protein,
    parallelized ACROSS ligand files using our own pool (see note in
    _profile_one_ligand about why ProLIF's internal n_jobs is left at 1).
    """

    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith(".sdf")]
    if not sdf_files:
        print("No SDF files found to profile.")
        return pd.DataFrame()

    print(total_cores)
    pool_size, _ = _concurrency_plan(len(sdf_files), total_cores=total_cores, max_cpu_per_job=1)
    print(f"\n[ProLIF profiling] {len(sdf_files)} ligand file(s) -> {pool_size} parallel worker(s)\n")

    job_args = [(os.path.join(sdf_dir, f), protein_path, os.path.splitext(f)[0]) for f in sdf_files]

    if pool_size == 1:
        raw = [_profile_one_ligand(a) for a in job_args]
    else:
        with mp.Pool(processes=pool_size) as pool:
            raw = pool.map(_profile_one_ligand, job_args)

    all_results = [r for r in raw if r is not None]
    if not all_results:
        print("WARNING: no ligand files were successfully profiled.")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. RESIDUE-NUMBER REMAPPING  (fixed PDB ↔ original PDB)  — NOT parallelized
# ══════════════════════════════════════════════════════════════════════════════

def _extract_residues(pdb_file: str) -> tuple[str, list[dict]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    seq, residues = [], []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                try:
                    aa = seq1(res.resname)
                except Exception:
                    continue
                seq.append(aa)
                residues.append({"chain": chain.id, "resid": res.id[1], "resname": res.resname})
    return "".join(seq), residues


def build_reverse_mapping(original_pdb: str, fixed_pdb: str) -> dict:
    """Return a dict mapping (resname, resid, chain) in *fixed_pdb* -> same tuple in *original_pdb*."""
    seq_orig, res_orig = _extract_residues(original_pdb)
    seq_fix,  res_fix  = _extract_residues(fixed_pdb)
    alignment = pairwise2.align.globalxx(seq_orig, seq_fix)[0]

    mapping = {}
    i_o = i_f = 0
    for a, b in zip(alignment.seqA, alignment.seqB):
        orig = res_orig[i_o] if a != "-" else None
        fix  = res_fix[i_f]  if b != "-" else None
        if a != "-": i_o += 1
        if b != "-": i_f += 1
        if orig and fix:
            mapping[(fix["resname"], fix["resid"], fix["chain"])] = (
                orig["resname"], orig["resid"], orig["chain"]
            )
    return mapping


def convert_interactions(interactions_obj, reverse_mapping: dict):
    """Translate residue numbers in an interaction list using *reverse_mapping*."""
    if isinstance(interactions_obj, list):
        interactions = interactions_obj
    elif isinstance(interactions_obj, str):
        interactions = ast.literal_eval(interactions_obj)
    else:
        return interactions_obj

    converted = []
    for residue_str, itype in interactions:
        try:
            left, chain = residue_str.split(".")
            fix_key = (left[:3], int(left[3:]), chain)
            if fix_key in reverse_mapping:
                rn, ri, rc = reverse_mapping[fix_key]
                residue_str = f"{rn}{ri}.{rc}"
        except Exception as exc:
            print(f"Residue conversion failed for {residue_str}: {exc}")
        converted.append((residue_str, itype))
    return converted


# ══════════════════════════════════════════════════════════════════════════════
# 8. SHAPE + INTERACTION SIMILARITY  (reference-pose scoring)        [PARALLEL]
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_mol(mol):
    mol = Chem.RemoveAllHs(mol, sanitize=False)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def _load_sdf_poses(sdf_path: str) -> list[tuple[int, Chem.Mol]]:
    return [(i + 1, m) for i, m in enumerate(Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)) if m]


def _compute_shape(ref_mol, pose_mol) -> tuple:
    try:
        td = rdShapeHelpers.ShapeTanimotoDist(_prepare_mol(ref_mol), _prepare_mol(pose_mol))
        pd_ = rdShapeHelpers.ShapeProtrudeDist(_prepare_mol(ref_mol), _prepare_mol(pose_mol))
        return round(1.0 - td, 4), round(pd_, 4)
    except Exception as exc:
        print(f"  Shape failed: {exc}")
        return None, None


def _parse_interaction_residues(interaction_str) -> set:
    if not interaction_str or pd.isna(interaction_str):
        return set()
    try:
        return {item[0] for item in ast.literal_eval(str(interaction_str)) if isinstance(item, (list, tuple))}
    except Exception:
        return set()


def _interaction_overlap(ref_res: set, pose_res: set) -> float:
    if not ref_res:
        return 0.0
    return round(len(ref_res & pose_res) / len(ref_res), 4)


def _combined_score(tanimoto, protrude, int_pct):
    if any(v is None for v in (tanimoto, protrude, int_pct)):
        return None
    return round(
        W_TANIMOTO    * tanimoto
        + W_PROTRUDE  * (1.0 - protrude)
        + W_INTERACTION * int_pct,
        4,
    )


def _score_one_pose_vs_ref(args: tuple) -> dict:
    """
    Worker (top-level function): scores ONE pose against the reference.

    RDKit Mol objects are passed as MolBlock strings (plain text) rather
    than Mol objects directly, then reconstructed inside the worker --
    this sidesteps inconsistent pickling behaviour for RDKit Mol objects
    across process boundaries on some RDKit builds.
    """
    ref_mol_block, pose_mol_block, sdf_name, pose_num, ref_residues_list, pose_residues_str = args
    ref_mol = Chem.MolFromMolBlock(ref_mol_block, removeHs=False)
    pose_mol = Chem.MolFromMolBlock(pose_mol_block, removeHs=False)
    ref_residues = set(ref_residues_list)
    pose_residues = _parse_interaction_residues(pose_residues_str)

    tanimoto, protrude = _compute_shape(ref_mol, pose_mol)
    int_ov = _interaction_overlap(ref_residues, pose_residues)
    score = _combined_score(tanimoto, protrude, int_ov)

    return {"sdf_name": sdf_name, "pose_num": pose_num,
            "shape_tanimoto": tanimoto, "shape_protrude": protrude,
            "interaction_overlap": int_ov, "combined_score": score}


def score_vs_reference(
    df: pd.DataFrame,
    sdf_dir: str,
    ref_sdf: str,
    ref_csv: str,
    total_cores: int,
) -> pd.DataFrame:
    """
    Add shape_tanimoto, shape_protrude, interaction_overlap and combined_score
    columns to *df* by comparing each pose against the reference (parallel
    across poses).
    """
    
    ref_mol = next((m for m in Chem.SDMolSupplier(ref_sdf, sanitize=True, removeHs=False) if m), None)
    if ref_mol is None:
        raise ValueError(f"Could not load reference SDF: {ref_sdf}")
    print(f"Reference SDF loaded: {ref_sdf}")
    ref_mol_block = Chem.MolToMolBlock(ref_mol)

    ref_df = pd.read_csv(ref_csv, index_col=0)
    ref_residues: set = set()
    for val in ref_df["Interactions"]:
        ref_residues |= _parse_interaction_residues(val)
    ref_residues_list = list(ref_residues)
    print(f"Reference residues ({len(ref_residues)}): {sorted(ref_residues)}\n")

    for col in ("shape_tanimoto", "shape_protrude", "interaction_overlap", "combined_score"):
        df[col] = np.nan

    job_args = []
    job_locations = []  # parallel list of boolean row masks, to write results back after scoring
    for sdf_path in glob.glob(os.path.join(sdf_dir, "*.sdf")):
        sdf_name = os.path.splitext(os.path.basename(sdf_path))[0]
        mask = df["Name"] == sdf_name
        if not mask.any():
            print(f"  No CSV rows for '{sdf_name}' - skipping")
            continue

        poses = _load_sdf_poses(sdf_path)
        print(f"  [{sdf_name}]  {len(poses)} pose(s) queued")

        for pose_num, pose_mol in poses:
            row_mask = mask & (df["Pose"] == pose_num)
            if not row_mask.any():
                continue
            pose_residues_str = df.loc[row_mask, "Interactions"].values[0]
            job_args.append((ref_mol_block, Chem.MolToMolBlock(pose_mol), sdf_name, pose_num,
                              ref_residues_list, pose_residues_str))
            job_locations.append(row_mask)

    pool_size, _ = _concurrency_plan(len(job_args), total_cores=total_cores, max_cpu_per_job=1)
    print(f"\n[Reference scoring] {len(job_args)} pose(s) -> {pool_size} parallel worker(s)\n")

    if pool_size == 1:
        results = [_score_one_pose_vs_ref(a) for a in job_args]
    else:
        with mp.Pool(processes=pool_size) as pool:
            results = pool.map(_score_one_pose_vs_ref, job_args)

    for row_mask, r in zip(job_locations, results):
        df.loc[row_mask, "shape_tanimoto"]     = r["shape_tanimoto"]
        df.loc[row_mask, "shape_protrude"]      = r["shape_protrude"]
        df.loc[row_mask, "interaction_overlap"] = r["interaction_overlap"]
        df.loc[row_mask, "combined_score"]      = r["combined_score"]
        print(f"    [{r['sdf_name']}] Pose {r['pose_num']:>2d} | "
              f"Tanimoto={r['shape_tanimoto']}  Protrude={r['shape_protrude']}"
              f"  IntOverlap={r['interaction_overlap']:.2%}  Score={r['combined_score']}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 9. FILTERING & DEDUPLICATION   — NOT parallelized (interactive + vectorized)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_residue(residue: str) -> str:
    return residue.upper().split(".")[0]


def _interaction_match_info(interactions_obj, user_interactions: list) -> tuple:
    try:
        if isinstance(interactions_obj, str):
            interactions_obj = ast.literal_eval(interactions_obj)
        residues = {r.upper() for r, _ in interactions_obj} | {_normalize_residue(r) for r, _ in interactions_obj}
    except Exception:
        residues = set()

    matched, missing = [], []
    for ui in user_interactions:
        ui_norm = _normalize_residue(ui)
        found = any(r == ui.upper() or _normalize_residue(r) == ui_norm for r in residues)
        (matched if found else missing).append(ui)
    return len(matched), matched, missing


def filter_and_deduplicate(final_df: pd.DataFrame) -> pd.DataFrame:
    """Interactive CLI filter: by interaction residues, mode, and affinity threshold."""
    print(f"\nInitial rows: {len(final_df)}")

    interaction_input = input(
        "\nEnter interactions (comma-separated, e.g. THR334,TYR370.R)\n"
        "Press Enter to skip interaction filtering:\n> "
    ).strip()

    print("\nFiltering modes:\n  1 -> ALL interactions required\n  2 -> MOST (at least one)")
    mode = input("Choose mode (1 or 2): ").strip()
    if mode not in ("1", "2"):
        mode = "2"

    affinity_threshold = float(input("\nAffinity threshold (keep affinity_kcal_mol <= value):\n> ").strip())

    user_interactions = [x.strip().upper() for x in interaction_input.split(",") if x.strip()] if interaction_input else []
    print(f"\nUser interactions: {user_interactions}")

    df = final_df.copy()

    if not user_interactions:
        df["match_count"] = df["matched_residues"] = df["missing_residues"] = 0
        df["satisfaction_percent"] = 0.0
        df = df.sort_values("affinity_kcal_mol")
    else:
        info = df["Mapped_Interactions"].apply(lambda x: _interaction_match_info(x, user_interactions))
        df["match_count"]         = info.apply(lambda x: x[0])
        df["matched_residues"]    = info.apply(lambda x: x[1])
        df["missing_residues"]    = info.apply(lambda x: x[2])
        df["satisfaction_percent"] = df["match_count"] / len(user_interactions) * 100

        if mode == "1":
            df = df[df["match_count"] == len(user_interactions)]
            df = df.sort_values("affinity_kcal_mol")
        else:
            df = df[df["match_count"] > 0]
            df = df.sort_values(by=["match_count", "affinity_kcal_mol"], ascending=[False, True])
            # Print match summary
            for _, row in df.iterrows():
                print(f"\n  Ligand: {row.get('Ligand','')} | SMILES: {row['smiles']}")
                print(f"  Matched: {row['match_count']}/{len(user_interactions)} ({row['satisfaction_percent']:.1f}%)")
                print(f"  Matched residues : {row['matched_residues']}")
                print(f"  Missing residues : {row['missing_residues']}")

    before = len(df)
    df = df.drop_duplicates(subset="smiles", keep="first")
    print(f"\nRows after deduplication: {len(df)} (removed {before - len(df)})")

    before = len(df)
    df = df[df["affinity_kcal_mol"] <= affinity_threshold]
    print(f"Rows after affinity threshold ({affinity_threshold}): {len(df)} (removed {before - len(df)})")

    for col in ("matched_residues", "missing_residues"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x))

    print(f"\nFinal rows: {len(df)}")
    return df.reset_index(drop=True)



