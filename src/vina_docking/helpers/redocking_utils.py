import os
import sys
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
import requests
import urllib.request, json, csv, os

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from Bio import pairwise2
from Bio.SeqUtils import seq1

import os
import subprocess
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

import re
import warnings
import pandas as pd
import prolif as plf
import MDAnalysis as mda
from MDAnalysis.converters.RDKitInferring import MDAnalysisInferrer
import ast
import glob
import tempfile
from rdkit.Chem import rdMolAlign, AllChem, rdFMCS, rdShapeHelpers




import subprocess

warnings.filterwarnings('ignore')

VINA_DEFAULTS = """
exhaustiveness = 16
num_modes      = 10
energy_range   = 4

"""

# ── Water / ion residue names to strip ────────────────────────────────────────
WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "D2O", "SOL", "TIP", "TIP3", "TIP4"}

COMMON_IONS = {
    "MG","ZN","CA","NA","CL","K","FE","MN","CU","CO","NI","HG",
    "CD","PB","SR","BA","CS","RB","LI","BR","IOD","SO4","PO4",
    "NO3","ACT","GOL","EDO","PEG","MPD","BME","DTT"
}

def is_water(residue):
    return residue.get_resname().strip() in WATER_NAMES

def is_ion(residue):
    return residue.get_resname().strip() in COMMON_IONS

def get_residue_coords(residue):
    """Return Nx3 array of atom coordinates for a residue."""
    return np.array([atom.get_vector().get_array() for atom in residue.get_atoms()])

def compute_hetatm_box(coords, padding=2.0):
    """Given Nx3 coords, return (center_xyz, size_xyz) with small padding."""
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)

    # Box center
    center = (mn + mx) / 2.0

    # Ligand span + padding on both sides
    size = (mx - mn) + padding * 2

    return center, size

def format_box(center, size):
    return (
        f"center_x = {center[0]:.3f}\n"
        f"center_y = {center[1]:.3f}\n"
        f"center_z = {center[2]:.3f}\n"
        f"size_x   = {size[0]:.3f}\n"
        f"size_y   = {size[1]:.3f}\n"
        f"size_z   = {size[2]:.3f}"
    )

class ChainWaterIonFilter(Select):
    def __init__(self, keep_chains):
        self.keep_chains = keep_chains

    def accept_chain(self, chain):
        return chain.get_id() in self.keep_chains

    def accept_residue(self, residue):
        if is_water(residue):
            return False
        if is_ion(residue):
            return False
        return True

def get_smiles_from_residue(res):
    """Fetch curated SMILES from RCSB's Chemical Component Dictionary,
    keyed on the residue's 3-letter code (e.g. 4WI, ATP)."""
    resname = res.get_resname().strip().upper()
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{resname}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            descriptors = data.get("rcsb_chem_comp_descriptor", {})
            smiles = (
                descriptors.get("SMILES_stereo")
                or descriptors.get("SMILES")
                or "NOT_FOUND"
            )
        else:
            smiles = "NOT_FOUND"
    except Exception as e:
        print(f"  Error retrieving SMILES for {resname}: {e}")
        smiles = "NOT_FOUND"

    # mol kept as None — the SDF-writing loop below still uses its own
    # RDKit parse of the residue's 3-D coordinates, unaffected by this change
    return smiles, None

def residue_to_pdb_block(res):
    """Serialize a single HETATM residue to a PDB-format string."""
    lines = ["REMARK  ligand extracted\n"]
    for atom in res.get_atoms():
        el   = (atom.element or "C").strip().capitalize()
        x, y, z = atom.get_coord()
        lines.append(
            f"HETATM{atom.get_serial_number():5d} {atom.get_name():<4s} "
            f"{res.get_resname():<3s} {res.get_parent().get_id()}"
            f"{res.get_id()[1]:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n"
        )
    lines.append("END\n")
    return "".join(lines)


# -----------------------------
# Extract sequence + residues
# -----------------------------
def extract_residues(pdb_file):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues = []
    sequence = []

    for model in structure:
        for chain in model:
            for residue in chain:

                if residue.id[0] != " ":
                    continue

                try:
                    aa = seq1(residue.resname)
                except:
                    continue

                sequence.append(aa)

                residues.append({
                    "chain": chain.id,
                    "resid": residue.id[1],
                    "resname": residue.resname,
                    "residue_obj": residue
                })

    return "".join(sequence), residues


# -----------------------------
# Build residue mapping
# -----------------------------
def build_mapping(original_pdb, fixed_pdb):

    seq_orig, residues_orig = extract_residues(original_pdb)
    seq_fix, residues_fix = extract_residues(fixed_pdb)

    alignment = pairwise2.align.globalxx(seq_orig, seq_fix)[0]

    aligned_orig = alignment.seqA
    aligned_fix = alignment.seqB

    mapping = {}

    i_orig = 0
    i_fix = 0

    for a, b in zip(aligned_orig, aligned_fix):

        orig_res = None
        fix_res = None

        if a != "-":
            orig_res = residues_orig[i_orig]
            i_orig += 1

        if b != "-":
            fix_res = residues_fix[i_fix]
            i_fix += 1

        if orig_res and fix_res:
            key = (
                orig_res["resname"],
                orig_res["resid"],
                orig_res["chain"]
            )

            mapping[key] = fix_res

    return mapping


# -----------------------------
# Get atoms from selected residues
# -----------------------------
def collect_atoms(mapping, query_residues):

    selected_atoms = []

    for query in query_residues:

        # Example: GLN10
        resname = query[:3]
        resid = int(query[3:])

        found = False

        for key, fixed_res in mapping.items():

            k_resname, k_resid, k_chain = key

            if k_resname == resname and k_resid == resid:

                residue_obj = fixed_res["residue_obj"]

                print(
                    f"{query} -> "
                    f"{fixed_res['resname']}{fixed_res['resid']} "
                    f"(Chain {fixed_res['chain']})"
                )

                for atom in residue_obj:
                    selected_atoms.append(atom.coord)

                found = True
                break

        if not found:
            print(f"{query} not found")

    return np.array(selected_atoms)


# -----------------------------
# Compute docking box
# -----------------------------
def compute_residue_box(atom_coords, padding=3.0):

    min_coords = atom_coords.min(axis=0)
    max_coords = atom_coords.max(axis=0)

    center = (min_coords + max_coords) / 2
    size = (max_coords - min_coords) + padding

    return center, size


# -----------------------------
# Write Vina config
# -----------------------------
def write_vina_config(
    output_file,
    center,
    size,
    receptor,
    # ligand="ligand.pdbqt"
):

    with open(output_file, "w") as f:

        f.write(f"receptor = {receptor}\n")
        # f.write(f"ligand = {ligand}\n\n")

        f.write(f"center_x = {center[0]:.3f}\n")
        f.write(f"center_y = {center[1]:.3f}\n")
        f.write(f"center_z = {center[2]:.3f}\n\n")

        f.write(f"size_x = {size[0]:.3f}\n")
        f.write(f"size_y = {size[1]:.3f}\n")
        f.write(f"size_z = {size[2]:.3f}\n\n")

        # Important docking parameters
        f.write("exhaustiveness = 16\n")
        f.write("num_modes = 10\n")
        f.write("energy_range = 4\n")

    print(f"\nVina config written to: {output_file}")


# -----------------------------
# Main workflow
# -----------------------------
def generate_vina_config(
    original_pdb,
    fixed_pdb,
    query_residues,
    basename,
    output_config="vina_config.txt"
):

    mapping = build_mapping(original_pdb, fixed_pdb)

    atom_coords = collect_atoms(mapping, query_residues)

    center, size = compute_residue_box(atom_coords)

    print("\nDocking box:")
    print("Center:", center)
    print("Size:", size)

    write_vina_config(
        output_config,
        center,
        size,
        receptor=f"{basename}_receptor.pdbqt"
    )


def prepare_receptor(input_pdb, output_pdbqt,mgl_python,prepare_vina_prot):

    command = [
        mgl_python,
        prepare_vina_prot,
        "-r", input_pdb,
        "-o", output_pdbqt,
        "-A", "checkhydrogens",
        "-U", "nphs_lps"
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(f"Successfully created {output_pdbqt}")
        return output_pdbqt

    except subprocess.CalledProcessError as e:
        print("STDOUT:")
        print(e.stdout)

        print("\nSTDERR:")
        print(e.stderr)

        print(f"\nError during conversion: {e}")

def prepare_ligands_from_csv(
    csv_file,
    output_dir,
    smiles_column,
    mgl_python,
    prepare_vina_lig,    
    top_k=None,

):

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------
    # Read CSV
    # -------------------------------------------------
    df = pd.read_csv(csv_file)

    if top_k is not None:
        df = df.head(top_k)

    if smiles_column not in df.columns:
        raise ValueError(
            f"Column '{smiles_column}' not found in CSV."
        )

    success_count = 0
    fail_count = 0

    # =================================================
    # Iterate through ligands
    # =================================================

    for idx, row in df.iterrows():

        smiles = str(row[smiles_column]).strip()

        ligand_name = ""

        if "Ligand" in row and pd.notna(row["Ligand"]):
            ligand_name = str(row["Ligand"]).strip()

        print("\n================================")
        print(f"Processing index {idx}")
        print(f"SMILES: {smiles}")

        try:

            # -------------------------------------------------
            # Create molecule
            # -------------------------------------------------
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                raise ValueError("Invalid SMILES")

            mol = Chem.AddHs(mol)

            # -------------------------------------------------
            # Generate 3D conformer
            # -------------------------------------------------
            embed_status = AllChem.EmbedMolecule(
                mol,
                AllChem.ETKDGv3()
            )

            print("Embed status:", embed_status)

            if embed_status != 0:
                raise ValueError("3D embedding failed")

            # -------------------------------------------------
            # MMFF optimization
            # -------------------------------------------------
            mmff_status = AllChem.MMFFOptimizeMolecule(mol)

            print("MMFF status:", mmff_status)

            if mmff_status == -1:
                print(
                    "WARNING: MMFF optimization failed, "
                    "proceeding with unoptimized geometry"
                )

            # -------------------------------------------------
            # Filenames
            # -------------------------------------------------
            pdb_filename = f"{idx}.pdb"
            pdbqt_filename = f"{idx}.pdbqt"

            pdb_path = os.path.normpath(
                os.path.abspath(
                    os.path.join(output_dir, pdb_filename)
                )
            )

            pdbqt_path = os.path.normpath(
                os.path.abspath(
                    os.path.join(output_dir, pdbqt_filename)
                )
            )

            # -------------------------------------------------
            # Generate PDB block
            # -------------------------------------------------
            pdb_block = rdmolfiles.MolToPDBBlock(mol)

            # -------------------------------------------------
            # Add metadata REMARK lines
            # -------------------------------------------------
            pdb_remarks = [
                f"REMARK SMILES: {smiles}",
                f"REMARK Ligand: {ligand_name}",
                f"REMARK Index: {idx}",
            ]

            full_pdb_block = (
                "\n".join(pdb_remarks)
                + "\n"
                + pdb_block
            )

            # -------------------------------------------------
            # Save PDB
            # -------------------------------------------------
            with open(pdb_path, "w") as f:
                f.write(full_pdb_block)

            if not os.path.exists(pdb_path):
                raise RuntimeError(
                    "PDB file was not created"
                )

            print(f"Saved PDB: {pdb_path}")

            # -------------------------------------------------
            # Run prepare_ligand4.py
            # -------------------------------------------------
            command = [
                mgl_python,
                prepare_vina_lig,
                "-l", pdb_filename,
                "-o", pdbqt_filename,
                "-A", "hydrogens"
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=output_dir
            )

            print("\nSTDOUT:")
            print(result.stdout)

            print("\nSTDERR:")
            print(result.stderr)

            if result.returncode != 0:
                raise RuntimeError(
                    "prepare_ligand4.py failed"
                )

            if not os.path.exists(pdbqt_path):
                raise RuntimeError(
                    "PDBQT file not created"
                )

            # -------------------------------------------------
            # Add metadata to PDBQT
            # -------------------------------------------------
            pdbqt_remarks = [
                f"REMARK SMILES: {smiles}",
                f"REMARK Ligand: {ligand_name}",
                f"REMARK Index: {idx}",
            ]

            with open(pdbqt_path, "r") as f:
                pdbqt_content = f.read()

            updated_pdbqt = (
                "\n".join(pdbqt_remarks)
                + "\n"
                + pdbqt_content
            )

            with open(pdbqt_path, "w") as f:
                f.write(updated_pdbqt)

            print(
                "Added metadata REMARK lines to PDBQT"
            )

            print(f"Saved PDBQT: {pdbqt_path}")

            success_count += 1

        except Exception as e:

            print(f"\nFAILED at index {idx}")
            print("Error:", e)

            fail_count += 1

        print("\n================================")
        print("SUMMARY")
        print("================================")

        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Output directory: {output_dir}")


def run_vina_batch(
    ligand_dir,
    config_file,
    output_dir="vina_results",
    csv_with_smiles=None,
    smiles_column="smiles",
    vina_executable="vina"
):
    """
    Run AutoDock Vina for all ligand PDBQT files in a directory.

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
    """

    # ---------------------------------------
    # Create output directories
    # ---------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    poses_dir = os.path.join(output_dir, "poses")
    logs_dir = os.path.join(output_dir, "logs")

    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ---------------------------------------
    # Load smiles if available
    # ---------------------------------------
    smiles_lookup = {}

    if csv_with_smiles:

        df_smiles = pd.read_csv(csv_with_smiles).head(10)

        if smiles_column not in df_smiles.columns:
            raise ValueError(
                f"Column '{smiles_column}' not found"
            )

        for idx, row in df_smiles.iterrows():
            smiles_lookup[str(idx)] = str(row[smiles_column])

    # ---------------------------------------
    # Find ligands
    # ---------------------------------------
    ligand_files = sorted(
        glob.glob(os.path.join(ligand_dir, "*.pdbqt"))
    )

    if not ligand_files:
        print("No ligand PDBQT files found.")
        return

    results = []

    # ---------------------------------------
    # Run docking
    # ---------------------------------------
    for ligand_path in ligand_files:

        ligand_name = os.path.basename(ligand_path)
        ligand_index = os.path.splitext(ligand_name)[0]

        print(f"\nDocking ligand: {ligand_name}")

        out_pose = os.path.join(
            poses_dir,
            f"{ligand_index}_out.pdbqt"
        )

        log_file = os.path.join(
            logs_dir,
            f"{ligand_index}.txt"
        )

        command = [
            vina_executable,
            "--config", config_file,
            "--ligand", ligand_path,
            "--out", out_pose
        ]

        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            vina_output = result.stdout

            # Save raw log
            with open(log_file, "w") as f:
                f.write(vina_output)

            # ---------------------------------------
            # Parse docking scores
            # ---------------------------------------
            mode_pattern = re.compile(
                r"^\s*(\d+)\s+(-?\d+\.\d+)\s+"
                r"(\d+\.\d+)\s+(\d+\.\d+)",
                re.MULTILINE
            )

            matches = mode_pattern.findall(vina_output)

            if matches:

                for match in matches:

                    mode = int(match[0])
                    affinity = float(match[1])
                    rmsd_lb = float(match[2])
                    rmsd_ub = float(match[3])

                    results.append({
                        "ligand_index": ligand_index,
                        "smiles": smiles_lookup.get(
                            ligand_index,
                            None
                        ),
                        "mode": mode,
                        "affinity_kcal_mol": affinity,
                        "rmsd_lb": rmsd_lb,
                        "rmsd_ub": rmsd_ub,
                        "pose_file": out_pose,
                        "log_file": log_file
                    })

            else:

                results.append({
                    "ligand_index": ligand_index,
                    "smiles": smiles_lookup.get(
                        ligand_index,
                        None
                    ),
                    "mode": None,
                    "affinity_kcal_mol": None,
                    "rmsd_lb": None,
                    "rmsd_ub": None,
                    "pose_file": out_pose,
                    "log_file": log_file
                })

            print("Docking completed.")

        except subprocess.CalledProcessError as e:

            print(f"Docking failed for {ligand_name}")

            with open(log_file, "w") as f:
                f.write(e.stdout or "")
                f.write("\n\nSTDERR:\n")
                f.write(e.stderr or "")

            results.append({
                "ligand_index": ligand_index,
                "smiles": smiles_lookup.get(
                    ligand_index,
                    None
                ),
                "mode": "FAILED",
                "affinity_kcal_mol": None,
                "rmsd_lb": None,
                "rmsd_ub": None,
                "pose_file": None,
                "log_file": log_file
            })

    # ---------------------------------------
    # Save results table
    # ---------------------------------------
    results_df = pd.DataFrame(results)

    csv_output = os.path.join(
        output_dir,
        "vina_results.csv"
    )

    results_df.to_csv(csv_output, index=False)

    print("\n================================")
    print("Batch docking complete.")
    print(f"Results CSV: {csv_output}")
    print(f"Poses folder: {poses_dir}")
    print(f"Logs folder: {logs_dir}")


# ---------------------------------------------------
# Parse Vina-style config file (key = value per line)
# ---------------------------------------------------
def parse_vina_config(config_path):
    """
    Reads a Vina config file with lines like:
        receptor = PKMYT1/8D6E_receptor.pdbqt
        center_x = 10.0
        ...
    Returns a dict of {key: value_as_string}.
    Ignores blank lines and lines starting with '#'.
    """
    config = {}

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()

    return config


def get_receptor_path_from_config(config_path, base_dir=None):
    """
    Extracts the 'receptor' entry from a Vina config file
    and resolves it to an absolute/usable path.

    base_dir: optional directory to join with a relative
              receptor path (e.g. your project root).
              If None, resolves relative to the config file's own directory.
    """
    config = parse_vina_config(config_path)

    if "receptor" not in config:
        raise KeyError(f"No 'receptor' entry found in config file: {config_path}")

    receptor_rel = config["receptor"]

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(config_path))

    receptor_path = os.path.normpath(os.path.join(base_dir, receptor_rel))

    if not os.path.exists(receptor_path):
        raise FileNotFoundError(
            f"Receptor path resolved from config does not exist:\n"
            f"  config value : {receptor_rel}\n"
            f"  resolved path: {receptor_path}\n"
            f"Check base_dir or the config's relative path."
        )

    return receptor_path


# ---------------------------------------------------
# Properly split vina multi-pose PDBQT
# ---------------------------------------------------
def split_vina_output(pdbqt_file):

    with open(pdbqt_file, "r") as f:
        lines = f.readlines()

    poses = []

    current_pose = []
    inside_model = False

    for line in lines:

        if line.startswith("MODEL"):

            inside_model = True
            current_pose = []

            continue

        elif line.startswith("ENDMDL"):

            inside_model = False

            pose_text = "".join(current_pose)

            poses.append(pose_text)

            current_pose = []

            continue

        if inside_model:
            current_pose.append(line)

    return poses


# ---------------------------------------------------
# Run vina --score_only
# ---------------------------------------------------
def score_pose(
    vina_executable,
    receptor_pdbqt,
    ligand_pose_path
):

    command = [
        vina_executable,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pose_path,
        "--score_only"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Vina failed:\n\n"
            f"{result.stdout}\n\n"
            f"{result.stderr}"
        )

    output = result.stdout + "\n" + result.stderr
    print(output)

    # ---------------------------------------
    # Regex helper
    # ---------------------------------------
    def extract(pattern):

        match = re.search(
            pattern,
            output,
            re.IGNORECASE
        )

        if match:
            return float(match.group(1))

        return None


    scores = {

        "affinity_kcal_mol": extract(
            r"Affinity:\s*([-0-9.]+)"
        ),

        "gauss1": extract(
            r"gauss\s*1\s*:\s*([-0-9.]+)"
        ),

        "gauss2": extract(
            r"gauss\s*2\s*:\s*([-0-9.]+)"
        ),

        "repulsion": extract(
            r"repulsion\s*:\s*([-0-9.]+)"
        ),

        "hydrophobic": extract(
            r"hydrophobic\s*:\s*([-0-9.]+)"
        ),

        "hydrogen": extract(
            r"Hydrogen\s*:\s*([-0-9.]+)"
        )
    }

    return scores


# ---------------------------------------------------
# Batch scoring
# ---------------------------------------------------
def batch_score_vina_outputs(
    poses_dir,
    receptor_pdbqt,
    output_csv,
    vina_executable
):

    if not os.path.exists(vina_executable):

        raise FileNotFoundError(
            f"Vina executable not found:\n"
            f"{vina_executable}"
        )

    pdbqt_files = glob.glob(
        os.path.join(poses_dir, "*.pdbqt")
    )

    all_results = []

    for pdbqt_file in pdbqt_files:

        ligand_name = os.path.basename(pdbqt_file)

        ligand_index = os.path.splitext(ligand_name)[0]
        print(ligand_name,ligand_index)

        print(f"\nProcessing: {ligand_name}")

        try:

            poses = split_vina_output(pdbqt_file)

            print(f"Found {len(poses)} poses")

            for pose_num, pose_text in enumerate(poses, start=0):

                # ---------------------------------------
                # Write SINGLE pose temp file
                # ---------------------------------------
                with tempfile.NamedTemporaryFile(
                    suffix=".pdbqt",
                    delete=False,
                    mode="w"
                ) as tmp:

                    tmp.write(pose_text)

                    tmp_path = tmp.name

                try:

                    scores = score_pose(
                        vina_executable,
                        receptor_pdbqt,
                        tmp_path
                    )

                    scores["Name"] = ligand_index
                    scores["Pose"] = pose_num

                    all_results.append(scores)

                    print(
                        f"Pose {pose_num} | "
                        f"Affinity: "
                        f"{scores['affinity_kcal_mol']}"
                    )

                finally:

                    os.remove(tmp_path)

        except Exception as e:

            print(f"FAILED: {ligand_name}")
            print(e)

    # ---------------------------------------
    # Save CSV
    # ---------------------------------------
    df = pd.DataFrame(all_results)

    df.to_csv(output_csv, index=False)

    print("\n===================================")
    print(f"Saved results:\n{output_csv}")

    return df


def extract_interaction_summary(name, df):
    results = []

    for frame_idx, row in df.iterrows():
        interactions = []
        for (_, residue, bondtype), value in row.items():
            if value is True or value == "True":
                interactions.append((residue, bondtype))
        results.append({
            "Name": name,
            "Pose": frame_idx,
            "Interactions": interactions,
            
        })

    return pd.DataFrame(results)

def docked_mols_interaction(
    docked_mols_directory,
    protein_path
):

    # -------------------------------------------------
    # Load protein
    # -------------------------------------------------
    u = mda.Universe(protein_path)

    # Sanitize=False bypasses the valence check entirely
    inferrer = MDAnalysisInferrer(sanitize=False)
    protein_mol = plf.Molecule.from_mda(u, inferrer=inferrer)

    # -------------------------------------------------
    # Interaction parameters
    # -------------------------------------------------
    parameters = {
        "HBDonor": {"distance": 4.0},
        "HBAcceptor": {"distance": 4.0},
        "Hydrophobic": {"distance": 5.0},
        "CationPi": {"distance": 5.0},
        "PiCation": {"distance": 5.0},
    }

    fp = plf.Fingerprint(parameters=parameters)

    sdf_dir = docked_mols_directory

    all_results = []

    # =================================================
    # Iterate through SDF files
    # =================================================
    for filename in os.listdir(sdf_dir):

        if filename.endswith(".sdf"):

            sdf_path = os.path.join(
                sdf_dir,
                filename
            )

            try:

                name = os.path.splitext(filename)[0]

                

                # -------------------------------------------------
                # Read RDKit molecules to extract properties
                # -------------------------------------------------
                rdkit_supplier = Chem.SDMolSupplier(
                    sdf_path,
                    removeHs=False
                )

                rdkit_mols = [
                    mol for mol in rdkit_supplier
                    if mol is not None
                ]

                mol = rdkit_mols[0]

                # Default values
                remark = mol.GetProp("REMARK") if mol.HasProp("REMARK") else ""

                # Extract using regular expressions
                smiles_match = re.search(r"SMILES:\s*(.+)", remark)
                ligand_match = re.search(r"Ligand:\s*(.+)", remark)

                smiles = smiles_match.group(1).strip() if smiles_match else ""
                ligand_name = ligand_match.group(1).strip() if ligand_match else ""
                # -------------------------------------------------
                # Load poses into ProLIF
                # -------------------------------------------------
                poses_mol = plf.sdf_supplier(sdf_path)

                # -------------------------------------------------
                # Run fingerprinting
                # -------------------------------------------------
                fp.run_from_iterable(
                    poses_mol,
                    protein_mol
                )

                df = fp.to_dataframe()

                summary_df = extract_interaction_summary(
                    name,
                    df
                )

                # -------------------------------------------------
                # Add metadata columns
                # -------------------------------------------------
                summary_df["smiles"] = smiles
                summary_df["Ligand"] = ligand_name

                all_results.append(summary_df)

            except Exception as e:

                print(
                    f"Error processing {filename}: {e}"
                )

                continue

    # =================================================
    # Combine all results
    # =================================================
    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    return final_df


def extract_residues(pdb_file):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues = []
    sequence = []

    for model in structure:
        for chain in model:
            for residue in chain:

                if residue.id[0] != " ":
                    continue

                try:
                    aa = seq1(residue.resname)
                except:
                    continue

                sequence.append(aa)

                residues.append({
                    "chain": chain.id,
                    "resid": residue.id[1],
                    "resname": residue.resname,
                    "residue_obj": residue
                })

    return "".join(sequence), residues


def build_reverse_mapping(original_pdb, fixed_pdb):

    seq_orig, residues_orig = extract_residues(original_pdb)
    seq_fix, residues_fix = extract_residues(fixed_pdb)

    alignment = pairwise2.align.globalxx(seq_orig, seq_fix)[0]

    aligned_orig = alignment.seqA
    aligned_fix = alignment.seqB

    reverse_mapping = {}

    i_orig = 0
    i_fix = 0

    for a, b in zip(aligned_orig, aligned_fix):

        orig_res = None
        fix_res = None

        if a != "-":
            orig_res = residues_orig[i_orig]
            i_orig += 1

        if b != "-":
            fix_res = residues_fix[i_fix]
            i_fix += 1

        if orig_res and fix_res:

            fix_key = (
                fix_res["resname"],
                fix_res["resid"],
                fix_res["chain"]
            )

            orig_value = (
                orig_res["resname"],
                orig_res["resid"],
                orig_res["chain"]
            )

            reverse_mapping[fix_key] = orig_value

    return reverse_mapping


def convert_interactions(interactions_obj, reverse_mapping):

    # If already a list, use directly
    if isinstance(interactions_obj, list):
        interactions = interactions_obj

    # If string, parse it
    elif isinstance(interactions_obj, str):
        interactions = ast.literal_eval(interactions_obj)

    else:
        return interactions_obj

    converted = []

    for residue_str, interaction_type in interactions:

        try:

            # Example: PHE51.A
            left, chain = residue_str.split(".")

            resname = left[:3]
            resid = int(left[3:])

            fix_key = (resname, resid, chain)

            # Convert to original numbering
            if fix_key in reverse_mapping:

                orig_resname, orig_resid, orig_chain = (
                    reverse_mapping[fix_key]
                )

                new_residue_str = (
                    f"{orig_resname}{orig_resid}.{orig_chain}"
                )

            else:
                new_residue_str = residue_str

            converted.append(
                (new_residue_str, interaction_type)
            )

        except Exception as e:

            print(
                f"Failed to process residue "
                f"{residue_str}: {e}"
            )

            converted.append(
                (residue_str, interaction_type)
            )

    return converted



# ── Helpers ───────────────────────────────────────────────────────────────────
def strip_h(mol):
    """Remove all H, try to sanitize, keep heavy atoms + coordinates."""
    mol = Chem.RemoveAllHs(mol, sanitize=False)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def load_ref_raw(ref_path):
    ext = os.path.splitext(ref_path)[1].lower()
    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(ref_path, sanitize=False, removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(ref_path, sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unsupported ref file type: {ext} (expected .sdf or .pdb)")
    if mol is None:
        raise ValueError(f"Could not load ref file: {ref_path}")
    return mol


def apply_template(mol, template):
    """Heavy-atom copy of `mol` with bond orders/aromaticity taken from the
    SMILES template. Returns (mol, used_template: bool)."""
    mol = strip_h(mol)
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, mol)
        try:
            Chem.SanitizeMol(fixed)
        except Exception:
            pass
        return fixed, True
    except Exception:
        return mol, False


def inplace_rmsd(ref_xyz, pose_xyz, ref_idx, pose_idx):
    d = pose_xyz[list(pose_idx)] - ref_xyz[list(ref_idx)]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def compute_rmsd(ref_fixed, pose_mol, template):
    """
    In-place, symmetry-corrected heavy-atom RMSD (no superposition).
      1. CalcRMS            - full-graph match, all symmetry mappings, min RMSD
      2. MCS + all matches  - for graph mismatches; min over all pose mappings
    Returns (rmsd, method, n_atoms_compared).
    """
    ref_c = ref_fixed
    pose_c, _ = apply_template(pose_mol, template)

    if ref_c.GetNumAtoms() != pose_c.GetNumAtoms():
        print(f"    ⚠ Heavy atom count mismatch: ref={ref_c.GetNumAtoms()}, "
              f"pose={pose_c.GetNumAtoms()}")

    # Tier 1: RDKit symmetry-aware in-place RMSD
    try:
        rmsd = rdMolAlign.CalcRMS(pose_c, ref_c)   # no map -> enumerates symmetry
        return round(rmsd, 4), "CalcRMS_sym", ref_c.GetNumAtoms()
    except Exception:
        pass

    # Tier 2: MCS, then minimum over ALL pose matches (symmetry-aware)
    try:
        mcs = rdFMCS.FindMCS(
            [ref_c, pose_c],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=20,
        )
        if mcs.numAtoms == 0:
            raise ValueError("Empty MCS")
        query = Chem.MolFromSmarts(mcs.smartsString)
        ref_match = ref_c.GetSubstructMatch(query)
        pose_matches = pose_c.GetSubstructMatches(
            query, uniquify=False, maxMatches=100000)
        if not ref_match or not pose_matches:
            raise ValueError("MCS substructure match failed")

        ref_xyz = ref_c.GetConformer().GetPositions()
        pose_xyz = pose_c.GetConformer().GetPositions()
        best = min(inplace_rmsd(ref_xyz, pose_xyz, ref_match, pm)
                   for pm in pose_matches)
        return round(best, 4), f"MCS_sym({mcs.numAtoms}atoms)", mcs.numAtoms
    except Exception as e:
        return None, f"failed: {e}", 0


def centroid_distance(ref_mol, pose_mol):
    a = strip_h(ref_mol).GetConformer().GetPositions().mean(axis=0)
    b = strip_h(pose_mol).GetConformer().GetPositions().mean(axis=0)
    return round(float(np.linalg.norm(a - b)), 3)


def shape_tanimoto(ref_mol, pose_mol):
    """Volume-overlap similarity (1 = identical), computed in place."""
    try:
        d = rdShapeHelpers.ShapeTanimotoDist(strip_h(pose_mol), strip_h(ref_mol))
        return round(1.0 - d, 3)
    except Exception:
        return None


















































