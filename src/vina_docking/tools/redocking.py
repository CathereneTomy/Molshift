from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from vina_docking.helpers.pdbqt_sdf import batch_pdbqt_to_sdf
from vina_docking.helpers.redocking_utils import *



def run_redocking(
        dock_config,
        PROT_DIR,
        BASE_NAME,
        OUT_DIR,
        CHAINS,
        LIGAND_ID,
        RESIDUE_LIST = None,
        k = None,
):
    with open(dock_config, "r") as f:
            dock_config = json.load(f)

    obabel_exe = dock_config['obabel_exe']
    vina_exe_path = dock_config["vina_exe_path"]
    mgl_python = dock_config["mgl_python"]
    vina_prepare_receptor = dock_config["vina_prepare_receptor"]
    vina_prepare_lig = dock_config["vina_prepare_lig"]
    obabel_exe = dock_config["obabel_exe"]

    PDB_FILE = f"{PROT_DIR}/{BASE_NAME}.pdb"
    assert Path(PDB_FILE).exists(), f"File not found: {PDB_FILE}"

    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", PDB_FILE)
    model = structure[0]  # always use first model

    all_chains = [c.get_id() for c in model.get_chains()]
    print(f"Chains present in {PDB_FILE}: {all_chains}")

    # for chain in model.get_chains():
    #     residues = list(chain.get_residues())
    #     aa_count = sum(1 for r in residues if is_aa(r, standard=True))
    #     het_count = sum(1 for r in residues if r.get_id()[0].startswith("H") and not is_water(r) and not is_ion(r))
    #     wat_count = sum(1 for r in residues if is_water(r))
    #     print(f"  Chain {chain.get_id():>3s}: {aa_count:4d} AA residues | {het_count:3d} HETATM ligands | {wat_count:4d} waters")

    KEEP_CHAINS = [c.strip().upper() for c in CHAINS.split(",") if c.strip()]
    invalid = [c for c in KEEP_CHAINS if c not in all_chains]
    if invalid:
        raise ValueError(f"Chain(s) not found in structure: {invalid}")

    print(f"Keeping chains: {KEEP_CHAINS}")

    base_name  = Path(PDB_FILE).stem

    cleaned_pdb = f"{PROT_DIR}/{base_name}_cleaned.pdb"

    io = PDBIO()
    io.set_structure(structure)
    io.save(cleaned_pdb, ChainWaterIonFilter(KEEP_CHAINS))
    print(f"Cleaned structure saved → {cleaned_pdb}")


    fixed_pdb = f"{PROT_DIR}/{base_name}_fixed.pdb"

    fixer = PDBFixer(filename=PDB_FILE)   # original file, SEQRES intact

    # 1. Keep only the requested chains
    drop_idx = [i for i, c in enumerate(fixer.topology.chains())
                if c.id not in KEEP_CHAINS]
    fixer.removeChains(drop_idx)
    print(f"Chains kept: {[c.id for c in fixer.topology.chains()]}")

    # 2. Convert nonstandard residues
    print("Finding nonstandard residues...")
    fixer.findNonstandardResidues()
    print(f"  Nonstandard residues found: {len(fixer.nonstandardResidues)}")
    fixer.replaceNonstandardResidues()

    # 3. Remove waters, ions, ligands, cofactors
    fixer.removeHeterogens(keepWater=False)

    # 4. Missing residues (internal gaps only)
    print("Finding missing residues...")
    fixer.findMissingResidues()

    chains_list = list(fixer.topology.chains())
    for key in list(fixer.missingResidues.keys()):
        chain_idx, res_idx = key
        n_res = len(list(chains_list[chain_idx].residues()))
        if res_idx == 0 or res_idx >= n_res:
            del fixer.missingResidues[key]

    n_missing = sum(len(v) for v in fixer.missingResidues.values())
    print(f"  Gap sites: {len(fixer.missingResidues)} | residues to add: {n_missing}")

    # 5. Missing heavy atoms, then hydrogens
    print("Finding missing heavy atoms...")
    fixer.findMissingAtoms()
    print(f"  Missing atoms to add: {sum(len(v) for v in fixer.missingAtoms.values())}")
    fixer.addMissingAtoms()

    print("Adding missing hydrogens at pH 7.4...")
    fixer.addMissingHydrogens(7.4)

    # 6. Write, preserving original numbering and chain IDs
    with open(fixed_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    print(f"Fixed structure saved → {fixed_pdb}")

    # Parse the *original cleaned* PDB (before fixer removes HETATMs)
    parser2 = PDBParser(QUIET=True)
    cleaned_struct = parser2.get_structure("cleaned", cleaned_pdb)
    cleaned_model  = cleaned_struct[0]

    hetatm_residues = []
    resnames = {}
    for chain in cleaned_model.get_chains():
        for res in chain.get_residues():
            het_flag = res.get_id()[0]
            if het_flag.startswith("H") and not is_water(res) and not is_ion(res):
                hetatm_residues.append(res)

    if not hetatm_residues:
        print("No HETATM ligands found in the selected chains.")
    else:
        print(f"Found {len(hetatm_residues)} HETATM ligand(s):")
        for res in hetatm_residues:
            name = res.get_resname()
            print(name)
            resnames[name] = res

    if LIGAND_ID:
        smiles_col = "smiles"

        if LIGAND_ID not in resnames.keys():
            raise ValueError(f"LIGAND_ID '{LIGAND_ID}' not found in resnames.")
        else:
            chosen_res = resnames[LIGAND_ID]
  
        selected = []
        chain_id = chosen_res.get_parent().get_id()
        resname  = chosen_res.get_resname().strip()
        resnum   = chosen_res.get_id()[1]
        coords   = get_residue_coords(chosen_res)
        center, size = compute_hetatm_box(coords)
        smiles, mol = get_smiles_from_residue(chosen_res)
        entry = dict(chain=chain_id, resname=resname, resnum=resnum,
                        center=center, size=size, smiles=smiles,
                        res_obj=chosen_res)          # ← store residue for SDF extraction later
        selected.append(entry)

        # ── Save ref.csv + SDF for selected ligands ───────────────────────────────
        # ── Save ref.csv + PDB + SDF for selected ligands ─────────────────────────
        ref_ligand_dir = f"{OUT_DIR}/ref_ligand"
        os.makedirs(ref_ligand_dir, exist_ok=True)
        csv_rows = []

        print("\nSaving selected ligand(s):")
        print("-" * 50)
        for e in selected:
            ligand_id = f"{e['resname']}_{e['chain']}_{e['resnum']}"

            # SMILES row (use PubChem SMILES already fetched)
            csv_rows.append({"ligand_id": ligand_id, "smiles": e["smiles"]})

            # Save raw PDB block for this ligand
            pdb_block = residue_to_pdb_block(e["res_obj"])
            pdb_path  = os.path.join(ref_ligand_dir, f"{ligand_id}.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb_block)
            print(f"  ✓ {ligand_id}  →  PDB: {pdb_path}")

            # 3-D SDF from the same residue
            mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True,  removeHs=False)
            if mol is None:
                mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)

            if mol is not None:
                sdf_path = os.path.join(ref_ligand_dir, f"{ligand_id}.sdf")
                writer = Chem.SDWriter(sdf_path)
                mol.SetProp("_Name", ligand_id)
                writer.write(mol)
                writer.close()
                print(f"  ✓ {ligand_id}  →  SDF: {sdf_path}")
            else:
                print(f"  ✗ {ligand_id}  →  RDKit could not parse residue; SDF skipped.")

        # Write ref.csv
        with open(f"{OUT_DIR}/ref.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ligand_id", "smiles"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n✓ ref.csv written with {len(csv_rows)} entry/entries.")

        config_generated = False   # track whether a config was made

        if selected:
            for e in selected:
                tag = f"{e['chain']}_{e['resname']}_{e['resnum']}"

                receptor_pdbqt = f"{PROT_DIR}/{base_name}_receptor.pdbqt"

                config_content = (
                    f"# Vina config — site: {tag}\n"
                    f"receptor = {receptor_pdbqt}\n"
                    f"\n{format_box(e['center'], e['size'])}\n"
                    f"{VINA_DEFAULTS}"
                )

                config_file = f"{PROT_DIR}/{base_name}_vina_config.txt"
                with open(config_file, "w") as fh:
                    fh.write(config_content)
                print(f"\nVina config saved → {config_file}")
                print(config_content)
                config_generated = True
        else:
            print("No HETATM site was selected earlier — will proceed to residue-based box.")

    elif RESIDUE_LIST:
        residue_input = RESIDUE_LIST.strip()

        # -----------------------------------
        # Parse residue input
        # -----------------------------------
        queries = []

        if residue_input:

            # Split by comma
            raw_residues = residue_input.split(",")

            for residue in raw_residues:

                residue = residue.strip().upper()

                # Remove optional chain notation
                # Example: B:LYS100 -> LYS100
                if ":" in residue:
                    residue = residue.split(":")[1]

                # Basic validation
                if len(residue) >= 4:
                    queries.append(residue)

            if queries:

                print("\nSelected residues:")
                for q in queries:
                    print(f"  - {q}")

                generate_vina_config(
                    original_pdb=PDB_FILE,
                    fixed_pdb=fixed_pdb,
                    query_residues=queries,
                    basename=base_name,
                    output_config=f"{base_name}_vina_config.txt"
                )

            else:
                print("\nNo valid residues detected. Skipping residue-based config generation.")

        else:
            print("\nNo residues chosen. Skipping residue-based config generation.")


    else:
        raise ValueError(
        "Pocket definition requires at least one input: LIGAND_ID/RESIDUE_LIST "
        "Please provide a valid input."
        )


    receptor_pdbqt = prepare_receptor(fixed_pdb, f"{PROT_DIR}/{base_name}_receptor.pdbqt",mgl_python=mgl_python, prepare_vina_prot= vina_prepare_receptor)

    outputs = [
    ("Cleaned PDB",           cleaned_pdb),
    ("Fixed PDB (PDBFixer)",  fixed_pdb),
    ("Receptor PDBQT",        receptor_pdbqt),
    ("Config file", f"{PROT_DIR}/{base_name}_vina_config.txt")
    ]

    for label, fp in outputs:
        exists = Path(fp).exists()
        status = f"({Path(fp).stat().st_size:,} bytes)" if exists else "[NOT CREATED]"
        print(f"  {'✓' if exists else '✗'} {label:<30s}  {fp}  {status}")

#===============================Ligand Preparation=====================================================

    prepare_ligands_from_csv(
    csv_file=f"{OUT_DIR}/ref.csv",
    output_dir=f"{OUT_DIR}/prepared_ligands",
    smiles_column=smiles_col,
    mgl_python = mgl_python,
    prepare_vina_lig = vina_prepare_lig,
    top_k=k,

    )

    run_vina_batch(
    ligand_dir=f"{OUT_DIR}/prepared_ligands",
    config_file=f"{PROT_DIR}/{base_name}_vina_config.txt",
    output_dir=f"{OUT_DIR}/vina_batch_output",
    csv_with_smiles=f"{OUT_DIR}/ref.csv",
    smiles_column=smiles_col,
    vina_executable= vina_exe_path,
    )

    receptor_pdbqt = get_receptor_path_from_config(
    f"{PROT_DIR}/{base_name}_vina_config.txt",
    base_dir=os.getcwd()   # the folder relative paths in config are resolved from
    )

    print(f"Using receptor: {receptor_pdbqt}")
    scores_df = batch_score_vina_outputs(
        poses_dir=fr"{OUT_DIR}\vina_batch_output\poses",
        receptor_pdbqt=receptor_pdbqt,
        output_csv=fr"{OUT_DIR}\vina_batch_output\vina_scores_detailed.csv",
        vina_executable=vina_exe_path,
    )

    batch_pdbqt_to_sdf(
    pdbqt_dir=rf"{OUT_DIR}/vina_batch_output/poses",
    pdb_dir=rf"{OUT_DIR}/vina_batch_output/pdb",
    output_dir=rf"{OUT_DIR}/vina_batch_output/sdfs",
    csv_file=rf"{OUT_DIR}/vina_batch_output/vina_results.csv",
    smiles_column=smiles_col,
    index_column="ligand_index",
    obabel_exe=obabel_exe
    )

    
    interaction_df = docked_mols_interaction(
        fr"{OUT_DIR}\vina_batch_output\sdfs",
        fixed_pdb
    )

    reverse_mapping = build_reverse_mapping(
    PDB_FILE,
    fixed_pdb
    )

    final_df = pd.merge(
        interaction_df,
        scores_df,
        how="inner",
        on = ['Name', 'Pose']
    )
    final_df["Mapped_Interactions"] = final_df["Interactions"].apply(
        lambda x: convert_interactions(
            x,
            reverse_mapping
        )
    )

    desired_order = [
        "smiles",
        "Ligand",
        "Name",
        "Pose",
        "affinity_kcal_mol",
        "gauss1",
        "gauss2",
        "repulsion",
        "hydrophobic",
        "hydrogen",
        "Interactions",
        "Mapped_Interactions",
    ]

    # Keep only columns that actually exist
    existing_columns = [
        col for col in desired_order
        if col in final_df.columns
    ]

    final_df = final_df[existing_columns]

    final_df.to_csv(f"{OUT_DIR}/vina_full_results.csv")

    # ── Paths ─────────────────────────────────────────────────────────────────────
    sdf_dir       = os.path.join(OUT_DIR, "vina_batch_output", "sdfs")
    full_csv      = os.path.join(OUT_DIR, "vina_full_results.csv")
    shortlist_dir = os.path.join(OUT_DIR, "shortlisted_pose")
    os.makedirs(shortlist_dir, exist_ok=True)

    # ── Find docked SDF ───────────────────────────────────────────────────────────
    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith(".sdf")]
    if not sdf_files:
        raise FileNotFoundError(f"No docked SDF found in {sdf_dir}")
    DOCKED_SDF = os.path.join(sdf_dir, sdf_files[0])
    print(f"Docked SDF: {DOCKED_SDF}")

    # ── Load reference ligand info ────────────────────────────────────────────────
    ref_csv = os.path.join(OUT_DIR, "ref.csv")
    if not os.path.exists(ref_csv):
        raise FileNotFoundError("ref.csv not found.")
    with open(ref_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("ref.csv is empty.")

    ref_ligand_id = rows[0]["ligand_id"]
    ref_smiles    = rows[0]["smiles"]
    print(f"Reference ligand : {ref_ligand_id}")
    print(f"SMILES           : {ref_smiles}")

    if ref_smiles == "NOT_FOUND" or not ref_smiles:
        raise ValueError("SMILES in ref.csv is NOT_FOUND — run the SMILES fetch fix first.")

    # Heavy-atom template used for BOTH reference and poses
    template = Chem.MolFromSmiles(ref_smiles)
    if template is None:
        raise ValueError(f"Could not parse SMILES: {ref_smiles}")
    template = Chem.RemoveAllHs(template)
    Chem.SanitizeMol(template)

    ref_pdb_path = os.path.join(OUT_DIR, "ref_ligand", f"{ref_ligand_id}.pdb")
    ref_raw = load_ref_raw(ref_pdb_path)
    ref_fixed, ref_ok = apply_template(ref_raw, template)
    print("  ✓ Reference bond orders assigned from SMILES template" if ref_ok
        else "  ✗ Reference template assignment failed — using raw ref")

    # ── Load poses ────────────────────────────────────────────────────────────────
    poses = [(i + 1, m)
            for i, m in enumerate(Chem.SDMolSupplier(DOCKED_SDF, sanitize=True, removeHs=False))
            if m is not None]
    print(f"\nLoaded {len(poses)} pose(s) — computing symmetry-corrected RMSD...")

    # ── Compute metrics per pose ──────────────────────────────────────────────────
    rmsd_map, n_atoms_map, method_map, cdist_map, shape_map = {}, {}, {}, {}, {}
    for pose_num, pose_mol in poses:
        rmsd, method, n_cmp = compute_rmsd(ref_fixed, pose_mol, template)
        rmsd_map[pose_num]    = rmsd
        method_map[pose_num]  = method
        n_atoms_map[pose_num] = n_cmp
        cdist_map[pose_num]   = centroid_distance(ref_fixed, pose_mol)
        shape_map[pose_num]   = shape_tanimoto(ref_fixed, pose_mol)
        status = f"{rmsd:.4f} Å" if rmsd is not None else "FAILED"
        print(f"  Pose {pose_num:>2d} | RMSD = {status:<10} | "
            f"centroid Δ = {cdist_map[pose_num]:.2f} Å | "
            f"shape Tc = {shape_map[pose_num]} | {method}")

    # ── Append to full results CSV ────────────────────────────────────────────────
    df = pd.read_csv(full_csv, index_col=0)
    df["rmsd_A"]         = df["Pose"].map(rmsd_map)
    df["rmsd_method"]    = df["Pose"].map(method_map)
    df["rmsd_n_atoms"]   = df["Pose"].map(n_atoms_map)
    df["centroid_dist_A"] = df["Pose"].map(cdist_map)
    df["shape_tanimoto"] = df["Pose"].map(shape_map)
    df.to_csv(full_csv)
    print(f"\n✓ rmsd_A, rmsd_method, rmsd_n_atoms, centroid_dist_A, shape_tanimoto "
        f"appended → {full_csv}")

    # ── Identify best pose ────────────────────────────────────────────────────────
    df_valid = df[df["rmsd_A"].notna()]
    if df_valid.empty:
        raise ValueError("No valid RMSD values — check mol parsing.")

    best_row      = df_valid.loc[df_valid["rmsd_A"].idxmin()]
    best_pose_num = int(best_row["Pose"])
    best_rmsd     = best_row["rmsd_A"]
    print(f"✓ Best pose: Pose {best_pose_num} | RMSD = {best_rmsd:.4f} Å")

    # ── Save shortlisted CSV row ──────────────────────────────────────────────────
    shortlist_csv = os.path.join(shortlist_dir, "shortlisted_pose.csv")
    best_row.to_frame().T.reset_index(drop=True).to_csv(shortlist_csv, index=False)
    print(f"✓ Shortlisted row saved → {shortlist_csv}")

    # ── Save best pose SDF ────────────────────────────────────────────────────────
    best_mol = next((m for pn, m in poses if pn == best_pose_num), None)
    if best_mol is None:
        raise ValueError(f"Could not retrieve Pose {best_pose_num} mol object.")

    best_sdf_out = os.path.join(shortlist_dir, "best_pose.sdf")
    writer = Chem.SDWriter(best_sdf_out)
    best_mol.SetProp("_Name",    f"best_pose_{best_pose_num}")
    best_mol.SetProp("rmsd_A",   str(best_rmsd))
    best_mol.SetProp("Ligand",   str(best_row.get("Ligand", "")))
    best_mol.SetProp("affinity", str(best_row.get("affinity_kcal_mol", "")))
    writer.write(best_mol)
    writer.close()
    print(f"✓ Best pose SDF saved  → {best_sdf_out}")

    print(f"\nShortlisted pose folder: {shortlist_dir}/")
    print(f"  ├── shortlisted_pose.csv")
    print(f"  └── best_pose.sdf")


if __name__ == '__main__': 
    PROT_DIR = r"C:\Users\cathe\Desktop\CI\Git_ready\Docking\Redocking\BTK"
    dock_config = "dock_config.json"
    run_redocking(
        dock_config=dock_config,
        PROT_DIR = PROT_DIR,
        BASE_NAME= "8FLL",
        OUT_DIR = f"{PROT_DIR}/REFERENCE",
        CHAINS = "A",
        LIGAND_ID = "Y7W",
        RESIDUE_LIST = "",
        k = 40,
    )

















































