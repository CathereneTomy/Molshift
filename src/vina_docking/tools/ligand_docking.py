from pathlib import Path
from configparser import ConfigParser
import pandas as pd
import rdkit
import json
from rdkit import Chem
import sys
import multiprocessing as mp
from vina_docking.helpers.ligand_docking_utils import (cluster_shortlist, prepare_ligands_from_csv, run_vina_batch, batch_score_vina_outputs,
                                   batch_pdbqt_to_sdf, docked_mols_interaction, build_reverse_mapping, convert_interactions,
                                   score_vs_reference,filter_and_deduplicate)



import os

def run_ligand_docking(
        PROT_DIR,
        BASE_NAME,
        OUT_DIR,
        LIGAND_FILE,
        SMILES_COL,
        K,
        HAS_REFERENCE,
        SCORE_THRESHOLD,
        REF_PATH,
        dock_config
):

    with open(dock_config, "r") as f:
        dock_config = json.load(f)

    obabel_exe = dock_config['obabel_exe']
    vina_exe_path = dock_config["vina_exe_path"]
    mgl_python = dock_config["mgl_python"]
    vina_prepare_lig = dock_config["vina_prepare_lig"]
    obabel_exe = dock_config["obabel_exe"]
    cpu_cores = dock_config["cpu_cores"]

    if HAS_REFERENCE and SCORE_THRESHOLD is None:
        print("ERROR: --score-threshold is required when --has-reference is set.",
              file=sys.stderr, flush=True)
        return 2
 
    PDB_FILE = f"{PROT_DIR}/{BASE_NAME}.pdb"

 
    os.makedirs(OUT_DIR, exist_ok=True)
 
    # ── 1. Clustering ─────────────────────────────────────────────────────────
    df = pd.read_csv(LIGAND_FILE)
    if len(df) > K:
        clustered = cluster_shortlist(LIGAND_FILE, n_clusters=K, k_selection=K, smiles_column=SMILES_COL)
        clustered.to_csv(f"{OUT_DIR}/clustered_selected_molecules.csv", index=False)
        print(f"Selected {len(clustered)} molecules -> clustered_selected_molecules.csv", flush=True)
        LIGAND_FILE = f"{OUT_DIR}/clustered_selected_molecules.csv"
 
    # ── 2. Ligand preparation (parallel) ──────────────────────────────────────
    prepare_ligands_from_csv(
        csv_file=LIGAND_FILE,
        output_dir=f"{OUT_DIR}/prepared_ligands",
        smiles_column=SMILES_COL,
        top_k=K,
        total_cores = cpu_cores,
        mgl_python = mgl_python,
        vina_prepare_lig = vina_prepare_lig,
    )
 
    # ── 3. Batch docking (parallel) ────────────────────────────────────────────
    run_vina_batch(
        ligand_dir=f"{OUT_DIR}/prepared_ligands",
        config_file=f"{PROT_DIR}/{BASE_NAME}_vina_config.txt",
        output_dir=f"{OUT_DIR}/vina_batch_output",
        csv_with_smiles=LIGAND_FILE,
        smiles_column=SMILES_COL,
        vina_exe_path=vina_exe_path,
        total_cores = cpu_cores,
    )
 
    # ── 4. Energy re-scoring (parallel) ───────────────────────────────────────
    scores_df = batch_score_vina_outputs(
        poses_dir=f"{OUT_DIR}/vina_batch_output/poses",
        receptor_pdbqt=f"{PROT_DIR}/{BASE_NAME}_receptor.pdbqt",
        output_csv=f"{OUT_DIR}/vina_batch_output/vina_scores_detailed.csv",
        vina_exe_path=vina_exe_path,
        total_cores=cpu_cores,
    )
 
    # ── 5. PDBQT → SDF (parallel) ─────────────────────────────────────────────
    sdf_dir = f"{OUT_DIR}/vina_batch_output/sdfs"
    os.makedirs(sdf_dir, exist_ok=True)

    batch_pdbqt_to_sdf(
    pdbqt_dir=rf"{OUT_DIR}/vina_batch_output/poses",
    pdb_dir=rf"{OUT_DIR}/vina_batch_output/pdb",
    obabel_exe=obabel_exe,
    output_dir=sdf_dir,
    csv_file=rf"{OUT_DIR}/vina_batch_output/vina_results.csv",
    smiles_column="smiles",
    index_column="ligand_index",
    )
 
 
    # ── 6. Interaction profiling (parallel) ───────────────────────────────────
    interaction_df = docked_mols_interaction(
        sdf_dir=sdf_dir,
        protein_path=f"{PROT_DIR}/{BASE_NAME}_fixed.pdb",
        total_cores = cpu_cores,
    )
 
    # ── 7. Residue remapping ──────────────────────────────────────────────────
    reverse_mapping = build_reverse_mapping(PDB_FILE, f"{PROT_DIR}/{BASE_NAME}_fixed.pdb")
 
    # ── Merge ─────────────────────────────────────────────────────────────────
    final_df = pd.merge(interaction_df, scores_df, how="inner", on=["Name", "Pose"])
    final_df["Mapped_Interactions"] = final_df["Interactions"].apply(
        lambda x: convert_interactions(x, reverse_mapping)
    )
 
    column_order = ["smiles", "Ligand", "Name", "Pose", "affinity_kcal_mol",
                    "gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen",
                    "Interactions", "Mapped_Interactions"]
    final_df = final_df[[c for c in column_order if c in final_df.columns]]
    final_df.to_csv(f"{OUT_DIR}/vina_full_results.csv")
 
    # ── 8 & 9. Reference scoring (parallel) OR plain filtering ───────────────
    if HAS_REFERENCE:
        SDF_DIR     = os.path.join(OUT_DIR, "vina_batch_output", "sdfs")
        LIGANDS_CSV = os.path.join(OUT_DIR, "vina_full_results.csv")
        REF_SDF     = os.path.join(REF_PATH, "shortlisted_pose", "best_pose_1.sdf")
        REF_CSV     = os.path.join(REF_PATH, "shortlisted_pose", "shortlisted_pose.csv")
 
        print("Reference pose available - computing shape & interaction similarity.", flush=True)
        ligands_csv = f"{OUT_DIR}/vina_full_results.csv"
        df = pd.read_csv(ligands_csv, index_col=0)
        df = score_vs_reference(
            df=df,
            sdf_dir=sdf_dir,
            ref_sdf=f"{REF_PATH}/shortlisted_pose/best_pose.sdf",
            ref_csv=f"{REF_PATH}/shortlisted_pose/shortlisted_pose.csv",
            total_cores = cpu_cores,
        )
        df.to_csv(ligands_csv)
 
        df_valid = df[df["combined_score"].notna()]
        print(f"\nCombined score stats ({len(df_valid)} poses):", flush=True)
        print(df_valid["combined_score"].describe().round(4), flush=True)
 
        score_threshold = SCORE_THRESHOLD
        print(f"\nUsing combined_score threshold (keep rows ABOVE this): {score_threshold}", flush=True)
        filtered = df_valid[df_valid["combined_score"] > score_threshold]
        filtered.to_csv(f"{OUT_DIR}/filtered_results.csv", index=False)
        print(f"\nSaved {len(filtered)} rows -> {OUT_DIR}/filtered_results.csv", flush=True)
 
        # ── STEP 3: Filter by affinity — keep rows <= ref best affinity ───────
        ref_df       = pd.read_csv(REF_CSV, index_col=0)
        ref_affinity = ref_df["affinity_kcal_mol"].min()
        affinity_thresh = -8
        print(f"\nRef best affinity : {ref_affinity:.3f} kcal/mol", flush=True)
        if ref_affinity < affinity_thresh:
            ref_affinity = affinity_thresh
            
        print(f"Keeping poses with affinity_kcal_mol <= {ref_affinity:.3f}", flush=True)
 
        df_filtered = filtered[filtered["affinity_kcal_mol"] <= ref_affinity].copy()
        print(f"After affinity filter       : {len(df_filtered)} poses "
              f"({df_filtered['Name'].nunique()} unique molecules)", flush=True)
 
        # ── STEP 4: Deduplicate — keep best combined_score pose per molecule ──
        df_dedup = (df_filtered
                    .sort_values("combined_score", ascending=False)
                    .drop_duplicates(subset="Name", keep="first")
                    .reset_index(drop=True))
 
        print(f"After deduplication         : {len(df_dedup)} unique molecules", flush=True)
        print("\nShortlisted molecules:", flush=True)
        print(df_dedup[["Name", "Pose", "affinity_kcal_mol",
                        "shape_tanimoto", "interaction_overlap",
                        "combined_score"]].to_string(index=False), flush=True)
 
        # ── STEP 5: Extract best pose SDF for each shortlisted molecule ───────
        shortlist_dir = os.path.join(OUT_DIR, "shortlisted_molecules")
        os.makedirs(shortlist_dir, exist_ok=True)
 
        extracted = []
        failed    = []
 
        for _, row in df_dedup.iterrows():
            name     = row["Name"]
            pose_num = int(row["Pose"])
            sdf_path = os.path.join(SDF_DIR, f"{name}.sdf")
 
            if not os.path.exists(sdf_path):
                print(f"  ✗ SDF not found: {sdf_path}", flush=True)
                failed.append(name)
                continue
 
            suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
            poses = [(i + 1, m) for i, m in enumerate(suppl) if m is not None]
 
            target_mol = next((m for pn, m in poses if pn == pose_num), None)
            if target_mol is None:
                print(f"  ✗ Pose {pose_num} not found in {name}.sdf", flush=True)
                failed.append(name)
                continue
 
            target_mol.SetProp("_Name",               name)
            target_mol.SetProp("Pose",                str(pose_num))
            target_mol.SetProp("affinity_kcal_mol",   str(row["affinity_kcal_mol"]))
            target_mol.SetProp("shape_tanimoto",      str(row["shape_tanimoto"]))
            target_mol.SetProp("shape_protrude",      str(row["shape_protrude"]))
            target_mol.SetProp("interaction_overlap", str(row["interaction_overlap"]))
            target_mol.SetProp("combined_score",      str(row["combined_score"]))
 
            out_sdf = os.path.join(shortlist_dir, f"{name}_pose{pose_num}.sdf")
            writer  = Chem.SDWriter(out_sdf)
            writer.write(target_mol)
            writer.close()
 
            extracted.append(name)
            print(f"  ✓ {name} | Pose {pose_num} | "
                  f"affinity={row['affinity_kcal_mol']:.2f} | "
                  f"score={row['combined_score']:.4f} → {os.path.basename(out_sdf)}", flush=True)
 
        # ── STEP 6: Save shortlisted CSV ───────────────────────────────────────
        out_csv = os.path.join(shortlist_dir, "shortlisted_molecules.csv")
        df_dedup.to_csv(out_csv, index=False)
 
        print(f"\n{'='*60}", flush=True)
        print(f"✓ Shortlisted CSV saved → {out_csv}", flush=True)
        print(f"✓ SDFs extracted        : {len(extracted)}", flush=True)
        if failed:
            print(f"✗ Failed               : {len(failed)} → {failed}", flush=True)
        print(f"\nShortlisted molecules folder: {shortlist_dir}/", flush=True)
        for name in extracted:
            row = df_dedup[df_dedup["Name"] == name].iloc[0]
            print(f"  ├── {name}_pose{int(row['Pose'])}.sdf", flush=True)
        print(f"  └── shortlisted_molecules.csv", flush=True)
 
    else:
        print("No reference pose - applying affinity-based filtering only.", flush=True)
        filtered = filter_and_deduplicate(final_df)
        filtered.to_csv(f"{OUT_DIR}/filtered_results.csv", index=False)
        print(f"\nSaved filtered results -> {OUT_DIR}/filtered_results.csv", flush=True)
 
    return 0



if __name__ == "__main__":
    mp.freeze_support()

    dock_config_path = r"dock_config.json"
    PROT_DIR = r"C:\Users\cathe\Desktop\CI\Git_ready\Docking\Multi_ligand_docking\BTK"
    BASE_NAME = "8FLL"
    OUT_DIR = r"C:\Users\cathe\Desktop\CI\FastAPI\Vina_multi_ligand_docking\Dock"
    LIGAND_FILE = r"C:\Users\cathe\Desktop\CI\Git_ready\Docking\Multi_ligand_docking\scaff_repl_diverse.csv"
    SMILES_COL = "product_smiles"
    K = 10
    HAS_REFERENCE = True
    SCORE_THRESHOLD = 0.5
    REF_PATH = r"C:\Users\cathe\Desktop\CI\Git_ready\Docking\Multi_ligand_docking\BTK\Ref"

    sys.exit(run_ligand_docking(
        PROT_DIR,
        BASE_NAME,
        OUT_DIR,
        LIGAND_FILE,
        SMILES_COL,
        K,
        HAS_REFERENCE,
        SCORE_THRESHOLD,
        REF_PATH,
        dock_config_path,
    ))