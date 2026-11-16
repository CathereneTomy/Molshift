from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import os
import pandas as pd
from huggingface_hub import hf_hub_download
import itertools
from molshift.helpers.scaffold_reactions import generate_products, generate_products_batch
from molshift.helpers.scaffold_replacement_screen import screen_scaffolds, load_library
from molshift.helpers.rgrp_enum_utils import find_frag_atts



def run_scaff_repl(
        whole_smiles,
        scaffold_smiles,
        top_n,
        resources,
        output_dir,
        sim_mode,

):
    os.makedirs(output_dir, exist_ok=True)
    
    
    whole_mol = Chem.MolFromSmiles(whole_smiles)
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)

    result = find_frag_atts(whole_mol, scaffold_mol, scaffold_smiles)
    selected_atts = result['selected_attachments']

    lib = load_library(
        main_path   = resources["frag_feat_path"],        # fragment_features.parquet
        extras_path = resources["frag_site_env_path"],    # fragment_site_extras.parquet
        usrcat_path = resources["frag_usrcat_path"],      # usrcat_cache.parquet
    )

    ranked, df = screen_scaffolds(scaffold_smiles, selected_atts,lib = lib,
                                save_dir = output_dir, sim_mode=sim_mode)


    products = generate_products(
        whole_smiles, scaffold_smiles, selected_atts, ranked,
        top_n=top_n, strict=False, out_dir=output_dir,          # or None to skip saving
    )


    return {
        "output_dir": str(output_dir),
        "n_scaffold_replaced": len(products),
        "Output_path": f"{output_dir}/scaff_replaced_products.csv",
    }


def run_scaff_repl_batch(
        molecule_df,
        smiles_col,
        scaffold_smiles,
        selected_atts,
        top_n,
        resources,
        output_dir,
        sim_mode,

):
    os.makedirs(output_dir, exist_ok=True)
    
    

    lib = load_library(
        main_path   = resources["frag_feat_path"],        # fragment_features.parquet
        extras_path = resources["frag_site_env_path"],    # fragment_site_extras.parquet
        usrcat_path = resources["frag_usrcat_path"],      # usrcat_cache.parquet
    )

    ranked, df = screen_scaffolds(scaffold_smiles, selected_atts,lib = lib,
                                save_dir = output_dir,sim_mode=sim_mode)


    products = generate_products_batch(
        molecule_df,
        smiles_col,
        scaffold_smiles,
        selected_atts,
        ranked,
        top_n=top_n,
        strict=False,
        n_jobs=max(1, os.cpu_count() - 4),
        out_dir=str(output_dir),
    )

    return {
        "out": df,
        "output_dir": str(output_dir),
        "n_scaffold_replaced": len(products),
        "Output_path": f"{output_dir}/scaff_replaced_products_batch.csv",
    }
