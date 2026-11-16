# data_loader.py

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "Catherene98/chembl29-brics-fragments"


def load_fragment_data():
    frag_feat_path = hf_hub_download(
        REPO,
        "fragment_features_organic_noalerts.parquet",
        repo_type="dataset",
    )

    frag_usrcat_path = hf_hub_download(
        REPO,
        "usrcat_cache.parquet",
        repo_type="dataset",
    )

    frag_site_env_path = hf_hub_download(
        REPO,
        "fragment_site_extras.parquet",
        repo_type="dataset",
    )

    fragment_desc = pd.read_parquet(frag_feat_path)


    return {
        "fragment_desc": fragment_desc,
        "frag_feat_path": frag_feat_path,
        "frag_usrcat_path": frag_usrcat_path,
        "frag_site_env_path": frag_site_env_path,
    }