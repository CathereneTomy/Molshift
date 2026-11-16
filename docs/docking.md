# Vina-Based Docking

Vina-based docking can be used as an accessory to the molecule generation process to assess the binding properties of the generated molecules within the target protein pocket.

## 1. Redocking the Reference Ligand

**Script:** `vina_docking/tools/redocking.py`

Given a protein-ligand PDB file, along with the chain ID and ligand ID from the PDB, the code automatically prepares the protein and generates the required docking files.

The workflow is as follows:

1. **Protein preparation:** Cleans and fixes the input protein structure and generates the protein PDBQT file.
2. **Docking configuration:** Generates the docking configuration file with the box dimensions and center defined based on the reference ligand.
3. **Ligand preparation and redocking:** Extracts the SMILES of the reference ligand, generates its PDBQT file and redocks it within the prepared protein pocket.
4. **Interaction extraction:** Uses ProLIF to extract the protein-ligand interactions from the docked poses.
5. **Pose validation:** Compares the docked poses against the original ligand conformation in the input PDB and calculates their RMSD. The pose with the lowest RMSD is shortlisted.

If the shortlisted pose has an RMSD of less than 2 Å, it is generally considered acceptable to proceed with further docking using the prepared protein structure and docking configuration.

## 2. Docking Generated Molecules

**Script:** `vina_docking/tools/ligand_docking.py`

If the above script was used to generate the initial protein PDBQT file and docking configuration, the resulting directory can be provided along with the protein PDB filename and the folder containing the reference ligand docking output.

The code then docks the given molecules within the prepared protein pocket. If only a subset of molecules needs to be docked, a value can be provided through the `K` argument. The code will cluster the molecules based on diversity and select K representative molecules for docking.

For each docked pose, the following information is extracted:

**1. Docking Scores**

- `affinity_kcal_mol`
- `gauss1`, `gauss2`
- `repulsion`
- `hydrophobic`
- `hydrogen`

**2. Protein-Ligand Interactions**

Interactions are extracted using ProLIF. These are later compared against the interaction profile of the reference ligand.

**3. Shape Similarity**

A shape overlay is performed between the generated molecule's docked pose and the reference docked structure to calculate:

- `shape_tanimoto`: Measures the extent of shape overlap between the two structures.
- `shape_protrude`: Measures how much of the generated molecule protrudes outside the reference docked structure.

**4. Interaction Overlap**

`interaction_overlap` represents the fraction of interactions observed in the reference ligand that are also retained by the generated molecule.

**5. Combined Score**

A combined score is generated using `shape_tanimoto`, `shape_protrude` and `interaction_overlap` by assigning weights to each metric and summing their contributions.

The weights currently used are:

```python
W_TANIMOTO    = 0.50
W_PROTRUDE    = 0.20  # After inversion
W_INTERACTION = 0.50
```

The `shape_protrude` contribution is inverted so that lower protrusion values contribute more positively to the final score.

**6. Pose Shortlisting**

A `SCORE_THRESHOLD` is provided to automatically filter the docked poses. For each molecule, the code retains only one pose: the pose with the highest `combined_score` that satisfies both of the following conditions:

- `combined_score` is above the specified `SCORE_THRESHOLD`.
- Docking affinity is better than or equal to -8 kcal/mol.

The shortlisted molecules are stored in a `shortlisted_mols` folder containing both the shortlisted molecular structures in SDF format and a corresponding CSV file with their docking scores and other extracted metrics.