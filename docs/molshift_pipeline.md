## 7. MolShift Pipeline

**Example:** `examples/molshift_pipeline.ipynb`

The MolShift pipeline generates molecules around a given reference molecule by combining R-group enumeration followed by scaffold replacement. It brings together the individual tools described above into a single workflow to generate and filter potential molecular candidates.

The script provided in `examples/molshift_pipeline.ipynb` is one way of putting together these different tools to implement the complete workflow. The pipeline can be run for a single input molecule, with the scaffold either specified by the user or automatically shortlisted by the code.

### Workflow

**1. Scaffold identification**

The pipeline starts with the input molecule and identifies the scaffold to be retained during the replacement process.

- If `scaffold_smiles` is provided, the specified scaffold is used.
- If `scaffold_smiles = None`, the code automatically shortlists a scaffold for the input molecule.

**2. R-group Enumeration**

The identified scaffold is used for R-group enumeration. The attached R-groups are replaced with similar fragments from the fragment library, generating different combinations of molecules while retaining the selected scaffold.

**3. Reference-Based Filtering**

The generated molecules are subjected to reference-based property filtering using the reference molecule as a baseline. This removes molecules that deviate beyond the permitted physicochemical property ranges, helping eliminate unsuitable chemical modifications introduced during fragment replacement.

**4. Structural Alert Filtering**

The filtered molecules are further screened to remove those containing PAINS or Brenk structural alerts that may have been introduced during R-group replacement.

**5. Scaffold Replacement**

The remaining molecules are then subjected to scaffold replacement using `run_scaff_repl_batch()`. Since the molecules share the same scaffold, the batch function can process them together using multiprocessing.

For each input molecule, the scaffold replacement step generates up to `top_n` replacement scaffolds, resulting in a total of approximately:

`Number of input molecules × top_n`

molecules before subsequent filtering.

**6. Final Filtering**

The molecules generated through scaffold replacement are again subjected to:

- Reference-based property filtering.
- PAINS and Brenk structural alert filtering.

This second round of filtering ensures that unsuitable physicochemical properties or structural alerts introduced during scaffold replacement are also removed.

The final set of molecules is then saved to the specified output directory.

### Docking Pipeline

The example is also followed by a Vina-based docking workflow that can be used to assess the binding properties of the generated molecules within the target protein pocket.

The docking pipeline allows the user to specify a `K` value to select a subset of molecules for docking. The molecules are clustered based on diversity, and K representative molecules are selected for docking.

**Note:** Reference ligand redocking should be performed before attempting to dock the generated molecules. This helps validate the prepared protein structure and docking configuration by checking whether the original ligand pose can be reproduced with an RMSD of less than 2 Å.

Refer to the individual sections on R-group enumeration, scaffold replacement, reference-based filtering, structural alert filtering and Vina-based docking for details on how each component works.