# MolShift

MolShift is a fragment-based molecular modification pipeline designed to generate new molecules around a given reference molecule. It combines **R-group enumeration** and **scaffold replacement** using a fragment library derived from ChEMBL molecules.

The generated molecules are subjected to physicochemical property filtering and structural alert screening to remove potentially unsuitable candidates. An optional Vina-based docking workflow is also available to assess their binding properties within the target protein pocket.

## Features

- **R-group Enumeration:** Replaces existing R-groups with similar fragments from the fragment library to generate molecular variants.
- **Scaffold Replacement:** Identifies and replaces a given scaffold with structurally and physicochemically similar alternatives.
- **Reference-Based Filtering:** Filters generated molecules based on physicochemical property ranges relative to the reference molecule.
- **Structural Alert Filtering:** Removes molecules containing PAINS and Brenk structural alerts.
- **MolShift Pipeline:** Combines R-group enumeration, scaffold replacement and filtering into a single workflow.
- **Vina-Based Docking:** Provides an optional docking workflow to evaluate binding poses, interactions and shape similarity against a reference ligand.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd MolShift
```

Install the package and its dependencies:

```bash
pip install -e .
```

This installs MolShift in editable mode.

## Usage

Example notebooks demonstrating the individual tools and the complete pipeline are available in the `examples/` directory.

- `rgrp_enum_run.ipynb` – R-group enumeration for a single molecule.
- `scaff_repl_run.ipynb` – Scaffold replacement for a single molecule.
- `molshift_pipeline.ipynb` – Complete MolShift workflow combining enumeration, scaffold replacement and filtering.

For detailed explanations of the individual modules and workflows, refer to the [`docs/`](docs/) directory.

## Vina-Based Docking

An optional Vina-based docking workflow is available under `vina_docking/` to assess the binding properties of the generated molecules.

**Note:** Reference ligand redocking should be performed first to validate the protein preparation and docking configuration before docking the generated molecules.

Refer to the docking documentation in `docs/` for details.

## Project Structure

```text
MolShift/
│
├── examples/           # Example notebooks
├── docs/               # Module and workflow documentation
├── molshift/           # Core molecular modification tools
├── vina_docking/       # Vina-based docking tools
├── pyproject.toml      # Package configuration and dependencies
└── README.md
```