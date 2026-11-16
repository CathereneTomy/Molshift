## 3. Reference-Based Property Filtering

**Script:** `molshift/tools/ref_filter.py`

When selecting and fitting a set of similar fragments onto a blueprint, the resulting molecules can sometimes be distorted due to several factors. These include the absence of sufficiently similar fragments in the library, differences in the arrangement of chemical groups, and an overall shift in chemistry and functional groups after fragment replacement.

To ensure that these changes do not result in molecules that are highly unsuitable for the target pocket, and to reduce the number of molecules to those with a higher likelihood of being suitable, reference-based filtering can be performed.

The code takes a DataFrame of molecules to be filtered along with a reference SMILES and applies hard cutoffs based on the following property ranges:

```python
LIPINSKI_BUFFERS = {
    "MW":   ("rel", 0.20),   # +/- 20 %
    "logP": ("abs", 1.0),    # +/- 1 log unit
    "HBD":  ("abs", 1),      # +/- 1
    "HBA":  ("abs", 2),      # +/- 2
    "TPSA": ("rel", 0.20),   # +/- 20 %
    "RotB": ("abs", 2),      # +/- 2
}

SA_BUFFER = 1.0     # keep SA <= ref_SA + 1.0
QED_BUFFER = 0.10   # keep QED >= ref_QED - 0.10
```

These buffers define how much deviation from the reference molecule is allowed for each property. They can be made more flexible or more restrictive by modifying the corresponding values in the code.

**Filtering approach:**

The molecules are not selected based on their similarity scores but rather on whether they satisfy the defined property cutoffs.

The underlying hypothesis is that since the fragments themselves are selected based on their similarity to the fragments being replaced, the final molecules should largely retain the overall chemical and physicochemical characteristics of the reference molecule.

However, in cases where the selected fragments are only proxies due to the absence of sufficiently similar candidates in the library, or where the resulting attachments introduce unsuitable chemistry, these hard cutoffs can help eliminate such molecules.

## 4. Structural Alert Filtering

**Script:** `molshift/tools/alerts.py`

Although the fragment library is already filtered to remove molecules containing PAINS and Brenk substructures, the newly generated attachments can sometimes introduce additional structural alerts that were not present in the original fragments.

This script takes the DataFrame of generated molecules and filters out those containing PAINS or Brenk structural alerts, returning a cleaned DataFrame for further processing.