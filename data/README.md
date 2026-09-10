# Data preparation

This repository does not distribute imaging data. ADNI and OASIS are each
governed by their own data use agreement (DUA); request access directly from
the cohorts and comply with their terms before use:

- ADNI: https://adni.loni.usc.edu/
- OASIS: https://www.oasis-brains.org/

Once you have access, arrange the files as follows (or point `--roots` at
wherever they already live):

```
data/
├── labels.csv
├── ADNI/       # T1-weighted .nii / .nii.gz files, any subdirectory depth
└── OASIS/      # T1-weighted .nii / .nii.gz files, any subdirectory depth
```

`train.py`, `eval.py`, and `explain.py` search each `--roots` directory
recursively for `*.nii` / `*.nii.gz` files (see `find_nifti_files` in
[`superager/data.py`](../superager/data.py)) and join them to `labels.csv` by
subject ID.

## `labels.csv` schema

One row per subject. Column names are matched case-insensitively; any of the
aliases below is accepted.

| Field   | Accepted column names                              | Required | Notes |
|---------|-----------------------------------------------------|----------|-------|
| subject | `subject`, `subject_id`, `subjectid`, `id`, `participant_id` | yes | Must match (or be a substring/superstring of) the subject ID parsed from the filename |
| label   | `label`, `diagnosis`, `group`, `class`, `dx`         | yes | `1`/`superager`/`sa`/`super`/`super_ager` → SuperAger; `0`/`normal`/`cn`/`control`/`typical`/`nondemented`/`normal_ager` → typical ager; anything else is dropped with a warning |
| age     | `age`, `age_at_scan`                                 | no       | Enables the age-only sanity baseline in `train.py` (paper Sec. 3.1) |

Example:

```csv
subject,label,age
003_S_1234,SuperAger,81
OAS2_0045_MR1,Normal,76
```

## Subject ID parsing

Subject IDs are extracted from the NIfTI filename/path, not assumed to be the
filename stem, using cohort-specific patterns (`subject_id` in
[`superager/data.py`](../superager/data.py)):

- **ADNI**: `\d{3}_S_\d{4}` (e.g. `003_S_1234`), which also encodes the
  acquisition site as `ADNI_<site>` for site-stratified splitting.
- **OASIS**: `OAS\d+[_-]\d+[_-]MR\d+` or `OAS2[_-]\d{4}` (e.g.
  `OAS2_0045_MR1`); all OASIS scans are treated as one site.
- Anything else falls back to the file's stem, uppercased.

Only one scan per subject is kept (`collect_records` in `superager/data.py`)
— this matters for OASIS-2, which is longitudinal, so that repeat sessions of
the same subject cannot land on both sides of a cross-validation split.

## Labelling criteria used in the paper

ADNI SuperAgers follow the operationalization of Keenan et al. (RAVLT-based,
≥60 age floor). OASIS lacks an episodic-memory instrument, so OASIS positives
are selected on the Northwestern age criterion (≥80) with intact global
cognition (CDR = 0, MMSE = 30) — a weaker selector than the ADNI one; see the
paper (Sec. 3.1) for the label-noise implications this has for the reported
AUCs.
