# Data

No imaging data is distributed with this repository. You must obtain ADNI and
OASIS access yourself and accept each provider's data use agreement:

- **ADNI** — https://adni.loni.usc.edu/data-samples/access-data/
- **OASIS** — https://www.oasis-brains.org/#access

This directory holds your label file and, by default, the image roots that
`train.py`, `eval.py`, and `explain.py` search.

## Expected layout

```
data/
├── labels.csv
├── ADNI/            # any nesting; searched recursively for *.nii / *.nii.gz
│   └── 003_S_1234/.../ADNI_003_S_1234_MR_MPRAGE.nii.gz
└── OASIS/
    └── OAS2_0045/.../OAS2_0045_MR1_mpr.nii.gz
```

Directory structure below each root does not matter. `find_nifti_files` in
[`superager/data.py`](../superager/data.py) walks each root recursively for
`*.nii` and `*.nii.gz`, skipping macOS `._` resource forks. Point `--roots` at
whatever directories you actually keep scans in:

```bash
python train.py --labels data/labels.csv --roots data/ADNI data/OASIS
```

## labels.csv

One row per subject. Column names are matched case-insensitively after
stripping whitespace, and several spellings are accepted:

| Field | Accepted column names | Required |
|---|---|---|
| Subject | `subject`, `subject_id`, `subjectid`, `id`, `participant_id` | yes |
| Label | `label`, `diagnosis`, `group`, `class`, `dx` | yes |
| Age | `age`, `age_at_scan` | no |

Example:

```csv
subject,label,age
003_S_1234,1,84
003_S_5678,0,81
OAS2_0045,superager,86
OAS2_0091,control,83
```

### Label values

Parsed by `_to_label` in [`superager/data.py`](../superager/data.py),
case-insensitively:

- **SuperAger (1)** — `1`, `superager`, `sa`, `super`, `super_ager`
- **Typical ager (0)** — `0`, `normal`, `cn`, `control`, `typical`,
  `nondemented`, `normal_ager`

Anything else is dropped with a warning, so check the `Dropped N rows with
unrecognised labels` line in the log before trusting a run.

### Age

Optional, but supplying it is worthwhile: `train.py` fits an age-only logistic
baseline (Sec. 3.1) and records it in `run_manifest.json`. If the imaging models
do not clear that baseline, they are recovering age, not SuperAger status.

## Subject ID parsing

IDs are recovered from **file paths**, not from directory names alone, and then
matched to the CSV:

- **ADNI** — the pattern `\d{3}_S_\d{4}` anywhere in the path, e.g. a file at
  `data/ADNI/.../ADNI_003_S_1234_MR.nii.gz` yields subject `003_S_1234`.
- **OASIS** — `OAS<n>_<digits>_MR<n>` or `OAS2_<4 digits>`, e.g. `OAS2_0045`.
- **Neither** — the filename stem, uppercased.

Before matching, CSV IDs are uppercased and `-` and `.` are converted to `_`, so
`oas2-0045` matches `OAS2_0045`. If an exact match fails, a substring match is
attempted in either direction. Unmatched subjects are reported:

```
WARNING  12 labelled subjects had no matching NIfTI (e.g. ['003_S_9999', ...])
```

A large count here almost always means an ID formatting mismatch rather than
missing files. Check the warning before interpreting any result.

## One scan per subject

`collect_records` keeps a **single scan per subject** (`drop_duplicates` on
subject, first match in sorted path order). This is deliberate: OASIS-2 is
longitudinal, and repeat sessions of one brain landing on both sides of a split
would leak. Splits are additionally subject-grouped, so the two guards stack.

## Sites

Site is derived from the path, not from the CSV:

- **ADNI** — the 3-digit site prefix, e.g. `003_S_1234` → site `ADNI_003`
- **OASIS** — one pooled site, `OASIS`

Splits are site-stratified where feasible. `train.py` writes `site_audit.csv`
to the output directory and warns about single-class sites:

```
WARNING  8/42 sites are single-class (e.g. ['ADNI_011', ...]).
```

In those sites, site *is* the label. Site-aware stratification mitigates this
confound but cannot remove it, so read `site_audit.csv` before quoting numbers.

## Preprocessing cache

`preprocess` caches the assembled `(N, 3, 224, 224)` array to
`<outdir>/cache/X_<hash>.npy`. The hash covers the cohort size, image paths, and
the preprocessing settings, so editing `labels.csv` or changing
`size_2d` / `harmonize` / `ref_size` / `mask_threshold` invalidates it
automatically. Pass `--no-cache` to force recomputation.
