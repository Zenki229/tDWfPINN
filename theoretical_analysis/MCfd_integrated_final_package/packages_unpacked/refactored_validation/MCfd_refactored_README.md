# MCfd refactored validation suite

This package was built from the uploaded `MCfd.ipynb` formulas.

## Files

- `MCfd_refactored_validation.py`: reusable Python script.
- `MCfd_refactored_validation.ipynb`: notebook wrapper with explanations and calls.
- `MCfd_refactored_outputs/`: generated figures and JSON data.

## Run

```bash
python MCfd_refactored_validation.py --outdir MCfd_refactored_outputs --seed 229 --mc-repeats 8
```

The script keeps the original MCfd choices:

- `tau ~ Beta(2-alpha, 1)` for MC-I/MC-II.
- `roots_jacobi(M, 0, 1-alpha)` and `tau=(x+1)/2` for GJ-I/GJ-II.
- MC-I, MC-II, GJ-I, GJ-II decompositions exactly follow the notebook cells.

Additional diagnostics are included for alpha -> 2, large-M GJ endpoint conditioning, and non-smooth functions.
