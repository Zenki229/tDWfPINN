# 1D Forward Analytic Reference

This folder generates the analytic reference GIF for the paper Section 4.3
forward diffusion-wave benchmark.

## PDE

```text
{}^C D_t^alpha u - lambda / (k^2 pi^2) u_xx = 0,
    (t, x) in (0, 2] x (0, 1)
u(t, 0) = u(t, 1) = 0
u(0, x) = a sin(k pi x)
u_t(0, x) = b sin(k pi x)
```

The analytic solution is

```text
u(t, x) = sin(k pi x)
          [a E_{alpha,1}(-lambda t^alpha)
           + b t E_{alpha,2}(-lambda t^alpha)].
```

The default generator uses the Section 4.3 setting:

```text
alpha = 1.75, lambda = 1, k = 1, a = 1, b = -0.5, T = 2
```

## Generate

From the repository root:

```bash
python reference_solvers/forward_1d/generate_forward_reference_gif.py
```

Equivalent explicit command:

```bash
python reference_solvers/forward_1d/generate_forward_reference_gif.py --alpha 1.75 --lam 1.0 --k 1 --a 1.0 --b -0.5 --t-max 2.0 --x-points 401
```

Output:

```text
data/reference_1d/forward_alpha1p75_reference.gif
data/reference_1d/forward_reference_summary.csv
```
