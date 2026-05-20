# 1D Forward Analytic Reference

This folder generates the analytic reference GIF for the Section 4.3 forward
diffusion-wave benchmark:

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

Run:

```bash
conda run -n sciml python reference_solvers/forward_1d/generate_forward_reference_gif.py
```
