# 1D Burgers Reference Solver

This folder contains the numerical reference code for the one-dimensional
time-fractional Burgers benchmark:

```text
{}^C D_t^alpha u + u u_x - (0.01 / pi) u_xx = 0,
    (t, x) in (0, 1.2] x (-1, 1)
u(t, -1) = u(t, 1) = 0
u(0, x) = -sin(pi x)
u_t(0, x) = beta sin(pi x)
```

The time discretization uses backward-Euler convolution quadrature weights for
the Caputo term. The spatial discretization code is kept here with the WENO and
Lax-Friedrichs components used by the existing Burgers reference data.
