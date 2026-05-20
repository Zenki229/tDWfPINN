# 2D Circular-Hole Reference Solver

This folder is reserved for the numerical reference solver for the
two-dimensional circular-hole benchmark. The PDE currently used by the JAX case
is:

```text
Omega = (-1, 1)^2 \ B_0.25((-0.3, 0.2))

{}^C D_t^alpha u - div(a(x, y) grad u)
    + b(x, y) . grad u + lambda u^3 = f(t, x, y),
    (t, x, y) in (0, 1] x Omega

u = 0 on partial Omega
u(0, x, y) = 0
u_t(0, x, y) = 0

a(x, y) = 1 + 0.3 sin(pi x) cos(pi y)
b(x, y) = (1 + y, x - 1)
lambda = 1
```

The final numerical reference generation workflow and GIF outputs will be added
after the reference method is fixed.
