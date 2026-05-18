# MCfd rate verification: $M^{-1/2}$ and $\rho^{-2M}$

This note adds the last two rate checks suggested by the paper:

1. the Monte Carlo root-mean-square rate

   $$
   \left(\mathbb E|D^\alpha f-D^\alpha_{M,\mathrm{MC}}f|^2\right)^{1/2}
   = O(M^{-1/2}),
   $$

2. the analytic Gauss--Jacobi rate

   $$
   |D^\alpha f-D^\alpha_{M,\mathrm{GJ}}f|=O(\rho^{-2M}).
   $$

The experiment uses the stable kernels in the paper rather than raw difference quotients, because the previous precision experiment showed that raw quotients can hit cancellation error before the asymptotic quadrature rate is visible.

## Benchmark

For the rate test I used

$$
    f_a(s)=\frac1{a+s},\qquad t=1.5.
$$

The stable kernels are available in closed form:

$$
K_f(t,\tau)=\frac{f'(t)-f'(t-t\tau)}{t\tau}
=\frac{2A-h}{A^2(A-h)^2},
$$

$$
H_f(t,\tau)=\frac{f(t)-f(t-t\tau)-t\tau f'(t)}{(t\tau)^2}
=-\frac1{A^2(A-h)},
$$

where $A=a+t$ and $h=t\tau$.  The pole is located at

$$
    \tau_* = 1+\frac a t>1.
$$

Under the Bernstein map $x=2\tau-1$, the corresponding ellipse parameter is

$$
    \rho_* = x_*+\sqrt{x_*^2-1},\qquad x_*=2\tau_*-1.
$$

For the GJ test I use $a=0.05,0.10,0.20$, giving three different $\rho_*$ values.

## Figure 9: Monte Carlo $M^{-1/2}$ rate

![MC sqrt M rate](fig09_mc_sqrtM_rate.png)

The dots are empirical RMS derivative errors over repeated Monte Carlo runs.  The dashed curves are the exact variance formula induced by

$$
Q_M^{\mathrm{MC}}[\phi]
=\frac1{(2-\alpha)M}\sum_{j=1}^M\phi(\xi_j),
\qquad \xi_j\sim \mathrm{Beta}(2-\alpha,1).
$$

The fitted slopes are:

| alpha | type | expected_slope | empirical_slope | exact_formula_slope |
| --- | --- | --- | --- | --- |
| 1.25 | Type-I | -0.5000 | -0.5005 | -0.5 |
| 1.25 | Type-II | -0.5000 | -0.5050 | -0.5 |
| 1.5 | Type-I | -0.5000 | -0.4950 | -0.5 |
| 1.5 | Type-II | -0.5000 | -0.4959 | -0.5 |
| 1.75 | Type-I | -0.5000 | -0.5139 | -0.5 |
| 1.75 | Type-II | -0.5000 | -0.5103 | -0.5 |

All empirical slopes are very close to $-1/2$, across $\alpha=1.25,1.50,1.75$ and both Type-I / Type-II.

## Figure 10: Gauss--Jacobi $\rho^{-2M}$ rate

![GJ rho spectral rate](fig10_gj_rho_spectral_rate.png)

For the analytic test, the expected exponential slope in a plot of $\log(error)$ against $M$ is

$$
    -2\log\rho_*.
$$

For Type-I, $K_f$ has a second-order pole at $\tau_*$ for this rational benchmark, so the asymptotic form is more accurately fitted as

$$
    error \approx C M^p \rho_*^{-2M}.
$$

Therefore the slope table reports both a naive linear fit and a two-term fit

$$
    \log(error) \approx c+sM+p\log M.
$$

The exponential slope $s$ from the two-term fit is the quantity that should match $-2\log\rho_*$.

| a | rho_star | type | expected_exp_slope | linear_slope | exp_slope_with_logM | logM_power | exp_slope_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 1.438 | Type-I | -0.7263 | -0.6733 | -0.7285 | 1.069 | 1.003 |
| 0.05 | 1.438 | Type-II | -0.7263 | -0.7217 | -0.7318 | 0.147 | 1.008 |
| 0.1 | 1.667 | Type-I | -1.0217 | -0.9560 | -1.0203 | 1.003 | 0.999 |
| 0.1 | 1.667 | Type-II | -1.0217 | -1.0160 | -1.0289 | 0.143 | 1.007 |
| 0.2 | 2.044 | Type-I | -1.4299 | -1.3325 | -1.4322 | 1.017 | 1.002 |
| 0.2 | 2.044 | Type-II | -1.4299 | -1.4228 | -1.4387 | 0.131 | 1.006 |

The Type-II kernel has only a first-order pole, so the naive linear slope is already very close to the expected value.  Type-I shows a visible polynomial prefactor, but after adding the $\log M$ term, the extracted exponential slope also matches $-2\log\rho_*$.

## Figure 11: slope summary

![Rate slope summary](fig11_rate_slope_summary.png)

The left panel summarizes the MC log-log slopes and confirms $M^{-1/2}$.  The right panel summarizes the normalized GJ exponential slopes

$$
    \frac{s_{\mathrm{fit}}}{-2\log\rho_*},
$$

which cluster around 1.

## Interpretation for the paper

The numerical evidence supports the two remaining theoretical rates:

- Monte Carlo has the canonical RMS rate $O(M^{-1/2})$, independent of the fractional order.  The fractional order changes the variance constant through the Beta sampling density, but not the rate.
- For analytic kernels, Gauss--Jacobi error decays exponentially with $M$.  The observed exponential slope is governed by the nearest complex singularity through $\rho_*^{-2M}$.
- The Type-I rational test has a higher-order pole, producing a mild polynomial prefactor.  This does not contradict the theorem: the theorem gives $C_\rho\rho^{-2M}$ for any ellipse strictly inside the analytic domain, and polynomial prefactors are absorbed by using any $\rho<\rho_*$.  The fitted exponential part still matches the expected $-2\log\rho_*$.

## Reproducibility

Run:

```bash
python MCfd_mc_gj_rate_verification.py --outdir MCfd_rate_verification_report --seed 229 --mc-repeats 384
```

Generated CSV files:

- `mc_sqrtM_rate_data.csv`
- `mc_sqrtM_slope_summary.csv`
- `gj_rho_spectral_error_data.csv`
- `gj_rho_spectral_slope_summary.csv`
