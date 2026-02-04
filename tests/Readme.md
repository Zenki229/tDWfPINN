# Testing for Fractional Derivatives 
According to the paper eqn(19)
$$
u(t,x)=(2E_{\alpha,1}(-\pi^2t^{\alpha})-tE_{\alpha, 2}(-\pi^2 t^\alpha))\sin(\pi x),
$$
we have 
$$
\partial_t^\alpha u(t,x)=-\pi^2u(t,x)
$$
so we pytest our `_gj_i`, `_mc_i`... with the above equation in `PROJECT_ROOT/tests/test_physics.py`.