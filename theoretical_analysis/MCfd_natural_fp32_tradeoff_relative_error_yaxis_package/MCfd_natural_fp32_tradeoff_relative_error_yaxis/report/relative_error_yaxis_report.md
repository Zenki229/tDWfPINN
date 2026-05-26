# Annotated slope figure with y-axis changed to "relative error"

All y-axis labels have been changed from `relative operator error` to `relative error`.

The numerical data and fitted slopes are unchanged.

## Figures

- `figures/fig16_natural_fp32_tradeoff_annotated_slopes.png/pdf`
- `figures/fig16_natural_fp32_mechanism_annotated_slopes.png/pdf`

## Slope table

|   alpha | type   | small_quantity   | small_fit_range   |   small_fitted_slope |   small_expected_slope | large_quantity     | large_fit_range   |   large_fitted_slope |   large_expected_slope |
|--------:|:-------|:-----------------|:------------------|---------------------:|-----------------------:|:-------------------|:------------------|---------------------:|-----------------------:|
|    1.25 | I      | raw32-stable gap | (1e-06, 0.0003)   |            -0.318301 |                  -0.25 | stable cutoff bias | (0.003, 0.1)      |             0.755041 |                   0.75 |
|    1.25 | II     | raw32-stable gap | (1e-06, 0.0003)   |            -1.34812  |                  -1.25 | stable cutoff bias | (0.003, 0.1)      |             0.753865 |                   0.75 |
|    1.5  | I      | raw32-stable gap | (1e-06, 0.0003)   |            -0.514252 |                  -0.5  | stable cutoff bias | (0.003, 0.1)      |             0.503701 |                   0.5  |
|    1.5  | II     | raw32-stable gap | (1e-06, 0.0003)   |            -1.49735  |                  -1.5  | stable cutoff bias | (0.003, 0.1)      |             0.502933 |                   0.5  |
|    1.75 | I      | raw32-stable gap | (1e-06, 0.0003)   |            -0.616066 |                  -0.75 | stable cutoff bias | (0.003, 0.1)      |             0.252058 |                   0.25 |
|    1.75 | II     | raw32-stable gap | (1e-06, 0.0003)   |            -1.79223  |                  -1.75 | stable cutoff bias | (0.003, 0.1)      |             0.251707 |                   0.25 |
