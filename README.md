# Hierarchical Time Series Forecasting with Robust Reconciliation

## Repository Structure

.
├── data/                   # datasets
├── hierarchical_forecast/  # python files for forecasting and evaluation
├── main.ipynb              # jupyter notebook for experiments
├── README.md
└── requirements.txt

## Requirements

The following and related packages are required.
See `requirements.txt` for details.

- python = ">=3.12,<3.14"
- cvxpy = "^1.6.5"
- jupyter = "^1.1.1"
- numpy = "^2.2.5"
- mosek = "^11.0.18"
- pandas = "^2.2.3"
- prophet = "^1.1.6"

Also, a MOSEK license is required to run the optimization solver.

## Usage

You can perform the numerical experiments described in the paper by running `main.ipynb`.
The roles of each section in `main.ipynb` are as follows.

- **Data**: Read one of the data sets used in the numerical experiment and obtain the necessary information (observed values, number of time points, number of series)
- **Forecast**: Obtain a DataFrame of predicted values in the same format as the observed values for each method
- **Evaluation**: Obtain an evaluation index of forecast accuracy for each method for each series

## References

- **Australian birth data**
  - Paper: [11] R. J. Hyndman and G. Athanasopoulos. *Forecasting: Principles and practice*. OTexts, Melbourne, Australia, 3rd edition, 2021. URL [https://OTexts.com/fpp3](https://OTexts.com/fpp3). Accessed on 2025-05-06.
  - Data: [https://github.com/robjhyndman/fpp3/tree/master/data-raw](https://github.com/robjhyndman/fpp3/tree/master/data-raw)
- **Australian tourism data**
  - Paper: [2] G. Athanasopoulos, R. A. Ahmed, and R. J. Hyndman. Hierarchical forecasts for australian domestic tourism. *International Journal of Forecasting*, 25(1):146–166, 2009.
  - Data: [https://github.com/robjhyndman/fpp3/tree/master/data-raw](https://github.com/robjhyndman/fpp3/tree/master/data-raw)
- **Walmart sales data**
  - Paper: [17] P. Mancuso, V. Piccialli, and A. M. Sudoso. A machine learning approach for forecasting hierarchical time series. *Expert Systems with Applications*, 182, 2021.
  - Data: [https://www.kaggle.com/competitions/m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- **Swiss electricity demand data**
  - Paper: [19] L. Nespoli, V. Medici, K. Lopatichki, and F. Sossan. Hierarchical demand forecasting benchmark for the distribution grid. *Electric Power Systems Research*, 189:106755, 2020.
  - Data: [https://zenodo.org/records/3463137#.XY3GqvexWV4](https://zenodo.org/records/3463137#.XY3GqvexWV4)
