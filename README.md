# Fuel Dispatching

## Introduction

![Dispatchers impact in mining](intro.pdf)
*Source: Hustrulid, W. A., Kuchta, M., and Martin, R. K. (2013). Open pit mine planning and design, two volume set & CD-ROM pack. CRC Press.*
*Moradi, A. and Askari-Nasab, H. (2019). Mining fleet management systems: a review of models and algorithms. International Journal of Mining, Reclamation and Environment, 33(1):42–60.*

## Research Problem & Hypothesis

![Research Problem & Hypothesis](resarchpro.pdf)
1. Machine learning - fuel consumption
2. Optimization model - dispatching

## Literature Review

### Machine learning predicting fuel consumption review

| Author                         | Algorithm | Features                      | Metric | Perf. |
|--------------------------------|-----------|--------------------------------|--------|-------|
| 1. Dindarloo and S. (2015)[^34] | ANN       | Payload, Cycle status          | MAPE   | 10\%  |
| 2. Dindarloo and S. (2016)[^34] | PLSR      | Cycle status                   | MAPE   | 6\%   |
| 3. Wang et al. (2021)[^5]       | XGBoost   | Distance, Time, Uphill distance | MAPE   | 8.8\% |
| **4. Soofastaei (2022)[^6]**    | **ANN**   | **Payload, Resistance, Speed** | **R^2**| **90\%** |

### Fuel dispatching review

| Approach                     | Features                                      | Lacking                                                                                          |
|------------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1. Caceres and Well (2017)[^7]| Automated fuel dispatching, better filling volumes, lower queues and person-hours | - Math formulation  |
| 2. Modular Mining Systems (2019)[^8]| Set minimum fuel level, assign manually     | - Trial and error approaches; - Needs customization; - No benefits and consequences.  |
| 3. Leonida (2022)[^9]         | Maximizes fuel utilization, minimizes trips to fuel locations                | - Math formulation; - Multi-objective function; - No proven results |

## Methodology

### Data Collection

![Mine Z Haulage Fleet](methodata.pdf)

- Data Retrieved from FMS
- MF is the match factor, calculated by dividing the sum of truck loading times by the product of the total loaders and the average truck cycle time (Burt and Caccetta (2007)).

### Machine Learning

- Features & Labels: Feature: EFH (m), truck model, payload (tons); Label: Fuel consumed (L)
- Machine learning algorithms: 6 supervised ML algorithms
- Tuning: Grid search, cross-validation
- Selection: Accuracy (R-squared), simplicity (tuning time)
- Coding Environment: Python, Scikit-learn, Tensorflow

### Optimization

- Optimization model: Binary integer programming model
- Objective Functions: Maximize match factor
- Decision variables: If truck h is sent to fuel at time t
- Constraints:
  - -<20% fuel ratio trucks refueled;
  - -Trucks avoid simultaneous refueling;
  - -Trucks fueling only once;
  - -15-minute refueling window;
  - -0% minimum fuel level ensured.
- Coding Environment: Python & Gurobi

## Results

### Machine Learning

| Algorithm                          | R^2 (Test) | Training Time | Tuning Time   |
|------------------------------------|------------|---------------|---------------|
| Multi-Linear Regression (Base Model)| 75%        | 126ms         | 0s            |
| Support Vector Regression (SVR)    | 75%        | 31ms          | 5m 21s        |
| Decision Tree Regression            | 55%        | 107ms         | 5m 25s        |
| Gradient Boosting Regression        | 66%        | 139ms         | 3m 29s        |
| Random Forest Regression            | 64%        | 109ms         | 1m 07s        |
| **Artificial Neural Network**       | **90%**    |  **65ms**     | **21hr 17min**|

### Optimization

- First Formulation (Fuel time window):
  - (1) Infeasible model,
  - (2) impractical to adjust truck-shovel allocations.

- Adjustment First Formulation:
  - (1) Penalty for overlapping fueling,
  - (2) non-linearity issue, and
  - (3) variables and constraints = more time.

- Second Formulation (No fuel time window):
  - (1) Feasible (+optimal) model,
  - (2) Match factor improvement (avg.): 1 point, and
  - (3) arrival difference (avg.): 10-minute difference.

## Results - Demo

![Demo](initial.png)

## Results - Summary

- Machine learning:
  - ANN: 90% accuracy
  - Reliable than a radio-call
- Optimization model:
  - Math formulation
  - Integral objective function
  - <1sec. to run for a fleet of 57 trucks

## Future work

1. Investigating the impact of different variables;
2. Extending the sample size;
3. Implementation;
4. Mobile stations in the optimization model;
5. Model a complete shift with the truck-shovel allocation engine.

## Acknowledgments

- Dr. Hall ![Dr. Hall](robert.png)
- Dr. Brickey ![Dr. Brickey](andrea.png)
- Dr. Kumar ![Dr. Kumar](dubey.png)
- Graduate Students ![Graduate Students](gradstudents.pdf)
