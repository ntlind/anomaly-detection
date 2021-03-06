﻿# Automated Anomaly Detection for Hierarchical Time Series

![python_utilities](https://github.com/ntlind/anomaly-detection/workflows/build/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://codecov.io/gh/ntlind/anomaly-detection/branch/master/graph/badge.svg)](https://codecov.io/gh/ntlind/anomaly-detection)


`anomaly-detection` builds on Facebook's `fbprophet` library, enabling you to identify unusual outliers and trends within hierarchical time series data in only a few lines of code. This library:

* Flags and prioritizes anomalies based on configurable Prophet forecasts
* Identifies changepoints in your data to help you spot sudden trend shifts
* Enables you to plot and measure trend differences between hierarchical groups 

What makes this package different from other anomaly detection libs?
* Leverages Facebook's Prophet algorithm, rather than older, classical approaches (e.g., KNN, smoothing algorithms, etc.)
* Explores differences in parameters derived from generative models, rather than focusing only on discrimant boundaries
* Overrides Prophet's methods to provide an easier usage and debugging experience

![Photo of an anomaly graph](https://github.com/ntlind/anomaly-detection/blob/master/examples/example_graph.PNG)

## Installation

Start by installing [`pystan` and and `fbprophet`](https://facebook.github.io/prophet/docs/installation.html), then install this repo using `git clone https://github.com/ntlind/anomaly-detection`.


## Examples

Check out the two .ipynb examples in `/examples`
