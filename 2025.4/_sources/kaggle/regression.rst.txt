.. Copyright 2021 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Kaggle Kernels for Regression Tasks
************************************

The following Kaggle kernels show how to patch scikit-learn with |sklearnex| for various regression tasks.
These kernels usually include a performance comparison between stock scikit-learn and scikit-learn patched with |sklearnex|.

.. include:: /kaggle/note-about-tps.rst

Using a Single Regressor
++++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 20 30

   * - Kernel
     - Goal
     - Content
   * - `Baseline Nu Support Vector Regression (nuSVR) with RBF Kernel
       <https://www.kaggle.com/kppetrov/tps-jul-2021-baseline-with-nusvr-model/notebook>`_

       **Data:** [TPS Jul 2021] Synthetic pollution data
     - Predict air pollution measurements over time based on weather and input values from multiple sensors
     -

       - data preprocessing
       - search for optimal paramters using Optuna
       - training and prediction using scikit-learn-intelex
   * - `Nu Support Vector Regression (nuSVR)
       <https://www.kaggle.com/alexeykolobyanin/tps-aug-nusvr-with-intel-extension-for-sklearn>`__

       **Data:** [TPS Aug 2021] Synthetic loan data
     - Calculate loss associated with a loan defaults
     -

       - data preprocessing
       - feature engineering
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn

   * - `Nu Support Vector Regression (nuSVR)
       <https://www.kaggle.com/alexeykolobyanin/house-prices-nusvr-sklearn-intelex-4x-speedup>`__

       **Data:** House Prices dataset
     - Predict sale prices for a property based on its characteristics
     -

       - data preprocessing
       - exploring outliers
       - feature engineering
       - filling missing values
       - search for optimal parameters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Random Forest Regression
       <https://www.kaggle.com/pahandrovich/tps-jul-2021-fast-randomforest-with-sklearnex>`_

       **Data:** [TPS Jul 2021] Synthetic pollution data
     - Predict air pollution measurements over time based on weather and input values from multiple sensors
     -

       - checking correlation between features
       - search for best paramters using GridSearchCV
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn

   * - `Random Forest Regression with Feature Engineering
       <https://www.kaggle.com/alexeykolobyanin/tps-jul-rf-with-intel-extension-for-scikit-learn>`_

       **Data:** [TPS Jul 2021] Synthetic pollution data
     - Predict air pollution measurements over time based on weather and input values from multiple sensors
     -

       - data preprocessing
       - feature engineering
       - search for optimal parameters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Random Forest Regression with Feature Importance Computation
       <https://www.kaggle.com/code/lordozvlad/tps-mar-fast-workflow-using-scikit-learn-intelex>`_

       **Data:** [TPS Mar 2022] Spatio-temporal traffic data
     - Forecast twelve-hours of traffic flow in a major U.S. metropolitan area
     -

       - feature engineering
       - computing feature importance with ELI5
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Ridge Regression
       <https://www.kaggle.com/alexeykolobyanin/tps-sep-ridge-with-sklearn-intelex-2x-speedup>`_


       **Data:** [TPS Sep 2021] Synthetic insurance data
     - Predict the probability of a customer making a claim upon an insurance policy
     -

       - data preprocessing
       - filling missing values
       - search for optimal parameters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn

Stacking Regressors
+++++++++++++++++++

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 20 30

   * - Kernel
     - Goal
     - Content
   * - `Stacking Regressor with Random Fores, SVR, and LASSO
       <https://www.kaggle.com/alexeykolobyanin/tps-jul-stacking-with-scikit-learn-intelex>`_

       **Data:** [TPS Jul 2021] Synthetic pollution data
     - Predict air pollution measurements over time based on weather and input values from multiple sensors
     -

       - feature engineering
       - creating a stacking regressor
       - search for optimal parameters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn

   * - `Stacking Regressor with ElasticNet, LASSO, and Ridge Regression for Time-series data
       <https://www.kaggle.com/alexeykolobyanin/predict-sales-stacking-with-scikit-learn-intelex>`_

       **Data:** Predict Future Sales dataset
     - Predict total sales for every product and store in the next month based on daily sales data
     -

       - data preprocessing
       - creating a stacking regressor
       - search for optimal parameters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
