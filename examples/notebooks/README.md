<!--
  ~ Copyright 2021 Intel Corporation
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# :snake: Intel(R) Extension for Scikit-learn* notebooks

This folder contains examples of python notebooks that use Intel(R) extension for Scikit-learn for popular datasets.

#### :rocket: Jupyter startup guide
You can use python notebooks with the help of Jupyter* notebook to run the following files:

```bash
conda install -c conda-forge notebook scikit-learn-intelex
```
or
```bash
pip install notebook scikit-learn-intelex
```
Run Jupyter after installation:
```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

#### :pencil: Table of contents

| Algorithm               | Workload       | Task            | Notebook       | Scikit-learn estimator|
| :----------------------:| :------------: | :---------------:| :------------: | :-------------------:|
|    LogisticRegression  |    CIFAR-100    |    Сlassification    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/logistictic_regression_cifar.ipynb)    | [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) |
|          SVC           |     Adult       |    Сlassification    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/svc_adult.ipynb) | [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| KNeighborsClassifier   |       MNIST     |    Сlassification    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/knn_mnist.ipynb) |    [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
| RandomForestClassifier | Rain in Australia | Classification | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/random_forest_rain.ipynb) | [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
|        NuSVR           | Medical charges |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/nusvr_medical_charges.ipynb) | [sklearn.svm.NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html) |
| RandomForestRegressor  |     Yolanda     |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/random_forest_yolanda.ipynb) | [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) |
|        Ridge           | Airlines DepDelay |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/ridge_regression.ipynb) | [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) |
|        Ridge           | New York City Bike Share |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/ridge_regression_bike_sharing.ipynb) | [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) |
|        Ridge           | IEEE-CIS Fraud Detection |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/ridge_regression_fraud_detection.ipynb) | [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) |
|        Ridge           | Higgs |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/ridge_regression_higgs.ipynb) | [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) |
| ElasticNet  |    Airlines DepDelay     |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/ElasticNet.ipynb) | [sklearn.linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) |
|        Lasso           | YearPredictionMSD |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/lasso_regression.ipynb) | [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) |
| Linear Regression  |    YearPredictionMSD     |    Regression    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/linear_regression.ipynb) | [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) |
|        KMeans           | Spoken arabic digit |    Clustering    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/kmeans.ipynb) | [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) |
| DBSCAN  |     Spoken arabic digit     |    Clustering    | [View source on GitHub](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/notebooks/dbscan.ipynb) | [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) |
