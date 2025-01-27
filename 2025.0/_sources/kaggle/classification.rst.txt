.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

Kaggle Kernels for Classification Tasks
***************************************

The following Kaggle kernels show how to patch scikit-learn with |intelex| for various classification tasks.
These kernels usually include a performance comparison between stock scikit-learn and scikit-learn patched with |intelex|.

.. include:: /kaggle/note-about-tps.rst

Binary Classification
+++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 20 30

   * - Kernel
     - Goal
     - Content
   * - `Logistic Regression for Binary Classification
       <https://www.kaggle.com/alexeykolobyanin/tps-nov-log-regression-with-sklearnex-17x-speedup>`_
     
       **Data:** [TPS Nov 2021] Synthetic spam emails data

     - Identify spam emails via features extracted from the email
     - 

       - data preprocessing (normalization)
       - search for optimal parameters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Feature Importance in Random Forest for Binary Classification
       <https://www.kaggle.com/code/lordozvlad/fast-feature-importance-using-scikit-learn-intelex/notebook>`_
     
       **Data:** [TPS Nov 2021] Synthetic spam emails data

     - Identify spam emails via features extracted from the email
     - 
      
       - reducing DataFrame memory usage
       - computing feature importance with ELI5 and the default scikit-learn permutation importance
       - training using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Random Forest for Binary Classification
       <https://www.kaggle.com/andreyrus/tps-apr-rf-with-intel-extension-for-scikit-learn>`_
     
       **Data:** [TPS Apr 2021] Synthetic data based on Titanic dataset
     - Predict whether a passenger survivies
     - 
      
       - data preprocessing
       - feature construction
       - search for optimal parameters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Support Vector Classification (SVC) for Binary Classification
       <https://www.kaggle.com/napetrov/tps04-svm-with-intel-extension-for-scikit-learn>`_
       
       **Data:** [TPS Apr 2021] Synthetic data based on Titanic dataset
     - Predict whether a passenger survivies
     - 
     
       - data preprocessing
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Support Vector Classification (SVC) with Feature Preprocessing for Binary Classification
       <https://www.kaggle.com/napetrov/tps04-svm-with-scikit-learn-intelex>`_
      
       **Data:** [TPS Apr 2021] Synthetic data based on Titanic dataset
     - Predict whether a passenger survivies
     - 
     
       - data preprocessing
       - feature engineering
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn

MultiClass Classification
+++++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 20 30

   * - Kernel
     - Goal
     - Content
   * - `Logistic Regression for MultiClass Classification with Quantile Transformer
       <https://www.kaggle.com/kppetrov/tps-jun-fast-logreg-with-scikit-learn-intelex>`_
       
       **Data:** [TPS Jun 2021] Synthetic eCommerce data
     - Predict the category of an eCommerce product
     - 

       - data preprocessing with Quantile Transformer
       - training and prediction using scikit-learn-intelex
       - search for optimal paramters using Optuna
       - performance comparison to scikit-learn
   * - `Support Vector Classification (SVC) for MultiClass Classification
       <https://www.kaggle.com/napetrov/svm-tps-may-2021-with-scikit-learn-intelex>`_
       
       **Data:** [TPS May 2021] Synthetic eCommerce data
     - Predict the category of an eCommerce product
     - 
       - data preprocessing
       - training and prediction using scikit-learn-intelex

   * - `Stacking Classifer with Logistic Regression, kNN, Random Forest, and Quantile Transformer
       <https://www.kaggle.com/owerbat/tps-jun-fast-stacking-with-scikit-learn-intelex>`_
      
       **Data:** [TPS Jun 2021] Synthetic eCommerce data
     - Predict the category of an eCommerce product
     - 

       - data preprocessing: one-hot encoding, dimensionality reduction with PCA, normalization
       - creating a stacking classifier with logistic regression, kNN, and random forest,
         and a pipeline of Quantile Transformer and another logistic regression as a final estimator
       - searching for optimal parameters for the stacking classifier
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Support Vector Classification (SVC) for MultiClass Classification
       <https://www.kaggle.com/code/alexeykolobyanin/tps-dec-svc-with-sklearnex-20x-speedup>`_
       
       **Data:** [TPS Dec 2021] Synthetic Forest Cover Type data
     - Predict the forest cover type
     - 
       - data preprocessing
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Feature Importance in Random Forest for MultiClass Classification
       <https://www.kaggle.com/code/lordozvlad/tps-dec-fast-feature-importance-with-sklearnex>`_
     
       **Data:** [TPS Dec 2021] Synthetic Forest Cover Type data

     - Predict the forest cover type
     - 
      
       - reducing DataFrame memory usage
       - computing feature importance with ELI5
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `k-Nearest Neighbors (kNN) for MultiClass Classification
       <https://www.kaggle.com/code/alexeykolobyanin/tps-feb-knn-with-sklearnex-13x-speedup>`_
       
       **Data:** [TPS Feb 2022] Bacteria DNA
     - Predict bacteria species based on repeated lossy measurements of DNA snippets
     - 
       - data preprocessing
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn

Classification Tasks in Computer Vision
+++++++++++++++++++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 20 30

   * - Kernel
     - Goal
     - Content
   * - `Support Vector Classification (SVC) for MultiClass Classification (CV task)
       <https://www.kaggle.com/kppetrov/fast-svc-using-scikit-learn-intelex-for-mnist?scriptVersionId=58739300>`_
       
       **Data:** Digit Recognizer (MNIST)
     - Recognize hand-written digits
     - 
       
       - data preprocessing
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `k-Nearest Neighbors (kNN) for MultiClass Classification (CV task)
       <https://www.kaggle.com/kppetrov/fast-knn-using-scikit-learn-intelex-for-mnist?scriptVersionId=58738635>`_
       
       **Data:** Digit Recognizer (MNIST)
     - Recognize hand-written digits
     - 
     
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn

Classification Tasks in Natural Language Processing
+++++++++++++++++++++++++++++++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 20 30

   * - Kernel
     - Goal
     - Content
   * - `Support Vector Classification (SVC) for a Binary Classification (NLP task)
       <https://www.kaggle.com/kppetrov/fast-svc-using-scikit-learn-intelex-for-nlp?scriptVersionId=58739339>`_
       
       **Data:** Natural Language Processing with Disaster Tweets
     - Predict which tweets are about real disasters and which ones are not
     - 
     
       - data preprocessing
       - TF-IDF calculation
       - search for optimal paramters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `One-vs-Rest Support Vector Machine (SVM) with Text Data for MultiClass Classification
       <https://www.kaggle.com/kppetrov/using-scikit-learn-intelex-for-what-s-cooking?scriptVersionId=58739642>`_
      
       **Data:** What's Cooking
     - Use recipe ingredients to predict the cuisine
     - 
     
       - feature extraction using TfidfVectorizer
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
   * - `Support Vector Classification (SVC) for Binary Classification with Sparse Data (NLP task)
       <https://www.kaggle.com/code/alex97andreev/fast-svm-for-sparse-data-from-nlp-problem>`_
       
       **Data:** Stack Overflow questions
     - Predict the binary quality rating for Stack Overflow questions
     - 
     
       - data preprocessing
       - TF-IDF calculation
       - search for optimal paramters using Optuna
       - training and prediction using scikit-learn-intelex
       - performance comparison to scikit-learn
