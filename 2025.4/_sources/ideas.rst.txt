.. Copyright Contributors to the oneDAL project
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

#####
Ideas
#####

As an open-source project, we welcome community contributions to Extension for Scikit-learn.
This document suggests contribution directions which we consider good introductory projects with meaningful
impact. You can directly contribute to next-generation supercomputing, or just learn in depth about key 
aspects of performant machine learning code for a range of architectures. This list is expected to evolve 
with current available projects described in the latest version of the documentation.

Every project is labeled in one of three tiers based on the time commitment: 'small' (90 hours), 'medium' 
(175 hours) or 'large' (350 hours). Related topics can be combined into larger packages, though not 
completely additive due to similarity in scope (e.g. 3 'smalls' may make a 'medium' given a learning 
curve). Others may increase in difficulty as the scope increases (some 'smalls' may become large with 
in-depth C++ coding). Each idea has a linked GitHub issue, a description, a difficulty, and possibly an 
extended goal. They are grouped into relative similarity to allow for easy combinations.

Implement Covariance Estimators for Supercomputers
--------------------------------------------------

The Extension for Scikit-learn contains an MPI-enabled covariance algorithm, showing high performance
from SBCs to multi-node clusters. It directly matches the capabilities of Scikit-Learn's EmpiricalCovariance
estimator. There exist a number of closely related algorithms which modify the outputs of EmpiricalCovariance
which can be created using our implementation. This includes Oracles Approximated Shrinkage (OAS) and Shrunk 
Covariance (ShrunkCovariance) algorithms. Adding these algorithms to our codebase will assist the community 
in their analyses. The total combined work of the two sub-projects is an easy difficulty with a medium time
requirement. With the extended goals, it becomes a hard difficulty with large time requirement.

Oracle Approximating Shrinkage Estimator (small)
************************************************

The output of EmpiricalCovariance is regularized by a shrinkage value impacted by the overall mean of the data.
The goal would be to implement this estimator with post-processing changes to the fitted empirical covariance.
This project is very similar to the ShrunkCovariance project and would combine into a medium project.
When implemented in python re-using our EmpiricalCovariance estimator, this would be an easy project with a 
small time commitment. Implementing the super-computing distributed version using python would only work for
distributed-aware frameworks. Extended goals would make this a hard difficulty, medium commitment project. This
would require implementing the regularization in C++ in oneDAL both for CPU and GPU. Then this must be made 
available in Scikit-learn-intelex for making a new estimator. This would hopefully follow the design strategy 
used for our Ridge Regression estimator.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2305>`__.


ShrunkCovariance Estimator (small)
**********************************

The output of EmpiricalCovariance is regularized by a shrinkage value impacted by the overall mean of the data.
The goal would be to implement this estimator with post-processing changes to the fitted empirical covariance.
This is very similar to the OAS project and would combine into a medium project.
When implemented in python re-using our EmpiricalCovariance estimator, this would be an easy project with a 
small time commitment. Implementing the super-computing distributed version using python would only work for
distributed-aware frameworks. Extended goals would make this a hard difficulty, medium commitment project. This
would require implementing the regularization in C++ in oneDAL both for CPU and GPU. Then this must be made 
available in Scikit-learn-intelex for making a new estimator. This would hopefully follow the design strategy 
used for our Ridge Regression estimator.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2306>`__.


Implement the Preprocessing Estimators for Supercomputers
---------------------------------------------------------

The Extension for Scikit-learn contains two unique estimators used to get vital metrics from large datasets,
known as BasicStatistics and IncrementalBasicStatistics. They generate relevant values like 'min', 'max', 'mean' 
and 'variance' with special focus on multithreaded performance. It is also MPI-enabled working on SBCs to multi-node 
clusters, and can prove very useful for important big data pre-processing steps which may be otherwise unwieldly. 
Several pre-processsing algorithms in Scikit-learn use these basic metrics where BasicStatistics could be used instead. 
The overall goal would be to use the online version, IncrementalBasicStatistics, to create advanced pre-processing 
scikit-learn-intelex estimators which can be used on supercomputing clusters. The difficulty of this project is easy,
with a combined time commitment of a large project. It does not have any extended goals.


StandardScaler Estimator (small)
********************************

The StandardScaler estimator scales the data to zero mean and unit variance. Use the IncrementalBasicStatistics estimator
to generate the mean and variance to scale the data. Investigate where the new implementation may be low performance and 
include guards in the code to use Scikit-learn as necessary. The final deliverable would be to add this estimator to the 'spmd'
interfaces which are effective on MPI-enabled supercomputers, this will use the underlying MPI-enabled mean and variance 
calculators in IncrementalBasicStatistics. This is an easy difficulty project, and would be a medium time commitment 
when combined with other pre-processing projects.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2307>`__.


MaxAbsScaler Estimator (small)
******************************

The MaxAbScaler estimator scales the data by its maximum absolute value. Use the IncrementalBasicStatistics estimator
to generate the min and max to scale the data. Investigate where the new implementation may be low performance and 
include guards in the code to use Scikit-learn as necessary. The final deliverable would be to add this estimator to the 'spmd'
interfaces which are effective on MPI-enabled supercomputers, this will use the underlying MPI-enabled minimum and maximum 
calculators in IncrementalBasicStatistics. This is similar to the MinMaxScaler and can be combined into a small project.
This is an easy difficulty project.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2308>`__.

MinMaxScaler Estimator (small)
******************************

The MinMaxScaler estimator scales the data to a range set by the minimum and maximum. Use the IncrementalBasicStatistics 
estimator to generate the min and max to scale the data. Investigate where the new implementation may be low performance and 
include guards in the code to use Scikit-learn as necessary. The final deliverable would be to add this estimator to the 'spmd'
interfaces which are effective on MPI-enabled supercomputers, this will use the underlying MPI-enabled minimum and maximum
calculators in IncrementalBasicStatistics. This is similar to the MaxAbsScaler and can be combined into a small project.
This is an easy difficulty project.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2309>`__.

Normalizer Estimator (small)
****************************

The normalizer estimator scales the samples independently by the sample's norm (l1, l2). Use the IncrementalBasicStatistics 
estimator to generate the sum squared data and use it for generating only the l2 version of the normalizer. Investigate where 
the new implementation may be low performance and include guards in the code to use Scikit-learn as necessary.  The final 
deliverable would be to add this estimator to the 'spmd' interfaces which are effective on MPI-enabled supercomputers, this 
will use the underlying MPI-enabled mean and variance calculators in IncrementalBasicStatistics. This is an easy difficulty project, 
and would be a medium time commitment when combined with other pre-processing projects.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2310>`__.


Expose Accelerated Kernel Distance Functions
--------------------------------------------

The Extension for Scikit-learn contains several kernel functions which have not been made available in our public API but
are available in our onedal package.  Making these available to the users is an easy, python-only project good for learning about 
Scikit-learn, testing and the underlying math of kernels. The goal would be to make them available in a similar fashion as in Scikit-Learn.
Their general nature makes them have high utility for both scikit-learn and scikit-learn-intelex as they can be used as plugins for a 
number of other estimators (see the Kernel trick).


sigmoid_kernel Function (small)
*******************************

The sigmoid kernel converts data via tanh into a new space. This is easy difficulty, but requires significant benchmarking to find when
the scikit-learn-intelex implementation provides better performance. This project will focus on the public API and including the benchmarking 
results for a seamless, high-performance user experience. Combines with the other kernel projects to a medium time commitment.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2311>`__.


polynomial_kernel Function (small)
**********************************

The polynomial kernel converts data via a polynomial into a new space. This is easy difficulty, but requires significant benchmarking to find when
the scikit-learn-intelex implementation provides better performance. This project will focus on the public API and including the benchmarking 
results for a seamless, high-performance user experience. Combines with the other kernel projects to a medium time commitment.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2312>`__.


rbf_kernel Function (small)
***************************

The rbf kernel converts data via a radial basis function into a new space. This is easy difficulty, but requires significant benchmarking to find when
the scikit-learn-intelex implementation provides better performance. This project will focus on the public API and including the benchmarking 
results for a seamless, high-performance user experience. Combines with the other kernel projects to a medium time commitment.

Questions, status, and additional information can be tracked on `GitHub <https://github.com/uxlfoundation/scikit-learn-intelex/issues/2313>`__.
