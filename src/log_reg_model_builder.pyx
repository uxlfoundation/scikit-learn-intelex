#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# We expose oneDAL's directly through Cython
# The model builder object is retrieved through calling model_builder.
# We will extend this once we know how other model builders will work in oneDAL

import numpy

cimport numpy


cdef extern from "log_reg_model_builder.h":
    cdef cppclass c_logistic_regression_model_builder:
        c_logistic_regression_model_builder(size_t n_features, size_t n_classes) except +
        void setBeta(data_management_NumericTablePtr ptrBeta)

    cdef logistic_regression_ModelPtr * get_logistic_regression_model_builder_model(c_logistic_regression_model_builder *)
    cdef data_management_NumericTablePtr getTable(const data_or_file &t)


cdef class logistic_regression_model_builder:
    '''
    Model Builder for logistic regression.
    '''
    cdef c_logistic_regression_model_builder * c_ptr
    cdef data_management_NumericTablePtr numTableBeta

    def __cinit__(self, size_t n_features, size_t n_classes):
        self.c_ptr = new c_logistic_regression_model_builder(n_features, n_classes)

    def __dealloc__(self):
        del self.c_ptr

    def set_beta(self, beta, intercept):
        '''
        Concatenate beta and intercept, convert to daal4py model

        :param beta: beta from scikit-learn model
        :param intercept: intercept from scikit-learn model
        '''
        if numpy.any(intercept):
            beta_for_daal = intercept.reshape(-1, 1)
            beta_for_daal = numpy.concatenate((beta_for_daal, beta), axis=1)
        else :
            beta_for_daal = beta
        numTableBeta = getTable(data_or_file(<PyObject*>beta_for_daal))
        return self.c_ptr.setBeta(numTableBeta)

    @property
    def model(self):
        '''
        Get built model

        :rtype: logistic_regression_model
        '''
        cdef logistic_regression_model res = logistic_regression_model.__new__(logistic_regression_model)
        res.c_ptr = get_logistic_regression_model_builder_model(self.c_ptr)
        return res
