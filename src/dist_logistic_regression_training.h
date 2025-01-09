/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef _DIST_LOG_REG_INCLUDED_
#define _DIST_LOG_REG_INCLUDED_

#include "dist_custom.h"
#include "transceiver.h"
#include <algorithm>
#include <vector>

template <typename fptype, daal::algorithms::logistic_regression::training::Method method>
struct logistic_regression_training_manager;

namespace dist_custom
{

// We need our own Model-class which needs to be derived from DAAL's
// because we cannot create the actual Model from the outside
class LRModel : public daal::algorithms::logistic_regression::Model
{
public:
    explicit LRModel(const daal::data_management::NumericTablePtr & coefs) : coefs_(coefs) {}

    size_t getNumberOfBetas() const override { return coefs_->getNumberOfColumns(); }

    size_t getNumberOfFeatures() const override { return coefs_->getNumberOfColumns() - 1; }

    bool getInterceptFlag() const override
    {
        return true; //FIXME
    }

    daal::data_management::NumericTablePtr getBeta() override { return coefs_; }

    const daal::data_management::NumericTablePtr getBeta() const override { return coefs_; }

private:
    daal::data_management::NumericTablePtr coefs_;
};

// custom distribution class for logistic regression
template <typename fptype, daal::algorithms::logistic_regression::training::Method method>
class dist_custom<logistic_regression_training_manager<fptype, method> >
{
public:
    typedef logistic_regression_training_manager<fptype, method> Algo;

    // the "map" phase: computing loss/gradient
    static daal::data_management::NumericTablePtr compute_loss(Algo & algo, const daal::data_management::NumericTablePtr & x, const daal::data_management::NumericTablePtr & y, daal::data_management::NumericTablePtr & weights)
    {
        // we use logistic_loss directly
        auto loss = daal::algorithms::optimization_solver::logistic_loss::Batch<>::create(x->getNumberOfRows());
        // set parameters as in our logistic regression class
        loss->parameter().interceptFlag = true; // FIXME algo._interceptFlag;
        // FIXME loss->parameter().penaltyL1 = algo._penaltyL1; raises an error
        // FIXME loss->parameter().penaltyL2 = algo._penaltyL2; raises an error
        loss->parameter().resultsToCompute = daal::algorithms::optimization_solver::objective_function::value | daal::algorithms::optimization_solver::objective_function::gradient;
        loss->input.set(daal::algorithms::optimization_solver::logistic_loss::data, x);
        loss->input.set(daal::algorithms::optimization_solver::logistic_loss::dependentVariables, y);
        loss->input.set(daal::algorithms::optimization_solver::logistic_loss::argument, weights);

        loss->compute();

        return loss->getResult()->get(daal::algorithms::optimization_solver::objective_function::gradientIdx);
    };

    static daal::data_management::NumericTablePtr column_to_row(const daal::data_management::NumericTablePtr & x)
    {
        const auto x_array = daal::data_management::HomogenNumericTable<>::cast(x)->getArraySharedPtr();
        return daal::data_management::HomogenNumericTable<>::create(x_array, x->getNumberOfRows(), x->getNumberOfColumns());
    }

    static typename Algo::iomb_type::result_type map_reduce(Algo & algo, const daal::data_management::NumericTablePtr & x, const daal::data_management::NumericTablePtr & y, size_t epoch_number, float learning_rate)
    {
        int rank, nRanks;
        MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        /* In case of logistic_loss and interceptFlag == true,
             * the number of weights is nFeatures + 1 */
        // FIXME: else?
        size_t N             = x->getNumberOfColumns() + 1;
        fptype * weights_ptr = static_cast<fptype *>(daal::services::daal_malloc(N * sizeof(fptype)));
        std::fill_n(weights_ptr, N, 1e-2);
        daal::data_management::NumericTablePtr weights(new daal::data_management::HomogenNumericTable<fptype>(weights_ptr, 1, N));

        for (size_t e = 0; e < epoch_number; e++)
        {
            for (size_t b = 0; b < 1; b++)
            { // FIXME iterate over batches
                const auto gradient_tbl = compute_loss(algo, x, y, weights);
                fptype * gradient       = daal::services::dynamicPointerCast<daal::data_management::HomogenNumericTable<fptype> >(gradient_tbl)->getArray();
                get_transceiver()->allreduce(gradient, N, MPI_SUM); // in-place all-reduce, result available on all proces
                // we compute the "reduce" on all processes (we want the result on all anyway)
                for (int i = 0; i < N; i++)
                {                                                             // FIXME: #weights == #gradient?
                    weights_ptr[i] -= learning_rate * (gradient[i] / nRanks); // div by nRanks because we need avrg
                }
            }
        }

        const auto weights_T = column_to_row(weights);
        auto model           = daal::algorithms::logistic_regression::ModelPtr(new LRModel(weights_T));
        auto result          = daal::algorithms::logistic_regression::training::ResultPtr(new daal::algorithms::logistic_regression::training::Result);
        result->set(daal::algorithms::classifier::training::model, model);

        return result;
    }

    static typename Algo::iomb_type::result_type compute(Algo & algo, const data_or_file & x, const data_or_file & y)
    {
        get_transceiver()->init();
        return map_reduce(algo, get_table(x), get_table(y), 50 /* FIXME */, 0.00001 /* FIXME */);
    }
};

} // namespace dist_custom
#endif // _DIST_LOG_REG_INCLUDED_
