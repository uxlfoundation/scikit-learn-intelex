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

#ifndef _DIST_KMEANS_INCLUDED_
#define _DIST_KMEANS_INCLUDED_

#include <cmath>
#include "dist_custom.h"
#include "map_reduce_tree.h"

namespace dist_custom
{

template <typename fptype, daal::algorithms::kmeans::Method method>
class dist_custom<kmeans_manager<fptype, method> >
{
public:
    typedef kmeans_manager<fptype, method> Algo;

    /*
      We basically iterate over a map_reduce_star.
      Step1 requires 2 inputs: the data per node and the current centroids.
      step2 combines in a reduce_star partial results of step1 and produces new centroids.
      except for first iter we need to bcast centroids.
      root detects convergence/end of iteration and bcasts to others
     */
    template <typename T1, typename T2>
    typename Algo::iomstep2Master__final_type::result_type static map_reduce(Algo & algo, const T1 & input1, const T2 & input2)
    {
        auto tcvr    = get_transceiver();
        T2 centroids = input2;
        bool done    = false;
        typename Algo::iomstep2Master__final_type::result_type fres;
        size_t iter              = 0;
        double goal              = std::numeric_limits<double>::max();
        double accuracyThreshold = use_default(algo._accuracyThreshold) ? typename Algo::algob_type::ParameterType(algo._nClusters, algo._maxIterations).accuracyThreshold : algo._accuracyThreshold;
        do
        {
            if (iter) tcvr->bcast(centroids);
            ++iter;
            auto s1_result = algo.run_step1Local(input1, centroids);
            // reduce all partial results
            auto pres = map_reduce_tree::map_reduce_tree<Algo>::reduce(algo, s1_result);
            // finalize and check convergence/end of iteration
            if (tcvr->me() == 0)
            {
                fres = algo.run_step2Master__final(std::vector<daal::algorithms::kmeans::PartialResultPtr>(1, pres));
                // now check if we convered/reached max_iter
                if (iter < algo._maxIterations)
                {
                    double new_goal = fres->get(daal::algorithms::kmeans::objectiveFunction)->daal::data_management::NumericTable::template getValue<double>(0, 0);
                    if (std::abs(goal - new_goal) > accuracyThreshold)
                    {
                        centroids = fres->get(daal::algorithms::kmeans::centroids);
                        goal      = new_goal;
                        tcvr->bcast(done);
                        continue;
                    }
                }
                // when we get here we either reached maxIter or desired accuracy
                done = true;
                // we have to provide the number of iterations in result
                daal::data_management::NumericTablePtr nittab(new daal::data_management::HomogenNumericTable<int>(1, 1, daal::data_management::NumericTable::doAllocate, static_cast<int>(iter)));
                fres->set(daal::algorithms::kmeans::nIterations, nittab);
            }
            // root gets here if done, other ranks always
            tcvr->bcast(done);
        } while (done == false);
        // bcast final result
        tcvr->bcast(fres);
        return fres;
    }

    template <typename... Ts>
    static typename Algo::iomstep2Master__final_type::result_type compute(Algo & algo, const Ts &... inputs)
    {
        return map_reduce(algo, get_table(inputs)...);
    }
};

} // namespace dist_custom

#endif // _DIST_KMEANS_INCLUDED_
