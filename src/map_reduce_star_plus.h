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

#ifndef _MAP_REDUCE_STAR_PLUS_INCLUDED_
#define _MAP_REDUCE_STAR_PLUS_INCLUDED_

#include "transceiver.h"

namespace map_reduce_star_plus
{

template <typename Algo>
class map_reduce_star_plus
{
public:
    template <typename... Ts>
    typename Algo::iomstep3Local_type::result_type static map_reduce(Algo & algo, Ts &... inputs)
    {
        auto tcvr = get_transceiver();

        // run step1 and gather all partial results
        auto s1Res         = algo.run_step1Local(inputs...);
        auto s1OutForStep2 = s1Res->get(algo.outputOfStep1ForStep2);
        auto s2InFromStep1 = tcvr->gather(s1OutForStep2);

        typename Algo::iomstep2Master_type::result_type s2Res;
        const int S23TAG = 4004;
        daal::data_management::DataCollectionPtr inputOfStep3FromStep2;
        if (tcvr->me() == 0)
        {
            s2Res = algo.run_step2Master(s2InFromStep1);
            // get intputs for step3 and send them to all processes
            auto outputOfStep2ForStep3 = std::get<1>(s2Res)->get(algo.outputOfStep2ForStep3);
            inputOfStep3FromStep2      = daal::services::staticPointerCast<daal::data_management::DataCollection>((*outputOfStep2ForStep3)[0]);
            for (size_t i = 1; i < tcvr->nMembers(); i++)
            {
                tcvr->send((*outputOfStep2ForStep3)[i], i, S23TAG);
            }
        }
        else
        {
            inputOfStep3FromStep2 = tcvr->recv<daal::data_management::DataCollectionPtr>(0, S23TAG);
        }

        // bcast result of step2 to all
        auto result = std::get<0>(s2Res);
        tcvr->bcast(result);

        // perform step3
        auto inputOfStep3FromStep1 = s1Res->get(algo.outputOfStep1ForStep3);
        auto step3Output           = algo.run_step3Local(inputOfStep3FromStep1, inputOfStep3FromStep2);

        // add result of step3
        result->set(algo.step3Res, step3Output->get(algo.step3Res));

        return result;
    }

    template <typename... Ts>
    static typename Algo::iomstep3Local_type::result_type compute(Algo & algo, Ts &... inputs)
    {
        return map_reduce(algo, get_table(inputs)...);
    }
};

} // namespace map_reduce_star_plus

#endif // _MAP_REDUCE_STAR_PLUS_INCLUDED_
