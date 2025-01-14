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

#ifndef _MAP_REDUCE_TREE_INCLUDED_
#define _MAP_REDUCE_TREE_INCLUDED_

#include "transceiver.h"

namespace map_reduce_tree
{

template <typename Algo>
class map_reduce_tree
{
public:
    static int get_power2(size_t x)
    {
        int power = 1;
        while (power < x) power *= 2;
        return power;
    }

    static typename Algo::iomstep1Local_type::result_type reduce(Algo & algo, typename Algo::iomstep1Local_type::result_type inp)
    {
        auto tcvr  = get_transceiver();
        int rank   = tcvr->me();
        int nRanks = tcvr->nMembers();

        if (nRanks == 1)
        {
            std::vector<typename Algo::iomstep1Local_type::result_type> p_results(1, inp);
            inp = algo.run_step2Master(p_results);
        }
        else
        {
            size_t N         = get_power2(nRanks);
            const int REDTAG = 5534;

            for (size_t cN = N / 2; cN > 0; cN /= 2)
            {
                if (rank >= cN)
                {
                    // Upper half of processes send their stuff to lower half
                    tcvr->send(inp, rank - cN, REDTAG);
                    break;
                }
                else if (rank + cN < nRanks)
                {
                    // lower half of processes receives message and computes partial reduction
                    std::vector<typename Algo::iomstep1Local_type::result_type> p_results(2);
                    p_results[0] = inp;
                    p_results[1] = tcvr->recv<typename Algo::iomstep1Local_type::result_type>(rank + cN, REDTAG);
                    inp          = algo.run_step2Master(p_results);
                }
            }
        }
        return inp;
    }

    template <typename... Ts>
    typename Algo::iomstep2Master__final_type::result_type static map_reduce(Algo & algo, const Ts &... inputs)
    {
        auto s1_result = algo.run_step1Local(inputs...);
        // reduce all partial results
        auto pres = reduce(algo, s1_result);
        // finalize result
        auto res = algo.run_step2Master__final(std::vector<typename Algo::iomstep2Master_type::result_type>(1, pres));
        // bcast final result
        get_transceiver()->bcast(res);
        return res;
    }

    template <typename... Ts>
    static typename Algo::iomstep2Master__final_type::result_type compute(Algo & algo, const Ts &... inputs)
    {
        return map_reduce(algo, get_table(inputs)...);
    }
};

} // namespace map_reduce_tree

#endif // _MAP_REDUCE_TREE_INCLUDED_
