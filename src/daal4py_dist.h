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

#ifndef _HLAPI_DISTR_H_INCLUDED_
#define _HLAPI_DISTR_H_INCLUDED_

#ifdef _WIN32
    #define NOMINMAX
#endif
#include "daal4py.h"

#ifdef _DIST_
    #include <tuple>

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// Input/Output manager for simple algos with a single fixed result-component
// abstracts from input/output types
// also defines how to get results and finalize
template <typename A, typename O, typename E, int P>
struct IOManagerSingle : public IOManager<A, O>
{
    static O getResult(A & algo) { return algo.getResult()->get(static_cast<E>(P)); }
};

// Input/Output manager for intermediate steps
// abstracts from input/output types
// also defines how to get results and finalize
template <typename A, typename O>
struct PartialIOManager
{
    typedef O result_type;

    static result_type getResult(A & algo) { return daal::services::staticPointerCast<typename result_type::ElementType>(algo.getPartialResult()); }
    static bool needsFini() { return false; }
};

// Input/Output manager for intermediate steps with a single fixed result-component
// abstracts from input/output types
// also defines how to get results and finalize
template <typename A, typename O, typename E, int P>
struct PartialIOManagerSingle : public PartialIOManager<A, O>
{
    static typename PartialIOManager<A, O>::result_type getResult(A & algo) { return algo.getPartialResult()->get(static_cast<E>(P)); }
};

// Input/Output manager for intermediate steps, output is a tuple of Result and PartialResult of algo step
// abstracts from input/output types
// also defines how to get results and finalize
template <typename A, typename O1, typename O2>
struct DoubleIOManager
{
    typedef std::tuple<O1, O2> result_type;

    static result_type getResult(A & algo) { return std::make_tuple(algo.getResult(), algo.getPartialResult()); }
    static bool needsFini() { return false; }
};

    //////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////

    #include "map_reduce_star.h"
    #include "map_reduce_star_plus.h"
    #include "map_reduce_tree.h"
    #include "dist_custom.h"

#endif // _DIST_
#endif // _HLAPI_DISTR_H_INCLUDED_
