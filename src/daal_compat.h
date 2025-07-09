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

#ifndef _DAALCOMPAT_H_INCLUDED_
#define _DAALCOMPAT_H_INCLUDED_

#include <services/library_version_info.h>
#include <services/daal_shared_ptr.h>

// oneDAL version < 2018 is what we are looking for.
// Some oneDAL versions seem broken, e.g. '2199' so we need to check that, too
#if __INTEL_DAAL__ < 2019 || __INTEL_DAAL__ > 2100

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace iterative_solver
{
// BatchPtr typedef not existent in older oneDAL versions
typedef daal::services::SharedPtr<interface1::Batch> BatchPtr;

} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif

#endif // _DAALCOMPAT_H_INCLUDED_
