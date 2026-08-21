/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#pragma once

#include "services/library_version_info.h"

#define ONEDAL_VERSION    INTEL_DAAL_VERSION
#define MAJOR_VERSION     __INTEL_DAAL__
#define MINOR_VERSION     __INTEL_DAAL_MINOR__
#define UPDATE_VERSION    __INTEL_DAAL_UPDATE__
#define ONEDAL_BUILD_DATE __INTEL_DAAL_BUILD_DATE

// TEMPORARY: HDBSCAN was added to oneDAL in the middle of the 2026.2 development
// window (first nightly carrying it is 2026-08-14), so 'ONEDAL_VERSION >= 20260200'
// alone also matches oneDAL builds that do not provide the algorithm yet. Drop the
// build date part of the condition once oneDAL 2026.2 is released.
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20260200 && ONEDAL_BUILD_DATE >= 20260814
#define ONEDAL_HDBSCAN_SUPPORTED 1
#else
#define ONEDAL_HDBSCAN_SUPPORTED 0
#endif
