// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "shmem_context.h"

#include <iostream>
#include <memory>
#include <vector>

#include "../common.h"
#include "../half.h"
#include "../logging.h"

namespace horovod {
namespace common {

void SHMEMContext::Initialize() {
  if (!enabled_) {
    return;
  }
  // Initialize SHMEM if it was not initialized.
  auto shmem_threads_disable = std::getenv(HOROVOD_SHMEM_THREADS_DISABLE);
  int required = SHMEM_THREAD_MULTIPLE;
  if (shmem_threads_disable != nullptr &&
      std::strtol(shmem_threads_disable, nullptr, 10) > 0) {
    required = SHMEM_THREAD_SINGLE;
  }
  if (shmem_initialized) {
    int provided;
    shmem_query_thread(&provided);
    if (provided < SHMEM_THREAD_MULTIPLE) {
      LOG(WARNING)
          << "SHMEM has already been initialized without "
             "multi-threading support (SHMEM_THREAD_MULTIPLE). This will "
             "likely cause a segmentation fault.";
    }
  } else {
    // SHMEM environment has not been created, using manager to initialize.
    int provided;
    std::cout << "SHMEM Initialized with " << required << " threads" << std::endl;
    shmem_init_thread(required, &provided);
    should_finalize = true;
    shmem_initialized = true;
  }
}

void SHMEMContext::Finalize() {
  if (!enabled_) {
    return;
  }
  if (should_finalize) {
    shmem_finalize();
  }
}
} // namespace common
} // namespace horovod
