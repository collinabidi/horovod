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

#ifndef HOROVOD_SHMEM_CONTEXT_H
#define HOROVOD_SHMEM_CONTEXT_H

#include <iostream>
#include <memory>
#include <vector>

#include "../common.h"
#include "../half.h"
#include "../logging.h"

#include "shmem.h"

namespace horovod {
namespace common {

// Base class for managing SHMEM environment.
class SHMEMContextManager {
public:
  // Initialize SHMEM environment
  virtual void EnvInitialize();

  // Finalize SHMEM environment.
  virtual void EnvFinalize();
};

struct SHMEMContext {

  void Enable() {
    enabled_ = true;
    LOG(DEBUG) << "SHMEM context enabled.";
  };

  bool IsEnabled() { return enabled_; }

  // Take an argument of context manager pointer that will take care of
  // initialization of SHMEM environment.
  void Initialize();

  // Take an argument of context manager pointer that will take care of
  // finalization of SHMEM environment.
  void Finalize(SHMEMContextManager& ctx_manager);
  SHMEM_Datatype GetSHMEMDataType(std::shared_ptr<Tensor> tensor);

  SHMEM_Datatype GetSHMEMDataType(DataType dtype);

  SHMEM_Op GetSHMEMSumOp(DataType dtype);

  int GetSHMEMTypeSize(DataType dtype);

  // Flag indicating whether shmem is enabled.
  bool enabled_ = false;

  // Custom SHMEM synchronization variables
  int pWrk_int[SHMEM_REDUCE_SYNC_SIZE];
  float pWrk_float[SHMEM_REDUCE_SYNC_SIZE];
  double pWrk_double[SHMEM_REDUCE_SYNC_SIZE];
  static long pSync[SHMEM_BCAST_SYNC_SIZE];

};

} // namespace common
} // namespace horovod

#endif // HOROVOD_SHMEM_CONTEXT_H
