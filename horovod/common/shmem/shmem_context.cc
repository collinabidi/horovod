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

SHMEM_DataType SHMEMContext::GetSHMEMDataType(const std::shared_ptr<Tensor> tensor) {
  return GetSHMEMDataType(tensor->dtype());
}

SHMEM_DataType SHMEMContext::GetSHMEMDataType(const DataType dtype) {
  // As of OpenMPI 5.0.0 which implements the OpenSHMEM 1.4 specification, SHMEM can only support collective
  // operations with 32- and 64-bit datatypes. Creating a custom method of concatenating data into 32-bit or 
  // 64-bit arrays is on the to-do list.
  switch (dtype) {
  case HOROVOD_UINT8:
    return SHMEM_UINT8_T;
  case HOROVOD_INT8:
    return SHMEM_INT8_T;
  case HOROVOD_UINT16:
    //return SHMEM_UINT16_T;
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in SHMEM mode.");
  case HOROVOD_INT16:
    //return SHMEM_INT16_T;
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in SHMEM mode.");
  case HOROVOD_INT32:
    return SHMEM_INT32_T;
  case HOROVOD_INT64:
    return SHMEM_INT64_T;
  case HOROVOD_FLOAT16:
    //return shmem_float16_t;
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in SHMEM mode.");
  case HOROVOD_FLOAT32:
    return SHMEM_FLOAT;
  case HOROVOD_FLOAT64:
    return SHMEM_DOUBLE;
  case HOROVOD_BOOL:
    //return SHMEM_C_BOOL;
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in SHMEM mode.");
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in SHMEM mode.");
  }
}

int SHMEMContext::GetSHMEMTypeSize(DataType dtype) {
  switch (GetSHMEMDataType(dtype)) {
    case SHMEM_UINT8_T:
      return sizeof(uint8_t);
    case SHMEM_INT8_T:
      return sizeof(int8_t);
    case HOROVOD_UINT16:
      //return SHMEM_UINT16_T;
      throw std::logic_error("Type " + DataType_Name(dtype) +
                            " is not supported in SHMEM mode.");
    case HOROVOD_INT16:
      //return SHMEM_INT16_T;
      throw std::logic_error("Type " + DataType_Name(dtype) +
                            " is not supported in SHMEM mode.");
    case SHMEM_INT32_T:
      return sizeof(int32_t);
    case SHMEM_INT64_T:
      return sizeof(int64_t);
    case HOROVOD_FLOAT16:
      //return shmem_float16_t;
      throw std::logic_error("Type " + DataType_Name(dtype) +
                            " is not supported in SHMEM mode.");
    case SHMEM_FLOAT:
      return sizeof(float);
    case SHMEM_DOUBLE:
      return sizeof(double);
    case HOROVOD_BOOL:
      //return SHMEM_C_BOOL;
      throw std::logic_error("Type " + DataType_Name(dtype) +
                            " is not supported in SHMEM mode.");
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                            " is not supported in SHMEM mode.");
  }
}

void SHMEMContext::Initialize(SHMEMContextManager& ctx_manager) {
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
    ctx_manager.EnvInitialize(required);
    should_finalize = true;
    shmem_initialized = true;
  }
}

void SHMEMContext::Finalize(SHMEMContextManager& ctx_manager) {
  if (!enabled_) {
    return;
  }
  if (should_finalize) {
    ctx_manager.EnvFinalize();
  }
}

void SHMEMContextManager::EnvInitialize(int shmem_threads_required) {
  int shmem_threads_provided;
  shmem_init_thread(shmem_threads_required, &shmem_threads_provided);
}

void SHMEMContextManager::EnvFinalize() {
  shmem_finalize();
}

} // namespace common
} // namespace horovod
