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

#ifndef HOROVOD_SHMEM_OPERATIONS_H
#define HOROVOD_SHMEM_OPERATIONS_H

#include <iostream>

#include "shmem.h"

#include "collective_operations.h"
#include "../common.h"
#include "../global_state.h"
#include "../shmem/shmem_context.h"

namespace horovod {
namespace common {

// AllreduceOp parameters: 
class SHMEMAllreduce : public AllreduceOp {
public:
  SHMEMAllreduce(SHMEMContext* shmem_context, HorovodGlobalState* global_state);

  virtual ~SHMEMAllreduce() = default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  SHMEMContext* shmem_context_;
};

class SHMEMAllgather : public AllgatherOp {
public:
  SHMEMAllgather(SHMEMContext* shmem_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  SHMEMContext* shmem_context_;
};

class SHMEMHierarchicalAllgather : public SHMEMAllgather {
public:
  SHMEMHierarchicalAllgather(SHMEMContext* shmem_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

private:
  void Barrier();
};

class SHMEMBroadcast : public BroadcastOp {
public:
  SHMEMBroadcast(SHMEMContext* shmem_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, 
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  SHMEMContext* shmem_context_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_SHMEM_OPERATIONS_H
