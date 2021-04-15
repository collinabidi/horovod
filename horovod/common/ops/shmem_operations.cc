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

#include "shmem_operations.h"
#include "../shmem/shmem_context.h"
#include <shmem.h>

#define SHMEM_SUCCESS 0

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
  case HOROVOD_INT8:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_INT8\n", __LINE__);
    return SHMEM_INT8_T;
  case HOROVOD_INT16:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_INT16\n", __LINE__);
    return SHMEM_INT16_T;
  case HOROVOD_INT32:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_INT32\n", __LINE__);
    return SHMEM_INT32_T;
  case HOROVOD_INT64:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_INT64\n", __LINE__);
    return SHMEM_INT64_T;
  case HOROVOD_UINT8:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_UINT8\n", __LINE__);
    return SHMEM_UINT8_T;
  case HOROVOD_UINT16:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_UINT16\n", __LINE__);
    return SHMEM_UINT16_T;
  case HOROVOD_FLOAT32:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_FLOAT32\n", __LINE__);
    return SHMEM_FLOAT;
  case HOROVOD_FLOAT64:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_FLOAT64\n", __LINE__);
    return SHMEM_DOUBLE;
  case HOROVOD_BOOL:
    fprintf(stderr, "[DEBUG][%d] HOROVOD_DATATYPE -> HOROVOD_BOOL\n", __LINE__);
    return SHMEM_C_BOOL;
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
    case SHMEM_INT32_T:
      return sizeof(int32_t);
    case SHMEM_INT64_T:
      return sizeof(int64_t);
    case SHMEM_FLOAT:
      return sizeof(float_t);
    case SHMEM_DOUBLE:
      return sizeof(double_t);
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                            " is not supported in SHMEM mode.");
  }
}

SHMEMAllreduce::SHMEMAllreduce(SHMEMContext* shmem_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), shmem_context_(shmem_context) {}

Status SHMEMAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  // SHMEM variable initialization
  int world_size = shmem_n_pes();
  int world_rank = shmem_my_pe();

  int pWrk_int[SHMEM_REDUCE_SYNC_SIZE];
  float pWrk_float[SHMEM_REDUCE_SYNC_SIZE];
  double pWrk_double[SHMEM_REDUCE_SYNC_SIZE];
  long pSync[SHMEM_REDUCE_SYNC_SIZE];
  for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }

  const void* fused_input_data;
  void* buffer_data;
  void* symmetric_buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy tensors into the symmetric memory.
  // Note: naive assumption that all tensors will have same datatype
  auto& timeline = global_state_->timeline;
  int element_size = shmem_context_->GetSHMEMTypeSize(first_entry.tensor->dtype());
  auto dtype = shmem_context_->GetSHMEMDataType(first_entry.tensor->dtype());

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  // Copy memory from fusion buffer to symmetric memory (temporary intermediate step)
  symmetric_buffer_data = (void*) shmem_malloc(buffer_len);

  const void* sendbuf = entries.size() > 1 || fused_input_data == buffer_data
                        ? symmetric_buffer_data : fused_input_data;
  void* symmetric_sendbuf = (void*) shmem_malloc(buffer_len);
  memcpy(symmetric_sendbuf, sendbuf, buffer_len);

  // Do allreduce.
  shmem_barrier_all();
  timeline.ActivityStartAll(entries, SHMEM_ALLREDUCE);
  if (dtype == SHMEM_INT) {
    shmem_int_sum_to_all((int*)(symmetric_buffer_data), (int*)symmetric_sendbuf, num_elements, 0, 0, world_size, pWrk_int, pSync);
  }
  else if (dtype == SHMEM_FLOAT) {
    shmem_float_sum_to_all((float *)(symmetric_buffer_data), (float*)symmetric_sendbuf, num_elements, 0, 0, world_size, pWrk_float, pSync);
  }
  else if (dtype == SHMEM_DOUBLE) {
    shmem_double_sum_to_all((double *)(symmetric_buffer_data), (double*)symmetric_sendbuf, num_elements, 0, 0, world_size, pWrk_double, pSync);
  }
  else {
    throw std::logic_error("ALLREDUCE: REEE not done with typecasting yet ");
  }
  timeline.ActivityEndAll(entries);
  shmem_barrier_all();

  // Copy memory from symmetric back to the local variables
  memcpy(buffer_data, symmetric_buffer_data, buffer_len);
  shmem_barrier_all();

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  shmem_barrier_all();

  // Free symmetric variables
  shmem_free(symmetric_buffer_data);
  shmem_free(symmetric_sendbuf);

  // Return OK
  return Status::OK();
}

bool SHMEMAllreduce::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

SHMEMAllgather::SHMEMAllgather(SHMEMContext* shmem_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), shmem_context_(shmem_context) {}

bool SHMEMAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

Status SHMEMAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  int global_size = global_state_->controller->GetSize();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    /* Cleanup */
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      delete[] entry_component_sizes[ec];
      delete[] entry_component_offsets[ec];
    }   
    delete[] entry_component_sizes;
    delete[] entry_component_offsets;
    delete[] recvcounts;
    delete[] displcmnts;
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = shmem_context_->GetSHMEMTypeSize(first_entry.tensor->dtype());

  const void* sendbuf = nullptr;
  void* symmetric_sendbuf;
  void* buffer_data;
  void* symmetric_buffer_data;
  int64_t total_num_elements = NumElements(entries);
  int world_size = shmem_n_pes();
  int world_rank = shmem_my_pe();
  auto dtype = shmem_context_->GetSHMEMDataType(first_entry.tensor->dtype());
  int64_t offset;

  // SHMEM variable initialization
  long pSync[SHMEM_ALLTOALL_SYNC_SIZE];
  int pWrk_int[SHMEM_REDUCE_SYNC_SIZE];
  float pWrk_float[SHMEM_REDUCE_SYNC_SIZE];
  double pWrk_double[SHMEM_REDUCE_SYNC_SIZE];
  for (int i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }  

  if (entries.size() > 1) {
    // Copy memory to fusion buffer
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    timeline.ActivityEndAll(entries);
  } else {
    // Create single-entry buffers
    sendbuf = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
  }

  // Allocate symmetric variable with the calculated offset size and copy to it from buffer_data
  int64_t buffer_len = offset - displcmnts[global_state_->controller->GetRank()] * element_size;
  symmetric_buffer_data = (void*) shmem_malloc(buffer_len);
  offset = displcmnts[global_state_->controller->GetRank()] * element_size;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    void* symmetric_buffer_data_at_offset = (uint8_t*)symmetric_buffer_data + offset;
    memcpy(symmetric_buffer_data_at_offset, buffer_data_at_offset, (size_t)e.tensor->size());
    offset += e.tensor->size();
  }

  // Create symmetric sendbuf
  if (sendbuf != nullptr) {
    symmetric_sendbuf = (void*) shmem_malloc(first_entry.tensor->size());
    memcpy(symmetric_sendbuf, sendbuf, (size_t)(first_entry.tensor->size()));
  } else {
    symmetric_sendbuf = (void*) shmem_malloc(buffer_len);
    memcpy(symmetric_sendbuf, symmetric_buffer_data, (size_t)buffer_len);
  }

  global_state_->timeline.ActivityStartAll(entries, SHMEM_ALLGATHER);
  
 switch (dtype) {
    case SHMEM_INT32_T:
      shmem_collect32(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pSync);
    case SHMEM_UINT32_T:
      shmem_collect32(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pSync);
    case SHMEM_FLOAT:
      shmem_fcollect64(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pSync);
    case SHMEM_INT64_T:
      shmem_collect64(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pSync);
    case SHMEM_UINT64_T:
      shmem_collect64(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pSync);
    case SHMEM_DOUBLE:
      shmem_fcollect64(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pSync);
    default:
      throw std::logic_error("Allgather: REEE Not done with typecasting yet ");
  }


  global_state_->timeline.ActivityEndAll(entries);

  // Copy memory from symmetric back to the local variables
  memcpy(symmetric_buffer_data, buffer_data, buffer_len);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    timeline.ActivityEndAll(entries);
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  // Free symmetric memory
  shmem_free(symmetric_buffer_data);
  shmem_free(symmetric_sendbuf);

  // Wait until all are done
  shmem_barrier_all();

  // Return OK
  return Status::OK();
}

bool SHMEMHierarchicalAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return false;
}

SHMEMBroadcast::SHMEMBroadcast(SHMEMContext* shmem_context, HorovodGlobalState* global_state)
    : BroadcastOp(global_state), shmem_context_(shmem_context) {}

Status SHMEMBroadcast::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  //std::cout << "SHMEM Broadcast" << std::endl;

  // SHMEM variable initialization
  int pWrk_int[SHMEM_BCAST_SYNC_SIZE];
  float pWrk_float[SHMEM_BCAST_SYNC_SIZE];
  double pWrk_double[SHMEM_BCAST_SYNC_SIZE];
  long pSync[SHMEM_BCAST_SYNC_SIZE];
  for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i++) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }

  int world_rank = shmem_my_pe();
  int world_size = shmem_n_pes();
  const void* data_ptr = nullptr;

  // Dynamically shmallocate tensors in shared memory as symmetric variables.
  // This is inefficient, since we'll have to do this every single time we call broadcast
  // but it's good enough for now.
  auto dtype = shmem_context_->GetSHMEMDataType(e.tensor->dtype());
  if (dtype ==  SHMEM_INT32_T) {
    data_ptr = (void*) shmem_malloc(sizeof(int32_t) * e.tensor->shape().num_elements());
  }
  else if (dtype == SHMEM_INT64_T) {
    data_ptr = (void*) shmem_malloc(sizeof(int64_t) * e.tensor->shape().num_elements());
  }
  else if (dtype == SHMEM_FLOAT) {
    data_ptr = (void*) shmem_malloc(sizeof(float_t) * e.tensor->shape().num_elements());
  }
  else if (dtype == SHMEM_DOUBLE) {
    data_ptr = (void*) shmem_malloc(sizeof(double_t) * e.tensor->shape().num_elements());
  }
  else {
    throw std::logic_error("BROADCAST: REEE Not done with typecasting yet");
  }

  // Fill the symmetric memory with data if on the main PE
  if (global_state_->controller->GetRank() == e.root_rank) {
    memcpy(const_cast<void*>(data_ptr), e.tensor->data(), (size_t)e.tensor->shape().num_elements());
  } else {
    memcpy(const_cast<void*>(data_ptr), e.output->data(), (size_t)e.output->shape().num_elements());
  }

  // Barrier
  shmem_barrier_all();

  // Perform broadcast
  global_state_->timeline.ActivityStartAll(entries, SHMEM_BCAST);
  if (dtype == SHMEM_INT32_T) {
    shmem_broadcast32(const_cast<void*>(data_ptr), data_ptr, (size_t)e.tensor->shape().num_elements(), e.root_rank, 0, 0, world_size, pSync);
  }
  else if (dtype == SHMEM_INT64_T) {
    shmem_broadcast64(const_cast<void*>(data_ptr), data_ptr, (size_t)e.tensor->shape().num_elements(), e.root_rank, 0, 0, world_size, pSync);
  }
  else if (dtype == SHMEM_FLOAT) {
    shmem_broadcast32(const_cast<void*>(data_ptr), data_ptr, (size_t)e.tensor->shape().num_elements(), e.root_rank, 0, 0, world_size, pSync);
  }
  else if (dtype == SHMEM_DOUBLE) {
    shmem_broadcast64(const_cast<void*>(data_ptr), data_ptr, (size_t)e.tensor->shape().num_elements(), e.root_rank, 0, 0, world_size, pSync);
  }
  else {
    throw std::logic_error("BROADCAST: REEE not done with typecasting yet on actual broadcast");
  }
  global_state_->timeline.ActivityEndAll(entries);

  // Copy results back to the local variables from symmetric memory
  if (global_state_->controller->GetRank() == e.root_rank) {
    memcpy((uint8_t*)e.tensor->data(), data_ptr, (size_t)e.tensor->shape().num_elements());
  } else {
    memcpy((uint8_t*)e.output->data(), data_ptr, (size_t)e.output->shape().num_elements());
  }

  // Deallocate memory
  shmem_free(const_cast<void*>(data_ptr));

  // Wait until done
  shmem_barrier_all();

  // Return OK
  return Status::OK();
}

bool SHMEMBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
