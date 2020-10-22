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
#include <shmem.h>

#define SHMEM_SUCCESS 0

namespace horovod {
namespace common {

void SHMEMContext::Init() {
  // Initialize pSync values before calling shmem_init
  static long pSync[SHMEM_BCAST_SYNC_SIZE];
  for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i++)
    pSync[i] = SHMEM_SYNC_VALUE;

  LOG(DEBUG) << "Background thread start";

  // Initialize SHMEM
  shmem_init();
}

void SHMEMContext::Finalize() {
  LOG(DEBUG) << "Background thread destroy";

  // Finalize SHMEM
  shmem_finalize();
}

SHMEMAllreduce::SHMEMAllreduce(SHMEMContext* shmem_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), shmem_context_(shmem_context) {}

Status SHMEMAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  void* buffer_data;
  void* symmetric_buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy tensors into the symmetric memory.
  // Note: naive assumption that all tensors will have same datatype
  // #@#@ Need to fix this
  auto& timeline = global_state_->timeline;
  int element_size = shmem_context_->GetSHMEMTypeSize(first_entry.tensor->dtype());
  buffer_len = (size_t)(num_elements) * element_size;
  std::cout << "SHMEM AllReduce: Calculated buffer_len = " << buffer_len << std::endl;
  if (entries.size() > 1) {
    std::cout << "SHMEM AllReduce: More than one entry" << std::endl;
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    std::cout << "SHMEM AllReduce: Just one entry"
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }
  std::cout << "SHMEM AllReduce: True buffer_len = " << buffer_len << std::endl;

  symmetric_buffer_data = (void*) shmem_malloc(buffer_len);
  memcpy(symmetric_buffer_data, buffer_data, buffer_len);

  // Do allreduce.
  timeline.ActivityStartAll(entries, SHMEM_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? buffer_data : first_entry.tensor->data();
  const void* symmetric_sendbuf = (void*) shmem_malloc(buffer_len);
  memcpy(symmetric_sendbuf, sendbuf, buffer_len);
  shmem_barrier_all();
  switch (shmem_context_->GetSHMEMDataType(e.tensor->dtype())) {
    case SHMEM_INT:
      shmem_int_sum_to_all(symmetric_buffer_data, symmetric_sendbuf, (int)num_elements, 0, 0, world_size, pWrk_int, pSync);
    case SHMEM_FLOAT:
      shmem_float_sum_to_all(symmetric_buffer_data, symmetric_sendbuf, (int)num_elements, 0, 0, world_size, pWrk_float, pSync);
    case SHMEM_DOUBLE:
      shmem_double_sum_to_all(symmetric_buffer_data, symmetric_sendbuf, (int)num_elements, 0, 0, world_size, pWrk_double, pSync);
  }
  timeline.ActivityEndAll(entries);

  // Copy memory from symmetric back to the local variables
  memcpy(symmetric_buffer_data, buffer_data, buffer_len);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  // Free symmetric variables
  shmem_free(symmetric_buffer_data);
  shmem_free(symmetric_sendbuf);

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

  if (entries.size() > 1) {
    // Copy memory to fusion buffer
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    SHMEMMemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data, offset);
    timeline.ActivityEndAll(entries);
  } else {
    // Create single-entry buffers
    sendbuf = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
  }

  // Allocate symmetric variable with the calculated offset size and copy to it from buffer_data
  int64_t buffer_len = offset - displcmnts[global_state_->controller->GetRank()] * element_size;
  symmetric_buffer_data = (void*) shmem_malloc(buffer_len);
  int64_t offset = displcmnts[global_state_->controller->GetRank()] * element_size;
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
  shmem_barrier_all();
  switch (shmem_context_->GetSHMEMDataType(e.tensor->dtype())) {
    case SHMEM_INT:
      shmem_alltoall32(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pWrk_int, pSync);
    case SHMEM_FLOAT:
      shmem_alltoall32(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pWrk_float, pSync);
    case SHMEM_DOUBLE:
      shmem_alltoall64(symmetric_buffer_data, symmetric_sendbuf, (int)total_num_elements, 0, 0, world_size, pWrk_double, pSync);
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
  free(symmetric_buffer_data);
  free(symmetric_sendbuf);

  return Status::OK();
}

SHMEMHierarchicalAllgather::SHMEMHierarchicalAllgather(SHMEMContext* shmem_context,
                                                   HorovodGlobalState* global_state)
    : SHMEMAllgather(shmem_context, global_state) {}

Status SHMEMHierarchicalAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
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

  int64_t total_size = displcmnts[global_size - 1] +
                       recvcounts[global_size - 1];

  // If shared buffer is not initialized or is not large enough, reallocate
  int64_t total_size_in_bytes = total_size * element_size;
  if (global_state_->shared_buffer == nullptr || global_state_->shared_buffer_size < total_size_in_bytes) {
    if (global_state_->shared_buffer != nullptr) {
      MPI_Win_fence(0, shmem_context_->window);
      MPI_Win_free(&shmem_context_->window);
      global_state_->shared_buffer = nullptr;
    }

    // Allocate shared memory, give each rank their respective pointer
    timeline.ActivityStartAll(entries, ALLOCATE_SHARED_BUFFER);
    int64_t window_size = global_state_->controller->GetLocalRank() == 0 ? total_size_in_bytes : 0;
    MPI_Win_allocate_shared(window_size,
                            element_size,
                            SHMEM_INFO_NULL,
                            shmem_context_->GetSHMEMCommunicator(Communicator::LOCAL),
                            &global_state_->shared_buffer,
                            &shmem_context_->window);
    if (global_state_->controller->GetLocalRank() != 0) {
      int disp_unit;
      MPI_Aint winsize;
      MPI_Win_shared_query(shmem_context_->window,
                           0,
                           &winsize,
                           &disp_unit,
                           &global_state_->shared_buffer);
    }
    global_state_->shared_buffer_size = total_size_in_bytes;
    timeline.ActivityEndAll(entries);
  }

  // Compute cross-node allgather displacements and recvcounts for
  // homogeneous/parallelized case
  int cross_size = global_state_->controller->GetCrossSize();
  int local_size = global_state_->controller->GetLocalSize();
  int local_rank = global_state_->controller->GetLocalRank();
  auto* cross_recvcounts = new int[cross_size]();
  auto* cross_displcmnts = new int[cross_size]();

  if (global_state_->controller->IsHomogeneous()) {
    for (int i = 0; i < global_state_->controller->GetCrossSize(); ++i) {
      cross_recvcounts[i] = recvcounts[local_size * i + local_rank];
      cross_displcmnts[i] = displcmnts[local_size * i + local_rank];
    }
  } else if (global_state_->controller->GetLocalRank() == 0) {
    // In this case local rank 0 will allgather with all local data
    int offset = 0;
    for (int i = 0; i < cross_size; ++i) {
      for (int j = offset; j < offset + global_state_->controller->GetLocalSizeAtCrossRank(i);
           ++j) {
        cross_recvcounts[i] += recvcounts[j];
      }
      cross_displcmnts[i] = displcmnts[offset];
      offset += global_state_->controller->GetLocalSizeAtCrossRank(i);
    }
  }

  timeline.ActivityStartAll(entries, MEMCPY_IN_SHARED_BUFFER);

  int rank = global_state_->controller->GetRank();
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    void* shared_buffer_at_offset =
        (uint8_t*) global_state_->shared_buffer +
        entry_component_offsets[ec][rank] * element_size;

    // CPU copy to shared buffer
    memcpy(shared_buffer_at_offset, e.tensor->data(),
           (size_t) (entry_component_sizes[ec][rank] * element_size));
  }
  Barrier();
  timeline.ActivityEndAll(entries);

  // Perform the cross-node allgather. If the cluster is homogeneous all
  // local ranks participate, otherwise local rank 0 handles all data
  global_state_->timeline.ActivityStartAll(entries, SHMEM_CROSS_ALLGATHER);
  if (global_state_->controller->IsHomogeneous() || global_state_->controller->GetLocalRank() == 0) {
    int op = SHMEM_Allgatherv(SHMEM_IN_PLACE,
                            0,
                            SHMEM_DATATYPE_NULL,
                            global_state_->shared_buffer,
                            cross_recvcounts,
                            cross_displcmnts,
                            shmem_context_->GetSHMEMDataType(first_entry.tensor->dtype()),
                            shmem_context_->GetSHMEMCommunicator(Communicator::CROSS));
    if (op != SHMEM_SUCCESS) {
      throw std::runtime_error("SHMEM_Allgatherv failed, see SHMEM output for details.");
    }
  }
  Barrier();
  global_state_->timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
  MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                        global_state_->shared_buffer, element_size, entries);
  Barrier();
  timeline.ActivityEndAll(entries);

  // Free the buffers
  delete[] cross_displcmnts;
  delete[] cross_recvcounts;

  return Status::OK();
}

bool SHMEMHierarchicalAllgather::Enabled(const ParameterManager& param_manager,
                                       const std::vector<TensorTableEntry>& entries,
                                       const Response& response) const {
  return param_manager.HierarchicalAllgather();
}

void SHMEMHierarchicalAllgather::Barrier() {
  shmem_barrier_all();
}

SHMEMBroadcast::SHMEMBroadcast(SHMEMContext* shmem_context, HorovodGlobalState* global_state)
    : BroadcastOp(global_state), shmem_context_(shmem_context) {}

Status SHMEMBroadcast::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  std::cout << "Calling SHMEM Broadcast!" << std::endl;

  int world_rank = shmem_my_pe();
  int world_size = shmem_n_pes();
  void* data_ptr;

  // Dynamically shmallocate tensors in shared memory as symmetric variables.
  // This is inefficient, since we'll have to do this every single time we call broadcast
  // but it's good enough for now.
  switch (shmem_context_->GetSHMEMDataType(e.tensor->dtype())) {
    case HOROVOD_INT32:
      data_ptr = (void*) shmem_malloc(sizeof(int32_t) * (int)e.tensor->shape().num_elements());
    case HOROVOD_INT64:
      data_ptr = (void*) shmem_malloc(sizeof(int64_t) * (int)e.tensor->shape().num_elements());
    case HOROVOD_FLOAT32:
      data_ptr = (void*) shmem_malloc(sizeof(float_t) * (int)e.tensor->shape().num_elements());
    case HOROVOD_FLOAT64:
      data_ptr = (void*) shmem_malloc(sizeof(double_t) * (int)e.tensor->shape().num_elements());
  }

  // Fill the symmetric memory with data if on the main PE
  if (global_state_->controller->GetRank() == e.root_rank) {
    memcpy(data_ptr, e.tensor->data(), (size_t)(int)e.tensor->shape().num_elements());
  } else {
    memcpy(data_ptr, e.output->data(), (size_t)(int)e.output->shape().num_elements());
  }

  // Barrier
  shmem_barrier_all();

  global_state_->timeline.ActivityStartAll(entries, SHMEM_BCAST);

  // Perform broadcast
  if (shmem_context_->GetSHMEMDataType(e.tensor->dtype()) == HOROVOD_INT32 || shmem_context_->GetSHMEMDataType(e.tensor->dtype()) == HOROVOD_FLOAT32) {
    int op = shmem_broadcast32(data_ptr, data_ptr, (int)e.tensor->shape().num_elements(), e.root_rank, 0, 0, world_size, pSync);
    if (op != SHMEM_SUCCESS) {
      throw std::runtime_error("SHMEM_Broadcast failed, see SHMEM output for details.");
    }
  } else if (shmem_context_->GetSHMEMDataType(e.tensor->dtype()) == HOROVOD_INT64 || shmem_context_->GetSHMEMDataType(e.tensor->dtype()) == HOROVOD_FLOAT64) {
    int op = shmem_broadcast64(data_ptr, data_ptr, (int)e.tensor->shape().num_elements(), e.root_rank, 0, world_size, pSync);
    if (op != SHMEM_SUCCESS) {
      throw std::runtime_error("SHMEM_Broadcast failed, see SHMEM output for details.");
    }
  }
  global_state_->timeline.ActivityEndAll(entries);

  // Copy results back to the local variables from symmetric memory
  if (global_state_->controller->GetRank() == e.root_rank) {
    memcpy(e.tensor->data(), data_ptr, (size_t)(int)e.tensor->shape().num_elements());
  } else {
    memcpy(e.output->data(), data_ptr, (size_t)(int)e.output.shape().num_elements());
  }

  // Deallocate memory
  shmem_free(data_ptr);

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
