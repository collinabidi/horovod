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

// SHMEM Datatype and Operation type declarations
typedef enum _SHMEM_DataType {
  SHMEM_DATATYPE_NULL          = 0x0c000000,
  SHMEM_CHAR                   = 0x4c000101,
  SHMEM_UNSIGNED_CHAR          = 0x4c000102,
  SHMEM_SHORT                  = 0x4c000203,
  SHMEM_UNSIGNED_SHORT         = 0x4c000204,
  SHMEM_INT                    = 0x4c000405,
  SHMEM_UNSIGNED               = 0x4c000406,
  SHMEM_LONG                   = 0x4c000407,
  SHMEM_UNSIGNED_LONG          = 0x4c000408,
  SHMEM_LONG_LONG_INT          = 0x4c000809,
  SHMEM_LONG_LONG              = SHMEM_LONG_LONG_INT,
  SHMEM_FLOAT                  = 0x4c00040a,
  SHMEM_DOUBLE                 = 0x4c00080b,
  SHMEM_LONG_DOUBLE            = 0x4c00080c,
  SHMEM_BYTE                   = 0x4c00010d,
  SHMEM_WCHAR                  = 0x4c00020e,
  SHMEM_PACKED                 = 0x4c00010f,
  SHMEM_LB                     = 0x4c000010,
  SHMEM_UB                     = 0x4c000011,
  SHMEM_C_COMPLEX              = 0x4c000812,
  SHMEM_C_FLOAT_COMPLEX        = 0x4c000813,
  SHMEM_C_DOUBLE_COMPLEX       = 0x4c001614,
  SHMEM_C_LONG_DOUBLE_COMPLEX  = 0x4c001615,
  SHMEM_2INT                   = 0x4c000816,
  SHMEM_C_BOOL                 = 0x4c000117,
  SHMEM_SIGNED_CHAR            = 0x4c000118,
  SHMEM_UNSIGNED_LONG_LONG     = 0x4c000819,
  SHMEM_CHARACTER              = 0x4c00011a,
  SHMEM_INTEGER                = 0x4c00041b,
  SHMEM_REAL                   = 0x4c00041c,
  SHMEM_LOGICAL                = 0x4c00041d,
  SHMEM_COMPLEX                = 0x4c00081e,
  SHMEM_DOUBLE_PRECISION       = 0x4c00081f,
  SHMEM_2INTEGER               = 0x4c000820,
  SHMEM_2REAL                  = 0x4c000821,
  SHMEM_DOUBLE_COMPLEX         = 0x4c001022,
  SHMEM_2DOUBLE_PRECISION      = 0x4c001023,
  SHMEM_2COMPLEX               = 0x4c001024,
  SHMEM_2DOUBLE_COMPLEX        = 0x4c002025,
  SHMEM_REAL2                  = SHMEM_DATATYPE_NULL,
  SHMEM_REAL4                  = 0x4c000427,
  SHMEM_COMPLEX8               = 0x4c000828,
  SHMEM_REAL8                  = 0x4c000829,
  SHMEM_COMPLEX16              = 0x4c00102a,
  SHMEM_REAL16                 = SHMEM_DATATYPE_NULL,
  SHMEM_COMPLEX32              = SHMEM_DATATYPE_NULL,
  SHMEM_INTEGER1               = 0x4c00012d,
  SHMEM_COMPLEX4               = SHMEM_DATATYPE_NULL,
  SHMEM_INTEGER2               = 0x4c00022f,
  SHMEM_INTEGER4               = 0x4c000430,
  SHMEM_INTEGER8               = 0x4c000831,
  SHMEM_INTEGER16              = SHMEM_DATATYPE_NULL,
  SHMEM_INT8_T                 = 0x4c000133,
  SHMEM_INT16_T                = 0x4c000234,
  SHMEM_INT32_T                = 0x4c000435,
  SHMEM_INT64_T                = 0x4c000836,
  SHMEM_UINT8_T                = 0x4c000137,
  SHMEM_UINT16_T               = 0x4c000238,
  SHMEM_UINT32_T               = 0x4c000439,
  SHMEM_UINT64_T               = 0x4c00083a,
  SHMEM_AINT                   = 0x4c00083b,
  SHMEM_OFFSET                 = 0x4c00083c,
  SHMEM_FLOAT_INT              = 0x8c000000,
  SHMEM_DOUBLE_INT             = 0x8c000001,
  SHMEM_LONG_INT               = 0x8c000002,
  SHMEM_SHORT_INT              = 0x8c000003,
  SHMEM_LONG_DOUBLE_INT        = 0x8c000004
} SHMEM_DataType;

typedef enum _SHMEM_Op {
  SHMEM_OP_NULL  = 0x18000000,
  SHMEM_MAX      = 0x58000001,
  SHMEM_MIN      = 0x58000003,
  SHMEM_SUM      = 0x58000003,
  SHMEM_PROD     = 0x58000004,
  SHMEM_LAND     = 0x58000005,
  SHMEM_BAND     = 0x58000006,
  SHMEM_LOR      = 0x58000007,
  SHMEM_BOR      = 0x58000008,
  SHMEM_LXOR     = 0x58000009,
  SHMEM_BXOR     = 0x5800000a,
  SHMEM_MINLOC   = 0x5800000b,
  SHMEM_MAXLOC   = 0x5800000c,
  SHMEM_REPLACE  = 0x5800000d
} SHMEM_Op;

struct SHMEMContext {

  void Enable() {
    enabled_ = true;
    LOG(DEBUG) << "SHMEM context enabled.";
  };

  bool IsEnabled() { return enabled_; }

  // Take an argument of context manager pointer that will take care of
  // initialization of SHMEM environment.
  // Take an argument of context manager pointer that will take care of
  // initialization of SHMEM environment.
  void Initialize();

  // Take an argument of context manager pointer that will take care of
  // finalization of SHMEM environment.
  void Finalize();
  SHMEM_DataType GetSHMEMDataType(std::shared_ptr<Tensor> tensor);

  SHMEM_DataType GetSHMEMDataType(DataType dtype);

  int GetSHMEMTypeSize(DataType dtype);

  // Flag indicating whether shmem is enabled.
  bool enabled_ = false;
  // Flag indicating whether or not shmem_init() has been called
  bool shmem_initialized = false;
  // Whether shmem context should be finalize.
  bool should_finalize = false;

  // Custom SHMEM synchronization variables
  static int pWrk_int[SHMEM_REDUCE_SYNC_SIZE];
  static float pWrk_float[SHMEM_REDUCE_SYNC_SIZE];
  static double pWrk_double[SHMEM_REDUCE_SYNC_SIZE];
  static long pSync[SHMEM_BCAST_SYNC_SIZE];
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_SHMEM_CONTEXT_H
