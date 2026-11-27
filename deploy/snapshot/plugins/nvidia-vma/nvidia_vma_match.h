// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef SNAPSHOT_NVIDIA_VMA_MATCH_H
#define SNAPSHOT_NVIDIA_VMA_MATCH_H

/*
 * Returns 1 when dev_major is the major of an NVIDIA character device whose
 * mappings CRIU cannot dump as regular pages. A negative major means the
 * device node was not found and is ignored.
 *
 * The agent owns CUDA state through cuda-checkpoint-helper; this plugin only
 * tells CRIU to keep the VMA metadata and leave the mapping to the driver's
 * restore. It deliberately does not register the CUDA plugin's pause,
 * checkpoint, or resume hooks, which would run a second checkpoint against a
 * process the agent already checkpointed.
 */
static inline int
snapshot_nvidia_vma_handles_major(int dev_major, int nvidia_major, int uvm_major)
{
  return (nvidia_major >= 0 && dev_major == nvidia_major) ||
         (uvm_major >= 0 && dev_major == uvm_major);
}

#endif
