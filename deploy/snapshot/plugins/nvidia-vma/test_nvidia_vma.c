// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <stdio.h>

#include "nvidia_vma_match.h"

int
main(void)
{
  /* /dev/nvidiactl and /dev/nvidiaN share major 195. */
  assert(snapshot_nvidia_vma_handles_major(195, 195, 499));
  /* /dev/nvidia-uvm is major 499. */
  assert(snapshot_nvidia_vma_handles_major(499, 195, 499));
  /* A regular or unrelated device is not claimed. */
  assert(!snapshot_nvidia_vma_handles_major(1, 195, 499));
  /* Missing device nodes are ignored, not treated as major 0. */
  assert(!snapshot_nvidia_vma_handles_major(0, -1, -1));
  assert(!snapshot_nvidia_vma_handles_major(499, 195, -1));

  puts("nvidia vma match: ok");
  return 0;
}
