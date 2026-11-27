// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * snapshot_nvidia_vma is a CRIU hook-only plugin for NVIDIA character-device
 * mappings.
 *
 * vLLM and NCCL keep small shared mappings of /dev/nvidiaN open. CRIU treats
 * every non-regular mapping as unsupported unless a plugin claims it through
 * HANDLE_DEVICE_VMA, and the CUDA plugin does not register that hook. Without
 * this plugin the dump fails with:
 *
 *   handle_device_vma plugin failed: No such file or directory
 *   Can't handle non-regular mapping
 *
 * Returning 0 marks the VMA VMA_EXT_PLUGIN: CRIU saves the mapping metadata
 * and the CUDA restore recreates the mapping at its original address. The
 * plugin registers no pause/checkpoint/resume hooks, so it cannot race the
 * agent's cuda-checkpoint-helper.
 */

#include <errno.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include "criu-plugin.h"
#include "nvidia_vma_match.h"

/*
 * The hook can run after CRIU switches to the workload's root, where the
 * device nodes are not on that rootfs. Resolve them in init when possible and
 * fall back to the standard Linux NVIDIA char-device majors: 195 for
 * /dev/nvidiactl and /dev/nvidiaN, 499 for /dev/nvidia-uvm.
 */
static int nvidia_major = 195;
static int uvm_major = 499;

static int
nvidia_device_major(const char* path)
{
  struct stat st;
  if (stat(path, &st) < 0)
    return -1;
  return major(st.st_rdev);
}

static int
snapshot_nvidia_vma_handle_device_vma(int fd, const struct stat* st)
{
  (void)fd;

  if (snapshot_nvidia_vma_handles_major(major(st->st_rdev), nvidia_major, uvm_major))
    return 0;

  /* Let another plugin, or CRIU's own error path, handle it. */
  return -ENOENT;
}

static int
snapshot_nvidia_vma_init(int stage)
{
  (void)stage;

  int resolved_nvidia = nvidia_device_major("/dev/nvidiactl");
  if (resolved_nvidia < 0)
    resolved_nvidia = nvidia_device_major("/dev/nvidia0");
  if (resolved_nvidia >= 0)
    nvidia_major = resolved_nvidia;

  int resolved_uvm = nvidia_device_major("/dev/nvidia-uvm");
  if (resolved_uvm >= 0)
    uvm_major = resolved_uvm;

  return 0;
}

static void
snapshot_nvidia_vma_exit(int stage, int ret)
{
  (void)stage;
  (void)ret;
}

cr_plugin_desc_t CR_PLUGIN_DESC = {
    .name = "snapshot_nvidia_vma",
    .init = snapshot_nvidia_vma_init,
    .exit = snapshot_nvidia_vma_exit,
    .version = CRIU_PLUGIN_VERSION,
    .max_hooks = CR_PLUGIN_HOOK__MAX,
    .hooks =
        {
            [CR_PLUGIN_HOOK__HANDLE_DEVICE_VMA] = snapshot_nvidia_vma_handle_device_vma,
        },
};
