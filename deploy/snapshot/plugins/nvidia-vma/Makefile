# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra

test: test_nvidia_vma
	./test_nvidia_vma

test_nvidia_vma: test_nvidia_vma.c nvidia_vma_match.h
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f test_nvidia_vma

.PHONY: test clean
