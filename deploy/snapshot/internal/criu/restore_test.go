// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package criu

import (
	"testing"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// A restore-only CRIU option must follow the agent's live configuration, not
// the value persisted at capture time, or flipping it after a checkpoint exists
// has no effect.
func TestBuildRestoreOptsMntnsCompatMode(t *testing.T) {
	tests := []struct {
		name      string
		manifest  bool
		override  *bool
		wantValue bool
	}{
		{name: "inherit false from manifest", manifest: false, wantValue: false},
		{name: "inherit true from manifest", manifest: true, wantValue: true},
		{name: "override false manifest with true", manifest: false, override: boolPtr(true), wantValue: true},
		{name: "override true manifest with false", manifest: true, override: boolPtr(false), wantValue: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manifest := &types.CheckpointManifest{
				CRIUDump: types.CRIUDumpManifest{
					CRIU:   types.CRIUSettings{MntnsCompatMode: tt.manifest},
					ExtMnt: map[string]string{"/": "/", "/dev/nvidia0": "/dev/nvidia0"},
				},
			}

			opts, err := BuildRestoreOpts(manifest, "/checkpoints/artifacts/test", "", tt.override, logr.Discard())
			if err != nil {
				t.Fatalf("BuildRestoreOpts() error = %v", err)
			}
			if got := opts.GetMntnsCompatMode(); got != tt.wantValue {
				t.Errorf("MntnsCompatMode = %v, want %v", got, tt.wantValue)
			}
		})
	}
}

func boolPtr(v bool) *bool { return &v }
