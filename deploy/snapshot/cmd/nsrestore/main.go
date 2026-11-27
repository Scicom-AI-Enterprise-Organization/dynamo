package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
)

func main() {
	// Logs go to stderr so stdout is reserved for the structured result.
	log := logging.ConfigureLogger("stderr").WithName("nsrestore")

	checkpointPath := flag.String("checkpoint-path", "", "Path to checkpoint directory")
	cudaDeviceMap := flag.String("cuda-device-map", "", "CUDA device map for cuda-checkpoint-helper restore")
	cgroupRoot := flag.String("cgroup-root", "", "CRIU cgroup root remap path")
	targetPodIP := flag.String("target-pod-ip", "", "Restore pod IP for CRIU TCP socket remapping")
	mntnsCompatMode := flag.String("mntns-compat-mode", "inherit",
		"override CRIU mount-namespace compatibility mode: inherit, true, or false")
	flag.Parse()

	if *checkpointPath == "" {
		fatal(log, nil, "--checkpoint-path is required")
	}

	compatOverride, err := parseOptionalBool(*mntnsCompatMode)
	if err != nil {
		fatal(log, err, "invalid --mntns-compat-mode")
	}

	opts := executor.RestoreOptions{
		CheckpointPath:  *checkpointPath,
		CUDADeviceMap:   *cudaDeviceMap,
		CgroupRoot:      *cgroupRoot,
		TargetPodIP:     *targetPodIP,
		MntnsCompatMode: compatOverride,
	}

	result, err := executor.RestoreInNamespace(context.Background(), opts, log)
	if err != nil {
		fatal(log, err, "restore failed")
	}
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fatal(log, err, "Failed to write restore result")
	}
}

// parseOptionalBool distinguishes "inherit" (nil, use the checkpoint's value)
// from an explicit true or false.
func parseOptionalBool(raw string) (*bool, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "inherit":
		return nil, nil
	case "true":
		v := true
		return &v, nil
	case "false":
		v := false
		return &v, nil
	default:
		return nil, fmt.Errorf("expected inherit, true, or false, got %q", raw)
	}
}

func fatal(log logr.Logger, err error, msg string, keysAndValues ...interface{}) {
	if err != nil {
		log.Error(err, msg, keysAndValues...)
	} else {
		log.Info(msg, keysAndValues...)
	}
	os.Exit(1)
}
