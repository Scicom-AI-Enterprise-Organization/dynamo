#!/bin/bash
set -euo pipefail
NS=snapfix-lab
for role in prefill decode; do
  POD=$(sudo k3s kubectl -n $NS get pods -l snapfix-pd/role=$role -o jsonpath='{.items[0].metadata.name}')
  echo "capturing $role pod $POD"
  cat <<YAML | sudo k3s kubectl apply -f -
apiVersion: nvidia.com/v1alpha1
kind: PodSnapshot
metadata:
  name: snapfix-pd-$role
  namespace: $NS
spec:
  source:
    podRef:
      name: $POD
      containers: ["main"]
  containers:
  - name: main
    checkpointReadinessProbe:
      file:
        path: /snapshot-control/ready-for-snapshot
YAML
done
