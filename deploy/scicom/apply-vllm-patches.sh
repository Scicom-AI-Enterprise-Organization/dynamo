#!/bin/bash
# Apply the vLLM patches in deploy/scicom/patches/ to the vLLM installed in a venv.
#
#   deploy/scicom/apply-vllm-patches.sh /path/to/venv/bin/python
#
# The patches are Python only, so no rebuild is needed: the script patches the
# installed site-packages in place. A patch is applied only when the installed vLLM
# version matches the version in its file name (vllm-<version>-<name>.patch), and is
# skipped when it is already applied, so running it twice is safe. The tests/ sections
# of a patch are for a vLLM source checkout and are left out here.
set -euo pipefail

PY="${1:?usage: $0 <venv python>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
VER="$("$PY" -c 'import importlib.metadata as m; print(m.version("vllm"))')"
VER="${VER%%+*}"

for p in "$HERE"/patches/vllm-*.patch; do
  name="$(basename "$p")"
  want="$(echo "$name" | sed -E 's/^vllm-([0-9][0-9.]*)-.*/\1/')"
  if [ "$VER" != "$want" ]; then
    echo "skip $name: installed vLLM is $VER"
    continue
  fi

  only_vllm="$(mktemp)"
  python3 - "$p" > "$only_vllm" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
parts = re.split(r"(?m)^(?=diff --git )", text)
sys.stdout.write("".join(x for x in parts if x.startswith("diff --git a/vllm/")))
EOF

  if patch -p1 -d "$SITE" -R -s -f --dry-run < "$only_vllm" >/dev/null 2>&1; then
    echo "already applied: $name"
  else
    patch -p1 -d "$SITE" --forward < "$only_vllm"
    find "$SITE/vllm" -path "*/kv_transfer/*" -name "*.pyc" -delete
    echo "applied: $name -> $SITE"
  fi
  rm -f "$only_vllm"
done
