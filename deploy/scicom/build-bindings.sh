#!/bin/bash
# Rebuild ai-dynamo-runtime (_core.abi3.so) from this branch.
#
# WHY THIS EXISTS
#   The gemma4 channel-leak fix lives in lib/llm/src/preprocessor.rs, which is
#   compiled into _core.abi3.so. The stock prebuilt .so still carries the leak,
#   so a source refresh or a fresh node silently reverts it -- nothing fails
#   loudly, the delimiters just start appearing in customer-visible replies
#   again. Rebuild with this, or install the released wheel.
#
# TESTED ON
#   tm-h20 (Alibaba PAI DSW pod, Ubuntu 24.04, x86_64, 164 cores).
#   cargo check: ~1m30s. maturin build --release: ~5m38s.
#
# NO apt REQUIRED -- deliberately. The DSW pods have a wedged dpkg state
# (libavdevice60 needs libgl1; libgl1 cannot unpack because the hand-injected
# NVIDIA driver userspace at /usr/lib/x86_64-linux-gnu/libGL.so.1.7.0 is on a
# different mount device, so dpkg cannot make its backup hardlink). Every
# dependency below is therefore installed without touching the package manager.
set -euo pipefail

SRC="${SRC:-/mnt/data/dynamo/src/dynamo}"
VENV="${VENV:-/mnt/data/dynamo/venv-vllm023}"
OUT="${OUT:-/root/dyn-wheel}"
JOBS="${JOBS:-48}"   # box has 164 cores but runs inference; don't take them all

export CARGO_HOME="${CARGO_HOME:-/root/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/root/.rustup}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/root/dyn-target}"   # local disk, NOT the NFS source tree
export PATH="$CARGO_HOME/bin:/usr/local/bin:$PATH"
export CARGO_BUILD_JOBS="$JOBS"

# --- one-time prerequisites -------------------------------------------------
if ! command -v rustc >/dev/null; then
  # rust-toolchain.toml pins 1.96.1; rustup installs entirely under $CARGO_HOME
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
  sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain 1.96.1
fi

if ! command -v protoc >/dev/null; then
  # lib/llm depends on tonic-build/prost-build. Prebuilt binary, no apt.
  PV=29.3
  curl -sSLo /tmp/protoc.zip \
    "https://github.com/protocolbuffers/protobuf/releases/download/v${PV}/protoc-${PV}-linux-x86_64.zip"
  python3 -c 'import zipfile;zipfile.ZipFile("/tmp/protoc.zip").extractall("/usr/local")'
  chmod +x /usr/local/bin/protoc
fi

# nixl-sys runs bindgen, which needs libclang. The PyPI `libclang` package ships
# the shared library, so no libclang-dev is required.
if [ ! -e /root/libclang-pkg/clang/native/libclang.so ]; then
  uv pip install --target /root/libclang-pkg libclang
fi
export LIBCLANG_PATH=/root/libclang-pkg/clang/native

# bindgen ships no builtin headers with that .so, so wrapper.h fails on
# stdbool.h. gcc's own header directory satisfies it.
GCC_INC="$(dirname "$(ls /usr/lib/gcc/x86_64-linux-gnu/*/include/stdbool.h | head -1)")"
export BINDGEN_EXTRA_CLANG_ARGS="-I${GCC_INC}"

"$VENV/bin/maturin" --version >/dev/null 2>&1 || uv pip install --python "$VENV/bin/python" maturin patchelf

# --- build ------------------------------------------------------------------
cd "$SRC/lib/bindings/python"
echo "### HEAD:  $(git -C "$SRC" log --oneline -1)"
echo "### rustc: $(rustc --version)"
echo "### start: $(date -Is)"
"$VENV/bin/maturin" build --release --out "$OUT"
echo "### end:   $(date -Is)"

# --- install ----------------------------------------------------------------
# ATOMIC ON PURPOSE. A running job has the old .so mmap'd; overwriting it in
# place can SIGBUS those processes. Writing a sibling file and renaming keeps
# the old inode alive for anything already running.
WHEEL="$(ls -t "$OUT"/ai_dynamo_runtime-*.whl | head -1)"
DEST="$SRC/lib/bindings/python/src/dynamo"
rm -rf /tmp/dyn-whl && mkdir -p /tmp/dyn-whl
python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall('/tmp/dyn-whl')" "$WHEEL"

cp -n "$DEST/_core.abi3.so" "$(dirname "$SRC")/_core.abi3.so.orig-$(git -C "$SRC" rev-parse --short HEAD)" 2>/dev/null || true
cp "$(find /tmp/dyn-whl -name _core.abi3.so | head -1)" "$DEST/_core.abi3.so.new"
mv -f "$DEST/_core.abi3.so.new" "$DEST/_core.abi3.so"

echo "### installed: $(stat -c%s "$DEST/_core.abi3.so") bytes"
echo "### wheel:     $WHEEL"
echo
echo "Restart the Dynamo job to pick it up (running processes keep the old mapping)."
