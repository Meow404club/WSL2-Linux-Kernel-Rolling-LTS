#!/bin/bash
# WSL2 内核构建脚本（构建方式参考 ~/oplus13/build.sh）
#
# 用法:
#   ./build.sh                    # 默认: config-wsl-x86, gcc + ccache, 输出到 out/
#   BUILD_CONFIG=arch/x86/configs/config-wsl-x86-rt ./build.sh
#   ./build.sh clean              # 其余参数透传给 make, 如 clean / bzImage / modules
#   USE_CLANG=1 ./build.sh        # 切换 clang/LLVM 构建
#
# 说明: 配置文件即树内 KCONFIG_CONFIG 本身 (不会生成 out/.config),
#       新增 Kconfig 符号需手动 `make KCONFIG_CONFIG=$BUILD_CONFIG olddefconfig` 刷新。
set -euo pipefail

BUILD_CONFIG="${BUILD_CONFIG:-arch/x86/configs/config-wsl-x86}"
OUT_DIR="${OUT_DIR:-out}"
JOBS="${JOBS:-$(nproc --all)}"
USE_CLANG="${USE_CLANG:-0}"

# ccache 配置与 ~/.zshrc 保持一致, 本脚本不依赖交互式 shell
export USE_CCACHE="${USE_CCACHE:-1}"
export CCACHE_EXEC="${CCACHE_EXEC:-/usr/bin/ccache}"
export CCACHE_DIR="${CCACHE_DIR:-/mnt/ccache}"
export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-file_macro,include_file_mtime,time_macros}"
export CCACHE_UMASK="${CCACHE_UMASK:-002}"

if [[ "$USE_CLANG" == 1 ]]; then
	CC_NAME="clang"
	CXX_NAME="clang++"
	TOOLCHAIN_ARGS=(LLVM=1 LLVM_IAS=1)
else
	CC_NAME="gcc"
	CXX_NAME="g++"
	TOOLCHAIN_ARGS=()
fi

BUILD_CC="$CC_NAME"
BUILD_CXX="$CXX_NAME"
CCACHE_DESC="disabled"
if [[ "$USE_CCACHE" == 1 && -x "$CCACHE_EXEC" ]]; then
	BUILD_CC="$CCACHE_EXEC $CC_NAME"
	BUILD_CXX="$CCACHE_EXEC $CXX_NAME"
	CCACHE_DESC="$CCACHE_DIR"
fi

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

echo "== config   : $BUILD_CONFIG"
echo "== output   : $PWD/$OUT_DIR"
echo "== jobs     : $JOBS"
echo "== compiler : $BUILD_CC ${TOOLCHAIN_ARGS[*]}"
echo "== ccache   : $CCACHE_DESC"

make -j"$JOBS" O="$OUT_DIR" KCONFIG_CONFIG="$BUILD_CONFIG" \
	"${TOOLCHAIN_ARGS[@]}" \
	CC="$BUILD_CC" CXX="$BUILD_CXX" \
	HOSTCC="$BUILD_CC" HOSTCXX="$BUILD_CXX" \
	"$@"
