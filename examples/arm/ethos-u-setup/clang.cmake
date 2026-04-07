#
# Copyright (c) 2020-2022 Arm Limited. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set(TARGET_CPU
    "cortex-m55"
    CACHE STRING "Target CPU"
)
string(TOLOWER ${TARGET_CPU} CMAKE_SYSTEM_PROCESSOR)

if(DEFINED ENV{CLANG_TOOLCHAIN_ROOT} AND NOT "$ENV{CLANG_TOOLCHAIN_ROOT}"
                                         STREQUAL ""
)
  set(_default_clang_toolchain_root "$ENV{CLANG_TOOLCHAIN_ROOT}")
else()
  set(_default_clang_toolchain_root "")
endif()
set(CLANG_TOOLCHAIN_ROOT
    "${_default_clang_toolchain_root}"
    CACHE PATH
          "Path to the plain LLVM/Clang bare-metal toolchain bin directory"
)

if(CLANG_TOOLCHAIN_ROOT STREQUAL "")
  message(
    FATAL_ERROR
      "CLANG_TOOLCHAIN_ROOT must point to the plain bare-metal Clang bin directory"
  )
endif()

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_C_COMPILER "${CLANG_TOOLCHAIN_ROOT}/clang")
set(CMAKE_CXX_COMPILER "${CLANG_TOOLCHAIN_ROOT}/clang++")
set(CMAKE_ASM_COMPILER "${CLANG_TOOLCHAIN_ROOT}/clang")
set(CMAKE_LINKER "${CLANG_TOOLCHAIN_ROOT}/ld.lld")
set(CMAKE_AR "${CLANG_TOOLCHAIN_ROOT}/llvm-ar")
set(CMAKE_RANLIB "${CLANG_TOOLCHAIN_ROOT}/llvm-ranlib")
set(CMAKE_OBJCOPY "${CLANG_TOOLCHAIN_ROOT}/llvm-objcopy")
set(CMAKE_OBJDUMP "${CLANG_TOOLCHAIN_ROOT}/llvm-objdump")
set(CMAKE_SIZE "${CLANG_TOOLCHAIN_ROOT}/llvm-size")

set(CMAKE_EXECUTABLE_SUFFIX ".elf")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_POSITION_INDEPENDENT_CODE
    OFF
    CACHE BOOL "Disable PIC for bare-metal Clang" FORCE
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CLANG_COMMON_FLAGS
    --target=arm-none-eabi -mcpu=${CMAKE_SYSTEM_PROCESSOR} -mthumb
    -mfloat-abi=hard -fdata-sections -ffunction-sections
)

add_compile_options(
  ${CLANG_COMMON_FLAGS} -fno-pic -fno-pie "$<$<CONFIG:DEBUG>:-gdwarf-3>"
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables;-fno-rtti;-fno-exceptions>"
)

add_compile_definitions("$<$<NOT:$<CONFIG:DEBUG>>:NDEBUG>")

add_link_options(
  ${CLANG_COMMON_FLAGS} -fuse-ld=lld -fno-pie LINKER:--nmagic,--gc-sections
)

if(SEMIHOSTING)
  add_link_options(-nostartfiles -lcrt0-semihost -lsemihost)
else()
  add_link_options(-lnosys)
endif()

add_compile_options(
  -Wno-error=deprecated-declarations -Wno-error=shift-count-overflow
  -Wno-unknown-warning-option
)
