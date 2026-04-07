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

# Copied from core_platform/cmake/toolchain/armclang.cmake and adjusted to align
# better with the Corstone-300 ExecuTorch semihosting runner.

set(TARGET_CPU
    "cortex-m55"
    CACHE STRING "Target CPU"
)
string(TOLOWER ${TARGET_CPU} CMAKE_SYSTEM_PROCESSOR)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_C_COMPILER "armclang")
set(CMAKE_CXX_COMPILER "armclang")
set(CMAKE_ASM_COMPILER "armclang")
set(CMAKE_LINKER "armlink")

# Default to GNU objcopy
set(CMAKE_OBJCOPY "arm-none-eabi-objcopy")

set(CMAKE_EXECUTABLE_SUFFIX ".elf")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_POSITION_INDEPENDENT_CODE
    OFF
    CACHE BOOL "Disable PIC for bare-metal ArmClang" FORCE
)

# Select C/C++ version
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Link target
string(REGEX MATCH "^cortex-m([0-9]+)([a-z]*)" __LINK_TARGET
             ${CMAKE_SYSTEM_PROCESSOR}
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "nodsp")
  string(APPEND __LINK_TARGET ".no_dsp")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "nofp")
  string(APPEND __LINK_TARGET ".no_fp")
endif()

# Compile options
add_compile_options(
  --target=arm-arm-none-eabi -mcpu=${CMAKE_SYSTEM_PROCESSOR} -fdata-sections
  -ffunction-sections "$<$<CONFIG:DEBUG>:-gdwarf-3>"
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables;-fno-rtti;-fno-exceptions>"
)

# Compile defines
add_compile_definitions("$<$<NOT:$<CONFIG:DEBUG>>:NDEBUG>")

# Link options
add_link_options(
  --cpu=${__LINK_TARGET}
  --lto
  --info
  common,debug,sizes,totals,veneers,unused
  --remove
  --symbols
  --diag_suppress=L6439W,L6314W
)

# Compilation warnings
add_compile_options(
  -Wall
  -Wextra
  -Wcast-align
  -Wdouble-promotion
  -Wformat
  -Wmissing-field-initializers
  -Wnull-dereference
  -Wredundant-decls
  -Wshadow
  -Wswitch-default
  -Wunused
)
