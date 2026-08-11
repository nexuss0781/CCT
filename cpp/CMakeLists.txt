cmake_minimum_required(VERSION 3.20)
project(cct_native VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

add_library(cct_native STATIC
    src/event.cpp
    src/field.cpp
    src/fft_roundtrip.cpp
    src/sequence.cpp
    src/sequence_scan.cpp
    src/baselines.cpp
    src/causal.cpp
)
target_include_directories(cct_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cct_native PUBLIC PkgConfig::FFTW3)
target_compile_options(cct_native PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(cct_tests tests/cct_tests.cpp)
target_link_libraries(cct_tests PRIVATE cct_native)
target_compile_options(cct_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(cct_sequence_tests tests/sequence_tests.cpp)
target_link_libraries(cct_sequence_tests PRIVATE cct_native)
target_compile_options(cct_sequence_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(cct_stage0_gate tools/stage0_gate.cpp)
target_link_libraries(cct_stage0_gate PRIVATE cct_native)
target_compile_options(cct_stage0_gate PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(cct_stage1_gate tools/stage1_gate.cpp)
target_link_libraries(cct_stage1_gate PRIVATE cct_native)
target_compile_options(cct_stage1_gate PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(cct_stage2_gate tools/stage2_gate.cpp)
target_link_libraries(cct_stage2_gate PRIVATE cct_native)
target_compile_options(cct_stage2_gate PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(cct_causal_tests tests/causal_tests.cpp)
target_link_libraries(cct_causal_tests PRIVATE cct_native)
target_compile_options(cct_causal_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(cct_stage3_gate tools/stage3_gate.cpp)
target_link_libraries(cct_stage3_gate PRIVATE cct_native)
target_compile_options(cct_stage3_gate PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(CTest)
enable_testing()
add_test(NAME cct_native_tests COMMAND cct_tests)
add_test(NAME cct_sequence_tests COMMAND cct_sequence_tests)
add_test(NAME cct_stage0_gate COMMAND cct_stage0_gate --output ${CMAKE_BINARY_DIR}/artifacts/stage-0/gate)
add_test(NAME cct_stage1_gate COMMAND cct_stage1_gate --output ${CMAKE_BINARY_DIR}/artifacts/stage-1/gate)
add_test(NAME cct_stage2_gate COMMAND cct_stage2_gate --output ${CMAKE_BINARY_DIR}/artifacts/stage-2/gate)
add_test(NAME cct_causal_tests COMMAND cct_causal_tests)
add_test(NAME cct_stage3_gate COMMAND cct_stage3_gate --output ${CMAKE_BINARY_DIR}/artifacts/stage-3/gate)
