cmake_minimum_required(VERSION 3.3)

SET(TARGET voxl-tflite-server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_QRB5165 "Build the qrb5165 binary" OFF)

if(BUILD_QRB5165)
    add_definitions(-DBUILD_QRB5165)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")
elseif(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")
endif()


# Enable compile optimizations
# Enable debug flags (use if you want to debug in gdb)
# Build from all source files
file(GLOB_RECURSE all_src_files *.c*)

add_executable(${TARGET}
	${all_src_files}
)

include_directories(
    ../include/
    ..include/model_helper/
    /usr/include/opencv4/

    # # apq8096 SPECIFIC
    # /usr/include/tensorflow/lite/tools/make/downloads/absl/
    # /usr/include/tensorflow/lite/tools/make/downloads/flatbuffers/include/

    # qrb5165 SPECIFIC
    /usr/include/flatbuffers/include/
    /usr/include/abseil-cpp/
    /usr/include/ruy/
)

set(ABSL_LIBS_1
    "absl_flags"
    "absl_flags_internal"
    "absl_flags_marshalling"
    "absl_flags_reflection"
    "absl_flags_config"
    "absl_flags_program_name"
    "absl_flags_private_handle_accessor"
    "absl_flags_commandlineflag"
    "absl_flags_commandlineflag_internal"
    "absl_status"
    "absl_strerror"
)

set(ABSL_LIBS_2
    "absl_bad_any_cast_impl"
    "absl_bad_optional_access"
    "absl_bad_variant_access"
    "absl_base"
    "absl_city"
    "absl_civil_time"
    "absl_cord"
    "absl_cordz_info"
    "absl_cord_internal"
    "absl_cordz_functions"
    "absl_cordz_handle"
    "absl_crc_cord_state"
    "absl_crc32c"
    "absl_crc_internal"
    "absl_crc_cpu_detect"
    "absl_debugging_internal"
    "absl_demangle_internal"
    "absl_exponential_biased"
    "absl_graphcycles_internal"
    "absl_hash"
    "absl_hashtablez_sampler"
    "absl_int128"
    "absl_log_severity"
    "absl_malloc_internal"
    "absl_string_view"
    "absl_low_level_hash"
    "absl_kernel_timeout_internal"
    "absl_raw_hash_set"
    "absl_raw_logging_internal"
    "absl_spinlock_wait"
    "absl_stacktrace"
    "absl_str_format_internal"
    "absl_strings"
    "absl_strings_internal"
    "absl_symbolize"
    "absl_synchronization"
    "absl_throw_delegate"
    "absl_time"
    "absl_time_zone"
)

set(RUY_LIBS
    "ruy_context_get_ctx"
    "ruy_context"
    "ruy_frontend"
    "ruy_kernel_arm"
    "ruy_kernel_avx"
    "ruy_kernel_avx2_fma"
    "ruy_kernel_avx512"
    "ruy_apply_multiplier"
    "ruy_pack_arm"
    "ruy_pack_avx"
    "ruy_pack_avx2_fma"
    "ruy_pack_avx512"
    "ruy_prepare_packed_matrices"
    "ruy_trmul"
    "ruy_ctx"
    "ruy_allocator"
    "ruy_prepacked_cache"
    "ruy_system_aligned_alloc"
    "ruy_have_built_path_for_avx"
    "ruy_have_built_path_for_avx2_fma"
    "ruy_have_built_path_for_avx512"
    "ruy_thread_pool"
    "ruy_blocking_counter"
    "ruy_wait"
    "ruy_denormal"
    "ruy_block_map"
    "ruy_tune"
    "ruy_cpuinfo"
    "ruy_profiler_instrumentation"
)

set(OPENCV_LIBS
    "opencv_core"
    "opencv_highgui"
    "opencv_imgproc"
    "opencv_imgcodecs"
    "opencv_dnn"
)

set(QCOM_LIBS
    "llvm-qcom"
    "adreno_utils"
    "OpenCL"
    "CB"
    "EGL_adreno"
    "GLESv2_adreno"
)

set(LINK_LIBS
    "modal_pipe"
    "modal_json"
    "tensorflow-lite"
    "glib-2.0"
    "stdc++"
    "pthread"
    "z"
    "cutils"
    "log"
    "sync"
    "gthread-2.0"
    "pcre"
    ${OPENCV_LIBS}
    "gsl"
    ${QCOM_LIBS}
    "rt"
    "dl"
    "cpuinfo"
    ${ABSL_LIBS_1}
    "farmhash"
    "fft2d_fftsg2d"
    "fft2d_fftsg"
    "flatbuffers"
    "eight_bit_int_gemm"
    ${RUY_LIBS}
    ${ABSL_LIBS_2}
)

if (BUILD_QRB5165)
set(LINK_LIBS "${LINK_LIBS}"
    "XNNPACK"
    "pthreadpool"
    "cpuinfo"
    "pthread"
)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-fallthrough")

target_link_libraries(${TARGET}
    -Wl,-rpath-link,/usr/lib64/
    -L/usr/lib64/
    ${LINK_LIBS}
)
# make sure everything is installed where we want
# LIB_INSTALL_DIR comes from the parent cmake file
install(
	TARGETS			${TARGET}
	LIBRARY			DESTINATION ${LIB_INSTALL_DIR}
	RUNTIME			DESTINATION /usr/bin
	PUBLIC_HEADER	DESTINATION /usr/include
)
