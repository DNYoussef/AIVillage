cmake_minimum_required(VERSION 3.14)
project(betanet_c_examples VERSION 0.1.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(ENABLE_SBOM "Generate SBOM for artifacts" OFF)

# Find threads (required for pthreads on Linux)
find_package(Threads REQUIRED)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Rust library paths
set(BETANET_RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BETANET_RUST_BUILD_DIR "${BETANET_RUST_TARGET_DIR}/debug")
else()
    set(BETANET_RUST_BUILD_DIR "${BETANET_RUST_TARGET_DIR}/release")
endif()

# Platform-specific library naming
if(WIN32)
    set(BETANET_LIB_NAME "betanet_c.dll.lib")
    set(BETANET_DLL_NAME "betanet_c.dll")
elseif(APPLE)
    set(BETANET_LIB_NAME "libbetanet_c.dylib")
else()
    set(BETANET_LIB_NAME "libbetanet_c.so")
endif()

# Build Rust library first
add_custom_command(
    OUTPUT ${BETANET_RUST_BUILD_DIR}/${BETANET_LIB_NAME}
    COMMAND cargo build --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml $<IF:$<CONFIG:Debug>,,--release>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Rust Betanet library..."
)

add_custom_target(build_rust_lib ALL
    DEPENDS ${BETANET_RUST_BUILD_DIR}/${BETANET_LIB_NAME}
)

# Create imported library target
add_library(betanet_c SHARED IMPORTED)
set_target_properties(betanet_c PROPERTIES
    IMPORTED_LOCATION ${BETANET_RUST_BUILD_DIR}/${BETANET_LIB_NAME}
    IMPORTED_IMPLIB ${BETANET_RUST_BUILD_DIR}/${BETANET_LIB_NAME}
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_dependencies(betanet_c build_rust_lib)

# Copy DLL to output directory on Windows
if(WIN32)
    add_custom_command(TARGET build_rust_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${BETANET_RUST_BUILD_DIR}/${BETANET_DLL_NAME}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${BETANET_DLL_NAME}
    )
endif()

# Build examples
if(BUILD_EXAMPLES)
    # Echo client
    add_executable(c_echo_client examples/c_echo_client.c)
    target_link_libraries(c_echo_client PRIVATE betanet_c Threads::Threads)
    target_include_directories(c_echo_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Echo server
    add_executable(c_echo_server examples/c_echo_server.c)
    target_link_libraries(c_echo_server PRIVATE betanet_c Threads::Threads)
    target_include_directories(c_echo_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Add Windows-specific libraries
    if(WIN32)
        target_link_libraries(c_echo_client PRIVATE ws2_32 userenv bcrypt ntdll)
        target_link_libraries(c_echo_server PRIVATE ws2_32 userenv bcrypt ntdll)
    endif()
endif()

# Install rules
install(FILES include/betanet.h DESTINATION include)
install(TARGETS c_echo_client c_echo_server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Generate pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/betanet.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/betanet.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/betanet.pc
    DESTINATION lib/pkgconfig
)

# Generate SBOM if requested
if(ENABLE_SBOM)
    add_custom_target(generate_sbom ALL
        COMMAND ${CMAKE_COMMAND} -E echo "Generating SBOM..."
        COMMAND cargo sbom --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating Software Bill of Materials"
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Betanet C Library Configuration:")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build examples:    ${BUILD_EXAMPLES}")
message(STATUS "  Generate SBOM:     ${ENABLE_SBOM}")
message(STATUS "  Rust target dir:   ${BETANET_RUST_BUILD_DIR}")
message(STATUS "")
