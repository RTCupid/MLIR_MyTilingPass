llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MYTILINGPASS_TEST_DEPENDS
        FileCheck count not
        my-tiling-opt
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND MYTILINGPASS_TEST_DEPENDS MyTilingPassPythonModules)
endif()

add_lit_testsuite(check-my-tiling "Running the my tiling regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MYTILINGPASS_TEST_DEPENDS}
        )
set_target_properties(check-my-tiling PROPERTIES FOLDER "Tests")

add_lit_testsuites(MYTILINGPASS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MYTILINGPASS_TEST_DEPENDS})

