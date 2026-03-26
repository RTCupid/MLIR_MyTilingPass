add_llvm_executable(my-tiling-opt
  my-tiling-opt.cpp
)

target_link_libraries(my-tiling-opt PRIVATE
  MLIRMyTilingPass
  MLIRIR
  MLIRMlirOptMain
  LLVMSupport
)

llvm_update_compile_flags(my-tiling-opt)

set_target_properties(my-tiling-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
