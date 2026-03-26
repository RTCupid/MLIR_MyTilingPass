add_mlir_dialect_library(MLIRMyTilingPass
  MyTilingPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/MyTiling

  DEPENDS
  MLIRMyTilingPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRPass
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
)
