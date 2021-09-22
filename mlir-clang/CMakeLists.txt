add_clang_tool(mlir-clang
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1_main.cpp"
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1as_main.cpp"
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1gen_reproducer_main.cpp"
  mlir-clang.cc
  Lib/utils.cc
  Lib/pragmaHandler.cc
  Lib/syntaxHandler.cc

  #Linalg plugin
  Lib/Plugin/Linalg/mlirAffineExprGen.cpp
  Lib/Plugin/Linalg/mlirGen.cpp
  Lib/Plugin/Linalg/tc/lexer.cc
  Lib/Plugin/Linalg/tc/parser.cc
  Lib/Plugin/Linalg/tc/tree.cpp
)

target_include_directories(mlir-clang PRIVATE
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
)

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
  )

list(TRANSFORM LLVM_LINK_COMPONENTS PREPEND LLVM)


target_compile_definitions(mlir-clang PUBLIC -DLLVM_OBJ_ROOT="${LLVM_BINARY_DIR}")
target_link_libraries(mlir-clang PRIVATE
    #    ${LLVM_LINK_COMPONENTS}
    ${LLVM_AVAILABLE_LIBS}

  MLIRLinalgTransforms
  MLIRTensorTransforms
  MLIRReconcileUnrealizedCasts
  MLIRSCFTransforms
  MLIRPolygeist

  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMIR
  MLIRNVVMIR
  MLIROpenMP
  MLIRGPUOps
  MLIRTransforms
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRTargetLLVMIRImport
  MLIRPolygeistTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation

  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangParse
  clangSerialization
)
add_dependencies(mlir-clang MLIRPolygeistOpsIncGen MLIRPolygeistPassIncGen)
add_subdirectory(Test)
