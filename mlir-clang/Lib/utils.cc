#include "utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

#include "Plugin/Linalg/MLIRGen.h"
#include "Plugin/Linalg/tc/lang/parser.h"
#include "Plugin/Linalg/tc/lang/sema.h"

using namespace mlir;
using namespace llvm;

Operation *mlirclang::buildLinalgOp(const AbstractOperation *op, OpBuilder &b,
                                    SmallVectorImpl<Value> &input,
                                    SmallVectorImpl<Value> &output) {
  StringRef name = op->name;
  if (name.compare("linalg.copy") == 0) {
    assert(input.size() == 1 && "linalg::copyOp requires 1 input");
    assert(output.size() == 1 && "linalg::CopyOp requires 1 output");
    return b.create<linalg::CopyOp>(b.getUnknownLoc(), input[0], output[0]);
  } else {
    llvm::report_fatal_error(llvm::Twine("builder not supported for: ") + name);
    return nullptr;
  }
}

Operation *mlirclang::replaceFuncByOperation(FuncOp f, StringRef opName,
                                             OpBuilder &b,
                                             SmallVectorImpl<Value> &input,
                                             SmallVectorImpl<Value> &output) {
  MLIRContext *ctx = f->getContext();
  assert(ctx->isOperationRegistered(opName) &&
         "Provided lower_to opName should be registered.");

  const AbstractOperation *op = AbstractOperation::lookup(opName, ctx);

  if (opName.startswith("linalg"))
    return buildLinalgOp(op, b, input, output);

  // NOTE: The attributes of the provided FuncOp is ignored.
  OperationState opState(b.getUnknownLoc(), op->name, input,
                         f.getCallableResults(), {});
  return b.createOperation(opState);
}

Operation *mlirclang::buildFunctionBodyWithPlugin(
    FuncOp func, std::string composedExpr, OpBuilder &builder,
    ScopedHashTable<StringRef, Value> &operandsMap) {
  std::string delimiter = ":";
  size_t pos = composedExpr.find(delimiter);
  std::string pluginame = composedExpr.substr(0, pos);
  std::string expr = composedExpr.substr(pos + delimiter.length());

  teckyl::MLIRGenImpl generator(func.getContext(), builder, operandsMap);
  lang::Parser parser(expr);
  lang::TreeRef comp = parser.parseStmt();
  // lang::Sema sema;
  // lang::TreeRef checked = sema.checkStmt(comp);
  mlir::Operation *op = generator.buildComprehension(lang::Comprehension(comp));
  llvm::errs() << "------- generated module ------\n";
  mlir::ModuleOp m = op->getParentOfType<ModuleOp>();
  m.dump();
  llvm::errs() << "\n\n --------------------------\n\n";
  return op;
}
