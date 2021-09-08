#include "syntaxHandler.h"
#include "clang/Parse/Parser.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

using namespace clang;

using llvm::yaml::MappingTraits;
using llvm::yaml::Output;

namespace {

struct LinalgPlugin {
  std::string funcName;
  std::string body;
};

} // namespace

namespace llvm {
namespace yaml {

template <> struct MappingTraits<LinalgPlugin> {
  static void mapping(IO &io, LinalgPlugin &info) {
    io.mapRequired("funcName", info.funcName);
    io.mapRequired("body", info.body);
  }
};

} // namespace yaml
} // namespace llvm

namespace {

// XXX: The llvm registry requires registered subclass to have a default
// constructor.
// But we need to export the information parsed here. For now write them on a
// file.
class PrintTokensHandler : public SyntaxHandler {
public:
  PrintTokensHandler() : SyntaxHandler("tokens") {}

  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override {
    std::string funcName = D.getIdentifier()->getName().str();

    std::string content = "";
    for (auto &Tok : Toks)
      content += PP.getSpelling(Tok);
    content += "\n";
    LinalgPlugin p = {funcName, content};

    Output yout(llvm::outs());
    yout << p;
  }

  void AddToPredefines(llvm::raw_string_ostream &OS) override {}
};

} // end namespace

void addSyntaxHandlers() {
  static SyntaxHandlerRegistry::Add<PrintTokensHandler> X("tokens",
                                                          "collect all tokens");
}
