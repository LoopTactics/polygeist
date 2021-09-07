#include "syntaxHandler.h"
#include "clang/Parse/Parser.h"

using namespace clang;

namespace {

class PrintTokensHandler : public SyntaxHandler {
public:
  PrintTokensHandler() : SyntaxHandler("tokens") {}

  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override {
    StringRef ti = D.getIdentifier()->getName();
    llvm::errs() << ti << "\n";
    std::string content = "";
    for (auto &Tok : Toks)
      content += PP.getSpelling(Tok);
    content += "\n";
    llvm::errs() << content << "\n";
  }

  void AddToPredefines(llvm::raw_string_ostream &OS) override {}
};

} // end namespace

void addSyntaxHandlers() {
  static SyntaxHandlerRegistry::Add<PrintTokensHandler> X("tokens",
                                                          "collect all tokens");
}
