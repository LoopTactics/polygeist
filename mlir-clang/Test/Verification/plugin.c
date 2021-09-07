// RUN: mlir-clang %s %stdinclude --function=foo | FileCheck %s

// C++
//[[clang::syntax(tokens)]] void foo(int A[5][5], int b[5], int x[5]) {
//    x(i) +=! A(i,j) * b(j)
//}

// C
//__attribute__((syntax(tokens))) void foo(int A[5][5], int b[5], int x[5]) {
//  x(i) +=! A(i,j) * b(j)
//}

__attribute__((syntax(tokens))) void func_matvec(int A[3][3], int x[3],
                                                 int b[3]) {
  x(i) += !A(i, j) * b(j)
}

int main() {
  int A[3][3];
  int x[3];
  int b[3];
  func_matvec(A, x, b);
  return 0;
}
