// RUN: mlir-clang %s | FileCheck %s

#pragma plugin(func_matvec, "linalg", "x(i) +=! A(i,j) * b(j)")
int *func_matvec(int A[3][3], int x[3], int b[3]) {
  for (int i = 0; i < 3; i++)
    x[i] = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      x[i] += A[i][j] * b[j];
  return x;
}

int main() {
  int A[3][3];
  int x[3];
  int b[3];
  int *r = func_matvec(A, x, b);
  return r[0];
}
