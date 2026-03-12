#ifndef FUNC
#define FUNC

#ifdef __cplusplus
extern "C" {
#endif

void vector_add(const float* A, const float* B, float* C, int N);

void matrix_multiplication(const float* A, const float* B, float* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif 