#ifndef MATRICES_H
#define MATRICES_H

typedef struct{
  int w;
  int h;
  double** values;
} matrix_t;

matrix_t* new_matrix(int, int);
int populate_matrix_stdin(matrix_t*);
matrix_t* dot_product(const matrix_t*, const matrix_t*);
matrix_t* add_matrices(const matrix_t*, const matrix_t*);
matrix_t* transpose(const matrix_t*);
matrix_t* scalar_multiplication(const matrix_t*, double);
matrix_t* element_multiplication(const matrix_t*, const matrix_t*);
matrix_t* softmax(const matrix_t*);
int argmax(const matrix_t*);
int print_matrix(const matrix_t*);
int free_matrix(matrix_t*);
int matrix_to_file(FILE*, matrix_t*);
int matrix_from_file(FILE*, matrix_t*);

#endif
