#ifndef MATRICES_H
#define MATRICES_H

typedef struct{
  int w;
  int h;
  double** values;
} matrix_t;

matrix_t* new_matrix(int, int);
int populate_matrix_stdin(matrix_t*);
matrix_t* dot_product(matrix_t*, matrix_t*);
matrix_t* add_matrices(matrix_t*, matrix_t*);
matrix_t* transpose(matrix_t*);
matrix_t* scalar_multiplication(matrix_t*, double);
matrix_t* element_multiplication(matrix_t*, matrix_t*);
matrix_t* softmax(matrix_t*);
int argmax(matrix_t*);
int print_matrix(matrix_t*);
int free_matrix(matrix_t*);
int matrix_to_file(FILE*, matrix_t*);

#endif
