#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "log.h"
#include "matrices.h"

matrix_t* new_matrix(int height, int width){
  int i, j;
  matrix_t* m;

  m = (matrix_t*)malloc(sizeof(matrix_t));
  if(m==NULL){
    print_err("Malloc failed");
    exit(1);
  }
  m->w = width;
  m->h = height;
  m->values = (double**)malloc(height*sizeof(double*));
  if(m->values==NULL){
    print_err("Malloc failed");
    exit(1);
  }
  for(i=0; i<height; i++){
    m->values[i] = (double*)malloc(width*sizeof(double));
    if(m->values[i]==NULL){
      print_err("Malloc failed");
      exit(1);
    }
  }

  for(i=0; i<m->h; i++){
    for(j=0; j<m->w; j++){
      m->values[i][j] = 0;
    }
  }

  return m;
}

int populate_matrix_stdin(matrix_t* m){
  int i, j;

  for(i=0; i<m->h; i++){
    for(j=0; j<m->w; j++){
      scanf(" %f", &(m->values[i][j]));
    }
  }

  return 0;
}

matrix_t* dot_product(matrix_t* a, matrix_t* b){
  int i, j, h, k, sum;
  matrix_t* m;

  m=NULL;
  
  if(a->w != b->h){
    print_err("Dimensions mismatch\n");
    return m;
  }

  m = new_matrix(a->h, b->w);
  
  for(k=0; k<b->w; k++){
    for(i=0; i<a->h; i++){
      for(j=0, sum=0; j<a->w; j++){
        m->values[i][k] += a->values[i][j] * b->values[j][k];
      }
    }
  }

  return m;
}

matrix_t* add_matrices(matrix_t* a, matrix_t* b){
  matrix_t* m;
  int i, j;

  m=NULL;

  if(a->h != b->h || a->w != b->w){
    print_err("Dimensions mismatch\n");
    return m;
  }

  m = new_matrix(a->h, a->w);
  
  for(i=0; i<a->h; i++){
    for(j=0; j<a->w; j++){
      m->values[i][j] = a->values[i][j] + b->values[i][j];
    }
  }

  return m;
}

matrix_t* transpose(matrix_t* m){
  matrix_t* t;
  int i, j;

  t=NULL;
  
  t = new_matrix(m->w, m->h);

  for(i=0; i<m->h; i++){
    for(j=0; j<m->w; j++){
      t->values[j][i] = m->values[i][j];
    }
  }

  return t;
}

matrix_t* scalar_multiplication(matrix_t* a, double n){
  matrix_t* b;
  int i, j;

  b = NULL;

  b = new_matrix(a->h, a->w);
  
  for(i=0; i<b->h; i++){
    for(j=0; j<b->w; j++){
      b->values[i][j] = a->values[i][j] * n;
    }
  }

  return b;
}

matrix_t* element_multiplication(matrix_t* a, matrix_t* b){
  matrix_t* m;
  int i, j;
  
  m=NULL;

  if(a->h != b->h || a->w != b->w) return m;
  
  m = new_matrix(a->h, a->w);
  
  for(i=0; i<a->h; i++){
    for(j=0; j<a->w; j++){
      m->values[i][j] = a->values[i][j] * b->values[i][j];
    }
  }

  // print_matrix(a);
  // print_matrix(b);
  // print_matrix(m);

  return m;
}

matrix_t* softmax(matrix_t* in){
  matrix_t* o;
  int i, j;
  double sum;

  o = NULL;
  
  if(in->w != 1)
    return o;

  o = new_matrix(in->h, 1);
  
  for(i=0, sum=0; i<in->h; i++){
    sum += exp(in->values[i][0]);
  }

  if(sum==0){ 
    print_err("Division by 0!");
    return NULL;
  }

  for(i=0; i<in->h; i++){
    o->values[i][0] = exp(in->values[i][0])/sum;
    if(o->values[i][0] != o->values[i][0]){
      print_err("Not a Number! (%lf / %lf)", exp(in->values[i][0]), sum);
      return NULL;
    }
  }

  return o;
}

int argmax(matrix_t* in){
  double m;
  int i;

  if(in == NULL || in->w != 1) return -1;
  
  for(i=1, m=in->values[0][0]; i<in->h; i++){
    if(in->values[i][0]>m)
      m = in->values[i][0];
  }

  for(i=0; i<in->h && in->values[i][0]!=m; i++);

  return i;
}

int print_matrix(matrix_t *m){
  int i, j;

  if(m==NULL){
    print_err("NULL pointer");
    return -1;
  }
  if(m->values==NULL){
    print_err("NULL pointer");
    return -1;
  } 

  printf("Height: %d\nWidth: %d\nValues:\n", m->h, m->w);
  
  for(i=0; i<m->h; i++){
    for(j=0; j<m->w; j++){
      if(m->values[i][j]<0)
        printf("%.3lf ", m->values[i][j]);
      else
        printf(" %.3lf ", m->values[i][j]);
    }
    puts("");
  }
  puts("");

  return 0;
}

int free_matrix(matrix_t* m){
  int i;
  
  for(i=0; i<m->h; i++){
    free(m->values[i]);
  }
  free(m->values);
  free(m);

  return 0;
}

int matrix_to_file(FILE* fp, matrix_t* m){
  int i;

  fwrite(&(m->w), sizeof(int), 1, fp);
  fwrite(&(m->h), sizeof(int), 1, fp);
  for(i=0; i<m->h; i++){
    fwrite(m->values[i], sizeof(float), m->w, fp);
  }

  return 0;
}
