#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "Resources/log.h"
#include "Resources/matrices.h"
#include "Resources/csv.h"

#define LEARNING_RATE 0.03
#define MAX_EPOCHS 10

typedef struct{
  matrix_t* weights;
  matrix_t* biases;
  matrix_t* unactivated;
  matrix_t* output;
  int neurons;
} layer_t;

static int gradient_descent(matrix_t*, layer_t*, layer_t*, layer_t*);
static int forward_propagation(layer_t*, layer_t*, layer_t*);
static int backward_propagation(layer_t*, layer_t*, layer_t*, matrix_t*);
static int update_params(layer_t*, matrix_t*, matrix_t*);
static int make_predictions(matrix_t*, layer_t*, layer_t*, layer_t*);
static matrix_t* relu(matrix_t*);
static matrix_t* deriv_relu(matrix_t*);
static void init_layer(layer_t*, int, int);
static double random_double();

int main(int argc, char** argv){
  layer_t output_layer, hidden_layer, input_layer;
  matrix_t *x, *y; 
  matrix_t *ref;
  int ret;

  /* initialize random seed */
  srand((unsigned)time(NULL));

  x = NULL;
  y = NULL;
  
  /* get contents of dataset */
  ref = parse_csv("mnist_train.csv");
  x = transpose(ref);
  free_matrix(ref);

  init_layer(&input_layer, (x->h)-1, 0); /* subtracting 1 for the label */
  init_layer(&hidden_layer, 10, input_layer.neurons);
  init_layer(&output_layer, 10, hidden_layer.neurons);
  
  print_info("Training model..."); 
  ret = gradient_descent(x, &input_layer, &hidden_layer, &output_layer);
  if(ret!=0) return ret;
  print_ok("Done");
  
  free_matrix(x);
  ref = parse_csv("mnist_test.csv");
  x = transpose(ref);
  free_matrix(ref);

  print_info("Testing model...");
  ret = make_predictions(x, &input_layer, &hidden_layer, &output_layer);
  if(ret!=0) return ret;
  print_ok("Done");

  return ret;
}

static int gradient_descent(matrix_t* dataset, layer_t* input, layer_t* hidden, layer_t* output){
  matrix_t* y;
  int i, j, e, ret, sum, label;
  float accuracy;
  
  y = NULL;
 
  /* loop for iterations */ 
  for(e=0; e<MAX_EPOCHS; e++){
    /* loop for each column of the dataset */ 
    for(i=0, sum=0; i<dataset->w; i++){
      label = dataset->values[0][i];
      if(y != NULL) free_matrix(y);
      y = new_matrix(output->neurons, 1); /* one hot y */
      y->values[label][0] = -1;

      if(input->output != NULL) free_matrix(input->output);
      input->output = new_matrix(input->neurons, 1); 
      for(j=1; j<dataset->h; j++){
        input->output->values[j-1][0] = dataset->values[j][i] / 255.00;
      }

      ret = forward_propagation(input, hidden, output);
      if(ret!=0) return ret;
    
      ret = backward_propagation(input, hidden, output, y);
      if(ret!=0) return ret;
      
      sum += argmax(output->output)==label;
    }

    accuracy = (float)sum / (float)dataset->w;

    print_info("Iteration: %d\n    Accuracy: %f", e, accuracy);
  }

  return 0;
}

static void init_layer(layer_t* l, int dim, int dim_prec){ 
  int i, j;
  
  l->weights = new_matrix(dim, dim_prec);
  l->biases = new_matrix(dim, 1);
  l->output = NULL;
  l->unactivated = NULL;
  l->neurons = dim;

  /* random weights */
  for(i=0; i<l->weights->h; i++){
    for(j=0; j<l->weights->w; j++){
      l->weights->values[i][j] = random_double();
    }
  }
  
  /* random biases */
  for(i=0; i<l->biases->h; i++){
    for(j=0; j<l->biases->w; j++){
      l->biases->values[i][j] = random_double();
    }
  }

}

static int forward_propagation(layer_t* input, layer_t* hidden, layer_t* output){
  matrix_t* ref;

  /* Z = WX+B */
  ref = dot_product(hidden->weights, input->output); /* keep reference to free */
  hidden->unactivated = add_matrices(ref, hidden->biases);
  free_matrix(ref);
  if(hidden->unactivated == NULL) return -1;
  /* A = ReLU(Z) */
  hidden->output = relu(hidden->unactivated);
  if(hidden->output == NULL) return -1;
  
  ref = dot_product(output->weights, hidden->output);
  output->unactivated = add_matrices(ref, output->biases);
  free_matrix(ref);
  if(output->unactivated == NULL) return -1;
  output->output = softmax(output->unactivated);
  if(output->output == NULL) return -1;

  return 0;
}

static int backward_propagation(layer_t* input, layer_t* hidden, layer_t* output, matrix_t* label){
  matrix_t *dz2, *dz1, *dw2, *db2, *dw1, *db1;
  matrix_t *ref, *ref2;
  int ret;
  
  ret = 0;

  /* output layer gradient */
  /* mean squared error */
  ref = add_matrices(output->output, label);
  dz2 = scalar_multiplication(ref, 2.00/((double)(output->neurons)));
  free_matrix(ref);
  if(dz2==NULL) return -1;
  ref = transpose(hidden->output);
  dw2 = dot_product(dz2, ref);
  free_matrix(ref);
  db2 = dz2;
  
  /* hidden layer gradient */
  ref = transpose(output->weights);
  ref2 = dot_product(ref, dz2);
  free_matrix(ref);
  ref = deriv_relu(hidden->unactivated);
  dz1 = element_multiplication(ref2, ref);
  free_matrix(ref);
  free_matrix(ref2);
  if(dz1==NULL) return -1;
  ref = transpose(input->output);
  dw1 = dot_product(dz1, ref);
  free_matrix(ref);
  db1 = dz1;
  
  /* update parameters */
  ret = update_params(output, dw2, db2);
  if(ret!=0) return ret;
  ret = update_params(hidden, dw1, db1);
  if(ret!=0) return ret;
  
  return 0;
}

static int update_params(layer_t* l, matrix_t* dw, matrix_t* db){
  matrix_t *ref, *new_values;

  ref = scalar_multiplication(dw, -LEARNING_RATE);
  new_values = add_matrices(l->weights, ref);
  free_matrix(l->weights);
  l->weights = new_values;
  free_matrix(ref);
  if(l->weights==NULL) return -1;
  free_matrix(dw);
  
  ref = scalar_multiplication(db, -LEARNING_RATE);
  new_values = add_matrices(l->biases, ref);
  free_matrix(l->biases);
  l->biases = new_values;
  free_matrix(ref);
  if(l->biases==NULL) return -1;
  free_matrix(db);

  return 0;
}

static int make_predictions(matrix_t* dataset, layer_t* input, layer_t* hidden, layer_t* output){
  int i, j, ret, sum, label;
  float accuracy;

  for(i=0, sum=0; i<dataset->w; i++){
    label = dataset->values[0][i];

    if(input->output != NULL) free_matrix(input->output);
    input->output = new_matrix(input->neurons, 1);
    for(j=1; j<dataset->h; j++){
      input->output->values[j-1][0] = dataset->values[j][i] / 255.00;
    }

    ret = forward_propagation(input, hidden, output);
    if(ret!=0) return ret;
    
    sum += argmax(output->output)==label;
  }

  accuracy = (float)sum / (float)dataset->w;
  print_info("Accuracy on test set: %f", accuracy);

  return ret;
}

static matrix_t* relu(matrix_t* in){
  matrix_t* o;
  int i, j;

  o = NULL;

  o = new_matrix(in->h, in->w);

  for(i=0; i<o->h; i++){
    for(j=0; j<o->w; j++){
      o->values[i][j] = in->values[i][j]<=0 ? 0 : in->values[i][j];
    }
  }

  return o;
}

static matrix_t* deriv_relu(matrix_t* in){
  matrix_t* o;
  int i, j;

  o = NULL;

  o = new_matrix(in->h, in->w);

  for(i=0; i<o->h; i++){
    for(j=0; j<o->w; j++){
      o->values[i][j] = in->values[i][j]>0;
    }
  }

  return o;
}

static double random_double(){
  return (double)rand()/(double)(RAND_MAX) - (double).5;
}
