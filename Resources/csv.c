#include <stdio.h>
#include <stdlib.h>

#include "log.h"
#include "matrices.h"
#include "csv.h"


matrix_t* parse_csv(const char* filename){
  FILE *fp;
  char *s;
  size_t len;
  int cols, rows, i, j;
  double n;
  matrix_t* m;
  
  m=NULL;
  s = NULL;
  len = 0;
  n = 0;

  print_info("Parsing csv file: %s", filename);
  
  if(get_dimensions(filename, &cols, &rows)!=0){
    return m;
  }

  m = new_matrix(rows-1, cols);

  fp=fopen(filename, "r");
  if(fp==NULL){
    print_err("Error opening file");
    return m;
  }
  
  if(!feof(fp)){
    getline(&s, &len, fp);
  }
  
  i=0;
  j=0;
  while(fscanf(fp, "%lf", &n)!=EOF){
    getc(fp);
    
    m->values[i][j] = n;
    j++;
    if(j==m->w){
      j=0;
      i++;
    }
  }


  fclose(fp);

  print_ok("Done");
  
  return m;
}

int get_dimensions(const char* filename, int* cols, int* rows){
  FILE *fp;
  char c;
  
  *cols = 0;
  *rows = 0;

  fp=fopen(filename, "r");
  if(fp==NULL){
    print_err("Error opening file");
    return -1;
  }

  while(!feof(fp)){
    fscanf(fp, "%c", &c);
    if((c==',' || c=='\n') && *rows==0){
      (*cols)++;
    } 
    if(c=='\n' && !feof(fp)){
      (*rows)++;
    }
  }

  // printf("Dim:%dx%d\n", *cols, *rows); 

  fclose(fp);

  return 0;
}
