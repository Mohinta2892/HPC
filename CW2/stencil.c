#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, float *  image, float *  tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
void halo_exchange(int rank, int size, int local_ncols, float * w, int ny);
double wtime(void);

int main(int argc, char *argv[]) {
  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  //MPI Initialisation
  int rank;               /* 'rank' of process among it's cohort */
  int size;               /* size of cohort, i.e. num processes started */
  int flag;               /* for checking whether MPI_Init() has been called */
  int tag = 0;
  int strlen;             /* length of a character array */
  MPI_Status status;     /* struct used by MPI_Recv */

  MPI_Init( &argc, &argv ); /* initialise our MPI environment */
  /* check whether the initialisation was successful */
    
  int row_offset, nrows_for_ranks, nrows;
  double tic, toc;
    
  MPI_Initialized(&flag);
  if ( flag != 1 ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  MPI_Comm_size( MPI_COMM_WORLD, &size ); /* determine how many processes associated with the communicator */
  MPI_Comm_rank( MPI_COMM_WORLD, &rank ); /* determine the RANK of the current process [0:SIZE-1] */

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  //printf("nx %d ny %d niters %d in rank %d\n", nx, ny, niters,rank);

   if (rank == 0) {
    printf("nx %d ny %d niters %d\n", nx, ny, niters);
    }

  if (rank == 0) {
    if (size==1){
      nrows = nx;
      // printf(" for size %d nrows %d\n",size,nrows);
}
    else
      nrows = (nx/size)+1;
  }
  else if (rank == size-1) {
    nrows = nx-((nx/size)*(size-1))+1;
  } else {
    nrows = (nx/size)+2;
  }

  // Allocate the image
  float *image = (float *)_mm_malloc(sizeof(float)*nx*ny,64);
  float *tmp_image = (float *)_mm_malloc(sizeof(float)*nx*ny,64);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // Allocate chunk
  float *w = (float *)_mm_malloc(sizeof(float)*nrows*ny,64);
  float *u = (float *)_mm_malloc(sizeof(float)*nrows*ny,64);


  if (rank == MASTER) {
    // Set rank 0 chunk
    
    for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ny; j++) {
	w[j+i*ny] = image[j+i*ny];
	//printf("w[%d] <-- %2.1f \n",j+i*ny,  w[j+i*ny]);
      }
    }
    //printf("Outside ranks block\n");
    for (int i = 1; i < size-1; i++) {
      // printf("Inside rank %d\n",i);
      row_offset = i*(nx/size)-1;
      nrows_for_ranks= (nx/size)+2;
      for (int j = 0; j < nrows_for_ranks; j++) {
	MPI_Send(&image[(row_offset+j)*ny], ny, MPI_FLOAT, i, tag, MPI_COMM_WORLD);
      }
    }
   
    if(size > 1){
      //printf(" inside last rank %d\n",rank);
    row_offset = ((nx/size)*(size-1))-1;
    nrows_for_ranks = nx-((nx/size)*(size-1))+1;
    // printf(" rows %d last rank %d\n",nrows_for_ranks,rank);

    for (int j = 0; j < nrows_for_ranks; j++) {
      MPI_Send(&image[(row_offset+j)*ny], ny, MPI_FLOAT, size-1, tag, MPI_COMM_WORLD);
   }
    }

    // Call the stencil kernel
    tic = wtime();
    for (int t = 0; t < niters; ++t) {
      if (size>1){
      halo_exchange(rank, size, nrows, w, ny);
      stencil(nrows, ny, w, u);
      halo_exchange(rank, size, nrows, u, ny);
      stencil(nrows, ny, u, w);
      }
      else {
	//printf("Calling simple stencil without halo\n");
	stencil(nrows, ny, w, u);
	stencil(nrows, ny, u, w);
      }
    }
    toc = wtime();
    //printf("Saving output for rank %d\n",rank);
    // Save rank 0 chunk
    if (size > 1){
    for (int i = 0; i < nrows-1; i++) {
      for (int j = 0; j < ny; j++) {
	image[j+i*ny] = w[j+i*ny];
	//printf("image[%d] <-- %2.1f\n",j+i*ny, image[j+i*ny]);

      }
    }
    }
    else {
      for (int i = 0; i < nrows; i++) {
	for (int j = 0; j < ny; j++) {
	  image[j+i*ny] = w[j+i*ny];
	  //printf("image[%d] <-- %2.1f\n",j+i*ny, image[j+i*ny]);
	}
      }
    }
    // printf("Saved output for rank %d\n",rank);

    // Receive chunks from cores
    for (int i = 1; i < size-1; i++) {
      for (int j = 0; j < nx/size; j++) {
	MPI_Recv(&image[(j+(i*(nx/size)))*ny], BUFSIZ, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
      } 
    }
    
    if(size>1){
      // printf("Inside rank %d\n",rank);
      for (int j = 0; j < nx-((nx/size)*(size-1)); j++) {
      MPI_Recv(&image[(j+((size-1)*(nx/size)))*ny], BUFSIZ, MPI_FLOAT, size-1, tag, MPI_COMM_WORLD, &status);
    }
    }
    
    output_image(OUTPUT_FILE, nx, ny, image);
  }
  else {
    // Receive chunk from core 0
    // printf("Rows in rank %d %d\n", nrows, rank);
    for (int j = 0; j < nrows; j++) {
      MPI_Recv(&w[j*ny], BUFSIZ, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
      // printf("w[%d] <-- %2.1f\n",j*ny,w[j*ny]);
    }

    // Call the stencil kernel
     tic = wtime();
    for (int t = 0; t < niters; ++t) {
      halo_exchange(rank, size, nrows, w, ny);
      stencil(nrows, ny, w, u);
      halo_exchange(rank, size, nrows, u, ny);
      stencil(nrows, ny, u, w);
    }
     toc = wtime();

    // Send chunk back to master
    if (rank == size-1) {
      row_offset = 1;
      nrows_for_ranks = nrows - 1;
    } else {
      row_offset = 1;
      nrows_for_ranks = nrows - 2;
    }
    for (int j = 0; j < nrows_for_ranks; j++) {
      // printf("sending back to master from rank %d\n", rank);
      MPI_Send(&w[(row_offset+j)*ny], ny, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    }

  }

  // Output
  printf("------------------------------------\n");
  printf(" process %d of %d runtime: %lf s\n", rank+1, size, toc-tic);
  printf("------------------------------------\n");
        
        
  MPI_Finalize(); /* finialise the MPI enviroment */
    
  _mm_free(image);
  _mm_free(tmp_image);
  _mm_free(w);
  _mm_free(u);
        
        
        
}
void halo_exchange(int rank, int size, int local_nrows, float * w, int ny) {
  int tag = 0;
  MPI_Status status;     /* struct used by MPI_Recv */
  if (rank == MASTER) {
    MPI_Send(&w[(local_nrows-2)*ny], ny, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
    MPI_Recv(&w[(local_nrows-1)*ny], BUFSIZ, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &status);
  } else if (rank == size-1) {
    MPI_Send(&w[ny], ny, MPI_FLOAT, size-2, tag, MPI_COMM_WORLD);
    MPI_Recv(&w[0], BUFSIZ, MPI_FLOAT, size-2, tag, MPI_COMM_WORLD, &status);
  } else {
    MPI_Send(&w[ny], ny, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD);
    MPI_Recv(&w[0], BUFSIZ, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD, &status);
    MPI_Send(&w[(local_nrows-2)*ny], ny, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD);
    MPI_Recv(&w[(local_nrows-1)*ny], BUFSIZ, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD, &status);
  }
}


void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  int i,j;
  for (i=1;i<nx-1;++i){
    for(j=1;j<ny-1;++j) {
      tmp_image[i*ny+j] =image[i*ny+j]*0.6f+ (image[(i+1)*ny+j]+ image[(i-1)*ny+j]+image[i*ny+j-1]+ image[i*ny+j+1])*0.1f;
    }
  }
  tmp_image[0] =image[0]*0.6f+(image[1]+image[ny])*0.1f;
  tmp_image[ny-1] =image[ny-1]*0.6f+ (image[ny-2]+ image[ny*2-1])*0.1f;
  tmp_image[(nx-1)*ny] =image[(nx-1)*ny]*0.6f+(image[(nx-2)*ny]+image[(nx-1)*ny+1])*0.1f;
  tmp_image[ny-1+(nx-1)*ny] =image[ny-1+(nx-1)*ny]*0.6f+(image[ny-1+(nx-2)*ny]+image[ny-1+(nx-1)*ny])*0.1f;

  for (i=1; i<nx-1 ; ++i) {
    tmp_image[i*ny] = image[i*ny] * 0.6f + (image[(i-1)*ny] + image[(i+1)*ny]+ image[1+i*ny]) * 0.1f;
    tmp_image[(ny-1)+i*ny] = image[(ny-1)+i*ny] * 0.6f + (image[(ny-1)+(i-1)*ny] + image[(ny-1)+(i+1)*ny]+ image[(ny-2) +i*ny]) * 0.1f;
  }
  for (j=1; j<ny-1; ++j) {
    tmp_image[j] = image[j] * 0.6f+ (image[j-1]+ image[j+1]+ image[j+ny]) * 0.1f;
    tmp_image[j+(nx-1)*ny] = image[j+(nx-1)*ny] * 0.6f + (image[j-1+(nx-1)*ny] + image[j+1+(nx-1)*ny] + image[j +(nx-2)*ny]) * 0.1f;
  }
}

void init_image(const int nx, const int ny, float *  image, float *  tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
	    image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}


void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
