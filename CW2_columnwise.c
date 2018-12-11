#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <sys/time.h>

#define MASTER 0

#define OUTPUT_FILE "stencil.pgm"
void stencil(const int nx, const int ny, float *  image, float *  tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
double wtime(void);

/* function prototypes */
int calc_ncols_from_rank(int rank, int size,int NCOLS);


int main(int argc, char* argv[] )
{
  // printf("%d, %d, %d",atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));
  int nx =atoi(argv[1]); 
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

 float *image =malloc(sizeof(float)*nx*ny);
  float *tmp_image =malloc(sizeof(float)*nx*ny);
 

  int NROWS=nx;
  int NCOLS=ny;
  int ii,jj;             /* row and column indices for the grid */
  int kk;                /* index for looping over ranks */
  int start_col,end_col; /* rank dependent looping indices */
  int iter;              /* index for timestep iterations */ 
  int rank;              /* the rank of this process */
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */
  int remote_ncols;      /* number of columns apportioned to a remote rank */
  
  float *u;            /* local temperature grid at time t - 1 */
  float *w;            /* local temperature grid at time t     */
  float *sendbuf;       /* buffer to hold values to send */
  float *recvbuf;       /* buffer to hold received values */
  float *printbuf;      /* buffer to hold values for printing */
  double tic,toc;
  /* MPI_Init returns once it has started up processes */
  /* get size and rank */ 
  MPI_Init(&argc, &argv);
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  
  printf("Initialised grid in rank %d",rank);
  printf("\n");
  init_image(nx, ny, image, tmp_image);
 
 /* 
 *   ** determine process ranks to the left and right of rank
 *     ** respecting periodic boundary conditions
 *       */
  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  /* 
 *   ** determine local grid size
 *     ** each rank gets all the rows, but a subset of the number of columns
 *       */
  local_nrows = NROWS;
  local_ncols = calc_ncols_from_rank(rank, size, NCOLS);

  /*
 *   ** allocate space for:
 *     ** - the local grid (2 extra columns added for the halos)
 *       ** - we'll use local grids for current and previous timesteps
 *         ** - buffers for message passing
 *           */
  u = (float*)malloc(sizeof(float*) * local_nrows * (local_ncols + 2));
  
  w = (float*)malloc(sizeof(float*) * local_nrows * (local_ncols + 2));
  
  sendbuf = (float*)malloc(sizeof(float) * local_nrows);
  recvbuf = (float*)malloc(sizeof(float) * local_nrows);
  /* The last rank has the most columns apportioned.
 *      printbuf must be big enough to hold this number */ 

  remote_ncols = calc_ncols_from_rank(size-1, size,NCOLS);
  printbuf = (float*)malloc(sizeof(float) * (remote_ncols + 2));
  
  /*
 *   ** initialize the local grid for the present time (w):
* - set boundary conditions for any boundaries that occur in the local grid
 *       ** - initialize inner cells to the average of all boundary cells
 *         ** note the looping bounds for index jj is modified 
 *           ** to accomodate the extra halo columns
 *             ** no need to initialise the halo cells at this point
*               */


int img_ind_on_rank=rank*(int)(ny/size);
 for(ii=0;ii<local_nrows;ii++) {
    for(jj=0; jj<local_ncols +2 ; jj++) {
	w[ii*(local_ncols+2)+jj+1]=image[ii*ny+img_ind_on_rank+jj];
}
}

/*
   ** time loop
 *     */

  for(iter=0;iter<niters*2;iter++) {
    /*
 *     ** halo exchange for the local grids w:
 *         ** - first send to the left and receive from the right,
 *             ** - then send to the right and receive from the left.
 *                 ** for each direction:
 *                     ** - first, pack the send buffer using values from the grid
 *                         ** - exchange using MPI_Sendrecv()
 *                             ** - unpack values from the recieve buffer into the grid
 *                                 */

    /* send to the left, receive from right */
    for(ii=0;ii<local_nrows;ii++)
      sendbuf[ii] =w[ii * (local_ncols + 2) + 1];;
    MPI_Sendrecv(sendbuf, local_nrows, MPI_FLOAT, left, tag,
		 recvbuf, local_nrows, MPI_FLOAT, right, tag,
		 MPI_COMM_WORLD, &status);
    for(ii=0;ii<local_nrows;ii++)
        w[ii * (local_ncols + 2) + local_ncols + 1] = recvbuf[ii];

    /* send to the right, receive from left */
    for(ii=0;ii<local_nrows;ii++)
      sendbuf[ii] = w[ii * (local_ncols + 2) + local_ncols];
    MPI_Sendrecv(sendbuf, local_nrows, MPI_FLOAT, right, tag,
		 recvbuf, local_nrows, MPI_FLOAT, left, tag,
		 MPI_COMM_WORLD, &status);
    for(ii=0;ii<local_nrows;ii++)
       w[ii * (local_ncols + 2)] = recvbuf[ii];

    /*
 *     ** copy the old solution into the u grid
 *         */ 
    for(ii=0;ii<local_nrows;ii++) {
      for(jj=0;jj<local_ncols + 2;jj++) {
	u[ii*(local_ncols + 2)+jj] = w[ii*(local_ncols + 2)+jj];   
    }
}
 tic = wtime();
stencil(local_nrows,local_ncols + 2,w,u);
 toc = wtime();
}

 printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");


for(ii=0; ii < local_nrows; ii++) {
    if(rank == MASTER) {
      for(jj=0; jj < local_ncols + 2; jj++) {
	image[ii*ny+img_ind_on_rank+jj]=w[ii * (local_ncols + 2) + jj];
      }
       for(kk=1; kk < size; kk++) { /* loop over other ranks */
	remote_ncols = calc_ncols_from_rank(kk, size,NCOLS);
	MPI_Recv(printbuf, remote_ncols + 2, MPI_FLOAT, kk, tag, MPI_COMM_WORLD, &status);
	offset_rank=kk*(int)(ny/size);
	for(jj=0; jj < remote_ncols + 2; jj++) {
	 image[ii*ny+offset_rank+jj]=printbuf[jj+1];
	}
      }
    }
    else {
      MPI_Send(&w[ii * (local_ncols + 2)], local_ncols + 2, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
    }
  }
 if(rank==MASTER){
  output_image(OUTPUT_FILE, nx, ny, image);
 }


MPI_Finalize();
  free(w);
  free(sendbuf);
  free(recvbuf);
  free(printbuf);
return EXIT_SUCCESS;
}

void stencil(const int nx, const int ny, float * restrict  image, float *  restrict tmp_image) {
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

void init_image(const int nx, const int ny,  float *restrict   image, float * restrict tmp_image) {
 int i,j;
    for ( j = 0; j < ny; ++j) {
    for ( i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
}
 int jj,ii;
  for ( j = 0; j < 8; ++j) {
    for ( i = 0; i < 8; ++i) {
      for ( jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for ( ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
   }
  }
 /*
   for ( j = 0; j < ny; ++j) {
    for ( i = 0; i < nx; ++i) {
      printf("[%d]- %2.1f ",(j+i*ny),image[j+i*ny]);
    }
    printf("\n");
  }*/
 
}

void output_image(const char * file_name, const int nx,  int ny, float * restrict image) {
 FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }
 fprintf(fp, "P5 %d %d 255\n", nx, ny);

double maximum = 0.0;
  int i,j;
  for ( j = 0; j < ny; ++j) {
    for ( i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }
  for ( j = 0; j < ny; ++j) {
    for ( i = 0; i < nx; ++i) {
      printf("[%d]- %2.1f ",(j+i*ny),(image[j+i*ny]/maximum));
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }
 fclose(fp);

}
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
} 
 
 int calc_ncols_from_rank(int rank, int size, int NCOLS)
{
  int ncols;

  ncols = NCOLS / size;       /* integer division */
  if ((NCOLS % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += NCOLS % size;  /* add remainder to last rank */
  }
  
  return ncols;
}
