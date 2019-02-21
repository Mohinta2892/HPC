
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny,  float * restrict   image,  float * restrict tmp_image);
void init_image(const int nx, const int ny, float * restrict  image, float *restrict  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *  restrict image);
double wtime(void);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float *image =malloc(sizeof(float)*nx*ny);
  float *tmp_image =malloc(sizeof(float)*nx*ny);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // Call the stencil kernel
  double tic = wtime();
  int t;
  for ( t = 0; t < niters; ++t) {
    stencil(nx, ny, image, tmp_image);
    stencil(nx, ny, tmp_image, image);
  }
  double toc = wtime();


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, image);
  free(image);
}

void stencil(const int nx, const int ny, float * restrict  image, float *  restrict tmp_image) {
  int i,j;
 
    //middle cells of the matrix
        
    for (i=1;i<nx-1;++i){
      for(j=1;j<ny-1;++j) {
	tmp_image[i*nx+j] =image[i*nx+j]*0.6f+ (image[(i+1)*nx+j]+ image[(i-1)*nx+j]+image[i*ny+j-1]+ image[i*ny+j+1])*0.1f;
	/*tmp_image[i*nx+j] += image[(i-1)*nx+j]*0.1f;
	tmp_image[i*nx+j] += image[i*ny+j-1]*0.1f;
	tmp_image[i*nx+j] += image[i*ny+j+1]*0.1f;*/
      }
    }
    // corners of the matrix

    tmp_image[0] =image[0]*0.6f+(image[1]+image[nx])*0.1f;
    //tmp_image[0] +=image[nx]*0.1f;

    tmp_image[nx-1] =image[nx-1]*0.6f+ (image[nx-2]+ image[nx*2-1])*0.1f;
    //tmp_image[nx-1] += image[nx*2-1]*0.1f;

    tmp_image[nx*ny-1] =image[nx*ny-1]*0.6f+(image[(ny-1)*nx-1]+image[nx*ny-2])*0.1f;
    // tmp_image[nx*ny-1] +=image[nx*ny-2]*0.1f;

    tmp_image[(nx-1)*ny] =image[(nx-1)*ny]*0.6f+(image[(nx-2)*ny]+image[(nx-1)*ny+1])*0.1f;
    // tmp_image[(nx-1)*ny] +=image[(nx-1)*ny+1]*0.1f;

    //borders of the matrix : left-right, top- bottom

    for (i=1; i<nx-1 ; ++i) {
      tmp_image[i] = image[i]*0.6f+(image[i-1]+image[i+1]+image[i*nx+1])*0.1f;
      // tmp_image[i] +=image[i+1]*0.1f;
      // tmp_image[i] +=image[i*nx+1]*0.1f;
      tmp_image[nx*ny-i-1] =image[nx*ny-i-1]*0.6f+(image[nx*ny-i-2]+image[nx*ny-i]+image[nx*(ny-1)-i-1])*0.1f;
      //tmp_image[nx*ny-i-1] +=image[nx*ny-i]*0.1f;
      //tmp_image[nx*ny-i-1] +=image[nx*(ny-1)-i-1]*0.1f;
    }

    for (j=1; j<ny-1; ++j) {
      tmp_image[j*ny] =image[j*ny]*0.6f+(image[(j-1)*ny]+image[ny*j+1]+image[(j+1)*ny])*0.1f;
      //tmp_image[j*ny] +=image[ny*j+1]*0.1f;
      // tmp_image[j*ny] +=image[(j+1)*ny]*0.1f;
      tmp_image[ny*(nx-j)-1] =image[ny*(nx-j)-1]*0.6f+(image[ny*(nx-j)-2]+image[ny*(nx-j)-1-ny]+image[ny*(nx-j)-1+ny])*0.1f;
      //tmp_image[ny*(nx-j)-1] +=image[ny*(nx-j)-1-ny]*0.1f;
      //  tmp_image[ny*(nx-j)-1] +=image[ny*(nx-j)-1+ny]*0.1f;
    }
   
}

// Create the input image
void init_image(const int nx, const int ny,  float *restrict   image, float * restrict tmp_image) {
    // Zero everything
   int i,j;
    for ( j = 0; j < ny; ++j) {
    for ( i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
    }
    //memset(image,0,nx*ny*sizeof(float));
    // memset(image,0,nx*ny*sizeof(float));

  // Checkerboard
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
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx,  int ny, float * restrict image) {

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
  int i,j;
  for ( j = 0; j < ny; ++j) {
    for ( i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }
  
  // Output image, converting to numbers 0-255
  
  for ( j = 0; j < ny; ++j) {
    for ( i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
