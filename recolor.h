#ifndef RECOLOR_H
#define RECOLOR_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Image structure
typedef struct
{
    unsigned char *data; // RGB data (host)
    int width;
    int height;
    int channels; // Usually 3 for RGB
} Image;

// Color mapping structure (customize based on your recoloring algorithm)
typedef struct
{
    unsigned char src_r, src_g, src_b;
    unsigned char dst_r, dst_g, dst_b;
    float tolerance; // For color matching
} ColorMap;

typedef struct ColorsInUse
{
    unsigned char src_r, src_g, src_b;
};

// Function declarations

// Image I/O
Image *loadImage(const char *filename);
void saveImage(const char *filename, Image *img);
void freeImage(Image *img);

// CUDA kernel launcher
void launchRecolorKernel(unsigned char *d_input, unsigned char *d_output,
                         int width, int height, int channels,
                         ColorMap *d_colorMaps, int numMaps);

// Utility
void printImageInfo(Image *img);

#endif // RECOLOR_H