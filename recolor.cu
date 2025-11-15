#include "recolor.h"
#include <curand_kernel.h>


__global__ void setup_kernel_randomness(curandState *state, unsigned long seed, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_id = y * width + x;  // Simple linear index
    curand_init(seed, thread_id, 0, &state[thread_id]);
}

// CUDA kernel for recoloring
__global__ void recolorKernel(unsigned char* input, unsigned char* output,
                               int width, int height, int channels,
                               ColorMap* colorMaps, curandState* d_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * channels;
    
    int thread_id = y * width + x;  // Simple linear index


int pixelOffset = 20;
// Calculate offset pixel position
int x2 = curand(&d_state[thread_id]) % width;
int y2 = curand(&d_state[thread_id]) % height;

// Bounds check
if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) {
    int offsetIdx = (y2 * width + x2) * channels;
    
    
    
    output[idx] = input[offsetIdx];
    output[idx + 1] = input[offsetIdx + 1];
    output[idx + 2] = input[offsetIdx + 2];
} else {
    // Out of bounds - copy original pixel
    output[idx] = 0;
    output[idx + 1] = 0;
    output[idx + 2] = 0;
}
}

// Kernel launcher
void launchRecolorKernel(unsigned char* d_input, unsigned char* d_output,
                         int width, int height, int channels,
                         ColorMap* d_colorMaps, int numMaps) {
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Allocate state on device for random
    int n = width * height;
    curandState *d_state;
    cudaMalloc(&d_state, n * sizeof(curandState));

    setup_kernel_randomness<<<gridSize, blockSize>>>(d_state, time(NULL), width);
    
    // Launch kernel
    recolorKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                           channels, d_colorMaps, d_state);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}



// Main function
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }
    
    // Load input image
    printf("Loading image: %s\n", argv[1]);
    Image* img = loadImage(argv[1]);
    if (!img) return 1;
    printImageInfo(img);
    
    // Allocate device memory
    size_t imageSize = img->width * img->height * img->channels;
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, img->data, imageSize, cudaMemcpyHostToDevice));
    
    // get color map of image (quantize r, g, and b into N number of buckets)
    // pass color map to kernel
    // kernel sample randomly from initial color map per pixel

    // Define color mappings (example: swap red with blue)
    ColorMap hostMaps[] = {
        {255, 0, 0,    0, 0, 255,    50.0f},  // Red -> Blue
        {0, 255, 0,    255, 255, 0,  50.0f},  // Green -> Yellow
    };
    int numMaps = sizeof(hostMaps) / sizeof(ColorMap);
    
    // Copy color maps to device
    ColorMap* d_colorMaps;
    CUDA_CHECK(cudaMalloc(&d_colorMaps, sizeof(ColorMap) * numMaps));
    CUDA_CHECK(cudaMemcpy(d_colorMaps, hostMaps, sizeof(ColorMap) * numMaps,
                          cudaMemcpyHostToDevice));
    
    // Launch kernel
    printf("Processing image...\n");
    launchRecolorKernel(d_input, d_output, img->width, img->height,
                        img->channels, d_colorMaps, numMaps);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(img->data, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Save output image
    printf("Saving image: %s\n", argv[2]);
    saveImage(argv[2], img);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_colorMaps));
    freeImage(img);
    
    printf("Done!\n");
    return 0;
}