#include "recolor.h"

// CUDA kernel for recoloring
__global__ void recolorKernel(unsigned char* input, unsigned char* output,
                               int width, int height, int channels,
                               ColorMap* colorMaps, int numMaps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * channels;
    
    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];
    
    // Simply swap red and green channels
    output[idx] = g;        // Put green in red channel
    output[idx + 1] = r;    // Put red in green channel
    output[idx + 2] = b;    // Keep blue the same
}

// Kernel launcher
void launchRecolorKernel(unsigned char* d_input, unsigned char* d_output,
                         int width, int height, int channels,
                         ColorMap* d_colorMaps, int numMaps) {
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    recolorKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                           channels, d_colorMaps, numMaps);
    
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