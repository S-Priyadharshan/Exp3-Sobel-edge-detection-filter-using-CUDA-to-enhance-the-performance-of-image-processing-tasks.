# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>AIM:</h3>
<h3>ENTER YOUR NAME</h3>
<h3>ENTER YOUR REGISTER NO</h3>
<h3>EX. NO</h3>
<h3>DATE</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,  
                            unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        int gx = srcImage[y * width + (x + 1)] - srcImage[y * width + (x - 1)];
        int gy = srcImage[(y + 1) * width + x] - srcImage[(y - 1) * width + x];

        int mag = abs(gx) + abs(gy);
        mag = min(mag, 255);

        dstImage[idx] = (unsigned char)mag;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

void analyzePerformance(const std::vector<std::pair<int, int>>& sizes, 
                        const std::vector<int>& blockSizes, unsigned char *d_inputImage, 
                        unsigned char *d_outputImage) {
                        
    for (auto size : sizes) {
        int width = size.first;
        int height = size.second;

        printf("CUDA - Size: %dx%d\n", width, height);
        
        dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));
        for (auto blockSize : blockSizes) {
            dim3 blockDim(blockSize, blockSize);
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            sobelFilter<<<gridSize, blockDim>>>(d_inputImage, d_outputImage, width, height);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("    Block Size: %dx%d Time: %f ms\n", blockSize, blockSize, milliseconds);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
}

int main() {
    Mat image = imread("/content/images.jpg", IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    // Convert to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    int width = grayImage.cols;
    int height = grayImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage,grayImage.data,imageSize,cudaMemcpyHostToDevice));

    // Performance analysis
    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {1024, 1024}};
    std::vector<int> blockSizes = {8, 16, 32};

    analyzePerformance(sizes, blockSizes, d_inputImage, d_outputImage);

    // Execute CUDA Sobel filter one last time for the original image
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));
    dim3 blockDim(16, 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockDim>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    checkCudaErrors(cudaMemcpy(h_outputImage,d_outputImage,imageSize,cudaMemcpyDeviceToHost));

    // Output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel_cuda.jpeg", outputImage);

    // OpenCV Sobel filter for comparison
    Mat opencvOutput;
    auto startCpu = std::chrono::high_resolution_clock::now();
    cv::Sobel(grayImage, opencvOutput, CV_8U, 1, 0, 3);
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCpu - startCpu;

    // Save and display OpenCV output
    imwrite("output_sobel_opencv.jpeg", opencvOutput);
    
    printf("Input Image Size: %d x %d\n", width, height);
    printf("Output Image Size (CUDA): %d x %d\n", outputImage.cols, outputImage.rows);
    printf("Total time taken (CUDA): %f ms\n", milliseconds);
    printf("OpenCV Sobel Time: %f ms\n", cpuDuration.count());

    // Cleanup
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

## OUTPUT:
SHOW YOUR OUTPUT HERE

## RESULT:
Thus the program has been executed by using CUDA to ________________.

Questions:

What challenges did you face while implementing the Sobel filter for color images?
How did changing the block size influence the performance of your CUDA implementation?
What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Suggest potential optimizations for improving the performance of the Sobel filter.

Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:

