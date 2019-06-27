#include <iostream>
#include <stdlib.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <time.h>
#include <cstdio>
#include "stdio.h"

using namespace std;
using namespace cv;


__global__ void make_histogram (unsigned char *image, int width, int height, int *histogram)
// make the histogram and convert the image from RGB to YUV
{

	//gridDim.x -> the number of thread blocks
	//blockDim.x -> the number of threads in each block
	//blockIdx.x -> the index the current block within the grid
	//threadIdx.x -> the index of the current thread within the block
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	long index;

	for(int i = idx; i < width * height; i += blockDim.x * gridDim.x){

		index = i * 3;

		int R = image[index];
		int G = image[index + 1];
		int B = image[index + 2];

		int Y = R * .299000 + G * .587000 + B * .114000;
		int U = R * -.168736 + G * -0.331264 + B * .500000 + 128;
		int V = R * .500000 + G * -.418688 + B * -.081312 + 128;

		//perform an add operation in the memory location pointed by the 1st parameter
		atomicAdd(&(histogram[Y]),1);

		image[index] = Y;
		image[index + 1] = U;
		image[index + 2] = V;
	}

	__syncthreads();
}
__global__ void normalizeCdf (int *equalized, int *cumulative_dist, int *histogram, int width, int height)
// Compute the equalize (cumulative_dist normalization )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int k = idx; k < 256; k += blockDim.x * gridDim.x){
		equalized[k] = (int)(((float)cumulative_dist[k] - histogram[0])/((float)width * height - 1) * 255);
	}
}

__global__ void equalize (unsigned char *image, int *cumulative_dist,int *histogram, int *equalized, int width, int height)
{
// edit Y channel with equalized vector and reconvert from YUV to RGB
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long index;

	for(int i = idx; i < width * height; i += blockDim.x * gridDim.x){

		index = i * 3;

		int Y = equalized[image[index]];
		int U = image[index + 1];
		int V = image[index + 2];

		unsigned char R = (unsigned char)max(0, min(255,(int)(Y + 1.4075 * (V - 128))));
		unsigned char G = (unsigned char)max(0, min(255,(int)(Y - 1.3455 * (U - 128) - (.7169 * (V - 128)))));
		unsigned char B = (unsigned char)max(0, min(255,(int)(Y + 1.7790 * (U - 128))));

		image[index] = R;
		image[index + 1] = G;
		image[index + 2] = B;

	}
}

Mat vectorToMat(unsigned char *image,int height,int width){
	Mat img(Size(width,height),CV_8UC3);

	int index = 0;
	for (int i = 0; i<height;i++){
		for (int j = 0; j<width;j++){
			unsigned char R = image[index];
			unsigned char G = image[index + 1];
			unsigned char B = image[index + 2];

			Vec3b intensity = img.at<Vec3b>(i,j);

			intensity.val[0] = R;
			intensity.val[1] = G;
			intensity.val[2] = B;
			//printf("%d %d %d",R,G,B);
			img.at<Vec3b>(i,j) = intensity;
			index = index+3;

		}
	}
	return img;
}

int main(){
	string folder_path = "/home/giuliocalamai/eclipse-workspace/CUDA/img/";
	string image_path = "car.jpg";

	Mat image = imread(folder_path + image_path,IMREAD_COLOR);		//load the image

	//Size size (100, 100);
	//resize(image, image, size);

	if(!image.data){
		cout << "no image found";
		return -1;
	}
	//imshow("Original Image", image);

	struct timeval start, end;
	gettimeofday(&start, NULL);

	int width = image.cols;
	int height = image.rows;

	int host_equalized[256];					//cpu equalized histogram
	int host_cumulative_dist[256];				//cpu cumulative dist
	unsigned char *host_image = image.ptr();	//Mat image to array image
	int host_histogram[256] = {0};				//cpu histogram

	int *device_equalized;				//gpu equalized histogram
	int *device_cumulative_dist;		//gpu cumulative dist
	unsigned char *device_image;		//gpu image
	int *device_histogram;				//gpu histogram


	//GPU space allocation
	cudaMalloc((void **)&device_image, sizeof(char) * (width * height * 3));
	cudaMalloc((void **)&device_histogram, sizeof(int) * 256);
	cudaMalloc((void **)&device_equalized, sizeof(int) * 256);
	cudaMalloc((void **)&device_cumulative_dist, sizeof(int) * 256);

	//copy to gpu, cudaMemcpyHostToDevice specifies the direction of the copy "Host-->Device(CPU-->GPU)"
	cudaMemcpy(device_image, host_image, sizeof(char) * (width * height * 3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_histogram, host_histogram, sizeof(int) * 256, cudaMemcpyHostToDevice);

	int block_size = 256;													// number of threads per block
	int grid_size = (width * height + (block_size - 1))/block_size;     	// number of blocks in the grid

	//call first kernel
	make_histogram<<<grid_size, block_size>>> (device_image, width, height, device_histogram);

	//copy to cpu, cudaMemcpyDeviceToHost specifies the direction of the copy "Device-->Host(GPU-->CPU)"
	cudaMemcpy(host_histogram, device_histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost);

	//compute cumulative_dist in cpu
	host_cumulative_dist[0] = host_histogram[0];
	for(int i = 1; i < 256; i++){
		host_cumulative_dist[i] = host_histogram[i] + host_cumulative_dist[i-1];
	}

	//copy to gpu
	cudaMemcpy(device_cumulative_dist, host_cumulative_dist, sizeof(int) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(device_equalized, host_equalized, sizeof(int) * 256, cudaMemcpyHostToDevice);

	//call second kernel
	normalizeCdf<<<grid_size, block_size>>>(device_equalized, device_cumulative_dist, device_histogram, width, height);
	//call third kernel
	equalize<<<grid_size, block_size>>>(device_image, device_cumulative_dist, device_histogram, device_equalized, width, height);

	//copy to cpu
	cudaMemcpy(host_image, device_image, sizeof(char) * (width * height * 3), cudaMemcpyDeviceToHost);


	cudaFree(device_image);						//free gpu
	cudaFree(device_histogram);					//
	cudaFree(device_equalized);					//
	cudaFree(device_cumulative_dist);			//

	gettimeofday(&end, NULL);

	double elapsed = ((end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/1000)/1.e3;

	cout << elapsed;

	cout << "correctly freed memory \n";


	// Call vectorToMat for cast ptr image to Mat
	Mat final_image = vectorToMat(host_image,height,width);
	//imshow("Equalized",final_image);

	string save_folder_path = "/home/giuliocalamai/eclipse-workspace/CUDA/img_after_eq/";
	string save_image_path = "car.jpg";

	imwrite(save_folder_path + save_image_path, final_image);	//save equalized RGB image

	cout << "correctly saved image";


	//waitKey(0);
	return 0;




}
