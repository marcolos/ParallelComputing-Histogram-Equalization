#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/time.h>
#include <time.h>

using namespace std;
using namespace cv;

void make_histogram(Mat image, int histogram[], int *yuv_vector)
// make the histogram and convert from RGB to YUV creating a coloumn vector yuv_vector with col*row*3 dimension
{
    // initialize all intensity values to 0
    for(int i = 0; i < 256; i++){
        histogram[i] = 0;
    }

    int index = 0;
    for(int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++)
        {

            Vec3b intensity = image.at<Vec3b>(i,j);

            int R = intensity.val[0]; // R is red
            int G = intensity.val[1]; // G is green
            int B = intensity.val[2]; // B is blue

            // Convert from RGB into YUV
            int Y = R * .299000 + G * .587000 + B * .114000; // Y is the luminous level
            int U = R * -.168736 + G * -.331264 + B * .500000 + 128;
            int V = R * .500000 + G * -.418688 + B * -.081312 + 128;

            histogram[Y]++;

            // memorize in a column vector all the pixels of the image in RGB format ( the yuv_vector is initialized like rows*cols*3 )
        
            yuv_vector[index] = Y;
            yuv_vector[index + 1] = U;
            yuv_vector[index + 2] = V;

            index = index +3;
        }
    }
}


void cumulative_histogram(int histogram[], int equalized[], int cols, int rows)
// compute the cdf(named cumulative_histogram) and normalize it (named equalized)
{
    int cumulative_histogram[256];

    cumulative_histogram[0] = histogram[0];

    for(int i = 1; i < 256; i++)
    {
        cumulative_histogram[i] = histogram[i] + cumulative_histogram[i-1];
        equalized[i] = (int)(((float)cumulative_histogram[i] - histogram[0])/((float)cols * rows - 1)*255);
    }
}

void equalize(Mat image, int equalized[], int *yuv_vector)
// edit Y channel with equalized vector and reconvert from YUV to RGB 
{
    int index = 0;
    for(int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {


            int Y = equalized[yuv_vector[index]];
            int U = yuv_vector[index+1];
            int V = yuv_vector[index+2];

            unsigned char R = (unsigned char)max(0, min(255,(int)(Y + 1.4075 * (V - 128))));
            unsigned char G = (unsigned char)max(0, min(255,(int)(Y - 0.3455 * (U - 128) - (0.7169 * (V - 128)))));
            unsigned char B = (unsigned char)max(0, min(255,(int)(Y + 1.7790 * (U - 128))));

            Vec3b intensity = image.at<Vec3b>(i,j);

            intensity.val[0] = R;
            intensity.val[1] = G;
            intensity.val[2] = B;

            image.at<Vec3b>(i,j) = intensity;

            index = index + 3;
        }
    }
}

int main(){

    // Load the image in RGB format. For each matrix cell there are 3 bytes that identifies RGB pixel
    Mat image = imread("../img/car.jpg", IMREAD_COLOR);

    //resize(image, image, Size(10000, 10000));
    //imshow("Original Image", image);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // For big image dimension is required to allocate array for avoid the segmentation fault
    int *yuv_vector = new int[image.rows * image.cols * 3];

    // Generate the histogram
    int histogram[256];
    make_histogram(image, histogram, yuv_vector);

    // Generate the equalized histogram
    int equalized[256];

    cumulative_histogram(histogram,equalized, image.cols, image.rows);

    equalize(image, equalized, yuv_vector);

    gettimeofday(&end, NULL);
    double elapsed_time = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    cout << elapsed_time << endl;

    // Display equalized image
    //resize(image, image, Size(800,800));
    //imshow("Equalized Image",image);

    imwrite("/Users/marco/Project/ParallelComputing-Histogram-Equalization/Sequential/img_after_eq/car.jpg", image);
    //waitKey(0);
    return 0;

}