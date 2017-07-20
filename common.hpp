#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

//lmath.cpp
extern void get_a(Mat &a, int w);
extern double cal_a_det(int n);
extern void print_float_mat(Mat m);
extern void cal_a_inv(int n, Mat &ans);
extern bool less_than( Mat &x, Mat &y );
extern void solve_FR( Mat &A, Mat &b, Mat &ans, double delta );
Mat cal_Ap( int rw, int rh, Mat &p );
void solve_FR_SparseA( int rw, int rh, Mat &b, Mat &ans, double delta );

//poisson.cpp
extern int get_ind(int i, int j, int w);
extern void getA(Mat &A, int h, int w);
extern void getB( Mat &img_front, Mat &img_back, Rect roi, Point pt, Mat &B);
extern void poisson(Mat &img_front, Mat &img_back, Rect roi, Point pt, Mat &ans);

//tools.cpp
extern int print_mat_info(Mat mat, const char* s);
extern void simple_replace(const Mat& obj, Point pt, const Mat &src, Rect roi, Mat& ans);
extern bool valid_roi(const Mat& img, Rect roi);
