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
void get_a(Mat &a, int w);
double cal_a_det(int n);
void print_float_mat(Mat m);
void cal_a_inv(int n, Mat &ans);
bool less_than( Mat &x, Mat &y );
void solve_FR( Mat &A, Mat &b, Mat &ans, double delta );
Mat cal_Ap( int rw, int rh, Mat &p );
void solve_FR_SparseA( int rw, int rh, Mat &b, Mat &ans, double delta );

//poisson.cpp
int get_ind(int i, int j, int w);
void getA(Mat &A, int h, int w);
vector< vector< int > >  get_sparseA(int h, int w);
void getB( Mat &img_front, Mat &img_back, Rect roi, Point pt, Mat &B);
void poisson(Mat &img_front, Mat &img_back, Rect roi, Point pt, Mat &ans);

//tools.cpp
int print_mat_info(Mat mat, const char* s);
void simple_replace(const Mat& obj, Point pt, const Mat &src, Rect roi, Mat& ans);
bool valid_roi(const Mat& img, Rect roi);
