#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <math.h>

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

//Mat tst = imread( "1_ori.jpg" , CV_LOAD_IMAGE_COLOR);

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
void getMaskMapTable(Mat &Mask, Rect roi, vector<vector<int> > & MapId, vector<pair<int,int> > &IdMap); 
void polygonPoisson(Mat &img_front, Mat &img_back, Mat &mask, Rect roi, Point pt, Mat &ans);


//tools.cpp
int print_mat_info(Mat mat, const char* s);
void simple_replace(const Mat& obj, Point pt, const Mat &src, Rect roi, Mat& ans);
bool valid_roi(const Mat& img, Rect roi);

//test.cpp
void dst(const Mat& src, Mat& dest, bool invert = false);
void idst(const Mat& src, Mat& dest);
void solve_dft(const Mat &img, Mat& mod_diff, Mat &result);

//api.cpp
Mat getPoissonMat(const Mat &back, const Mat &front, Rect b_roi, int type);
Mat getPolygonPoissonMat(const Mat &back, const Mat &front, const Mat &mask, Rect b_roi, int type);
