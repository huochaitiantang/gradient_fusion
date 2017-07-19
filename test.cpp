#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <math.h>

using namespace cv;
using namespace std;

void compute_gradient( const Mat &img, Mat &ans){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat Ans_gray, grad, grad_x, grad_y, abs_grad_x, abs_grad_y;
	GaussianBlur( img, img, Size(3,3), 0, 0, BORDER_DEFAULT );
	//cvtColor( img, Ans_gray, CV_RGB2GRAY );
	Sobel( img, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	Sobel( img, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	//Scharr( img, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//Scharr( img, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, ans );
	//addWeighted( grad_x, 0.5, grad_y, 0.5, 0, grad );
}

/* Print mat infomation
 * mat: img
 * s: mat name
 */
int print_mat_info(Mat mat, const char* s){
	cout << " -- " << s << ":" << endl;
	cout << "\tcti:" << mat.isContinuous();
	cout << " cha:" << mat.channels();
	cout << " col:" << mat.cols;
	cout << " row:" << mat.rows << endl;
	cout << endl;
	return 0;
}

/* Simply replace the part of img obj with the part of img src
 * obj : object img
 * src : source img
 * pt : object cordinate anchor
 * roi : source roi part
 * ans : result img
 */ 
void simple_replace(const Mat& obj, Point pt, const Mat &src, Rect roi, Mat& ans){
	ans = obj.clone();
	int c = src.channels();
	int orow = obj.rows;
	int ocol = obj.cols;
	if(obj.channels() != src.channels()){
		cout << "ERROR: obj and src channels not matched!" << endl;
		return;
	}
	for(int i = pt.y; i < pt.y + roi.height && i < orow; i++){
		for(int j = pt.x; j < pt.x + roi.width && j < ocol; j++){
			for(int k = 0; k < c; k++){
				ans.at<Vec3b>(i,j)[k] = src.at<Vec3b>( i - pt.y + roi.y, j - pt.x + roi.x )[k];
			}
		}
	}
}

/* Compute the index
 * i,j : cordinate
 * w : size of column
 */
int get_ind(int i, int j, int w){
	return i * w + j;
}

/* Get the matrix A in Ax = b
 * A : result
 * h : img height
 * w : img width
 */
void  getA(Mat &A, int h, int w){
	Mat M, temp, roimat;
	A = Mat::eye( h * w, h * w, CV_64FC1);
	A *= -4;
	M = Mat::zeros( h, w, CV_64FC1);

	temp = Mat::ones( h, w - 2, CV_64FC1);
	roimat = M( Rect( 1, 0, w - 2, h) );
	temp.copyTo(roimat);

	temp = Mat::ones( h - 2, w, CV_64FC1);
	roimat = M( Rect( 0, 1, w, h - 2) );
	temp.copyTo(roimat);
	
	temp = Mat::ones( h - 2, w - 2, CV_64FC1);
	temp *= 2;
	roimat = M( Rect( 1, 1, w - 2, h - 2) );
	temp.copyTo(roimat);
	//corner has 2 neighbors, border has 3 neighbors, other has 4 neighbors
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			int label = get_ind(i, j, w);
			if( M.at<double>(i, j) == 0){
				if( i == 0 ) A.at<double>( get_ind( i + 1, j, w ), label ) = 1;
				else if( i == h - 1 ) A.at<double>( get_ind( i - 1, j, w ), label ) = 1;
				if( j == 0 ) A.at<double>( get_ind( i, j + 1, w), label ) = 1;
				else if( j = w - 1 ) A.at<double>( get_ind( i, j - 1, w), label ) = 1;
			}
			else if( M.at<double>(i, j) == 1){
				if( i == 0 ){
					A.at<double>( get_ind( i + 1, j, w ), label ) = 1;
					A.at<double>( get_ind( i, j - 1, w ), label ) = 1;
					A.at<double>( get_ind( i, j + 1, w ), label ) = 1;
				}
				else if( i == h - 1 ){
					A.at<double>( get_ind( i - 1, j, w ), label ) = 1;
					A.at<double>( get_ind( i, j - 1, w ), label ) = 1;
					A.at<double>( get_ind( i, j + 1, w ), label ) = 1;
				}
				if( j == 0 ){
					A.at<double>( get_ind( i, j + 1, w ), label ) = 1;
					A.at<double>( get_ind( i - 1, j, w ), label ) = 1;
					A.at<double>( get_ind( i + 1, j, w ), label ) = 1;
				}
				else if( j == w - 1 ){
					A.at<double>( get_ind( i, j - 1, w ), label ) = 1;
					A.at<double>( get_ind( i - 1, j, w ), label ) = 1;
					A.at<double>( get_ind( i + 1, j, w ), label ) = 1;
				}
			}
			else{
				A.at<double>( get_ind( i, j - 1, w ), label ) = 1;
				A.at<double>( get_ind( i, j + 1, w ), label ) = 1;
				A.at<double>( get_ind( i - 1, j, w ), label ) = 1;
				A.at<double>( get_ind( i + 1, j, w ), label ) = 1;
			}
		}
	}
}

void  getA1(Mat &A, int h, int w){
	Mat M, temp, roimat;
	A = Mat::eye( h * w, h * w, CV_64FC1);
	A *= -4;
	M = Mat::zeros( h, w, CV_64FC1);

	temp = Mat::ones( h, w - 2, CV_64FC1);
	roimat = M( Rect( 1, 0, w - 2, h) );
	temp.copyTo(roimat);

	temp = Mat::ones( h - 2, w, CV_64FC1);
	roimat = M( Rect( 0, 1, w, h - 2) );
	temp.copyTo(roimat);
	
	temp = Mat::ones( h - 2, w - 2, CV_64FC1);
	temp *= 2;
	roimat = M( Rect( 1, 1, w - 2, h - 2) );
	temp.copyTo(roimat);

	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			int label = get_ind(i, j, w);
			if( M.at<double>(i, j) == 0 || M.at<double>(i, j) == 1){
				A.at<double>( label, label ) = 1;
			}
			else{
				A.at<double>( get_ind( i, j - 1, w ), label ) = 1;
				A.at<double>( get_ind( i, j + 1, w ), label ) = 1;
				A.at<double>( get_ind( i - 1, j, w ), label ) = 1;
				A.at<double>( get_ind( i + 1, j, w ), label ) = 1;
			}
		}
	}
}

void  compute_lap(Mat &img, Mat &ans){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat img_blur,gradX, gradY, gradXabs, gradYabs, lapX, lapY;
	GaussianBlur( img, img_blur, Size(3,3), 0, 0, BORDER_DEFAULT );
	Sobel( img_blur, gradX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );	
	Sobel( img_blur, gradY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );	
	//Scharr( img_blur, gradX, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );	
	//Scharr( img_blur, gradY, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );	
	convertScaleAbs( gradX, gradXabs );
	convertScaleAbs( gradY, gradYabs );
	
	Mat kx = Mat::zeros(1, 3, CV_8S);
	kx.at<char>(0,0) = -1;
	kx.at<char>(0,1) = 1;
	filter2D( gradXabs, lapX, CV_32F, kx );

	Mat ky = Mat::zeros(1, 3, CV_8S);
	ky.at<char>(0,0) = -1;
	ky.at<char>(1,0) = 1;
	filter2D( gradYabs, lapY, CV_32F, ky );

	ans = lapX + lapY;
}

/* Compute matrix b for Ax = b
 * img_front : front img
 * img_back : background img
 * roi : roi of front
 * pt : anchor of background
 * B : result
 */
void getB( Mat &img_front, Mat &img_back, Rect roi, Point pt, Mat &B){
	Mat Lap;
	int rh = roi.height;
	int rw = roi.width;
	Mat lk = Mat::zeros( 3, 3, CV_64FC1 );
	lk.at<double>( 0, 1 ) = 1.0;
	lk.at<double>( 1, 0 ) = 1.0;
	lk.at<double>( 1, 2 ) = 1.0;
	lk.at<double>( 2, 1 ) = 1.0;
	lk.at<double>( 1, 1 ) = -4.0;
	B = Mat::zeros( rh * rw, 1, CV_64FC1);
	//get Lap matrix
	filter2D( img_front, Lap, -1, lk);
	//compute_lap( img_front, Lap );
	for(int i = 0; i < rh; i++){
		for( int j = 0; j < rw; j++){
			double tmp = 0.0;
			tmp += Lap.at<double>( (i + roi.y ), ( j + roi.x ) );
			//border lap value should substract background pixel value
			if( i == 0 ) tmp -= img_back.at<double>( i - 1 + pt.y, j + pt.x );
			else if( i == rh - 1) tmp -= img_back.at<double>( i + 1 + pt.y, j + pt.x );
			if( j == 0 ) tmp -= img_back.at<double>( i + pt.y, j - 1 + pt.x );
			else if( j == rw - 1) tmp -= img_back.at<double>( i + pt.y, j + 1 + pt.x );
			B.at<double>( get_ind( i, j, rw ), 0) = tmp;
		}
	}
}

/* Solve for Ax = b
 * A : matrix A, (width*height)x(width*height)
 * B : matrix b, (wight*height)x1
 * h : img height
 */
Mat get_result(Mat &A, Mat &B, int h){
	Mat ans;
	long start, end;
	start = time(NULL);
	//2500x2500 about 10s
	solve( A, B, ans, DECOMP_LU );
	//2500x2500 about 46s
	//Mat A_inv;
	//invert( A, A_inv, DECOMP_LU );
	//ans = A_inv * B;
	end = time(NULL);
	cout << "\tSolve cost " << (end - start) * 1000 << " ms." << endl;
	ans = ans.reshape(0, h);
	return ans;
}

/*
 * Poisson fusion
 * img_front : front img
 * img_back : background img
 * roi : roi of front
 * pt : anchor of background
 * ans : result img
 */
void poisson(Mat &img_front, Mat &img_back, Rect roi, Point pt, Mat &ans){
	int rh = roi.height;
	int rw = roi.width;
	Mat A, B;
	long start, end;
	getA( A, rh, rw);
	print_mat_info(A,"A");
	vector <Mat> rgb_f,rgb_b,result;
	split(img_front, rgb_f);
	split(img_back, rgb_b);
	start = time(NULL);
	for(int k = 0; k < 3; k++){
		cout << " For rgb[" << k << "]..." << endl;
		getB(rgb_f[k], rgb_b[k], roi, pt, B);	
		result.push_back( get_result( A, B, rh ) );
	}
	end = time(NULL);
	cout << " Poisson cost " << (end - start) * 1000 << " ms." << endl;
	merge( result, ans );
}

/* 
 * Check if the roi of img is valid
 */
bool valid_roi(const Mat& img, Rect roi){
	if(	roi.x < 0 || roi.x >= img.cols ||
		roi.y < 0 || roi.y >= img.rows ||
		roi.x + roi.width > img.cols || 
		roi.y + roi.height > img.rows ) return false;
	else return true;
}

/*
 * Handle for simple fusion and poisson edit, 
 * put the f_roi of front to the b_roi of background
 */
void handle(const char* background_name, const char* front_name, Rect b_roi, Rect f_roi, const char* save_name ){
	char fname[128];
	Mat back, fron, roi_front, tmp;
	back = imread( background_name, CV_LOAD_IMAGE_COLOR );
	fron = imread( front_name, CV_LOAD_IMAGE_COLOR );
	if ( !back.data || !fron.data ){
		cout << "No Image Data!" << endl;
		return;
	}
	// check if the roi valid
	if ( !valid_roi( back, b_roi ) || !valid_roi( fron, f_roi ) ){
		cout << "Invalid Roi In Image!" << endl;
	}
	//the roi part should be same as background roi
	Rect roi( 0, 0, b_roi.width, b_roi.height );
	//rezie the front roi to the size of background roi, and get the matrix roi_front
	fron( f_roi ).copyTo( roi_front );
	resize( roi_front, roi_front, roi.size() );
	print_mat_info( back, "background" );
	print_mat_info( roi_front, "roi_front" );
	//simple replace the background roi with the front roi
	Mat smp_rep;
	simple_replace( back, b_roi.tl(), roi_front, roi, smp_rep );
	sprintf( fname, "simple_replace_%s.jpg", save_name );
	imwrite( fname, smp_rep );
	//poisson fusion
	Mat res,in1, in2;
	back.convertTo( in1, CV_64FC3 );
	roi_front.convertTo( in2, CV_64FC3 );
	poisson(in2, in1, roi, b_roi.tl(), res);
	res.convertTo(res, CV_8UC1);
	//copy the roi part to the background
	Mat roimat = back( b_roi );
	res.copyTo(roimat);
	sprintf( fname, "poisson_%s.jpg", save_name );
	imwrite( fname, back );
	imshow( "poisson", back );
} 

void get_a(Mat &a, int w){
	a = Mat::zeros( w, w, CV_64FC1 );
	for(int i = 0; i < w; i++){
		a.at<double>(i,i) = -4;
		if(i > 0) a.at<double>(i,i-1) = 1;
		if(i < w - 1) a.at<double>(i,i+1) = 1;
	}
}

double cal_a_det(int n){
	double sq3, x1, x2, y1, y2, res;
	sq3 = sqrt(3);
	x1 = 2 + ( 7.0 / 6 ) * sq3;
	x2 = 2 - ( 7.0 / 6 ) * sq3;
	y1 = -2 - sq3;
	y2 = -2 + sq3;
	//cout << x1 << " " << x2 << " " << y1 << " " << y2 << endl;
	res = x1 * pow( y1, n - 1 ) + x2 * pow( y2, n - 1 );
	res = -res;
	return res;	
}

void print_float_mat(Mat m){
	cout << "[" << endl;
	for(int i = 0; i < m.rows; i++) {
		for( int j = 0; j < m.cols; j++){
			double x = m.at<double>(i,j);
			if((x == 0.0)||(x>0 && x<0.0001)||(x<0 && x>-0.0001)) cout << "0, ";
			//else cout << setprecision(10) << (x*abs(cal_a_det(m.rows))) << ", ";	
			else cout << setprecision(5) << (int)(1/x) << ", ";	
		}
		cout << ";" << endl;
	}
	cout << "]" << endl;

}


void cal_a_inv(int n, Mat &ans){
	ans = Mat::zeros( n, n, CV_64FC1 );
	double fac[10000];
	for( int i = 0; i <= n; i++ ) fac[i] = abs( cal_a_det(i) );
	for ( int i = 0; i < n; i++ ){
		for ( int j = i; j < n; j++ ){
			ans.at<double>(i,j) = - fac[i] * fac[n-j-1] / fac[n];
		}
	}
		
	
	for( int i = 0; i < n; i++ ){
		for( int j = i + 1; j < n; j++) ans.at<double>(j,i) = ans.at<double>(i,j);
	}
	//return ans;
}


int main (int argc, char **argv)
{	
	Rect b_roi, f_roi;
	/*
	b_roi = Rect( 93, 44, 52, 51 );
	f_roi = Rect( 0, 0, 62, 62 );
	handle("img/1_ori.jpg", "img/1.jpg", b_roi, f_roi, "1" );
	*/
	/*	
	b_roi = Rect( 800, 150, 50, 50);
	f_roi = Rect( 160, 40, 110, 110);
	handle("img/mountains.JPG", "img/moon.JPG", b_roi, f_roi, "moon");
	*/
	/*
	for(int i = 1; i <= 10 ; i++) cout << cal_a_det(i) << " ";
	cout << endl;

	Mat A, A_inv, a, a_inv, ans;
	int h = atoi(argv[1]), w = atoi(argv[2]);
	getA(A, h, w);
	get_a(a, w);
	invert(A, A_inv);
	print_float_mat(A_inv);
	//cout << (a * a) << endl;
	
	//invert(a, a_inv, DECOMP_LU);
	//cal_a_inv(a.cols, ans); 
	//cout << ans << endl;
	//cout << a << endl;
	//cout << A << endl;
	//cout << determinant(a) << endl;
	//cout << cal_a_det(a.cols) << endl;
	
	
	//print_float_mat(a_inv);
	//print_float_mat(ans);
	//print_float_mat(a*a_inv);
	//print_float_mat(ans*a);
	//print_float_mat(A_inv);
	//print_float_mat(A*A_inv);
	*/
	waitKey(0);
	return 0;
}

