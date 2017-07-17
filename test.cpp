#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int print_mat_info(Mat mat){
	cout << "continuous:" << mat.isContinuous() << endl;
	cout << "channels:" << mat.channels() << endl;
	cout << "cols:" << mat.cols << endl;
	cout << "rows:" << mat.rows << endl;
	return 0;
}

int get_ind(int i, int j, int w){
	return i * w + j;
}

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

	cout << "start A" << endl;
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
	cout << "end A" << endl;
}


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
	cout << "start B" << endl;
	filter2D( img_front, Lap, -1, lk);
	for(int i = 0; i < rh; i++){
		for( int j = 0; j < rw; j++){
			double tmp = 0.0;
			tmp += Lap.at<double>( (i + roi.y ), ( j + roi.x ) );
			if( i == 0 ) tmp -= img_back.at<double>( i - 1 + pt.y, j + pt.x );
			else if( i == rh - 1) tmp -= img_back.at<double>( i + 1 + pt.y, j + pt.x );
			if( j == 0 ) tmp -= img_back.at<double>( i + pt.y, j - 1 + pt.x );
			else if( j == rw - 1) tmp -= img_back.at<double>( i + pt.y, j + 1 + pt.x );
			B.at<double>( get_ind( i, j, rw ), 0) = tmp;
		}
	}
	cout << "end B" << endl;
}

Mat get_result(Mat &A, Mat &B, int h){
	Mat ans;
	cout << "start solve" << endl;
	solve(A, B, ans);
	cout << "end solve" << endl;
	ans = ans.reshape(0, h);
	return ans;
}

void poisson(Mat &img_front, Mat &img_back, Rect roi, Point pt, Mat &ans){
	int rh = roi.height;
	int rw = roi.width;
	Mat A, B;
	getA( A, rh, rw);
	vector <Mat> rgb_f,rgb_b,result;
	split(img_front, rgb_f);
	split(img_back, rgb_b);
	for(int k = 0; k < 3; k++){
		getB(rgb_f[k], rgb_b[k], roi, pt, B);	
		result.push_back( get_result( A, B, rh ) );
	}
	merge( result, ans );
}

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

/* Simply replace the part of img obj with the part of img src
 * obj : object img
 * src : source img
 * x,y : object cordinate
 * sx,sy : source cordinate
 * w,h : part size
 * Return the fusion img
 */ 
void simple_replace(const Mat& obj, int x, int y, const Mat &src, int sx, int sy, int w, int h, Mat& ans){
	ans = obj.clone();
	int c = src.channels();
	int srow = src.rows;
	int scol = src.cols;
	int orow = obj.rows;
	int ocol = obj.cols;
	if(obj.channels() != src.channels()){
		cout << "ERROR: obj and src channels not matched!" << endl;
		return;
	}
	if(	x < 0 || x >= obj.cols || 
		y < 0 || y >= obj.rows ){
		cout << "ERROR: obj location error!" << endl;
		return;
	}
	if(	sx < 0 || sx >= src.cols ||
		sy < 0 || sy >= src.rows ||
		sx + w > src.cols ||
		sy + h > src.rows){
		cout << "ERROR: src location error!" << endl;
		return ;
	}
	for(int i = y; i < y + h && i < orow; i++){
		for(int j = x; j < x + w && j < ocol; j++){
			for(int k = 0; k < c; k++){
				ans.at<Vec3b>(i,j)[k] = src.at<Vec3b>(i-y+sy,j-x+sx)[k];
			}
		}
	}
}

int main (int argc, char **argv)
{
	Mat img1,img2;
	img1 = imread("fg.jpg", CV_LOAD_IMAGE_COLOR );
	img2 = imread("fg.jpg", CV_LOAD_IMAGE_COLOR );
	
	if (!img1.data || !img2.data) {
		cout << "No image data\n";
		return -1;
	}
	
	print_mat_info(img1);
	print_mat_info(img2);

	int x = 97;
	int y = 257;
	int sx = 25;
	int sy = 260;
	int w = 30;
	int h = 45;
	Mat Ans;
        simple_replace(img1, x, y, img2, sx, sy, w, h, Ans);
	imwrite("simple_fusion.jpg",Ans);
	imshow("simple_fusion", Ans);
	//Mat grad;
	//compute_gradient(Ans,grad);
	//print_mat_info(grad);
	//imshow("Ans2", grad);
	
	//Mat grad_b,grad_f,grad_fusion,grad_lap;
	//compute_gradient(img1, grad_b);
	//compute_gradient(img2, grad_f);
	//imshow("grad_b",grad_b);
	//imshow("grad_f",grad_f);
	//simple_replace(grad_b, x, y, grad_f, sx, sy, w, h, grad_fusion);
	//print_mat_info(grad_fusion);
	//imshow("grad_fusion",grad_fusion);

	//imshow("img1",img1);
	//compute_laplacian(grad_fusion, grad_lap);
	//Laplacian(grad_fusion, grad_lap,grad_fusion.depth());
	//print_mat_info(grad_lap);
	//imshow("grad_lap", grad_lap);
	/*
	vector<Mat> img_chan; 
	vector<Mat> lap_chan; 
	
       	split(img1, img_chan);
	split(grad_lap, lap_chan);
	
	for(int k = 0; k < 3; k++){
		solve(img_chan[k], lap_chan[k], img_chan[k]);
	}	
	
	Mat result;
	merge(img_chan, result);
	imshow("result",result);	
	*/
	//Mat A;
	//getA( A, 20, 30);
	Mat res,in1, in2;
	img1.convertTo( in1, CV_64FC3 );
	img2.convertTo( in2, CV_64FC3 );
	poisson(in2, in1, Rect(sx, sy, w, h), Point(x, y), res);
	res.convertTo(res, CV_8UC1);
	Mat roimat = img1( Rect( x, y, w, h ) );


	res.copyTo(roimat);

	//imshow("roi",res);
	imshow("res",img1);
	imwrite("poisson.jpg",img1);
		
	waitKey(0);
	return 0;
}

