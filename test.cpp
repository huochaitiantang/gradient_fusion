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

void computeGradientX( const Mat &img, Mat &gx){  
	Mat kernel = Mat::zeros(1, 3, CV_8S);  
	kernel.at<char>(0,2) = 1;  
	kernel.at<char>(0,1) = -1;  
	      
	if(img.channels() == 3){  
		filter2D(img, gx, CV_32F, kernel);  
	}  
	else if (img.channels() == 1){  
		Mat tmp[3];  
		for(int chan = 0 ; chan < 3 ; ++chan){
			filter2D(img, tmp[chan], CV_32F, kernel);
		}
		merge(tmp, 3, gx); 
	}  
}  
  
void computeGradientY( const Mat &img, Mat &gy){  
	Mat kernel = Mat::zeros(3, 1, CV_8S);  
	kernel.at<char>(2,0) = 1;  
	kernel.at<char>(1,0) = -1;  	
	if(img.channels() == 3)  {  
		filter2D(img, gy, CV_32F, kernel); 		
       	}  
	else if (img.channels() == 1)  {  
		Mat tmp[3];  
		for(int chan = 0 ; chan < 3 ; ++chan)  {  
			filter2D(img, tmp[chan], CV_32F, kernel);  
		}  
		merge(tmp, 3, gy);  				
	}  
} 

void compute_laplacian(const Mat &img, Mat &ans){
	Mat lap_x, lap_y;
	
	Mat kernel_x = Mat::zeros(1, 3, CV_8S);  
	kernel_x.at<char>(0,0) = -1;  
	kernel_x.at<char>(0,1) = 1;  
	filter2D(img, lap_x, CV_32F, kernel_x);  
	
	Mat kernel_y = Mat::zeros(1, 3, CV_8S);  
	kernel_y.at<char>(0,0) = -1;  
	kernel_y.at<char>(0,1) = 1;  
	filter2D(img, lap_y, CV_32F, kernel_y);

	ans = lap_x + lap_y;	
}	

void dst(const Mat& src, Mat& dest, bool invert){  
	Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);  
	int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE: DFT_ROWS;  
	src.copyTo(temp(Rect(1,0, src.cols, src.rows)));  	
	for(int j = 0 ; j < src.rows ; ++j){  
		float * tempLinePtr = temp.ptr<float>(j);  
		const float * srcLinePtr = src.ptr<float>(j);  
		for(int i = 0 ; i < src.cols ; ++i){  	        			
			tempLinePtr[src.cols + 2 + i] = - srcLinePtr[src.cols - 1 - i];  
		}  
	}  		  
	Mat planes[] = {temp, Mat::zeros(temp.size(), CV_32F)};  
	Mat complex;  			  
	merge(planes, 2, complex);  
	dft(complex, complex, flag);  
	split(complex, planes);  
	temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);  				  
	for(int j = 0 ; j < src.cols ; ++j){  
		float * tempLinePtr = temp.ptr<float>(j);  
		for(int i = 0 ; i < src.rows ; ++i){  
			float val = planes[1].ptr<float>(i)[j + 1];  
			tempLinePtr[i + 1] = val;  
			tempLinePtr[temp.cols - 1 - i] = - val;  
		}  
	}  

	Mat planes2[] = {temp, Mat::zeros(temp.size(), CV_32F)};  

	merge(planes2, 2, complex);  
	dft(complex, complex, flag);  
	split(complex, planes2);  

	temp = planes2[1].t();  
	dest = Mat::zeros(src.size(), CV_32F);  
	temp(Rect( 0, 1, src.cols, src.rows)).copyTo(dest);  
}  
  
void idst(const Mat& src, Mat& dest){  
	dst(src, dest, true);  
}

void solve(const Mat &img, Mat& mod_diff, Mat &result) {  
	const int w = img.cols;  
	const int h = img.rows;  
	
	Mat res;  
	dst(mod_diff, res, true); 

	for(int j = 0 ; j < h-2; j++){  
		float * resLinePtr = res.ptr<float>(j);  
		for(int i = 0 ; i < w-2; i++){  
			resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);  
		}  
	}  

	idst(res, mod_diff);  

	unsigned char *  resLinePtr = result.ptr<unsigned char>(0);  
	const unsigned char * imgLinePtr = img.ptr<unsigned char>(0); 
	const float * interpLinePtr = NULL;  

	//first col  
	for(int i = 0 ; i < w ; ++i)  
		result.ptr<unsigned char>(0)[i] = img.ptr<unsigned char>(0)[i];  

	for(int j = 1 ; j < h-1 ; ++j)  { 
		resLinePtr = result.ptr<unsigned char>(j);  
		imgLinePtr  = img.ptr<unsigned char>(j);  
		interpLinePtr = mod_diff.ptr<float>(j-1);  

		//first row  
		resLinePtr[0] = imgLinePtr[0];  

		for(int i = 1 ; i < w-1 ; ++i)  {  
			float value = interpLinePtr[i-1];  
			if(value < 0.)  
				resLinePtr[i] = 0;  
			else if (value > 255.0)  
				resLinePtr[i] = 255;  
			else  
				resLinePtr[i] = static_cast<unsigned char>(value);  
		}  
		//last row  
		resLinePtr[w-1] = imgLinePtr[w-1];  
	}  
	//last col  
	resLinePtr = result.ptr<unsigned char>(h-1);  
	imgLinePtr = img.ptr<unsigned char>(h-1);  
	for(int i = 0 ; i < w ; ++i)
		resLinePtr[i] = imgLinePtr[i];  
}  

void compute_gradient( const Mat &img, Mat &ans){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat Ans_gray, grad, grad_x, grad_y, abs_grad_x, abs_grad_y;
	GaussianBlur( img, img, Size(3,3), 0, 0, BORDER_DEFAULT );
	//cvtColor( Ans, Ans_gray, CV_RGB2GRAY );
	Sobel( img, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	Sobel( img, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	//Scharr( Ans, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//Scharr( Ans, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	//computeGradientX(Ans, grad_x);
	//computeGradientY(Ans, grad_y);
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
	img1 = imread("3.jpg", CV_LOAD_IMAGE_COLOR );
	img2 = imread("5.jpeg", CV_LOAD_IMAGE_COLOR );
	
	if (!img1.data || !img2.data) {
		cout << "No image data\n";
		return -1;
	}
	
	print_mat_info(img1);
	print_mat_info(img2);

	int x = 324;
	int y = 420;
	int sx = 0;
	int sy = 55;
	int w = img2.cols;
	int h = img2.rows - 55 - 30;
	Mat Ans;
        simple_replace(img1, x, y, img2, sx, sy, w, h, Ans);
	imwrite("3_5_simple_fusion.jpg",Ans);
	imshow("simple_fusion", Ans);
	//Mat grad;
	//compute_gradient(Ans,grad);
	//print_mat_info(grad);
	//imshow("Ans2", grad);
	
	Mat grad_b,grad_f,grad_fusion,grad_lap;
	compute_gradient(img1, grad_b);
	compute_gradient(img2, grad_f);
	//imshow("grad_b",grad_b);
	//imshow("grad_f",grad_f);
	simple_replace(grad_b, x, y, grad_f, sx, sy, w, h, grad_fusion);
	print_mat_info(grad_fusion);
	imshow("grad_fusion",grad_fusion);

	compute_laplacian(grad_fusion, grad_lap);
	//print_mat_info(grad_lap);
	//imshow("grad_lap", grad_lap);
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

	waitKey(0);
	return 0;
}

