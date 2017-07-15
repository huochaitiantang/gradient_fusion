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




/* Simply replace the part of img obj with the part of img src
 * obj : object img
 * src : source img
 * x,y : object cordinate
 * sx,sy : source cordinate
 * w,h : part size
 * Return the fusion img
 */ 
Mat simple_replace(Mat obj, int x, int y, Mat src, int sx, int sy, int w, int h){
	Mat ans = obj.clone();
	int c = src.channels();
	int srow = src.rows;
	int scol = src.cols;
	int orow = obj.rows;
	int ocol = obj.cols;
	if(obj.channels() != src.channels()){
		cout << "ERROR: obj and src channels not matched!" << endl;
		return ans;
	}
	if(	x < 0 || x >= obj.cols || 
		y < 0 || y >= obj.rows ){
		cout << "ERROR: obj location error!" << endl;
		return ans;
	}
	if(	sx < 0 || sx >= src.cols ||
		sy < 0 || sy >= src.rows ||
		sx + w > src.cols ||
		sy + h > src.rows){
		cout << "ERROR: src location error!" << endl;
		return ans;
	}
	for(int i = y; i < y + h && i < orow; i++){
		for(int j = x; j < x + w && j < ocol; j++){
			int ind = ( i * ocol + j ) * c;
			int ind2 = ( ( i - y + sy ) * scol + ( j - x + sx ) ) * c;
			for(int k = 0; k < c; k++){
				ans.data[ ind + k ] = src.data[ ind2 + k ];
			}
		}
	}
	return ans;
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
	Mat Ans = simple_replace(img1, x, y, img2, sx, sy, w, h);
	
	imwrite("3_5_simple_fusion.jpg",Ans);
		
	namedWindow("Ans", CV_WINDOW_AUTOSIZE);
	imshow("Ans", Ans);

	waitKey(0);
	return 0;
}

