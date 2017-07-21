#include "common.hpp"

void handle(const char* background_name, const char* front_name, Rect b_roi, Rect f_roi, const char* save_name );

void sys_handle(const char* background_name, const char* front_name, Rect b_roi, Rect f_roi, const char* save_name );

int main (int argc, char **argv)
{	
	Rect b_roi, f_roi;
	
	/*	
	b_roi = Rect( 93, 44, 52, 51 );
	f_roi = Rect( 0, 0, 62, 62 );
	//sys_handle("img/1_ori.jpg", "img/1.jpg", b_roi, f_roi, "1" );
	handle("img/1_ori.jpg", "img/1.jpg", b_roi, f_roi, "1" );
	*/
	
	
	b_roi = Rect( 495, 266, 223, 223);
	f_roi = Rect( 0, 0, 185, 185);
	handle("img/0_ori.jpg", "img/0.jpg", b_roi, f_roi, "0");
	
	/*
	b_roi = Rect( 800, 150, 150, 150);
	f_roi = Rect( 160, 40, 110, 110);
	handle("img/mountains.JPG", "img/moon.JPG", b_roi, f_roi, "moon");
	*/
	
	waitKey(0);
	return 0;
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
 
/*
 * Handle for poisson edit by opencv system3.0 function, 
 * put the f_roi of front to the b_roi of background
 */
void sys_handle(const char* background_name, const char* front_name, Rect b_roi, Rect f_roi, const char* save_name ){
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
	/*Mat smp_rep;
	simple_replace( back, b_roi.tl(), roi_front, roi, smp_rep );
	sprintf( fname, "simple_replace_%s.jpg", save_name );
	imwrite( fname, smp_rep );
	*/
	//poisson fusion
	Mat mask = 255 * Mat::ones( roi_front.rows, roi_front.cols, roi_front.depth() );
	Mat sys_normal, sys_mixed;
	
	long t1 = time(NULL);
	//seamlesssClone( roi_front, back, mask, b_roi.tl(), sys_normal, NORMAL_CLONE );
	long t2 = time(NULL);
	cout << " System normal seamless clone cost " << (t2-t1)*1000 << " ms." << endl;
	//seamlesssClone( roi_front, back, mask, b_roi.tl(), sys_mixed, MIXED_CLONE );
	long t3 = time(NULL);
	cout << " System mixed seamless clone cost " << (t2-t1)*1000 << " ms." << endl;
	
	sprintf( fname, "sys_normal_poisson_%s.jpg", save_name );
	imwrite( fname, sys_normal );
	
	sprintf( fname, "sys_mixed_poisson_%s.jpg", save_name );
	imwrite( fname, sys_mixed );

}
