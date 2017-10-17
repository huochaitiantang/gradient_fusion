#include "common.hpp"

Mat getPoissonMat(const Mat &back, const Mat &front, Rect b_roi, int type){
	Mat ans = back.clone();
	Rect roi = Rect(0, 0, b_roi.width, b_roi.height);
	if( type == 1 ){
		cout << " Opencv3.2 Normal Clone " << endl;
		Mat mask = 255 * Mat::ones( front.rows, front.cols, front.depth() );
		Point center( b_roi.x + b_roi.width / 2, b_roi.y + b_roi.height / 2 );
		long t1 = time(NULL);
		seamlessClone( front, back, mask, center, ans, NORMAL_CLONE );
		long t2 = time(NULL);
		cout << " Normal seamless clone cost " << (t2-t1)*1000 << " ms." << endl;
		//imshow("ans",ans);
		return ans;
	}
	else if( type == 2 ){
		cout << " Opencv3.2 Mixed Clone " << endl;
		Mat mask = 255 * Mat::ones( front.rows, front.cols, front.depth() );
		Point center( b_roi.x + b_roi.width / 2, b_roi.y + b_roi.height / 2 );
		long t1 = time(NULL);
		seamlessClone( front, back, mask, center, ans, MIXED_CLONE );
		long t2 = time(NULL);
		cout << " Mixed seamless clone cost " << (t2-t1)*1000 << " ms." << endl;
		//imshow("ans",ans);
		return  ans;
	}
	else{
		cout << " FR Solver " << endl;
		Mat res, in1, in2;
		print_mat_info( back, "background" );
		print_mat_info( front, "roi_front" );
		back.convertTo(in1, CV_64FC3);
		front.convertTo(in2, CV_64FC3);	
		poisson(in2, in1, roi, b_roi.tl(), res);
		print_mat_info( res, "poisson res" );
		res.convertTo(res, CV_8UC3);
		Mat roimat = ans(b_roi);
		res.copyTo(roimat);
		return ans;
	}

}

Mat getPolygonPoissonMat(const Mat &back, const Mat &front, const Mat &mask, Rect b_roi, int type){
	Mat ans = back.clone();
	Rect roi = Rect(0, 0, b_roi.width, b_roi.height);
	if( type == 1 ){
		cout << " Opencv3.2 Normal Clone " << endl;
		Point center( b_roi.x + b_roi.width / 2, b_roi.y + b_roi.height / 2 );
		long t1 = time(NULL);
		seamlessClone( front, back, mask, center, ans, NORMAL_CLONE );
		long t2 = time(NULL);
		cout << " Normal seamless clone cost " << (t2-t1)*1000 << " ms." << endl;
		return ans;
	}
	else if(type == 2){
		cout << " Opencv3.2 Mixed Clone " << endl;
		Point center( b_roi.x + b_roi.width / 2, b_roi.y + b_roi.height / 2 );
		long t1 = time(NULL);
		seamlessClone( front, back, mask, center, ans, MIXED_CLONE );
		long t2 = time(NULL);
		cout << " Mixed seamless clone cost " << (t2-t1)*1000 << " ms." << endl;
	}
	else{
		return ans;
	}

}
