#include "common.hpp"

Mat getFusionMat(const Mat &back, const Mat &front, const Mat &mask, Rect b_roi, int type){
	Mat ans = back.clone();
	Rect roi = Rect(0, 0, b_roi.width, b_roi.height);
	Point center( b_roi.x + b_roi.width / 2, b_roi.y + b_roi.height / 2 );
	long t1 = time(NULL);
	if(type == 0){
		cout << "Poisson Opencv3.2 Normal Clone " << endl;
		seamlessClone( front, back, mask, center, ans, NORMAL_CLONE );
	}
	else if(type == 1){
		cout << "Poisson Opencv3.2 Mixed Clone " << endl;
		seamlessClone( front, back, mask, center, ans, MIXED_CLONE );
	}
	else if(type == 2){
		cout << "Poisson Own Rect ROI By FR Solver " << endl;
		Mat res, in1, in2;
		print_mat_info( back, "background" );
		print_mat_info( front, "roi_front" );
		back.convertTo(in1, CV_64FC3);
		front.convertTo(in2, CV_64FC3);	
		poisson(in2, in1, roi, b_roi.tl(), res);
		res.convertTo(res, CV_8UC3);
		Mat roimat = ans(b_roi);
		res.copyTo(roimat);
	}
	else if(type == 3){
		cout << "Poisson Own Poly ROI By FR Solver " << endl;
		Mat res, in1, in2, msk;
		vector<Mat> msk_v;
		back.convertTo(in1, CV_64FC3);
		front.convertTo(in2, CV_64FC3);	
		mask.convertTo(msk, CV_64FC3);
		split(msk,msk_v);
		msk = msk_v[0];
		print_mat_info( in1, "background-new" );
		print_mat_info( in2, "roi_front-new" );
		print_mat_info( msk, "mask-new" );
		polygonPoisson(in2, in1, msk, roi, b_roi.tl(), res);
		print_mat_info( res, "poisson res" );
		res.convertTo(res, CV_8UC3);
		Mat roimat = ans(b_roi);
		res.copyTo(roimat);
	}
	else if(type == 4){
		cout << "Poisson Own Drag Drop Solver" << endl;
		Mat res, msk, in1, in2;
		vector<Mat> msk_v;
		split(mask,msk_v);
		msk = msk_v[0];
		in1 = back;
		in2 = front;
		edgePoisson(in2, in1, msk, roi, b_roi.tl(), ans);
	}
	else cout << "No type [" << type << "] fusion!" << endl;
	long t2 = time(NULL);
	cout << "Fusion type [" << type << "] Cost "<<(t2-t1)*1000 << " ms." << endl;
	return ans;
}
