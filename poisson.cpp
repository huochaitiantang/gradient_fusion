#include "common.hpp"

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
//	getA( A, rh, rw);
	print_mat_info(A,"A");
	vector <Mat> rgb_f,rgb_b,result;
	split(img_front, rgb_f);
	split(img_back, rgb_b);
	start = time(NULL);
	for(int k = 0; k < 3; k++){
		cout << " For rgb[" << k << "]..." << endl;
		getB(rgb_f[k], rgb_b[k], roi, pt, B);	

		Mat ans,ans2,ans3;
		
		//cout << "B : \n" << B << endl;	
		
		long t1 = time(NULL);
		//solve( A, B, ans, DECOMP_LU );
		//cout << " ans : \n" << ans << endl;
		//cout << " A * ans : \n" << (A*ans) << endl;
		long t2 = time(NULL);
		//cout << "\tSolve cost " << (t2 - t1) * 1000 << " ms.\n" << endl;

		double delta = 0.000001;
	//	solve_FR( A, B, ans2, delta );
		//cout << "ans2 : \n" << ans2 << endl;
		//cout << " A * ans2 : \n" << (A*ans2) << endl;
		long t3 = time(NULL);
	//	cout << "\tSolve_FR cost " << (t3 - t2) * 1000 << " ms.\n" << endl;
		
		solve_FR_SparseA(rw, rh, B, ans3, delta );
		//cout << " A * ans3 : \n" << (A*ans3) << endl;
		long t4 = time(NULL);
		cout << "\tSolve_FR_Sparse cost " << (t4 - t3) * 1000 << " ms.\n" << endl;
		
		
		//ans = ans.reshape(0, rh);
		//ans2 = ans2.reshape(0, rh);
		ans3 = ans3.reshape(0, rh);
		result.push_back( ans3 );
	}
	end = time(NULL);
	cout << " Poisson cost " << (end - start) * 1000 << " ms." << endl;
	merge( result, ans );
}

