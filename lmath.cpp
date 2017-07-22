#include "common.hpp"

/* get matrix like 
 * -4 1 0 0 0 0 
 * 1 -4 1 0 0 0
 * 0 1 -4 1 0 0
 * 0 0 1 -4 1 0
 * 0 0 0 1 -4 1
 * 0 0 0 0 1 -4
 */

void get_a(Mat &a, int w){
	a = Mat::zeros( w, w, CV_64FC1 );
	for(int i = 0; i < w; i++){
		double * aLinePtr = a.ptr<double>(i);	
		aLinePtr[i] = -4;
		if(i > 0) aLinePtr[i-1] = 1;
		if(i < w - 1) aLinePtr[i+1] = 1;
	}
}

/* calculate the det(a)
 */
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

/* print a float mat 
 */
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

/* calculate the invert of a
 */
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

/* judge if x < y */
bool less_than( Mat &x, Mat &y ){
	int h = std::min( x.rows, y.cols );
	double * xPtr = x.ptr<double>();
	double * yPtr = y.ptr<double>();
	for( int i = 0; i < h; i++ ){
		if( abs(xPtr[i]) > yPtr[i] ) return false;
	}
	return true;
}

/*
 * conjugate gradient method solve for Ax = b
 * A, ans, b : matrix A x b
 * delta : precision
 */
void solve_FR( Mat &A, Mat &b, Mat &ans, double delta ){
	if ( A.cols != A.rows || A.rows != b.rows || b.cols != 1 ){
		cout << "ERROR : Solver_FR Failed!" << endl;
		return;
	}
	int k = 0, h = b.rows;
	Mat del = Mat::ones( h, 1, CV_64FC1 );
	del *= delta;
	Mat tmp;
	Mat x = Mat::zeros( h, 1, CV_64FC1 );
	Mat p = Mat::zeros( h, 1, CV_64FC1 );
	Mat r = b.clone();
	double rTr_2, alpha;
	tmp = r.t() * r ;
	double rTr_1 = tmp.at<double>(0,0);
	while( less_than( r , del ) == false ){
		k++;
		if( k == 1 ){
			r.copyTo( p );
		}
		else{
			p = ( r + ( rTr_1 / rTr_2 ) * p );
		}
		Mat Ap = A * p;	
		tmp = p.t() * Ap;
		alpha = rTr_1 / tmp.at<double>(0,0);
		x = ( x + alpha * p );
		r = ( r - alpha * Ap );
		
		rTr_2 = rTr_1;
		tmp = r.t() * r;
		rTr_1 = tmp.at<double>(0,0);
	}
	cout << " solve FR for step " << k << " ." << endl;
	x.copyTo(ans);
}

/* calculate for a*p
 */
Mat cal_ap( int rw, Mat &p ){
	Mat ans = Mat::zeros( rw, 1, CV_64FC1 );
	double * pPtr = p.ptr<double>();
	double * ansPtr = ans.ptr<double>();	
	ansPtr[0] = -4 * pPtr[0] + pPtr[1];
	for( int i = 0; i < rw -1; i++ )
		ansPtr[i] = -4 * pPtr[i] + pPtr[i - 1] + pPtr[i + 1];
	ansPtr[rw - 1] = -4 * pPtr[rw - 1] + pPtr[rw - 2];
	return ans;
}

/* calculate for A*p
 * A like 
 * a i o o o,
 * i a i o o,
 * o i a i 0,
* o o i a i,
 * o o o i a,
 * i,a : size(w,w)
 * p : size(w*h,1)
 * A can not be stored because it is too large and sparse
 */
Mat cal_Ap( int rw, int rh, Mat &p ){
	Mat a,ans;
	vector<Mat> P;
	vector<Mat> ANS;	
	//get_a( a, rw );
	for( int i = 0; i < rh; i++ ){
		Mat tmp;
		p( Rect( 0 , rw * i , 1, rw ) ).copyTo(tmp);
		P.push_back(tmp);
	}
	//ANS.push_back( a * P[0] + P[1] );
	ANS.push_back( cal_ap( rw, P[0] ) + P[1] );
	for( int i = 1; i < rh - 1; i++ ){
		//ANS.push_back( P[i-1] + a * P[i] + P[i+1] );
		ANS.push_back( P[i-1] + cal_ap( rw, P[i] ) + P[i+1] );
	}
	//ANS.push_back( P[rh-2] + a * P[rh-1] );
	ANS.push_back( P[rh-2] + cal_ap( rw, P[rh-1] ) );
	int k = 0;
	ans = Mat::zeros( rw * rh, 1, CV_64FC1 );
	double * ansPtr = ans.ptr<double>();
	for( int i = 0; i < rh; i++ ){
		double * tmpPtr = ANS[i].ptr<double>();
		for( int j = 0; j < rw; j++ ){
			ansPtr[k++] = tmpPtr[j];
		}		
	}
	return ans;
}


/* slower than cal_Ap */
Mat cal_Ap2( vector< vector< int > > A, Mat &p ){
	Mat ans = Mat::zeros( A.size(), 1, CV_64FC1 );
	double * pPtr = p.ptr<double>();
	double * ansPtr = ans.ptr<double>();	
	for(int i = 0; i < A.size(); i++ ){
		double tmp = -4 * pPtr[i];
		for(int j = 0; j < A[i].size(); j++ ){
			tmp += pPtr[ A[i][j] ];
		}
		ansPtr[i] = tmp;
	}
	return ans;
}

/*
 * conjugate gradient method solve for Ax = b, A 
 * ans, b : matrix x b
 * rw : child matrix width
 * rh : num of chlid matrix in each line
 * delta : precision
 */
void solve_FR_SparseA( int rw, int rh, Mat &b, Mat &ans, double delta ){
	if ( b.rows != rw * rh || b.cols != 1 ){
		cout << "ERROR : Solver_FR Failed!" << endl;
		return;
	}
	int k = 0, h = b.rows;
	//vector< vector < int > > A = get_sparseA( rh, rw );
	Mat del = Mat::ones( h, 1, CV_64FC1 );
	del *= delta;
	Mat tmp;
	Mat x = Mat::zeros( h, 1, CV_64FC1 );
	Mat p = Mat::zeros( h, 1, CV_64FC1 );
	Mat r = b.clone();
	double rTr_2, alpha;
	tmp = r.t() * r ;
	double rTr_1 = tmp.at<double>();
	while( less_than( r , del ) == false ){
		k++;
		if( k == 1 ){
			r.copyTo( p );
		}
		else{
			p = ( r + ( rTr_1 / rTr_2 ) * p );
		}
		Mat Ap = cal_Ap( rw, rh, p);
		//Mat Ap = cal_Ap2( A, p);
		tmp = p.t() * Ap;
		alpha = rTr_1 / tmp.at<double>();
		x = ( x + alpha * p );
		r = ( r - alpha * Ap );
		
		rTr_2 = rTr_1;
		tmp = r.t() * r;
		rTr_1 = tmp.at<double>();
	}
	cout << " solve FR_SpareA for step " << k << " ." << endl;
	x.copyTo(ans);
}
