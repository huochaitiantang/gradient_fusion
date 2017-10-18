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
	A = Mat::eye( h * w, h * w, CV_64FC1);
	A *= -4;
	//corner has 2 neighbors, border has 3 neighbors, other has 4 neighbors
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			int label = get_ind(i, j, w);
			double * ALinePtr = A.ptr<double>(label);
			if ( i > 0 ) ALinePtr[ get_ind( i - 1, j, w ) ] = 1;
			if ( j > 0 ) ALinePtr[ get_ind( i, j - 1, w ) ] = 1;
			if ( i < h - 1 ) ALinePtr[ get_ind( i + 1, j, w ) ] = 1;
			if ( j < w - 1 ) ALinePtr[ get_ind( i, j + 1, w ) ] = 1;
		}
	}
}

/* Get the sparse matrix A in Ax = b
 * A : result
 * h : img height
 * w : img width
 */
vector< vector< int > >  get_sparseA(int h, int w){
	vector< vector< int > > ans;
	//corner has 2 neighbors, border has 3 neighbors, other has 4 neighbors
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			vector<int> ind;
			if ( i > 0 ) ind.push_back( get_ind( i - 1, j, w ) );
			if ( j > 0 ) ind.push_back( get_ind( i, j - 1, w ) );
			if ( i < h - 1 ) ind.push_back( get_ind( i + 1, j, w ) );
			if ( j < w - 1 ) ind.push_back( get_ind( i, j + 1, w ) );
 			ans.push_back(ind);
		}
	}
	return ans;
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
	double * lkPtr = lk.ptr<double>();
	lkPtr[1] = 1.0;
	lkPtr[3] = 1.0;
	lkPtr[5] = 1.0;
	lkPtr[7] = 1.0;
	lkPtr[4] = -4.0;
	B = Mat::zeros( rh * rw, 1, CV_64FC1);
	//get Lap matrix
	filter2D( img_front, Lap, -1, lk);
	double * BPtr = B.ptr<double>();
	for(int i = 0; i < rh; i++){
		for( int j = 0; j < rw; j++){
			double tmp = 0.0;
			tmp += Lap.at<double>( (i + roi.y ), ( j + roi.x ) );
			//border lap value should substract background pixel value
			if( i == 0 ) tmp -= img_back.at<double>( i - 1 + pt.y, j + pt.x );
			else if( i == rh - 1) tmp -= img_back.at<double>( i + 1 + pt.y, j + pt.x );
			if( j == 0 ) tmp -= img_back.at<double>( i + pt.y, j - 1 + pt.x );
			else if( j == rw - 1) tmp -= img_back.at<double>( i + pt.y, j + 1 + pt.x );
			BPtr[ get_ind( i, j, rw ) ] = tmp;
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
	//getA( A, rh, rw);
	print_mat_info(A,"A");
	vector <Mat> rgb_f,rgb_b,result;
	split(img_front, rgb_f);
	split(img_back, rgb_b);
	start = time(NULL);
	for(int k = 0; k < img_back.channels(); k++){
		cout << " For rgb[" << k << "]..." << endl;
		getB(rgb_f[k], rgb_b[k], roi, pt, B);	

		Mat ans,ans2,ans3,ans4;
		double delta = 0.00001;
		
		//guassian solve	
		long t1 = time(NULL);
		//solve( A, B, ans, DECOMP_LU );
		long t2 = time(NULL);
		//cout << " ans : \n" << ans << endl;
		//cout << " A * ans : \n" << (A*ans) << endl;
		//cout << "\tSolve cost " << (t2 - t1) * 1000 << " ms.\n" << endl;

		//solve FR with big A	
		//solve_FR( A, B, ans2, delta );
		long t3 = time(NULL);
		//cout << "ans2 : \n" << ans2 << endl;
		//cout << " A * ans2 : \n" << (A*ans2) << endl;
		//cout << "\tSolve_FR cost " << (t3 - t2) * 1000 << " ms.\n" << endl;
	
		//solve FR with sparse A
		solve_FR_SparseA(rw, rh, B, ans3, delta );
		long t4 = time(NULL);
		//cout << " ans3: \n " << ans3 << endl;
		//cout << " A * ans3 : \n" << (A*ans3) << endl;
		cout << "\tCost " << (t4 - t3) * 1000 << " ms.\n" << endl;
	/*	
		Mat mod_diff, mod_img;
		mod_diff = B.reshape(0, rh).clone();
		//cout << "mod_diff: \n" << mod_diff << endl;
		rgb_b[k]( Rect ( pt.x, pt.y, roi.width, roi.height ) ).copyTo(mod_img);
		//cout << "mod_img: \n" << mod_img << endl;
		solve_dft(mod_img, mod_diff, ans4 );
		cout << " ans4: \n " << ans4 << endl;
	*/	
		
		//ans = ans.reshape(0, rh);
		//ans2 = ans2.reshape(0, rh);
		ans3 = ans3.reshape(0, rh);
		result.push_back( ans3 );
	}
	end = time(NULL);
	cout << " Poisson cost " << (end - start) * 1000 << " ms." << endl;
	cout << " ------------------------------------------------------ \n" << endl;
	merge( result, ans );
}



void getMaskMapTable(Mat &Mask, Rect roi, vector<vector<int> > & MapId, vector<pair<int,int> > &IdMap){	
	int rh = roi.height;
	int rw = roi.width;
	int ind = 0;
	MapId.clear();
	IdMap.clear();
	for(int i = 0; i < rh; i++){
		vector<int> cur_row;
		for(int j = 0; j < rw; j++){
			if(Mask.at<double>(i,j) == 0){
				cur_row.push_back(-1);
			}
			else{
				cur_row.push_back(ind);
				pair<int,int> p(i,j);
				IdMap.push_back(p);
				ind ++;
			}
		}
		MapId.push_back(cur_row);
	}
}

void polygonPoisson(Mat &img_front, Mat &img_back, Mat &mask, Rect roi, Point pt, Mat &ans){
	int rh = roi.height;
	int rw = roi.width;
	Mat A, B;
	long start, end;
	vector <Mat> rgb_f,rgb_b,result;
	split(img_front, rgb_f);
	split(img_back, rgb_b);

	
	//start = time(NULL);
	for(int k = 0; k < img_back.channels(); k++){
		cout << " For rgb[" << k << "]..." << endl;
		//getB(rgb_f[k], rgb_b[k], roi, pt, B);	

		Mat res;
		double delta = 0.00001;
		
		//solve_FR_SparseA(rw, rh, B, ans3, delta );
		//long t4 = time(NULL);
		
		//cout << "\tCost " << (t4 - t3) * 1000 << " ms.\n" << endl;
		
		//ans3 = ans.reshape(0, rh);
		//result.push_back( ans3 );
	}
	//end = time(NULL);
	//cout << " Poisson cost " << (end - start) * 1000 << " ms." << endl;
	//cout << " ------------------------------------------------------ \n" << endl;
	//merge( result, ans );
	ans = img_front;




}
