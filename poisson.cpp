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

vector<vector<int> > getPolySparseA( vector<vector<int> > &MapId, vector<pair<int,int> > &IdMap){
	vector<vector<int> > ans;
	int h = MapId.size();
	int w = MapId[0].size();
	for(int i = 0; i < IdMap.size(); i++){
		vector<int> ind;
		int x = IdMap[i].first;
		int y = IdMap[i].second;
		int cur;
		if( x > 0){
			cur = MapId[x-1][y];
			if(cur >= 0) ind.push_back(cur);
		}
		if( y > 0){
			cur = MapId[x][y-1];
			if(cur >= 0) ind.push_back(cur);
		}
		if( x < h -1){
			cur = MapId[x+1][y];
			if(cur >= 0) ind.push_back(cur);
		}
		if( y < w - 1){
			 cur = MapId[x][y+1];
			if(cur >= 0) ind.push_back(cur);
		}
		ans.push_back(ind);
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
			if( i == 0){ 
				int cur = max(i - 1 + pt.y, 0);
				tmp -= img_back.at<double>(cur, j + pt.x );
			}
			else if( i == rh - 1){
				int cur = min(i + 1 + pt.y, img_back.rows - 1); 
				tmp -= img_back.at<double>(cur, j + pt.x );
			}
			if( j == 0){
				int cur = max(j - 1 + pt.x, 0);
				tmp -= img_back.at<double>( i + pt.y, cur);
			}
			else if( j == rw - 1 && pt.x < img_back.cols - 1){
				int cur = min(j + 1 + pt.x, img_back.cols - 1);
				tmp -= img_back.at<double>( i + pt.y, cur);
			}
			BPtr[ get_ind( i, j, rw ) ] = tmp;
		}
	}
}

void getPolyB( Mat &img_front, Mat &img_back, Rect roi, Point pt, Mat &B, vector<vector<int> > &MapId, vector<pair<int,int> > &IdMap){
	Mat Lap;
	int rh = roi.height;
	int rw = roi.width;
	int siz = IdMap.size();
	Mat lk = Mat::zeros( 3, 3, CV_64FC1 );
	double * lkPtr = lk.ptr<double>();
	lkPtr[1] = 1.0;
	lkPtr[3] = 1.0;
	lkPtr[5] = 1.0;
	lkPtr[7] = 1.0;
	lkPtr[4] = -4.0;
	B = Mat::zeros( siz, 1, CV_64FC1);
	//get Lap matrix
	filter2D( img_front, Lap, -1, lk);
	double * BPtr = B.ptr<double>();
	for(int i = 0; i < siz; i++){
		int x = IdMap[i].first;
		int y = IdMap[i].second;
		double tmp = Lap.at<double>(x, y);
		//border
		if(x == 0 || (x > 0 && MapId[x-1][y] < 0)){
			int cur = max(0, pt.y + x -1);
			tmp -= img_back.at<double>(cur, pt.x + y);
		}
		if(x == rh -1 || (x < rh -1 && MapId[x+1][y] < 0)){
			int cur = min(x + 1 + pt.y, img_back.rows - 1); 
			tmp -= img_back.at<double>(cur, pt.x + y);
		}
		if(y == 0 || (y > 0 && MapId[x][y-1] < 0)){
			int cur = max(0, y - 1 + pt.x);
			tmp -= img_back.at<double>(x + pt.y, cur);
		}
		if(y == rw - 1 || (y < rw - 1 && MapId[x][y+1] < 0)){
			int cur = min(y + 1 + pt.x, img_back.cols - 1);
			tmp -= img_back.at<double>(x + pt.y, cur);
		}
		BPtr[i] = tmp;	
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
	vector<vector<int> > A;
	Mat B;
	long start, end;
	vector <Mat> rgb_f,rgb_b,result;
	split(img_front, rgb_f);
	split(img_back, rgb_b);
	vector<vector<int> > MapId;
	vector<pair<int,int> > IdMap;
	
	//imshow("msk",mask);
	cout << "rh:" << rh << " rw:" << rw << endl;
	cout << "mask: rows:" << mask.rows << ", cols:" << mask.cols << endl;
	getMaskMapTable(mask,roi,MapId,IdMap);
	cout << "Rect pixel : " << MapId.size() * MapId[0].size() << endl;
	cout << "IdMap size( num of x) : " << IdMap.size() << endl;
	/*
	for(int i = 0; i < rh; i++){
		for(int j = 0; j < rw; j++){
			if(MapId[i][j] >= 0) cout << "*";
			else cout << " ";	
		}
		cout << endl;
	}*/
	/*for(int i = 0; i < IdMap.size(); i++){
		cout << i << " : [" << IdMap[i].first << "," << IdMap[i].second <<"]"<< endl;
	}
	*/
	A = getPolySparseA(MapId, IdMap);
	/*
	for(int i = 0; i < 10; i++){
		cout << i << " : [" << IdMap[i].first << "," << IdMap[i].second <<"] ";
		for(int j = 0; j < A[i].size(); j++){
			cout << A[i][j] << " " ;
		}
		cout << endl;
	}
	*/
	start = time(NULL);
	for(int k = 0; k < img_back.channels(); k++){
		cout << " For rgb[" << k << "]..." << endl;
		getPolyB(rgb_f[k], rgb_b[k], roi, pt, B, MapId, IdMap);	

		Mat res;
		Mat res_rect = Mat::zeros(rh, rw, CV_64FC1);
		double delta = 0.00001;
		long t0 = time(NULL);
		solve_FR_PolySparseA(rw, rh, B, A, res, delta );
		long t1 = time(NULL);
		cout << "\tCost " << (t1 - t0) * 1000 << " ms.\n" << endl;
		for(int i = 0; i < rh; i++){
			double * Ptr = res_rect.ptr<double>(i);
			for(int j = 0; j < rw; j++){
				if(MapId[i][j] < 0) Ptr[j] = rgb_b[k].at<double>(pt.y + i, pt.x + j);
				else Ptr[j] = res.at<double>(MapId[i][j],0);
			}
		}
		result.push_back( res_rect );
	}
	end = time(NULL);
	cout << "Poly Poisson cost " << (end - start) * 1000 << " ms." << endl;
	cout << " ------------------------------------------------------ \n" << endl;
	merge( result, ans );
	//img_front.copyTo(ans);

}

void edgePoisson(Mat &img_front, Mat &img_back, Mat &mask, Rect roi, Point pt, Mat &ans){
	int rh = roi.height;
	int rw = roi.width;
	long start, end;
	Mat grab_mask, bgdModel, fgdModel;
	Mat obj_mask, edge_mask;
	mask.copyTo(grab_mask);
	for(int i = 0; i < rh; i++){
		uchar * Ptr = grab_mask.ptr<uchar>(i);
		for(int j = 0; j < rw; j++){
			if(Ptr[j] == 0)
				Ptr[j] = GC_BGD;
			else
				Ptr[j] = GC_PR_FGD;
		}
	}
	grabCut( img_front, grab_mask, Rect(0, 0, rw, rh), bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );
	grabCut( img_front, grab_mask, Rect(0, 0, rw, rh), bgdModel, fgdModel, 10);
	grab_mask.copyTo(obj_mask);
	grab_mask.copyTo(edge_mask);
	for(int i = 0; i < rh; i++){
		uchar * Ptr = grab_mask.ptr<uchar>(i);
		uchar * OPtr = obj_mask.ptr<uchar>(i);
		uchar * EPtr = edge_mask.ptr<uchar>(i);
		for(int j = 0; j < rw; j++){
			if(Ptr[j] == GC_BGD) 
				EPtr[j] = OPtr[j] = Ptr[j] = 0 ;
			else if(Ptr[j] == GC_PR_BGD){
				Ptr[j] = 128;
				OPtr[j] = 0;
				EPtr[j] = 255;
			}
			else if(Ptr[j] == GC_PR_FGD) {
				OPtr[j] = Ptr[j] = 255;
				EPtr[j] = 0;
			}
		}
	}
	imshow("grab_mask",grab_mask);
	//imshow("edge_mask",edge_mask);
	imshow("obj_mask",obj_mask);
	
	//Mat new_back, s_new_back, new_front;
	//img_back.copyTo(new_back);
	//img_front.copyTo(new_front);
	
	//Mat roimat = new_back(Rect(pt.x, pt.y, rw, rh));
	//roimat.copyTo(s_new_back);
	//s_new_back.copyTo(new_front, edge_mask);
	//new_front.copyTo(s_new_back, obj_mask);	
	//s_new_back.copyTo(roimat);	

	//imshow("s_new_back", s_new_back);
	//imshow("new_front", new_front);
	//imshow("new_back", new_back);
	Point center( pt.x + rw / 2, pt.y + rh / 2 );
	seamlessClone(img_front,img_back,obj_mask,center, ans, NORMAL_CLONE);

	//imshow("ans",ans);
	//img_front.copyTo(ans);
}
