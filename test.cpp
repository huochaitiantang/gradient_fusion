#include "common.hpp"


            std::vector<double> filter_X, filter_Y;
void initXY(const Mat &des){
    //init of the filters used in the dst
    const int w = des.cols;
    filter_X.resize(w - 2);
    for(int i = 0 ; i < w-2 ; ++i)
        filter_X[i] = 2.0f * std::cos(static_cast<double>(CV_PI) * (i + 1) / (w - 1));
    const int h  = des.rows;
    filter_Y.resize(h - 2);
    for(int j = 0 ; j < h - 2 ; ++j)
        filter_Y[j] = 2.0f * std::cos(static_cast<double>(CV_PI) * (j + 1) / (h - 1));
}
void dst(const Mat& src, Mat& dest, bool invert)
{
    Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_64FC1);

    int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE: DFT_ROWS;

    src.copyTo(temp(Rect(1,0, src.cols, src.rows)));

    for(int j = 0 ; j < src.rows ; ++j)
    {
        double * tempLinePtr = temp.ptr<double>(j);
        const double * srcLinePtr = src.ptr<double>(j);
        for(int i = 0 ; i < src.cols ; ++i)
        {
            tempLinePtr[src.cols + 2 + i] = - srcLinePtr[src.cols - 1 - i];
        }
    }

    Mat planes[] = {temp, Mat::zeros(temp.size(), CV_64FC1)};
    Mat complex;

    merge(planes, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes);
    temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_64FC1);

    for(int j = 0 ; j < src.cols ; ++j)
    {
        double * tempLinePtr = temp.ptr<double>(j);
        for(int i = 0 ; i < src.rows ; ++i)
        {
            double val = planes[1].ptr<double>(i)[j + 1];
            tempLinePtr[i + 1] = val;
            tempLinePtr[temp.cols - 1 - i] = - val;
        }
    }

    Mat planes2[] = {temp, Mat::zeros(temp.size(), CV_64FC1)};

    merge(planes2, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes2);

    temp = planes2[1].t();
    dest = Mat::zeros(src.size(), CV_64FC1);
    temp(Rect( 0, 1, src.cols, src.rows)).copyTo(dest);
}

void idst(const Mat& src, Mat& dest)
{
    dst(src, dest, true);
}

void solve_dft(const Mat &img, Mat& mod_diff, Mat &result)
{
    const int w = img.cols;
    const int h = img.rows;
	result = Mat::zeros(img.rows,img.cols,CV_64FC1);
	initXY(img);
    Mat res;
    dst(mod_diff, res);
	cout << " before: " << mod_diff << "\n"  << endl;
	cout << " after: " << res << "\n" << endl;
    for(int j = 0 ; j < h-2; j++)
    {
        double * resLinePtr = res.ptr<double>(j);
        for(int i = 0 ; i < w-2; i++)
        {
            resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
        }
    }

    idst(res, mod_diff);
	cout << " before2: " << res << "\n" << endl;
	cout << " after2: " << mod_diff << "\n" << endl;

    double *  resLinePtr = result.ptr<double>(0);
    const double * imgLinePtr = img.ptr<double>(0);
    const double * interpLinePtr = NULL;

     //first col
    for(int i = 0 ; i < w ; ++i)
        result.ptr<double>(0)[i] = img.ptr<double>(0)[i];

    for(int j = 1 ; j < h-1 ; ++j)
    {
        resLinePtr = result.ptr<double>(j);
        imgLinePtr  = img.ptr<double>(j);
        interpLinePtr = mod_diff.ptr<double>(j-1);

        //first row
        resLinePtr[0] = imgLinePtr[0];

        for(int i = 1 ; i < w-1 ; ++i)
        {
            //saturate cast is not used here, because it behaves differently from the previous implementation
            //most notable, saturate_cast rounds before truncating, here it's the opposite.
            double value = interpLinePtr[i-1];
            if(value < 0.)
                resLinePtr[i] = 0;
            else if (value > 255.0)
                resLinePtr[i] = 255;
            else
                resLinePtr[i] = static_cast<double>(value);
        }

        //last row
        resLinePtr[w-1] = imgLinePtr[w-1];
    }

    //last col
    resLinePtr = result.ptr<double>(h-1);
    imgLinePtr = img.ptr<double>(h-1);
    for(int i = 0 ; i < w ; ++i)
        resLinePtr[i] = imgLinePtr[i];
}

 
