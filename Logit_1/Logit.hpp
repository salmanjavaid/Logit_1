#ifndef _LOGIT_H
#define _LOGIT_H


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <fstream>




using namespace cv;
using namespace std;



class Logit{

public:
	void Train_(Mat &_theta_, Mat &_X_, Mat &_y_, const double &_alpha_, const double &_lambda_, size_t _num_iter_);
	void Predict_(Mat &_X_, Mat &_theta_, Mat &_pred_);
	void sigmoid_(Mat &_z_);
	void Classify_Correct_(Mat &_pred_, Mat &_y_);
	void batch_grad_descent_(Mat &_theta_, Mat &_X_, Mat &_y_, Mat &_Ones_, Mat &_grad_, const double &_alpha_, const double &_lambda_);

};

#endif