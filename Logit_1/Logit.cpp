#ifndef _LOGIT_H
#include "Logit.hpp"



void Logit::sigmoid_(Mat &_z_)
{
	Mat t;
	exp(-1 *_z_ , t);
	Mat _Ones_ = Mat::ones(_z_.rows, _z_.cols, CV_32F);
	_z_ = _Ones_ / (_Ones_ + t) ;

	
}



void Logit::Predict_(Mat &_X_, Mat &_theta_, Mat &_pred_)
{
	for (int i = 0; i < _X_.rows; i++)
	{
		//Mat t = _X_.row(i);
		
		Mat _check_ = _X_.row(i) * _theta_;
		sigmoid_(_check_);
		float k = _check_.at<float>(0 , 0);
		
		//cout<< k << endl;

		if  (k >= 0.5)
		{
			_pred_.row(i).col(0) = 1.0;
		}
		else
		{
			_pred_.row(i).col(0) = 0.0;
		}
	}
}


void Logit::Classify_Correct_(Mat &_pred_, Mat &_y_)
{
	int count = 1;
	for(int i = 0; i < _y_.rows; i++)
	{
		double a  = _pred_.at<float>( i , 0 );
		double b  = _y_.at<float>( i , 0 );

	//	cout<< a << "   " << b  << endl;
		if(a == b)
		{
			count++;	
		}
		else
		{
			int k = 0;
		}
	}

	cout<< (double)(((double)(count) / _y_.rows)) * 100 << endl;

}



void Logit::batch_grad_descent_(Mat &_theta_, Mat &_X_, Mat &_y_, Mat &_Ones_, Mat &_grad_, const double &_alpha_, const double &_lambda_)
{
		Mat _z_ = _X_ * _theta_;
		Mat temp;
		sigmoid_(_z_);
		_grad_.row(0).col(0) = ((double) (1 / (double)(_y_.rows))) * ((double)(sum( _z_ - _y_ ).val[0]));
		
		transpose(_X_.colRange(1, _X_.cols) , temp);
		_grad_.rowRange(1, _grad_.rows) = ( (1 / (double)(_y_.rows)) * (temp * (_z_ - _y_) ) ) + ( (_lambda_ / (double)(_y_.rows)) * _theta_.rowRange(1, _theta_.rows)); 

		/*if (j == 1)
		{
			cout<< _theta_.row(0).col(0) << endl;
			for (int i = 1; i < _theta_.rows; i++)
			{
				cout<< _theta_.row(i).col(0) << endl;
			}
		}*/

}




void Logit::Train_(Mat &_theta_, Mat &_X_, Mat &_y_, const double &_alpha_, const double &_lambda_, size_t _num_iter_)
{
	Mat _grad_ = Mat::zeros(_theta_.rows, _theta_.cols, CV_32F);
	Mat _Ones_ = Mat::zeros(_y_.rows, _y_.cols, CV_32F);
	for (size_t i = 0; i < _num_iter_; i++)
	{
		batch_grad_descent_(_theta_, _X_, _y_, _Ones_, _grad_, _alpha_, _lambda_);
		_theta_ = _theta_ - (_alpha_ * _grad_);
	}

}

#endif