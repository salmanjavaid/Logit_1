
#include "Logit.hpp"


int main()
{
	Mat M = Mat::zeros(1,1,CV_32F);
	CvMLData Train, Test;
	
	Train.read_csv("D:\\X_train_logit.csv");
	Test.read_csv("D:\\X_test_logit.csv");

	Mat X = Train.get_values();
	Mat y = Test.get_values();
	Mat _Ones_ = Mat::ones(y.rows, y.cols, CV_32F);
	Logit L;



	Mat theta = Mat::zeros(X.cols, 1, CV_32F);

	L.Train_(theta, X, y, 0.1, 0.00, 400);
	
	ofstream fout("D:\\theta.csv");

	for (int i = 0; i < theta.rows; i++)
	{
		fout<<theta.row(i).col(0)<<" "<<endl;
	}
	fout.close();

	Mat pred = Mat::zeros(y.rows, y.cols, CV_32F);
	L.Predict_(X, theta, pred);
	L.Classify_Correct_(pred, y);


	return 0;
}
