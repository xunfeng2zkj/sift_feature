#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include<opencv2/opencv.hpp>
#include <opencv/ml.h>
#include <opencv2/ml/ml.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include<opencv2/contrib/contrib.hpp>
#include<string>
#include<fstream>
//#include<io.h>
#include<vector>
#include <dirent.h> 

#define CLUSTERNUM 768

using namespace std;
using namespace cv;
class UBSelect {
private:
	string train_path,test_path,test_img;
	vector<string> files,file_name;

	Mat image,dst;
	int hisSize[1], hisWidth, hisHeight;
	float range[2];
	const float *ranges;
	Mat channelsRGB[3];
	MatND outputRGB[3];
	vector<float>feature1,feature2,feature3,feature;

	vector<vector<float> >hist_feats,sift_feats;
	vector<float>hist_labels,sift_labels;
	vector<float>sift;

	Mat HIST_FEAT, SIFT_FEAT,FEATS;
	Mat Labels;

public:
	UBSelect() {
		hisSize[0] = 256;
		hisWidth = 400;
		hisHeight = 400;
		range[0] = 0.0;
		range[1] = 255.0;
		ranges = &range[0];
	}
	void svm_train();
	void getAllFiles(string &train_path, vector<string>& files, vector<string> &file_name);
	bool importImage(const string path);
	void splitChannels();
	void getHistogram();
	void extrcat_hist_feature(const vector<string>& files, vector<vector<float> >& feat_, vector<float> &Labels);
	void extrcat_sift_feature(const vector<string>& files, vector<vector<float> >& sift_feats, vector<float> &Labels);
	string split_label(const string path);
	


};
