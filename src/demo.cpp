
#include"../include/demo.h"

void UBSelect::getAllFiles(string &train_path, vector<string> &files, vector<string>& file_name){
	/*
    intptr_t   hFile = 0;
	struct _finddata_t fileinfo;  
	string p;  
	if ((hFile = _findfirst(p.assign(train_path).append("//*").c_str(), &fileinfo)) != -1){
		do{
			if ((fileinfo.attrib &  _A_SUBDIR)) {
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0){
					file_name.push_back(p.assign(train_path).append("//").append(fileinfo.name));
					getAllFiles(p.assign(train_path).append("//").append(fileinfo.name), files,file_name);
				}
			}
			else{
				files.push_back(p.assign(train_path).append("//").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0); 
		_findclose(hFile);
	}
	//random_shuffle(files.begin(), files.end());
    //
    */
    vector<string> paths;
    string path;
    string file;
    vector<string> single_files;
    for(int j = 0; j < 69; ++j){
        path = train_path + "//" + to_string(j+1) + "//";
  //      cout << path << endl;
        paths.push_back(path);
    }
  //  cout << "test" << endl;
    Directory dir;
    for(int ii = 0; ii < 69; ++ii){
//        cout << paths[ii] << endl;
        single_files.clear();
        single_files = dir.GetListFiles(paths[ii],"*",false);
  //      cout << single_files.size() << endl; 
        for(int jj = 0; jj < single_files.size();++jj){
            file = single_files[jj];
            cout << paths[ii]+ file << endl;
            files.push_back(paths[ii] + file);
        }
    }
  //  cout << files[1] << files.size() << endl;
}
bool UBSelect::importImage(const string path) {
//    cout << "img_path="<<path << endl;
	image = imread(path, 1);
	resize(image, dst, cv::Size(256, 256), (0, 0), (0, 0), 1);

	if (!dst.data)
		return false;
	return true;
}
void UBSelect::splitChannels() {
	split(dst, channelsRGB);
}
void UBSelect::getHistogram() {
	feature1.clear();
	feature2.clear();
	feature3.clear();
	feature.clear();
	calcHist(&channelsRGB[0], 1, 0, Mat(), outputRGB[0], 1, hisSize, &ranges);
	calcHist(&channelsRGB[1], 1, 0, Mat(), outputRGB[1], 1, hisSize, &ranges);
	calcHist(&channelsRGB[2], 1, 0, Mat(), outputRGB[2], 1, hisSize, &ranges);
	
	int sum1 = 0,sum2 = 0,sum3 = 0;

	for (int i = 0; i < hisSize[0]; ++i) {
		feature1.push_back(outputRGB[0].at<float>(i));
		feature2.push_back(outputRGB[1].at<float>(i));
		feature3.push_back(outputRGB[2].at<float>(i));
		sum1 = sum1 + outputRGB[0].at<float>(i);
		sum2 = sum2 + outputRGB[1].at<float>(i);
		sum3 = sum3 + outputRGB[2].at<float>(i);
	}

	for (int i = 0; i < hisSize[0]; ++i) {
		/*cout << feature1[i] << endl;*/
		feature1[i] = feature1[i] / sum1;
		feature2[i] = feature2[i] / sum2;
		feature3[i] = feature3[i] / sum3;
	}
	
	feature.insert(feature.end(), feature1.begin(), feature1.end());
	feature.insert(feature.end(), feature2.begin(), feature2.end());
	feature.insert(feature.end(), feature3.begin(), feature3.end());

}

string UBSelect::split_label(const string path) {
    char temp[100];
    path.copy(temp, path.size());
	const char * d = "//";
	char *p;
	vector<string> labels;
	string Labels;
	labels.clear();
	Labels.clear();
	p = strtok(temp, d);
	while (p) {
		//cout << p << endl;
		labels.push_back(p);
		p = strtok(NULL, d);
	}
    cout << labels.size() << endl;
    if (labels.size() < 2) {
        int temp = 1;
    }
	Labels = labels[labels.size() - 2];
	return Labels;
}
void UBSelect::extrcat_hist_feature(const vector<string>& files,vector<vector<float> >& feat_ , vector<float> &Labels) {

	for (int i = 0; i < files.size(); ++i) {
		cout <<"file="<< files[i].c_str();
		importImage(files[i]);
		splitChannels();
		getHistogram();

		feat_.push_back(feature);
		string label = split_label(files[i]);
		cout << " " << label.c_str() << endl;
		Labels.push_back(atof(label.c_str()));
	}
}

void UBSelect::extrcat_sift_feature(const vector<string>& files, vector<vector<float> >& sift_feats, vector<float> &Labels) {
	Mat featuresUnclustered1;
    Mat featuresUnclustered2;
    Mat featuresUnclustered;
	for (int i = 0; i < 17000;++i) {
        cout << files[i] <<" " << files.size()<< endl;
		Mat img = imread(files[i],1);
		cout << " the " << img.size() << " " << i << endl;
		vector<KeyPoint> keypoints;
		Mat descriptors;
		SiftDescriptorExtractor extractor;
		extractor.detect(img, keypoints);
		extractor.compute(img, keypoints, descriptors);
        featuresUnclustered1.push_back(descriptors);
        cout << "features size:"<<featuresUnclustered1.size() << endl;
        img.release();
	}
	for (int i = 17000; i < files.size();++i) {
		Mat img = imread(files[i],1);
		vector<KeyPoint> keypoints;
		Mat descriptors;
		SiftDescriptorExtractor extractor;
		extractor.detect(img, keypoints);
		extractor.compute(img, keypoints, descriptors); 
        featuresUnclustered2.push_back(descriptors);
        cout << "features size:"<<featuresUnclustered2.size() << endl;
        img.release();
	}
    vconcat(featuresUnclustered1, featuresUnclustered2, featuresUnclustered);
    cout << "end of extrcat feature" << endl;
    cout << featuresUnclustered.size() << endl;
    featuresUnclustered.convertTo(featuresUnclustered,CV_32F);
    cout << "convert finish" << endl;
	BOWKMeansTrainer trainer(CLUSTERNUM);
	Mat vocabulary = trainer.cluster(featuresUnclustered);
	FileStorage fw("vocabulary.xml", FileStorage::WRITE);
	fw << "Mat" << vocabulary;
	fw.release();

    /*
    Mat vocabulary;
    FileStorage fr("vocabulary.xml",FileStorage::READ);
    fr["Mat"] >> vocabulary;
*/
	Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SIFT");
	Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce");
	cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);
	bowDE.setVocabulary(vocabulary);

	for (int i = 0;i < files.size();++i) {
		Mat img = imread(files[i]);
		vector<KeyPoint>keypoints;
		SiftFeatureDetector detector;
		Mat descriptors;
		detector.detect(img, keypoints);
		bowDE.compute(img, keypoints, descriptors);
		normalize(descriptors, descriptors, 1.0, 0.0, NORM_MINMAX);
		sift.clear();
		for (size_t nrow = 0; nrow < CLUSTERNUM; ++nrow) {
			sift.push_back(descriptors.at<float>(0, nrow));
		}
		sift_feats.push_back(sift);
		string label = split_label(files[i]);
		Labels.push_back(atof(label.c_str()));
	}

//	for (int i = 32000;i < 32010;++i) {
//		cout << files[i].c_str();
//    }
}

void vector2Mat(vector<vector<float> > & src, Mat & dst, int type)
{
	Mat temp(src.size(), src.at(0).size(), type);
	for (int i = 0; i<temp.rows; ++i)
		for (int j = 0; j<temp.cols; ++j)
			temp.at<float>(i, j) = src.at(i).at(j);
	temp.copyTo(dst);
}
void vector2Mat(vector<float> & src, Mat & dst, int type)
{
	Mat temp(src.size(), 1, type);
	for (int i = 0; i < temp.rows; ++i) {
		temp.at<float>(i) = src.at(i);
	}
	temp.copyTo(dst);
}
//void vector22Mat(vector<float> src, Mat & dst, int type)
//{
//	Mat temp(1, src.size(), type);
//	for (int i = 0; i < temp.cols; ++i) {
//		temp.at<float>(0, i) = src.at(i);
//	}
//	temp.copyTo(dst);
//}


void UBSelect::svm_train() {

	train_path = "//home//zhukj1//UBSelect//crop2";
	//-----------------------------
	getAllFiles(train_path, files, file_name);//file_name 是类别数
	cout << "extrcat sift feature..." << endl;
	extrcat_sift_feature(files, sift_feats, sift_labels);
	vector2Mat(sift_feats, SIFT_FEAT, CV_32FC1);
	FileStorage fw2("SIFT_FEAT.xml", FileStorage::WRITE);
	fw2 << "Mat" << SIFT_FEAT;
	fw2.release();

	vector2Mat(sift_labels, Labels, CV_32FC1);
	FileStorage fw3("Sift_Labels.xml", FileStorage::WRITE);
	fw3 << "Mat" << Labels;
	fw3.release();
    
    cout << "extrcat hist feature..." << endl;
    extrcat_hist_feature(files, hist_feats, hist_labels);
    cout << files.size() << " " << hist_feats.size() << endl;
	vector2Mat(hist_feats, HIST_FEAT, CV_32FC1);
	FileStorage fw1("HIST_FEAT.xml", FileStorage::WRITE);
	fw1 << "Mat" << HIST_FEAT;
	fw1.release();
	//------------------------------------------
	vector2Mat(hist_labels, Labels, CV_32FC1);
	FileStorage fw4("Hist_Labels.xml", FileStorage::WRITE);
	fw4 << "Mat" << Labels;
	fw4.release();
	//------------------------------------------
    cout <<" concation mat..." << endl;
	hconcat(HIST_FEAT, SIFT_FEAT, FEATS);

	//for (int i = 0; i < FEATS.rows; ++i) {
	//	Mat tmp = FEATS.row(i);
	//}
/*
	CvMat labelsMat = cvMat(LABEL_NUM, 1, CV_32FC1, label);
	CvMat trainingDataMat = cvMat(LABEL_NUM, DATA, CV_32FC1, feat);*/
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100000, FLT_EPSILON);

	CvSVM SVM;
	SVM.train_auto(FEATS, Labels, Mat(), Mat(), params);
	//int c = SVM.get_support_vector_count();
	SVM.save("SVM_DATA.xml");
	cout << "训练完成" << endl;
 //   
	//Mat Sift;
	//Mat Hist;
	//Mat feats;
	//vector22Mat(sift, Sift, CV_32FC1);
	//vector22Mat(feature, Hist, CV_32FC1);
	//hconcat(Hist, Sift, feats);
	//for (int i = 0; i < 10; ++i) {
	//	cout << feats.at<float>(0, i) << endl;
	//}
	//cout << "-----------------" << endl;

	//for (int i = FEATS.rows-1; i < FEATS.rows; ++i) {
	//	Mat temp = FEATS.row(i);
	//	for (int j = 0; j < 10; ++j) {
	//		cout << temp.at<float>(0, j) << endl;
	//	}
	//	cout << "-----------------" << endl;

	//	float res = SVM.predict(temp);
	//	cout << res << endl;
	//}
	//float response = SVM.predict(feats);
	//cout << response << endl;
}


int main() {

	UBSelect demo;
	demo.svm_train();
	return 0;

}

