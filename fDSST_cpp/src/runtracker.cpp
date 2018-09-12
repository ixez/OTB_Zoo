#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

#include "fdssttracker.hpp"


#include <windows.h>
//#include <dirent.h>



using namespace std;
using namespace cv;

std::vector <cv::Mat> imgVec;

int main(int argc, char* argv[]){

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;
	// Create KCFTracker object
	FDSSTTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// DSSTTracker tracker;


	std::string PATH_IMG_TOPCV = argv[1];
	std::string seq_name = argv[2];

	float time_sum = 0.0;

	int count = 1;
	cv::Mat image;
	char name[9];
	std::string imgName;
	std::string imgPath = PATH_IMG_TOPCV + "\\" + seq_name + "\\imgs\\";

	//get init target box params from information file
	std::ifstream initInfoFile;
	std::string fileName = imgPath + "groundtruth.txt";
	initInfoFile.open(fileName);
	std::string firstLine;
	std::getline(initInfoFile, firstLine);
	float initX, initY, initWidth, initHegiht;
	char ch;
	std::istringstream ss(firstLine);
	ss >> initX, ss >> ch;
	ss >> initY, ss >> ch;
	ss >> initWidth, ss >> ch;
	ss >> initHegiht, ss >> ch;

	cv::Rect_<float> initRect = cv::Rect(initX, initY, initWidth, initHegiht);

	double duration = 0;
	cv::Rect_<float> _roi = initRect;
	cv::Point2f center_point = cv::Point2f(_roi.x + _roi.width / 2.0, _roi.y + _roi.height / 2.0);
	cv::Size target_size = cv::Size(_roi.width, _roi.height);


	std::ifstream imagesFile;

	fileName = imgPath + "images.txt";
	imagesFile.open(fileName);
	std::string text;

	std::vector<std::string> filenames;

	while (getline(imagesFile, text))
	{
		filenames.push_back(text);
	}


	std::ofstream resultsFile;
	resultsFile.open(seq_name + ".txt");
	/*
	if (padded_w >= padded_h)  //fit to width
	_scale = padded_w / (float)template_size;
	else
	_scale = padded_h / (float)template_size;
	*/



	cv::Mat show_img;
	std::string imgFinalPath;

	float confidence;

	for (int i = 0; i < filenames.size(); i++)
	{
		std::string imgFinalPath = imgPath + "\\" + filenames[i];
		show_img = cv::imread(imgFinalPath, IMREAD_COLOR);

		//processImg = cv::imread(imgFinalPath, CV_LOAD_IMAGE_COLOR);

		if (show_img.empty())
		{
			break;
		}
		cv::Rect showRect;
		if (count == 1)
		{
			cv::Mat img;
			cv::cvtColor(show_img, img, cv::COLOR_RGB2GRAY);
			tracker.init(initRect, img);
			showRect = initRect;
		}
		else{
#ifndef WINDOWS
			LARGE_INTEGER t1, t2, tc;
			QueryPerformanceFrequency(&tc);
			QueryPerformanceCounter(&t1);
#endif
			cv::Mat img;
			cv::cvtColor(show_img, img, cv::COLOR_RGB2GRAY);
			showRect = tracker.update(img);
#ifndef WINDOWS
			QueryPerformanceCounter(&t2);
			printf("Use Time : %f\n", (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);
			time_sum += ((t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);
#endif
			// printf( "rect (w h): %d %d \n" , showRect.width, showRect.height);
		}


		cv::rectangle(show_img, showRect, cv::Scalar(0, 255, 0));
		cv::imshow("windows", show_img);
		cv::waitKey(1);
		count++;


	}
	std::cout << "FPS: " << count/ time_sum  << "\n";

	system("pause");
	return 0;

}
