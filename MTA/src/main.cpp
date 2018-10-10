#include "IterTracker.h"

#include "Rect.h"
#include "Config.h"

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>


using namespace std;
using namespace cv;

static const int kLiveBoxWidth = 80;
static const int kLiveBoxHeight = 80;

void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour, int thickness, int linetype)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour,thickness,linetype);
}
void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);
}

void  ImageMat2BYTEGRAY(Mat& img,unsigned char* pbyOut)
{
	for (int y=0;y<img.rows;y++)
	{
		for (int x=0;x<img.cols;x++)
		{			
			pbyOut[x+y*img.cols] = img.at<uchar>(y,x);				
		}
	}
}
void  ImageMat2INTRGB(Mat& img,int* pbyOut)
{
	for (int y=0;y<img.rows;y++)
	{
		for (int x=0;x<img.cols;x++)
		{
			for (int c=0;c<img.channels();c++)
			{
				pbyOut[(x+y*img.cols)*img.channels()+c] = (int)img.at<Vec3b>(y,x)[c];				
			}
		}
	}
}

void ShowFlowImage(Mat& flow)
{
	//extraxt x and y channels
	Mat xy[2]; //X,Y
	cv::split(flow, xy);

	//calculate angle and magnitude
	Mat magnitude, angle;
	cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

	//translate magnitude to range [0;1]
	double mag_min,mag_max;
	cv::minMaxLoc(magnitude, &mag_min, &mag_max);
	magnitude.convertTo(magnitude, -1, 1.0/mag_max);

	//build hsv image
	Mat _hsv[3], hsv;
	_hsv[0] = angle;
	_hsv[1] = magnitude;
	_hsv[2] = cv::Mat::ones(angle.size(), CV_32F);
	cv::merge(_hsv, 3, hsv);

	//convert to BGR and show
	Mat bgr;//CV_32FC3 matrix
	cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
	cv::imshow("optical flow", bgr);
}

float fOverlap(const FloatRect resultRect,const FloatRect gtRect)
{
	float gtx, gty, gtw, gth;
	float tx, ty, tw, th;
	float x0,x1,y0,y1,areaInt;

	gtx = gtRect.XMin();
	gty = gtRect.YMin();
	gtw = gtRect.Width();
	gth = gtRect.Height();

	tx = resultRect.XMin();
	ty = resultRect.YMin();
	tw = resultRect.Width();
	th = resultRect.Height();

	x0 = max(gtx,tx);
	x1 = min(gtx+gtw,tx+tw);
	y0 = max(gty,ty);
	y1 = min(gty+gth,ty+th);

	if (x0 >= x1 || y0 >= y1) 
	{
		areaInt  =  0.f;
	}else
	{
		areaInt =(float) (x1-x0)*(y1-y0);
		areaInt/=((float)gtw*gth+(float)tw*th-areaInt);
	}
	return areaInt;
}
void CalRatio(cv::Mat tmpFrame, FloatRect initRect, float& fratio,int& searchRang)
{
	float minScale;

	int frame_min_width = 320;
	int trackwin_max_dimension = 64;
	int template_max_numel = 144;
	int frame_width = tmpFrame.cols;
	int frame_height = tmpFrame.rows;
	float fImgScale;
	float fBBWidth,fBBHeight;

	fBBWidth = initRect.Width();
	fBBHeight = initRect.Height();

	if (max(fBBWidth,fBBHeight) <= trackwin_max_dimension||	frame_width<= frame_min_width)
		fImgScale = 1.f;
	else
	{
		minScale = (float)frame_min_width/frame_width;
		fImgScale = (float)max((float)trackwin_max_dimension/max(fBBWidth,fBBHeight),minScale);    
	}
	float fBBWidth_r = initRect.Width()*fImgScale;
	float fBBHeight_r = initRect.Height()*fImgScale;
	float win_area = fBBHeight_r*fBBWidth_r;
	float fBBRatio = sqrtf((float)template_max_numel/win_area);
	fratio = fImgScale;
}
int main(int argc, char* argv[])
{
	int frame_counter=0;
	double TIME_SUM=0;

	string configPath = "config_girl_mov.txt";
	Config conf;

	int startFrame;
	int endFrame;
	float xmin;
	float ymin;
	float width;
	float height;

	std::string sZeroNum;
	std::string file_ext;

	string sseqName = argv[1];
	string sseqPath = argv[2];
	string sFeat = "haar";
	string sKern = "gaussian";
	int nSeed = 0;
	int nSearchRadius = 30;
	double dsvmC  = 100.0;
	int svmBsize = 100;
	double dParam = 0.2;
		
	startFrame = atoi(argv[3]);
	endFrame = atoi(argv[4]);
	xmin = atoi(argv[5]);
	ymin = atoi(argv[6]);
	width = atoi(argv[7]);
	height = atoi(argv[8]);

	sZeroNum = argv[9];
	file_ext = argv[10];

	conf.quietMode = (strcmp(argv[11],"1")==0)?false:true;

	conf.setting(sseqName, sseqPath, 
		nSeed, nSearchRadius,dsvmC,svmBsize, sFeat, sKern, dParam);

	cout << conf << endl;

	ofstream outFile;
	if (conf.resultsPath != "")
	{
		outFile.open(conf.resultsPath.c_str(), ios::out);
		if (!outFile)
		{
			cout << "error: could not open results file: " << conf.resultsPath << endl;
			return EXIT_FAILURE;
		}
	}

	VideoCapture cap;
	string imgFormat;

	unsigned char* pcInImage = NULL;
	int* pnInImage = NULL;

	string gtLine;
	ifstream gtFile;
	imgFormat = conf.sequenceBasePath+"/%0"+sZeroNum+"d."+file_ext;	

	char imgPath[256];
	sprintf(imgPath, imgFormat.c_str(), startFrame);
	Mat tmp = cv::imread(imgPath, 0);

	FloatRect initBB;
	initBB = FloatRect(xmin, ymin, width, height);

	float fRatio = 1.f;	 
	int nSRange;
	float fSearchRatio = 1.0f;	
	CalRatio(tmp, initBB, fSearchRatio,nSRange);
	float win_area = width*height;
	//increase the search range for a large rectangle
	if (win_area>=800)
	{
		conf.searchRadius = 34;
	}
	else
	{
		conf.searchRadius = 30;
	}
	
	initBB = FloatRect(xmin*fRatio, ymin*fRatio, width*fRatio, height*fRatio);


	conf.frameWidth = (int)(tmp.cols*fRatio);
	conf.frameHeight = (int)(tmp.rows*fRatio);


	Mat result(conf.frameHeight, conf.frameWidth, CV_8UC3), gray, HSI;
	bool paused = false;
	bool doInitialise = false;
	
	srand(conf.seed);

	//Set Tracker Parameter 
	ITTrack ITTracker(conf);
	
	Mat frame,frame2;

	for (int frameInd = startFrame; frameInd <= endFrame; ++frameInd)
	{
		char imgPath[256];
		sprintf(imgPath, imgFormat.c_str(), frameInd);
		Mat frame = cv::imread(imgPath, 1);
		resize(frame,frame,Size(conf.frameWidth,conf.frameHeight));
		if (frame.empty())
		{
			cout << "error: could not read frame: " << imgPath << endl;
			return EXIT_FAILURE;
		}
		frame.copyTo(result);
				

		if (frameInd == startFrame)
		{	
			ITTracker.InitialiseMH(frame,initBB,endFrame-startFrame+1);
			
			doInitialise = true;
 		}		
		
		if (frameInd>startFrame)
		{		
 			cout<<frameInd-startFrame+1<<"-th frame"<<endl;
						
			ITTracker.TrackMH(frame);
			frame_counter++;
									
			//Draw Result box(red)
 			rectangle(result, ITTracker.GetBB(0), CV_RGB(0, 160, 233));
			
			//Draw Result box(green)
			if (ITTracker.GetNumberTracker()>=2)
			{
			rectangle(result, ITTracker.GetBB(1),CV_RGB(142, 195, 31));
			}
			if (ITTracker.GetNumberTracker()>=3)
			{
				rectangle(result, ITTracker.GetBB(2),CV_RGB(248, 182,44));
			}
		}	
		
		if (!conf.quietMode)
		{
			imshow("result", result);
			waitKey(1);
		}
	}

	for (int nframe = 0; nframe<ITTracker.m_vecFusioinBB.size(); nframe++)
	{
		outFile << ITTracker.m_vecFusioinBB[nframe].XMin() / fRatio << "," << ITTracker.m_vecFusioinBB[nframe].YMin() / fRatio << "," << ITTracker.m_vecFusioinBB[nframe].Width() / fRatio << "," << ITTracker.m_vecFusioinBB[nframe].Height() / fRatio << endl;
	}

	if (outFile.is_open())
	{
		outFile.close();
	}
	return EXIT_SUCCESS;
}