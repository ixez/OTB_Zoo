/* 
 * Struck: Structured Output Tracking with Kernels
 * 
 * Code to accompany the paper:
 *   Struck: Structured Output Tracking with Kernels
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   International Conference on Computer Vision (ICCV), 2011
 * 
 * Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of Struck.
 * 
 * Struck is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Struck is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Struck.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */
 
#include "Tracker.h"
#include "Config.h"

#include <iostream>
#include <fstream>

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;


void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);
}

int main(int argc, char* argv[])
{
	// read config file
	string configPath = "config.txt";
	Config conf(configPath);
	cout << conf << endl;
	
	if (conf.features.size() == 0)
	{
		cout << "error: no features specified in config" << endl;
		return EXIT_FAILURE;
	}
	
	ofstream outFile;
	conf.resultsPath=string(argv[1])+"_Struck.txt";
    outFile.open(conf.resultsPath.c_str(), ios::out);
	
	VideoCapture cap;
	
	int startFrame = atoi(argv[3]);
	int endFrame = atoi(argv[4]);
	FloatRect initBB;
	string imgFormat;
	float scaleW = 1.f;
	float scaleH = 1.f;

    conf.sequenceBasePath=argv[2];
    imgFormat = conf.sequenceBasePath+"/%0"+argv[9]+"d."+argv[10];

    // read first frame to get size
    char imgPath[256];
    sprintf(imgPath, imgFormat.c_str(), startFrame);
    Mat tmp = cv::imread(imgPath, 0);
    scaleW = (float)conf.frameWidth/tmp.cols;
    scaleH = (float)conf.frameHeight/tmp.rows;

    float xmin = atoi(argv[5]);
    float ymin = atoi(argv[6]);
    float width = atoi(argv[7]);
    float height = atoi(argv[8]);
    initBB = FloatRect(xmin*scaleW, ymin*scaleH, width*scaleW, height*scaleH);

    conf.quietMode = (strcmp(argv[11],"1")==0)?false:true;
	
	Tracker tracker(conf);
	if (!conf.quietMode)
	{
		namedWindow("result");
	}
	
	Mat result(conf.frameHeight, conf.frameWidth, CV_8UC3);
	srand(conf.seed);
	for (int frameInd = startFrame; frameInd <= endFrame; ++frameInd)
	{
        Mat frame;
        char imgPath[256];
        sprintf(imgPath, imgFormat.c_str(), frameInd);
        Mat frameOrig = cv::imread(imgPath, 0);
        if (frameOrig.empty())
        {
            cout << "error: could not read frame: " << imgPath << endl;
            return EXIT_FAILURE;
        }
        resize(frameOrig, frame, Size(conf.frameWidth, conf.frameHeight));
        cvtColor(frame, result, CV_GRAY2RGB);

        if (frameInd == startFrame)
        {
            tracker.Initialise(frame, initBB);
        }
		
		if (tracker.IsInitialised())
		{
			tracker.Track(frame);
			
			rectangle(result, tracker.GetBB(), CV_RGB(0, 255, 0));
			
			if (outFile)
			{
				const FloatRect& bb = tracker.GetBB();
				outFile << (int)(bb.XMin()/scaleW) << "," << (int)(bb.YMin()/scaleH) << "," << (int)(bb.Width()/scaleW) << "," << (int)(bb.Height()/scaleH) << endl;
			}
		}
		
		if (!conf.quietMode)
		{
			imshow("result", result);
			int key = waitKey(1);
		}
	}
	
	if (outFile.is_open())
	{
		outFile.close();
	}
	
	return EXIT_SUCCESS;
}
