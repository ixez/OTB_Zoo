#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "kcftracker.hpp"
#include "TrackTask.h"


using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
	TrackTask conf;
	conf.SetArgs(argc, argv);

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = true;

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	// Using min and max of X and Y for groundtruth rectangle
	float xMin		=	conf.Bbox.x;
	float yMin		=	conf.Bbox.y;
	float width		=	conf.Bbox.width;
	float height	=	conf.Bbox.height;
    
    float tha = 0.40;

	for (int frameId = conf.StartFrmId, i = 1; frameId <= conf.EndFrmId; ++frameId, ++i)
	{
		// Read each frame from the list
		frame = conf.GetFrm(frameId);

		// First frame, give the groundtruth to the tracker
		if (i == 1) {
			result = Rect(xMin, yMin, width, height);
			tracker.init(result, frame);
		}
		// Update
		else{
			result = tracker.update(frame);
		}
        
		conf.PushResult(result);
	}

	conf.SaveResults();

}
