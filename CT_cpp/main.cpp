#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "TrackTask.h"
#include "CompressiveTracker.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
	TrackTask conf;
	conf.SetArgs(argc, argv);

	// Create KCFTracker object
	CompressiveTracker ct;

	// Frame readed
	Mat frame;
	Mat frameGray;

	// Tracker results
	Rect result;

	// Using min and max of X and Y for groundtruth rectangle
	float xMin		=	conf.Bbox.x;
	float yMin		=	conf.Bbox.y;
	float width		=	conf.Bbox.width;
	float height	=	conf.Bbox.height;


	for (int frameId = conf.StartFrmId, i = 1; frameId <= conf.EndFrmId; ++frameId, ++i)
	{
		// Read each frame from the list
		frame = conf.GetFrm(frameId);
		cvtColor(frame, frameGray, CV_RGB2GRAY);

		// First frame, give the groundtruth to the tracker
		if (i == 1) {
			result = Rect(xMin, yMin, width, height);
            ct.init(frameGray, result);
		}
		// Update
		else{
            ct.processFrame(frameGray, result);
		}
        
		conf.PushResult(result);
	}

	conf.SaveResults();

}
