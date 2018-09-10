#include <fstream>
#include "TrackTask.h"

void TrackTask::SetArgs(string name, string basePath, int startFrame, int endFrame, string seqZeroNum, string seqFormat, cv::Rect bbox, bool enableMonitor)
{

	SeqName = name;
	SeqPathPre = basePath;
	StartFrmId = startFrame;
	EndFrmId = endFrame;
	Bbox = bbox;
	ZeroNum = seqZeroNum;
	Format = seqFormat;
	EnableMonitor = enableMonitor;
}

void TrackTask::SetArgs(int argc, char* argv[])
{
	if (argc >= 11)
	{
		SetArgs(
			argv[1],
			argv[2],
			stoi(argv[3]),
			stoi(argv[4]),
			argv[5],
			argv[6],
			cv::Rect(atoi(argv[7]), atoi(argv[8]), atoi(argv[9]), atoi(argv[10])),
			stoi(argv[11]) == 1
			);
	}
}

cv::Mat& TrackTask::GetFrm(int frmId, int flag)
{
	if (CurrentFrmId != frmId)
	{
		CurrentFrmId = frmId;
		CurrentFrm = cv::imread(GetFrmPath(frmId), flag);
	}
	return CurrentFrm;
}

string TrackTask::GetFrmPath(int frmId) const
{
	string SeqFullPath = SeqPathPre + PathSub + "%0" + ZeroNum + "d." + Format;
	char imgPath[1024];
	sprintf(imgPath, SeqFullPath.c_str(), frmId);
	return imgPath;
}

void TrackTask::PushResult(cv::Rect result)
{
	Results.push_back(result);
	if (EnableMonitor)
	{
		cv::Mat renderedFrm = CurrentFrm.clone();
		char text[100];
		sprintf(text, "Frame: %d", CurrentFrmId);
		cv::rectangle(renderedFrm, result, cv::Scalar(0, 255, 255), 2, 8);
		cv::putText(renderedFrm, text, cv::Point(10, 25), CV_FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 0), 1);
		cv::imshow("Tracking Monitor", renderedFrm);
		cv::waitKey(1);
	}
}

void TrackTask::SaveResults()
{
	//Saving Result
	ofstream outFile;
	outFile.open(GetResultOutputPath(), ios::out);
	for (auto t : Results)
	{
		outFile << t.x << "," << t.y << "," << t.width << "," << t.height << endl;
	}
	if (outFile.is_open())
	{
		outFile.close();
	}
}
