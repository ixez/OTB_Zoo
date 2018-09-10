#pragma once
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
class TrackTask
{
public:
	void SetArgs(string name, string basePath, int startFrame, int endFrame, string seqZeroNum, string seqFormat, cv::Rect bbox, bool enableMonitor);
	void SetArgs(int argc, char* argv[]);
	cv::Mat& GetFrm(int frmId, int flag = CV_LOAD_IMAGE_COLOR);
	string GetFrmPath(int frmId) const;
	string GetResultOutputPath() const { return SeqName + ResultSubfix; };
	void PushResult(cv::Rect result);
	void SaveResults();

	string SeqName = "Box";
	string SeqPathPre = "/Users/zeke/Codes/GitRepos/tracker_benchmark/data/Box/img/";
	string PathSub = "";
	int StartFrmId = 1;
	int EndFrmId = 1161;
    cv::Rect Bbox=cv::Rect(478,143,80,111);
	string ZeroNum = "4";
	string Format = "jpg";
	bool EnableMonitor = true;
	string ResultSubfix = "_result.txt";

	int CurrentFrmId=-1;
	cv::Mat CurrentFrm;
	vector<cv::Rect> Results;
};
