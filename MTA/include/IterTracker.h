#ifndef ITERTRACKER_H
#define ITERTRACKER_H
#include "Config.h"
#include "Rect.h"
#include <vector>
#include <opencv/cv.h>
#include <Eigen3.2/Core>

#include <opencv2/flann/flann.hpp>
#include <opencv2/highgui/highgui.hpp>

class Config;
class Features;
class Kernel;
class LaRank;

class ImageRep;
class MultiSample;

class ITTrack {

public:
	ITTrack::ITTrack(const Config& conf);
	ITTrack::~ITTrack();
	
	inline const FloatRect& GetBB() const { return m_bb; }
	inline bool IsInitialised() const { return m_initialised; }

	//////////////////////////////////////////////////////////////////////////
	//Single Tracker
	void Initialise(const cv::Mat& frame, const FloatRect& bb);
	void Track(const cv::Mat& frame);
	//////////////////////////////////////////////////////////////////////////
	
	void InitialiseMH(const cv::Mat& frame, const FloatRect& bb,int NUMofFrm);
	void TrackMH(const cv::Mat& frame);
	inline const FloatRect& GetBB(int T_ID) const { return m_vecBB[T_ID]; }
	inline int GetNumberTracker() { return m_NumTracker; }

	std::vector<FloatRect> m_vecFusioinBB;

	std::vector<int> T_counter;
	std::vector<int> T_Select;
private:

	bool m_bcolorimage;
	int m_frameInterval;
	int m_frameID;
	Config m_config;
	bool m_initialised;	
	bool m_needsIntegralImage;
	bool m_needsIntegralHist;

	std::vector<ImageRep> m_Images;
	std::vector<Features*> m_features;
	std::vector<Kernel*> m_kernels;
	LaRank* m_pLearner;

	FloatRect m_bb;

	//Functions
	void UpdateLearner(const ImageRep& image);
	void FeatureSet();
	
	//////////////////////////////////////////////////////////////////////////
	//Multihypothesis tracker
	int m_remainFrmNum;
	int m_NumofFrms;
	int m_MaxNumTracker;
	int m_NumTracker;
	int m_selectedID;
	std::vector<std::vector<FloatRect>> m_vecBackTrajectory;
	std::vector<std::vector<FloatRect>> m_vecTrajectory;
	
	std::vector<FloatRect> m_vecBB;
	std::vector<FloatRect> m_vecBackBB;
	std::vector<LaRank*> m_vecLearner;
	std::vector<LaRank*> m_vecLearnerSave;
	std::vector<LaRank*> m_vecBackLearner;
	
	std::vector<int> m_vecOccFlag;
	std::vector<std::vector<float>> m_vecBackCost;
	std::vector<float> m_InterTrackOvrlap;
	std::vector<cv::Mat> m_vecFrgImg;

	void UpdateLearner(const ImageRep& image,LaRank* pTmpLearner, FloatRect& inRect);
	void TrackSingle(const ImageRep& image,LaRank* pTmpLearner, FloatRect& inRect);
	void TrackSingleWOUpd(const ImageRep& image,LaRank* pTmpLearner, FloatRect& inRect);
	void Refine();
	void InitTbacker();
	void MemRelbacker();
	void TrackBack();
	void SelectTracker();
	void CalLastFrm(int NUMofFrm);
	void BasicBackTracking(const ImageRep& image);
	//////////////////////////////////////////////////////////////////////////
	//Debug
	int m_debugWindowWidth;
	int m_txtLocX;
	int m_txtLocY;
	double m_fontScale;
	int m_refineNUM;
	bool dirExists(const std::string& dirName);
	void DebugRefine();
	void DebugInit(const Config& conf);
	void rectangle(cv::Mat& rMat, const FloatRect& rRect, const cv::Scalar& rColour, int thickness, int linetype);
	void rectangle(cv::Mat& rMat, const FloatRect& rRect, const cv::Scalar& rColour);
	void cvDrawDottedRect2(cv::Mat& img,FloatRect rRect, CvScalar& rColour, int thickness, int lenghOfDots, int lineType); 
	void cvDrawDottedRect(cv::Mat& img,cv::Point pt1, cv::Point pt2,CvScalar color, int thickness, int lenghOfDots, int lineType);
	void DrawDottedLine(IplImage* img, CvPoint pt1, CvPoint pt2,CvScalar color, int thickness, int lenghOfDots, int lineType, int leftToRight);

	void drawArrow(cv::Mat& image, cv::Point p, cv::Point q, CvScalar color, int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0);
	void Text(cv::Mat& img,std::string& inText,int p_x,int p_y,double fontScale,int fontWidth=2);
	void TextwC(cv::Mat& img,std::string& inText,int p_x,int p_y,double fontScale,int fontWidth=2,CvScalar color = CV_RGB(146, 208, 80));
	void SelectedTrackerColor(CvScalar& color);
	CvScalar SelectTrackColor(int TID);
	//////////////////////////////////////////////////////////////////////////
	//Appearance
	std::vector<std::vector<float>> m_vecBBDiffval;
	float CalImgDiff(int TrackID, int nFirstID, int nFrameID,FloatRect& inRect);
	//////////////////////////////////////////////////////////////////////////
	// NEW Backtracker

	int m_OccPreCounter;
	int m_IsOcclusion;
	int m_OccFrmCount;
	void PerFrameBackTracking();
	void SelectTargetImage();
	void CheckFrameOcclusion();
	void IsFrameOcclusion();

	//////////////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////////////
	//Version2
	int m_nNumBckFrms;
	ImageRep* m_pImages;

	float CalIntTrackOvrlapRatio(std::vector<FloatRect>& vecResBBs);
	double TrackSingleWOUpdv2(const ImageRep& image,LaRank* pTmpLearner, FloatRect& inRect);
	void TrackWUpdv2(const cv::Mat& frame);
	int selectLocation(std::vector<FloatRect>& vecResBBs,std::vector<double>& vecBestScores);
	void MHREstimation();
	void InitBackTrackV2();
	void BackTrackV2();
	void CompareTrajectoryV2(std::vector<float>& TrajectoryDist,int T_ID,int nSFrmID);
	void TrajectoryEstimation();
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//Time
	__int64  m_i64Frequency;
	__int64  m_i64StartTime;
	__int64  m_i64FinishTime;
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//V3
	Eigen::VectorXd m_ImgDiffMask;
	void CalcMask();
	
	std::vector<std::vector<float>> m_vecOvl;
	void ITTrack::TimeDomainPattern(std::vector<float>& TrajectoryDist);
	//////////////////////////////////////////////////////////////////////////
	void ITTrack::SaveFBTrajectory();
	
};


#endif