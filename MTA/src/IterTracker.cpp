#include "IterTracker.h"

#include "HaarFeatures.h"
#include "RawFeatures.h"
#include "HistogramFeatures.h"
#include "MultiFeatures.h"
#include "Kernels.h"

#include "Config.h"
#include "ImageRep.h"
#include "Sampler.h"
#include "Sample.h"

#include "LaRank.h"

#include <iostream>
#include <fstream>

#include <direct.h>
#include <Windows.h>
#include <opencv/cv.h>
#include <opencv2/video/tracking.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <opencv2\contrib\contrib.hpp>
#include <Eigen/Core>

using namespace std;
using namespace cv;


ITTrack::ITTrack(const Config& conf)
{
	m_config = conf;
	m_initialised = false;	

	m_pLearner = NULL;	

	m_NumTracker = 3;
	m_frameID = 0;
	m_frameInterval = 30;
	m_IsOcclusion = 0;

	DebugInit(conf);

	m_pImages = NULL;
	T_counter.resize(m_NumTracker);
	for (int i=0;i<T_counter.size();i++)
	{
		T_counter[i] = 0;
	}
}

ITTrack::~ITTrack()
{	
	if (m_pLearner!=NULL)
	{
		delete m_pLearner;
	}
	if (m_pImages!=NULL)
	{
		delete m_pImages;
	}
}

void ITTrack::rectangle(cv::Mat& rMat, const FloatRect& rRect, const cv::Scalar& rColour, int thickness, int linetype)
{
	IntRect r(rRect);
	cv::rectangle(rMat, cv::Point(r.XMin(), r.YMin()), cv::Point(r.XMax(), r.YMax()),rColour,thickness,linetype);
}
void ITTrack::rectangle(cv::Mat& rMat, const FloatRect& rRect, const cv::Scalar& rColour)
{
	IntRect r(rRect);
	cv::rectangle(rMat, cv::Point(r.XMin(), r.YMin()), cv::Point(r.XMax(), r.YMax()), rColour);
}

void ITTrack::drawArrow(cv::Mat& image, cv::Point p, cv::Point q, CvScalar color, int arrowMagnitude , int thickness, int line_type, int shift) 
{
    //Draw the principle line
    cv::line(image, p, q, color, thickness, line_type, shift);
    const double PI = 3.141592653;
    //compute the angle alpha
    double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
    //compute the coordinates of the first segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle + PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle + PI/4));
    //Draw the first segment
    cv::line(image, p, q, color, thickness, line_type, shift);
    //compute the coordinates of the second segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle - PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle - PI/4));
    //Draw the second segment
    cv::line(image, p, q, color, thickness, line_type, shift);
} 

void ITTrack::DrawDottedLine(IplImage* img,CvPoint pt1, CvPoint pt2,CvScalar color, int thickness, int lenghOfDots, int lineType, int leftToRight) 
{ 
	CvLineIterator iterator; 
	int count = cvInitLineIterator( img, pt1, pt2, &iterator, lineType, leftToRight );	
	int offset,x,y; 


	for( int i = 0; i < count; i= i + (lenghOfDots*2-1) ) 
	{ 
		if(i+lenghOfDots > count) 
			break; 

		offset = iterator.ptr - (uchar*)(img->imageData); 
		y = offset/img->widthStep; 
		x = (offset - y*img->widthStep)/(3*sizeof(uchar) /* size of pixel */); 

		CvPoint lTemp1 = cvPoint(x,y); 
		for(int j=0;j<lenghOfDots-1;j++)	//I want to know have the last of these in the iterator 
			CV_NEXT_LINE_POINT(iterator); 

		offset = iterator.ptr - (uchar*)(img->imageData); 
		y = offset/img->widthStep; 
		x = (offset - y*img->widthStep)/(3*sizeof(uchar) /* size of pixel */); 

		CvPoint lTemp2 = cvPoint(x,y); 
		cvDrawLine(img,lTemp1,lTemp2,color,thickness,lineType); 
		for(int j=0;j<lenghOfDots;j++) 
			CV_NEXT_LINE_POINT(iterator); 
	} 
}

void ITTrack::cvDrawDottedRect(Mat& img,Point pt1, Point pt2,CvScalar color, int thickness, int lenghOfDots, int lineType) 
{	
	IplImage ipl_img = img.operator IplImage();
	//1---2 
	//|	  | 
	//4---3 
	//	 1 --> pt1, 2 --> tempPt1, 3 --> pt2, 4 --> tempPt2 
	//Convert to IplImage or CvMat, no data copying


	CvPoint tempPt1 = cvPoint(pt2.x,pt1.y); 
	CvPoint tempPt2 = cvPoint(pt1.x,pt2.y); 
	DrawDottedLine(&ipl_img,pt1,tempPt1,color,thickness,lenghOfDots,lineType, 0); 
	DrawDottedLine(&ipl_img,tempPt1,pt2,color,thickness,lenghOfDots,lineType, 0); 
	DrawDottedLine(&ipl_img,pt2,tempPt2,color,thickness,lenghOfDots,lineType, 1); 
	DrawDottedLine(&ipl_img,tempPt2,pt1,color,thickness,lenghOfDots,lineType, 1); 
} 

void ITTrack::cvDrawDottedRect2(Mat& img,FloatRect rRect, CvScalar& rColour, int thickness, int lenghOfDots, int lineType) 
{	//1---2 
	//|	  | 
	//4---3 
	IplImage ipl_img = img.operator IplImage();
	IntRect r(rRect);
	CvScalar color(rColour);

	CvPoint pt1 = cvPoint(r.XMin(), r.YMin()); 
	CvPoint pt2 = cvPoint(r.XMax(),r.YMin()); 
	CvPoint pt3 = cvPoint(r.XMax(), r.YMax()); 
	CvPoint pt4 = cvPoint(r.XMin(),r.YMax()); 

	DrawDottedLine(&ipl_img,pt1,pt2,color,thickness,lenghOfDots,lineType, 0); 
	DrawDottedLine(&ipl_img,pt2,pt3,color,thickness,lenghOfDots,lineType, 0); 
	DrawDottedLine(&ipl_img,pt3,pt4,color,thickness,lenghOfDots,lineType, 1); 
	DrawDottedLine(&ipl_img,pt4,pt1,color,thickness,lenghOfDots,lineType, 1); 
} 


void ITTrack::FeatureSet()
{	
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
	m_features.clear();
	m_kernels.clear();

	m_needsIntegralImage = false;
	m_needsIntegralHist = false;

	int numFeatures = m_config.features.size();
	vector<int> featureCounts;
// 	for (int i = 0; i < numFeatures; ++i)
// 	{
// 		switch (m_config.features[i].feature)
// 		{
// 		case Config::kFeatureTypeHaar:
// 			m_features.push_back(new HaarFeatures(m_config));
// 			m_needsIntegralImage = true;
// 			break;			
// 		case Config::kFeatureTypeRaw:
// 			m_features.push_back(new RawFeatures(m_config));
// 			break;
// 		case Config::kFeatureTypeHistogram:
// 			m_features.push_back(new HistogramFeatures(m_config));
// 			m_needsIntegralHist = true;
// 			break;
// 		}
// 		featureCounts.push_back(m_features.back()->GetCount());
// 
// 		switch (m_config.features[i].kernel)
// 		{
// 		case Config::kKernelTypeLinear:
// 			m_kernels.push_back(new LinearKernel());
// 			break;
// 		case Config::kKernelTypeGaussian:
// 			m_kernels.push_back(new GaussianKernel(m_config.features[i].params[0]));
// 			break;
// 		case Config::kKernelTypeIntersection:
// 			m_kernels.push_back(new IntersectionKernel());
// 			break;
// 		case Config::kKernelTypeChi2:
// 			m_kernels.push_back(new Chi2Kernel());
// 			break;
// 		}
// 	}
	
	//Tracker 1
 	m_features.push_back(new HaarFeatures(m_config));	
 	m_needsIntegralImage = true;
 	m_kernels.push_back(new IntersectionKernel());
		
	//Tracker 2
  	m_features.push_back(new HistogramFeatures(m_config,m_bcolorimage));	
  	m_needsIntegralHist = true;
  	m_kernels.push_back(new IntersectionKernel());

	//Tracker 3
	m_features.push_back(new RawFeatures(m_config));	 	
	m_kernels.push_back(new IntersectionKernel());

// 	//Tracker 1
// 	m_features.push_back(new HaarFeatures(m_config));	
// 	m_needsIntegralImage = true;
// 	m_kernels.push_back(new GaussianKernel(0.2));
// 
// 	//Tracker 2
// 	m_features.push_back(new HistogramFeatures(m_config,m_bcolorimage));	
// 	m_needsIntegralHist = true;
// 	m_kernels.push_back(new GaussianKernel(0.2));
// 
// 	//Tracker 3
// 	m_features.push_back(new RawFeatures(m_config));	 	
// 	m_kernels.push_back(new GaussianKernel(0.2));
	if (numFeatures > 1)
	{
		MultiFeatures* f = new MultiFeatures(m_features);
		m_features.push_back(f);

		MultiKernel* k = new MultiKernel(m_kernels, featureCounts);
		m_kernels.push_back(k);		
	}	
}


void ITTrack::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0
	vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 2*m_config.searchRadius, 5, 16);
	//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);

	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}

#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif
		
	MultiSample sample(image, keptRects);
	m_pLearner->Update(sample, 0);
}

void ITTrack::InitialiseMH(const cv::Mat& frame, const FloatRect& bb, int NUMofFrm)
{
	if (frame.channels()==3)
	{
		m_bcolorimage = true;
	}
	else
	{
		m_bcolorimage = false;
	}
	FeatureSet();	

	//cout<<m_features[1]<<m_features.back()<<endl;
	m_bb = IntRect(bb);
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist,m_bcolorimage);
	m_Images.push_back(image);

	for (int T_ID=0;T_ID<m_NumTracker;T_ID++)
	{
		m_vecBB.push_back(IntRect(bb));

		LaRank* pTmpLearner = new LaRank(m_config, *(m_features[T_ID]), *(m_kernels[T_ID]));		
		UpdateLearner(image,pTmpLearner,m_vecBB[T_ID]);
		m_vecLearner.push_back(pTmpLearner);

		std::vector<FloatRect> tmpTrajectory;
		tmpTrajectory.push_back(bb);
		m_vecTrajectory.push_back(tmpTrajectory);

	}	
	m_vecFusioinBB.push_back(m_bb);
	
	m_initialised = true;
	CalLastFrm(NUMofFrm);
	
	Mat LabImg = frame.clone();
	if (m_bcolorimage==true)
	{
		cvtColor(LabImg,LabImg,CV_RGB2Lab);
	}
	
	Mat roi = Mat(LabImg, cv::Rect((int)m_bb.XMin(), (int)m_bb.YMin(), (int)m_bb.Width(), (int)m_bb.Height()));
	cv::Mat tmpMat = roi.clone();
	m_vecFrgImg.push_back(tmpMat);

	m_vecLearnerSave.clear();
	for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
	{
		LaRank* pTmpLearner = new LaRank(m_config, *(m_features[T_ID]), *(m_kernels[T_ID]));
		m_vecLearner[T_ID]->CopyTo(*pTmpLearner);
		m_vecLearnerSave.push_back(pTmpLearner);
	}

	CalcMask();
}
void ITTrack::CalcMask()
{
	
	int nHeight,nWidth;
	nWidth = (int)m_bb.Width();
	nHeight = (int)m_bb.Height();

	Mat DiffWeight(nHeight, nWidth, CV_32FC1);	

	m_ImgDiffMask.resize(nHeight,nWidth);
	float fCenterX = (float)(nWidth-1.f)/2.f;
	float fCenterY = (float)(nHeight-1.f)/2.f;
	int nHeightSq = nHeight*nHeight;
	int nWidthSq = nWidth*nWidth;
	
	for (int h=0;h<nHeight;h++)
	{ 
		float fhDiff = (float)(h-fCenterY);
		for (int w=0;w<nWidth;w++)
		{
			float fwDiff = (float)(w-fCenterX);
			m_ImgDiffMask(h,w) = exp(-(double)3.2*((double)(fhDiff*fhDiff)/nHeightSq+(double)(fwDiff*fwDiff)/nWidthSq));
		}
	}
}
void ITTrack::CalLastFrm(int NUMofFrm)
{
	m_NumofFrms=NUMofFrm;
	m_remainFrmNum = ((m_NumofFrms-1)%m_frameInterval);
}

void ITTrack::TrackMH(const cv::Mat& frame)
{
	assert(m_initialised);
	
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist,m_bcolorimage);	
	m_Images.push_back(image);
		
	for (int T_ID=0;T_ID<m_NumTracker;T_ID++)
	{

		TrackSingleWOUpd(image,m_vecLearner[T_ID],m_vecBB[T_ID]);
		m_vecTrajectory[T_ID].push_back(m_vecBB[T_ID]);
	}
	m_frameID++;	

	BasicBackTracking(image);
}
void ITTrack::BasicBackTracking(const ImageRep& image)
{
	if (m_IsOcclusion!=1)
	{
		for (int T_ID=0;T_ID<m_NumTracker;T_ID++)
		{
			UpdateLearner(image,m_vecLearner[T_ID],m_vecBB[T_ID]);		
		}
	}
	if ((m_frameID%m_frameInterval)==0)
	{		
		Refine();

		for (int TID=0;TID<m_vecLearner.size();TID++)
		{
			delete m_vecLearner[TID];
		}
		m_vecLearner.erase(m_vecLearner.begin(),m_vecLearner.end());

		for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
		{					
			LaRank* pTmpLearner = new LaRank(m_config, *(m_features[T_ID]), *(m_kernels[T_ID]));
			m_vecLearnerSave[T_ID]->CopyTo(*pTmpLearner);
			m_vecLearner.push_back(pTmpLearner);
		}	

		if (m_IsOcclusion!=1)
		{
			int nFirstFrame = m_vecFusioinBB.size()-m_frameInterval-1;
			for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
			{
				for (int frmID = 1;frmID<=m_frameInterval;frmID++)
				{
					//cout<<"Back Cost: "<<m_vecBackCost[m_selectedID][frmID]<<endl;
					if (m_vecBackCost[m_selectedID][frmID]>=0.2f)
					{					
						UpdateLearner(m_Images[frmID],m_vecLearner[T_ID],m_vecFusioinBB[nFirstFrame+frmID]);			
					}
				}
			}

			for (int TID=0;TID<m_vecLearnerSave.size();TID++)
			{
				delete m_vecLearnerSave[TID];
			}
			m_vecLearnerSave.clear();
			for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
			{				
				LaRank* pTmpLearner = new LaRank(m_config, *(m_features[T_ID]), *(m_kernels[T_ID]));
				m_vecLearner[T_ID]->CopyTo(*pTmpLearner);
				m_vecLearnerSave.push_back(pTmpLearner);				
			}
		}
		int frmSZ = m_Images.size()-1;
		for (int fID=0;fID<frmSZ;fID++)
		{
			m_Images.erase(m_Images.begin());			
		}	
	}
	else if (m_frameID==(m_NumofFrms-1))
	{
		m_frameInterval = m_remainFrmNum;
		Refine();
		m_Images.clear();
	}
	MemRelbacker();
}

void ITTrack::Refine()
{
	InitTbacker();
	TrackBack();
	SelectTracker();
}

void ITTrack::Text(cv::Mat& img,string& inText,int p_x,int p_y,double fontScale,int fontWidth)
{
	string text = inText;

	int fontFace = FONT_HERSHEY_SIMPLEX;
	double fontScale2  = m_fontScale*fontScale;	
	cv::Point textOrg(p_x, p_y);
	cv::putText(img, text, textOrg, fontFace, fontScale2,  CV_RGB(255, 200, 20), fontWidth,8);
}
void ITTrack::TextwC(cv::Mat& img,string& inText,int p_x,int p_y,double fontScale,int fontWidth,CvScalar color)
{
	string text = inText;

	int fontFace = FONT_HERSHEY_SIMPLEX;
	double fontScale2  = m_fontScale*fontScale;	
	cv::Point textOrg(p_x, p_y);
	cv::putText(img, text, textOrg, fontFace, fontScale2,  color, fontWidth,8);
}
CvScalar ITTrack::SelectTrackColor(int TID)
{
	if (TID == 0)
	{
		return CV_RGB(0, 160, 233);
	}
	else if (TID == 1)
	{
		return  CV_RGB(142, 195, 31);
	}
	else if (TID == 2)
	{
		return  CV_RGB(248, 182,44);
	}
	else{
		return  CV_RGB(146, 208, 80);
	}
}
void ITTrack::SelectedTrackerColor(CvScalar& color)
{
	if (m_selectedID == 0)
	{
		color = CV_RGB(0, 160, 233);
	}
	else if (m_selectedID == 1)
	{
		color = CV_RGB(142, 195, 31);
	}
	else if (m_selectedID == 2)
	{
		color = CV_RGB(248, 182,44);
	}
	else
	{
		color = CV_RGB(146, 208, 80);
	}
}
void ITTrack::DebugRefine()
{		
	m_refineNUM++;
	stringstream ssRfNum;
	ssRfNum << setw(4) << setfill('0') <<m_refineNUM;	
	string strRf_num = ssRfNum.str();
	
	CvScalar selColor;	
	SelectedTrackerColor(selColor);

	int nFirstID = m_frameID - m_frameInterval;
	for (int frmID = 0;frmID<=m_frameInterval;frmID++)
	{		
		Mat oriImg;
		cv::merge(m_Images[frmID].GetImages(),oriImg);
		cvtColor(oriImg,oriImg,CV_Lab2RGB);
		
		Mat mergedImg = Mat(Size(oriImg.cols/*+m_debugWindowWidth*/,oriImg.rows),CV_8UC3);
		mergedImg.setTo(cv::Scalar(0,0,0));
		Mat roi = Mat(mergedImg, cv::Rect(0, 0, oriImg.cols, oriImg.rows));

		int nFrameID = nFirstID+frmID;
		oriImg.copyTo(roi);
	
		char framenumber[100] = "#%04d";
		sprintf(framenumber,framenumber,nFrameID);
		string text = (string)framenumber;		
		TextwC(mergedImg,text,10,m_txtLocY+40,1.2,5,CV_RGB(0, 0, 0));
		TextwC(mergedImg,text,10,m_txtLocY+40,1.2,3,CV_RGB(255, 241, 0));
		
		for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
		{
			if (T_ID == 0)
			{
				rectangle(mergedImg, m_vecTrajectory[T_ID][nFrameID], CV_RGB(0, 0, 0),4,8);
				rectangle(mergedImg, m_vecTrajectory[T_ID][nFrameID], CV_RGB(0, 160, 233),2,8);
				cvDrawDottedRect2(mergedImg, m_vecBackTrajectory[T_ID][frmID], CV_RGB(0,0,0),4,4,8);
				cvDrawDottedRect2(mergedImg, m_vecBackTrajectory[T_ID][frmID], CV_RGB(0, 160, 233),2,4,8);
				drawArrow(mergedImg,cv::Point(m_vecTrajectory[T_ID][nFrameID].XCentre(),m_vecTrajectory[T_ID][nFrameID].YCentre()),cv::Point(m_vecBackTrajectory[T_ID][frmID].XCentre(),m_vecBackTrajectory[T_ID][frmID].YCentre()),CV_RGB(0, 160, 233),3,1,CV_AA);

			}
			if (T_ID == 1)
			{
				rectangle(mergedImg, m_vecTrajectory[T_ID][nFrameID], CV_RGB(0, 0, 0),4,8);
				rectangle(mergedImg, m_vecTrajectory[T_ID][nFrameID], CV_RGB(142, 195, 31),2,8);
				cvDrawDottedRect2(mergedImg, m_vecBackTrajectory[T_ID][frmID], CV_RGB(0,0,0),4,4,8);
				cvDrawDottedRect2(mergedImg, m_vecBackTrajectory[T_ID][frmID], CV_RGB(142, 195, 31),2,4,8);
				drawArrow(mergedImg,cv::Point(m_vecTrajectory[T_ID][nFrameID].XCentre(),m_vecTrajectory[T_ID][nFrameID].YCentre()),cv::Point(m_vecBackTrajectory[T_ID][frmID].XCentre(),m_vecBackTrajectory[T_ID][frmID].YCentre()),CV_RGB(142, 195, 31),3,1,CV_AA);
			}
			if (T_ID == 2)
			{
				rectangle(mergedImg, m_vecTrajectory[T_ID][nFrameID], CV_RGB(0, 0, 0),4,8);
				rectangle(mergedImg, m_vecTrajectory[T_ID][nFrameID], CV_RGB(248, 182,44),2,8);
				cvDrawDottedRect2(mergedImg, m_vecBackTrajectory[T_ID][frmID], CV_RGB(0,0,0),4,4,8);
				cvDrawDottedRect2(mergedImg, m_vecBackTrajectory[T_ID][frmID], CV_RGB(248, 182,44),2,4,8);
				drawArrow(mergedImg,cv::Point(m_vecTrajectory[T_ID][nFrameID].XCentre(),m_vecTrajectory[T_ID][nFrameID].YCentre()),cv::Point(m_vecBackTrajectory[T_ID][frmID].XCentre(),m_vecBackTrajectory[T_ID][frmID].YCentre()),CV_RGB(248, 182,44),3,1,CV_AA);
			}
		}
		
				
		stringstream ssFrmNum;
		ssFrmNum << setw(5) << setfill('0') <<nFrameID;	
		string frame_num = ssFrmNum.str();

		std::string outImage = "result/"+m_config.sequenceName+"/oimg_r"+strRf_num+"f"+frame_num+".jpg";
		imwrite(outImage.c_str(),mergedImg);		
	}
}
void ITTrack::SaveFBTrajectory()
{
	int nFirstID = m_frameID - m_frameInterval;
	stringstream ssRfNum;
	ssRfNum << setw(4) << setfill('0') <<m_refineNUM+1;	
	string strRf_num = ssRfNum.str();
	if(dirExists("TextResult/")==false)
	{
		mkdir("TextResult");		
	}
	
	for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
	{
		stringstream ssTRKNum;
		ssTRKNum << setw(2) << setfill('0') <<T_ID+1;	
		string tracker_num = ssTRKNum.str();

		std::string outFolder= "TextResult/"+m_config.sequenceName+"/";
		if(dirExists(outFolder.c_str())==false)
		{
			mkdir(outFolder.c_str());		
		}
		std::string outTxt = "TextResult/"+m_config.sequenceName+"/tj_r"+strRf_num+"tracker"+tracker_num+".txt";
		ofstream tmpfile;
		tmpfile.open(outTxt.c_str());
		for (int frmID = 0;frmID<=m_frameInterval;frmID++)
		{
			int nFrameID = nFirstID+frmID;
			tmpfile<<m_vecTrajectory[T_ID][nFrameID].XMin()<<","<<m_vecTrajectory[T_ID][nFrameID].YMin()<<","<<m_vecTrajectory[T_ID][nFrameID].Width()<<","<<m_vecTrajectory[T_ID][nFrameID].Height()<<",";
			tmpfile<<m_vecBackTrajectory[T_ID][frmID].XMin()<<","<<m_vecBackTrajectory[T_ID][frmID].YMin()<<","<<m_vecBackTrajectory[T_ID][frmID].Width()<<","<<m_vecBackTrajectory[T_ID][frmID].Height()<<","<<endl;
		}
		tmpfile.close();
	}
}
void ITTrack::DebugInit(const Config& conf)
{
	if(dirExists("result/")==false)
	{
		mkdir("result");		
	}
	if(dirExists("result/")==true)
	{		
		string outfolder = "result/"+m_config.sequenceName;
		if (dirExists(outfolder.c_str())==false)
		{			
			mkdir(outfolder.c_str());
		}		
	}
	else
	{
		cout<<"========Folder Creation Error========"<<endl;
		exit(0);
	}
	m_refineNUM = 0;
	m_debugWindowWidth = (int)(100*conf.frameWidth/640+0.5);
	m_fontScale = (double)conf.frameWidth/1000;
	m_txtLocX = (int)(m_config.frameWidth);
	m_txtLocY = (int)(m_config.frameHeight/432);
}

bool ITTrack::dirExists(const std::string& dirName)
{
	unsigned long ftyp = GetFileAttributesA(dirName.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;  //something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;   // this is a directory!

	return false;    // this is not a directory!
}
void ITTrack::MemRelbacker()
{

	m_vecBackLearner.clear();
	for (int i=0;i<m_vecBackTrajectory.size();i++)
	{
		m_vecBackTrajectory[i].clear();
	}
	m_vecBackTrajectory.clear();
	m_vecBackBB.clear();
}
void ITTrack::InitTbacker()
{
	m_vecBackTrajectory.resize(m_NumTracker);
	m_vecBackBB.resize(m_NumTracker);
	ImageRep LastImage = m_Images.back();
	for (int T_ID=0;T_ID<m_NumTracker;T_ID++)
	{		

		LaRank* pTmpLearner = new LaRank(m_config, *(m_features[T_ID]), *(m_kernels[T_ID]));		
		UpdateLearner(LastImage,pTmpLearner,m_vecBB[T_ID]);		
		m_vecBackLearner.push_back(pTmpLearner);

		std::vector<FloatRect> tmpTrajectory;
		tmpTrajectory.resize(m_frameInterval+1);
		tmpTrajectory[m_frameInterval] = m_vecBB[T_ID];
		m_vecBackTrajectory[T_ID] = (tmpTrajectory);
		m_vecBackBB[T_ID] = m_vecBB[T_ID];
	}
}
void ITTrack::TrackBack()
{
	for (int frmID = m_frameInterval-1;frmID>=0;frmID--)
	{
		for (int T_ID=0;T_ID<m_NumTracker;T_ID++)
		{			

			TrackSingleWOUpd(m_Images[frmID],m_vecBackLearner[T_ID],m_vecBackBB[T_ID]);
			UpdateLearner(m_Images[frmID],m_vecBackLearner[T_ID],m_vecBackBB[T_ID]);
			m_vecBackTrajectory[T_ID][frmID] = m_vecBackBB[T_ID];
		}
	}
}
float ITTrack::CalImgDiff(int TrackID, int nFirstFrmID, int nFrameID,FloatRect& inRect)
{
	FloatRect FirstRect = m_vecTrajectory[TrackID][nFirstFrmID];
	int n1stX = FirstRect.XMin();
	int n1stY = FirstRect.YMin();
	
	int ntargetX = inRect.XMin();
	int ntargetY = inRect.YMin();

	double tmpSum=0.0;
	int nChannel = m_Images.back().m_channels;
	for (int imgID=0;imgID<m_vecFrgImg.size();imgID++)
	{
		for (int nh=0;nh<m_bb.Height();nh++)
		{
			for (int nw=0;nw<m_bb.Width();nw++)
			{				
				for (int nc=0;nc<nChannel;nc++)
				{

					int tmpDiff =( (int)m_vecFrgImg[imgID].at<Vec3b>(nh,nw)[nc]- (int)m_Images[nFrameID].m_images[nc].at<unsigned char>(ntargetY+nh,ntargetX+nw));
					tmpSum+=((double)(tmpDiff*tmpDiff)*m_ImgDiffMask(nh,nw));
				}
			}
		}	
	}		
	return (float)exp(-tmpSum/(double)(inRect.Width()*inRect.Height()*900*m_vecFrgImg.size()));
}

void ITTrack::TimeDomainPattern(std::vector<float>& TrajectoryDist)
{
	int nFrmEnd = min(4,m_frameInterval);
	int nFirstID = m_frameID - m_frameInterval;
	int nOccFlg = 0;
	for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
	{		
		int nflg = 0;
		for (int frmID = 0;frmID<=nFrmEnd;frmID++)
		{
			if (m_vecOvl[T_ID][frmID]<=0.3)
			{
				nflg++;				
			}
		}
		if (nflg <= 1)
		{
			TrajectoryDist[T_ID] = TrajectoryDist[T_ID]*1000000;
			nOccFlg++;
		}

	}
	if(nOccFlg==0)
	{
		m_OccPreCounter = 1;//OCCLUSION
	}
	else{
		m_OccPreCounter = 0;//NON-OCCLUSION
	}

}

void ITTrack::SelectTracker()
{

	int nFirstID = m_frameID - m_frameInterval;
	std::vector<float> TrajectoryDist;
	TrajectoryDist.reserve(m_NumTracker);

	m_vecBBDiffval.clear();
	m_vecBBDiffval.resize(m_NumTracker);

	m_vecOvl.clear();
	
	for (int T_ID=0;T_ID<m_NumTracker;T_ID++)
	{
		m_vecBBDiffval[T_ID].resize(m_frameInterval+1);
	}


	m_vecBackCost.clear();
	for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
	{
		std::vector<float> tmpOvrlCost;
		tmpOvrlCost.resize(m_frameInterval+1);
		std::vector<float> tmpCost;
		tmpCost.resize(m_frameInterval+1);
		TrajectoryDist[T_ID] = 0.f;
		float tmpTRJDist = 0.f;
// #pragma omp parallel
//  		{
// #pragma omp for reduction(+: tmpTRJDist )
			for (int frmID = 0;frmID<=m_frameInterval;frmID++)
			{
				float x_diff = m_vecBackTrajectory[T_ID][frmID].XCentre()-m_vecTrajectory[T_ID][nFirstID+frmID].XCentre();
				float y_diff = m_vecBackTrajectory[T_ID][frmID].YCentre()-m_vecTrajectory[T_ID][nFirstID+frmID].YCentre();
				float ImgDiff = CalImgDiff(T_ID, nFirstID,frmID,m_vecBackTrajectory[T_ID][frmID]);
				m_vecBBDiffval[T_ID][frmID] = ImgDiff;
				float costVal = (float)(ImgDiff*expf(-(x_diff*x_diff+y_diff*y_diff)/500.f));


				float foverlap = m_vecBackTrajectory[T_ID][frmID].Overlap(m_vecTrajectory[T_ID][nFirstID+frmID]);
				
				tmpOvrlCost[frmID] = foverlap;
								
				tmpTRJDist += (costVal);							
				tmpCost[frmID] = costVal;
				
			}
		//}
		TrajectoryDist[T_ID] = tmpTRJDist;
		m_vecBackCost.push_back(tmpCost);

		m_vecOvl.push_back(tmpOvrlCost);
	}
	TimeDomainPattern(TrajectoryDist);
	int maxID=0;
	float maxVal=FLT_MIN;
	for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
	{
		if (maxVal<TrajectoryDist[T_ID])
		{
			maxVal = TrajectoryDist[T_ID];
			maxID = T_ID;
		}
	}	
	T_counter[maxID]++;
	T_Select.push_back(maxID);
	CheckFrameOcclusion();
	IsFrameOcclusion();

	m_InterTrackOvrlap.clear();
	m_InterTrackOvrlap.resize(m_frameInterval+1);

	for (int frmID = 0;frmID<=m_frameInterval;frmID++)
	{
		float tmpOverlp = 0.f;
		for (int T_ID = 0;T_ID<m_NumTracker-1;T_ID++)
		{
			for (int T_ID2 = T_ID+1;T_ID2<m_NumTracker;T_ID2++)
			{
				tmpOverlp+=m_vecTrajectory[T_ID][nFirstID+frmID].Overlap(m_vecTrajectory[T_ID2][nFirstID+frmID]);
			}
		}
		if (tmpOverlp>0.f)
		{
			tmpOverlp/=(float)(m_NumTracker-1);
		}
				
		m_InterTrackOvrlap[frmID] = tmpOverlp;
	}
	

	//select tracjectory
	for (int frmID = 1;frmID<=m_frameInterval;frmID++)
	{
		m_vecFusioinBB.push_back(m_vecTrajectory[maxID][nFirstID+frmID]);	
	}	
	
	for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
	{
		m_vecTrajectory[T_ID][nFirstID+m_frameInterval] = m_vecBB[maxID];
		if (m_vecTrajectory[T_ID].size()<=nFirstID+m_frameInterval)
		{
			cout<<"Trajectory Size Error"<<endl;
			getchar();
			exit(1);
		}
		m_vecBB[T_ID] = m_vecBB[maxID];
	}
	m_selectedID = maxID;
	if (m_IsOcclusion==0)
	{
		SelectTargetImage();
	}	

}
void ITTrack::IsFrameOcclusion()
{
	m_IsOcclusion = 0;
	
	if(m_OccPreCounter==0)
	{
		int ncounter = 0;
		for (int frmID = 0;frmID<=m_frameInterval;frmID++)
		{
			if (m_vecOccFlag[frmID]==1)
			{
				ncounter++;
			}
			else
			{
				ncounter=0;
			}
			if (ncounter>=20)
			{
				m_IsOcclusion = 1;
				break;
			}
		}
	}
	else
	{
		m_IsOcclusion = 1;
	}
}
void ITTrack::CheckFrameOcclusion()
{
	m_vecOccFlag.clear();
	m_vecOccFlag.resize(m_frameInterval+1);

	for (int frmID = 0;frmID<=m_frameInterval;frmID++)
	{
		m_vecOccFlag[frmID] = 1;
		for (int T_ID = 0;T_ID<m_NumTracker;T_ID++)
		{
			if (m_vecBackCost[T_ID][frmID]>0.004f)
			{
				m_vecOccFlag[frmID] = 0;
				break;
			}
		}
	}	
}


void ITTrack::SelectTargetImage()
{
	std::vector<float> score;
	score.resize(m_frameInterval+1);
	for (int frmID=0;frmID<score.size();frmID++)
	{
		float BackCost = 1.f;
		for (int TID=0;TID<m_NumTracker;TID++)
		{
			BackCost = BackCost*m_vecBackCost[TID][frmID];
		}
		score[frmID] = m_InterTrackOvrlap[frmID]*BackCost;
	}
	std::vector<int> sortedIndice( m_frameInterval+1 );
	for (int frmID=0;frmID< m_frameInterval+1;frmID++ )
	{
		sortedIndice[frmID] = frmID;
	}
	std::sort(sortedIndice.begin(),sortedIndice.end(),[&score](size_t i1, size_t i2) {
		return score[i1] > score[i2];
	});

	int nFirstID = m_frameID - m_frameInterval;

	for (int ID=0;ID<min(3,m_frameInterval);ID++)
	{
		int fID =sortedIndice[ID];
		if (m_vecOccFlag[fID]!=1&&m_vecBackCost[m_selectedID][fID]>=0.2f)
		{
			Mat oriImg;
			cv::merge(m_Images[fID].GetImages(),oriImg);
			
			Mat roi = Mat(oriImg, cv::Rect((int)m_vecFusioinBB[nFirstID+fID].XMin(), (int)m_vecFusioinBB[nFirstID+fID].YMin(), (int)m_vecFusioinBB[nFirstID+fID].Width(), (int)m_vecFusioinBB[nFirstID+fID].Height()));
						
			cv::Mat tmpMat = roi.clone();
			m_vecFrgImg.push_back(tmpMat);
		}		
	}
	if (m_vecFrgImg.size()>4)
	{
		int ImgNUM =m_vecFrgImg.size();		
		m_vecFrgImg.erase(m_vecFrgImg.begin()+1,m_vecFrgImg.begin()+ImgNUM-3);		
	}
}
void ITTrack::TrackSingleWOUpd(const ImageRep& image,LaRank* pTmpLearner, FloatRect& inRect)
{
	vector<FloatRect> rects;
	if (m_IsOcclusion==1)
	{
		rects = Sampler::PixelSamples(inRect, m_config.searchRadius*4, 8);
	}
	else
	{
		rects = Sampler::PixelSamples(inRect, m_config.searchRadius, 1);
	}
	
	

	vector<FloatRect> keptRects;
	keptRects.reserve(rects.size());
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}

	MultiSample sample(image, keptRects);

	vector<double> scores;
	pTmpLearner->Eval(sample, scores);

	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];
			bestInd = i;
		}
	}

	if (bestInd != -1)
	{
		inRect = keptRects[bestInd];		
	}
	keptRects.clear();
	scores.clear();
	rects.clear();
}

void ITTrack::TrackSingle(const ImageRep& image,LaRank* pTmpLearner, FloatRect& inRect)
{
	vector<FloatRect> rects = Sampler::PixelSamples(inRect, m_config.searchRadius);

	vector<FloatRect> keptRects;
	keptRects.reserve(rects.size());
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}

	MultiSample sample(image, keptRects);

	vector<double> scores;
	pTmpLearner->Eval(sample, scores);

	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
	
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];
			bestInd = i;
		}
	}
	if (bestInd != -1)
	{
		inRect = keptRects[bestInd];
		UpdateLearner(image,pTmpLearner,inRect);
	}
	keptRects.clear();
	scores.clear();
	rects.clear();
}

void ITTrack::UpdateLearner(const ImageRep& image,LaRank* pTmpLearner, FloatRect& inRect)
{
	vector<FloatRect> rects = Sampler::RadialSamples(inRect, 2*m_config.searchRadius, 5, 16);

	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}

	MultiSample sample(image, keptRects);
	pTmpLearner->Update(sample, 0);
}

