#include "RawFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"

#include <iostream>
#include <fstream>
#include <string>
using namespace Eigen;
using namespace cv;

static const int kChannel = 4;
static const int kPatchSize = 16;
static const int kQLevel = 4;
static const double dweight [16][16] ={
{0.245061,0.291927,0.339171,0.384331,0.424752,0.457833,0.481307,0.493491,0.493491,0.481307,0.457833,0.424752,0.384331,0.339171,0.291927,0.245061},{0.291927,0.347757,0.404037,0.457833,0.505984,0.545392,0.573355,0.58787,0.58787,0.573355,0.545392,0.505984,0.457833,0.404037,0.347757,0.291927},{0.339171,0.404037,0.469423,0.531926,0.58787,0.633655,0.666144,0.683007,0.683007,0.666144,0.633655,0.58787,0.531926,0.469423,0.404037,0.339171},{0.384331,0.457833,0.531926,0.602752,0.666144,0.718026,0.75484,0.773948,0.773948,0.75484,0.718026,0.666144,0.602752,0.531926,0.457833,0.384331},{0.424752,0.505984,0.58787,0.666144,0.736203,0.793541,0.834227,0.855345,0.855345,0.834227,0.793541,0.736203,0.666144,0.58787,0.505984,0.424752},{0.457833,0.545392,0.633655,0.718026,0.793541,0.855345,0.8992,0.921963,0.921963,0.8992,0.855345,0.793541,0.718026,0.633655,0.545392,0.457833},{0.481307,0.573355,0.666144,0.75484,0.834227,0.8992,0.945303,0.969233,0.969233,0.945303,0.8992,0.834227,0.75484,0.666144,0.573355,0.481307},{0.493491,0.58787,0.683007,0.773948,0.855345,0.921963,0.969233,0.993769,0.993769,0.969233,0.921963,0.855345,0.773948,0.683007,0.58787,0.493491},{0.493491,0.58787,0.683007,0.773948,0.855345,0.921963,0.969233,0.993769,0.993769,0.969233,0.921963,0.855345,0.773948,0.683007,0.58787,0.493491},{0.481307,0.573355,0.666144,0.75484,0.834227,0.8992,0.945303,0.969233,0.969233,0.945303,0.8992,0.834227,0.75484,0.666144,0.573355,0.481307},{0.457833,0.545392,0.633655,0.718026,0.793541,0.855345,0.8992,0.921963,0.921963,0.8992,0.855345,0.793541,0.718026,0.633655,0.545392,0.457833},{0.424752,0.505984,0.58787,0.666144,0.736203,0.793541,0.834227,0.855345,0.855345,0.834227,0.793541,0.736203,0.666144,0.58787,0.505984,0.424752},{0.384331,0.457833,0.531926,0.602752,0.666144,0.718026,0.75484,0.773948,0.773948,0.75484,0.718026,0.666144,0.602752,0.531926,0.457833,0.384331},{0.339171,0.404037,0.469423,0.531926,0.58787,0.633655,0.666144,0.683007,0.683007,0.666144,0.633655,0.58787,0.531926,0.469423,0.404037,0.339171},{0.291927,0.347757,0.404037,0.457833,0.505984,0.545392,0.573355,0.58787,0.58787,0.573355,0.545392,0.505984,0.457833,0.404037,0.347757,0.291927},{0.245061,0.291927,0.339171,0.384331,0.424752,0.457833,0.481307,0.493491,0.493491,0.481307,0.457833,0.424752,0.384331,0.339171,0.291927,0.245061}
};
RawFeatures::RawFeatures(const Config& conf) :
m_patchImage(kPatchSize, kPatchSize, CV_8UC1)
{
	SetCount(kPatchSize*kPatchSize*kChannel);
}

void RawFeatures::UpdateFeatureVector(const Sample& s)
{
//    	Eigen::VectorXd weight;
//    	CalcMask(weight,kPatchSize,kPatchSize);
	int nPatchSZ = kPatchSize*kPatchSize;
	IntRect rect = s.GetROI(); // note this truncates to integers
	cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
	cv::resize(s.GetImage().GetMagImg()(roi), m_patchImage, m_patchImage.size());
	//equalizeHist(m_patchImage, m_patchImage);
	VectorXd vecTmp = VectorXd::Zero(kPatchSize*kPatchSize*kChannel);;
	int ind = 0;
	for (int i = 0; i < kPatchSize; ++i)
	{
		uchar* pixel = m_patchImage.ptr(i);
		for (int j = 0; j < kPatchSize; ++j, ++pixel, ++ind)
		{
			m_featVec[ind] = (double)*pixel*dweight[i][j];
		}
	}
	cv::resize(s.GetImage().GetImage(0)(roi), m_patchImage, m_patchImage.size());
	ind = 0;
	for (int i = 0; i < kPatchSize; ++i)
	{
		uchar* pixel = m_patchImage.ptr(i);
		for (int j = 0; j < kPatchSize; ++j, ++pixel, ++ind)
		{
			m_featVec[nPatchSZ+ind] = (double)*pixel*dweight[i][j];
		}
	}	
	cv::resize(s.GetImage().GetImage(1)(roi), m_patchImage, m_patchImage.size());
	ind = 0;
	for (int i = 0; i < kPatchSize; ++i)
	{
		uchar* pixel = m_patchImage.ptr(i);
		for (int j = 0; j < kPatchSize; ++j, ++pixel, ++ind)
		{
			m_featVec[nPatchSZ+ind] = (double)*pixel*dweight[i][j];
		}
	}	
	cv::resize(s.GetImage().GetImage(2)(roi), m_patchImage, m_patchImage.size());
	ind = 0;
	for (int i = 0; i < kPatchSize; ++i)
	{
		uchar* pixel = m_patchImage.ptr(i);
		for (int j = 0; j < kPatchSize; ++j, ++pixel, ++ind)
		{
			m_featVec[nPatchSZ*2+ind] = (double)*pixel*dweight[i][j];
		}
	}	
	m_featVec/=m_featVec.sum();
}

void RawFeatures::CalcMask(Eigen::VectorXd& inVec,int nW,int nH)
{
	std::ofstream out("weight_output.txt");
	
	int nHeight,nWidth;
	nWidth = nW;
	nHeight = nH;
	inVec.resize(nHeight,nWidth);
	
	float fCenterX = (float)(nWidth+1.f)*0.5f;
	float fCenterY = (float)(nHeight+1.f)*0.5f;
	int nHeightSq = nHeight*nHeight;
	int nWidthSq = nWidth*nWidth;

	for (int h=0;h<nHeight;h++)
	{
		float fhDiff = (float)((1.f+h)-fCenterX);
		std::cout<<fhDiff<<std::endl;
		out<<"{";
		for (int w=0;w<nWidth;w++)
		{
			float fwDiff = (float)((1.f+w)-fCenterY);
			inVec(h,w) = exp(-(double)3.2*((double)(fhDiff*fhDiff)/nHeightSq+(double)(fwDiff*fwDiff)/nWidthSq));
			out<<inVec(h,w);
			if (w!=nWidth-1)
			{
				out<<",";
			}
		}
		if (h!=nHeight-1)
		{
			out<<"},";
		}
		else
		{
			out<<"}";
		}		
	}
		
	out.close();
	getchar();
}