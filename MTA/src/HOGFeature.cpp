
#include "HOGFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"

#include <iostream>

using namespace Eigen;
using namespace cv;
using namespace std;

static const int kNumBins = 16;
static const int kNumLevels = 1;
static const int kNumCellsX = 4;
static const int kNumCellsY = 4;

HOGFeatures::HOGFeatures(const Config& conf)
{
	
	SetCount(kNumBins*kNumLevels*kNumCellsX*kNumCellsY);
	
}

void HOGFeatures::UpdateFeatureVector(const Sample& s)
{
	IntRect rect = s.GetROI(); // note this truncates to integers
	//cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
	//cv::resize(s.GetImage().GetImage(0)(roi), m_patchImage, m_patchImage.size());

	m_featVec.setZero();
	// 	VectorXd hist(kNumBins);
	// 	
	// 	int histind = 0;
	// 	for (int il = 0; il < kNumLevels; ++il)
	// 	{
	// 		int nc = il+1;
	// 		float w = s.GetROI().Width()/nc;
	// 		float h = s.GetROI().Height()/nc;
	// 		FloatRect cell(0.f, 0.f, w, h);
	// 		for (int iy = 0; iy < nc; ++iy)
	// 		{
	// 			cell.SetYMin(s.GetROI().YMin()+iy*h);
	// 			for (int ix = 0; ix < nc; ++ix)
	// 			{
	// 				cell.SetXMin(s.GetROI().XMin()+ix*w);
	// 				s.GetImage().Hist(cell, hist);
	// 				m_featVec.segment(histind*kNumBins, kNumBins) = hist;
	// 				++histind;
	// 			}
	// 		}
	// 	}
	VectorXd hist(kNumBins*m_nChannel);

	int histind = 0;

	float w = s.GetROI().Width()/kNumCellsX;
	float h = s.GetROI().Height()/kNumCellsY;
	FloatRect cell(0.f, 0.f, w, h);
	for (int iy = 0; iy < kNumCellsY; ++iy)
	{
		cell.SetYMin(s.GetROI().YMin()+iy*h);
		for (int ix = 0; ix < kNumCellsY; ++ix)
		{
			cell.SetXMin(s.GetROI().XMin()+ix*w);
			s.GetImage().Hist2(cell, hist);
			m_featVec.segment(histind*kNumBins*m_nChannel, kNumBins*m_nChannel) = hist;
			++histind;
		}
	}

	m_featVec /= histind;
}
