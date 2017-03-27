
#include "HistogramFeatures.h"
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


HistogramFeatures::HistogramFeatures(const Config& conf,const bool bcolor)
{
	if (bcolor==true)
	{
		m_nChannel = 3;
	}
	else
		m_nChannel = 1;
	
// 	int nc = 0;
// 	for (int i = 0; i < kNumLevels; ++i)
// 	{
// 		//nc += 1 << 2*i;
// 		nc += (i+1)*(i+1);
// 	}
// 	SetCount(kNumBins*nc);
// 	cout << "histogram bins: " << GetCount() << endl;

	if (m_nChannel==3)
	{
		SetCount(kNumBins*m_nChannel*kNumCellsX*kNumCellsY);
	}
	else
	{
		SetCount(kNumBins*kNumCellsX*kNumCellsY);
	}
}

void HistogramFeatures::UpdateFeatureVector(const Sample& s)
{
	FloatRect rect = s.GetROI(); // note this truncates to integers
	//cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
	//cv::resize(s.GetImage().GetImage(0)(roi), m_patchImage, m_patchImage.size());
	
	m_featVec.setZero();

// 	float x[] = {0.3f, 0.5f, 0.7f};
// 	float y[] = {0.3f, 054f, 0.7f};	
// 	float s[] = {0.4f};
// 	for (int iy = 0; iy < 4; ++iy)
// 	{
// 		for (int ix = 0; ix < 4; ++ix)
// 		{
// 			FloatRect r(x[ix]-0.3f, y[iy]-0.3f, 0.6f, 0.6f);
// 			IntRect sampleRect((int)(rect.XMin()+r.XMin()*rect.Width()+0.5f), (int)(rect.YMin()+r.YMin()*rect.Height()+0.5f),
// 				(int)(rect.Width()*r.Width()), (int)(rect.Height()*r.Height()));
// 			value += m_weights[i]*image.Sum(sampleRect);
// 		}
// 	}


	int histind = 0;
	if (m_nChannel==3)
	{
		VectorXd hist(kNumBins*m_nChannel);

		float w = s.GetROI().Width()/kNumCellsX;
		float h = s.GetROI().Height()/kNumCellsY;
		FloatRect cell(0.f, 0.f, w, h);
		for (int iy = 0; iy < kNumCellsY; ++iy)
		{
			cell.SetYMin(s.GetROI().YMin()+iy*h);
			for (int ix = 0; ix < kNumCellsX; ++ix)
			{
				float fweight=1.f;
				if (ix==0||ix==kNumCellsX-1||iy==0||iy==kNumCellsY-1)
				{
					fweight=0.7f;
					//fweight=0.65f;
				}
				cell.SetXMin(s.GetROI().XMin()+ix*w);
				s.GetImage().Hist2(cell, hist);
				m_featVec.segment(histind*kNumBins*m_nChannel, kNumBins*m_nChannel) = hist*fweight;
				++histind;
			}
		}
	}else
	{
		VectorXd hist(kNumBins);

		float w = s.GetROI().Width()/kNumCellsX;
		float h = s.GetROI().Height()/kNumCellsY;
		FloatRect cell(0.f, 0.f, w, h);
		for (int iy = 0; iy < kNumCellsY; ++iy)
		{
			cell.SetYMin(s.GetROI().YMin()+iy*h);
			for (int ix = 0; ix < kNumCellsY; ++ix)
			{
				float fweight=1.f;
				if (ix==0||ix==kNumCellsX-1||iy==0||iy==kNumCellsY-1)
				{
					fweight=0.7f;
				}
				cell.SetXMin(s.GetROI().XMin()+ix*w);
				s.GetImage().Hist2(cell, hist);
				m_featVec.segment(histind*kNumBins, kNumBins) = hist*fweight;
				++histind;
			}
		}
	}
	

	m_featVec /= histind;
}
