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

#include "ImageRep.h"

#include <cassert>

#include <opencv/highgui.h>
#include <opencv/cv.h>

using namespace std;
using namespace cv;

static const int kNumBins = 16;
static const bool LUVCOLOR = true;
static const bool COLOR_INTGR_HIST_IMG = true;
static const bool COLOR_INTGR_IMG = false;

ImageRep::ImageRep(const Mat& image, bool computeIntegral, bool computeIntegralHist, bool colour) :	
	m_channels(colour ? 3 : 1),
	m_rect(0, 0, image.cols, image.rows)
{	
	//m_ColorImage = image.clone();
	for (int i = 0; i < m_channels; ++i)
	{
		m_images.push_back(Mat::zeros(image.rows, image.cols, CV_8UC1));

	}
	if (computeIntegralHist)
	{
		m_integralHistImages.clear();
		if (COLOR_INTGR_HIST_IMG==true&&image.channels() == 3)
		{
			m_integralHistImages.resize(kNumBins*m_channels);			
			for (int i = 0; i < m_channels; ++i)
			{	
// #pragma omp parallel
// 				{
// #pragma omp for
					for (int j = 0; j < kNumBins; ++j)
					{
						m_integralHistImages[i*kNumBins+j]=Mat::zeros(image.rows+1, image.cols+1, CV_32SC1);
					}				
//				}	
			}
		}	
		else
		{
			m_integralHistImages.resize(kNumBins);			
// #pragma omp parallel
// 			{
// #pragma omp for
				for (int j = 0; j < kNumBins; ++j)
				{
					m_integralHistImages[j]=Mat::zeros(image.rows+1, image.cols+1, CV_32SC1);
				}				

//			}
		}
	}
		
	
	assert(image.channels() == 1 || image.channels() == 3);
	if (image.channels() == 3)
	{
		cvtColor(image, m_grayimages, CV_RGB2GRAY);
	}
	else if (image.channels() == 1)
	{
		image.copyTo(m_grayimages);
	}
	if (m_channels==3)
	{
		if (LUVCOLOR)
		{
			Mat HSIimage2;
			cvtColor(image,HSIimage2,CV_RGB2Lab);
			split(HSIimage2, m_images);		
		}
		else
		{
			split(image, m_images);		
		}
	}
	else
	{
		image.copyTo(m_images[0]);
	}
	

	
	if (computeIntegral)
	{
		m_integralImages.clear();
		if (COLOR_INTGR_IMG==true)
		{
			for (int i = 0; i < m_channels; ++i)
			{
				m_integralImages.push_back(Mat(image.rows+1, image.cols+1, CV_32SC1));
			}
			for (int i = 0; i < m_channels; ++i)
			{
				integral(m_images[i], m_integralImages[i]);
			}
		}
		else
		{			
			m_integralImages.push_back(Mat(image.rows+1, image.cols+1, CV_32SC1));
			integral(m_grayimages, m_integralImages[0]);			
		}
	}
	
	if (computeIntegralHist)
	{
		if (image.channels() == 3)
		{
// #pragma omp parallel
// 			{
// #pragma omp for
				for (int i = 0; i < m_channels; ++i)
				{
					Mat tmp(image.rows, image.cols, CV_8UC1);
					tmp.setTo(0);
					for (int j = 0; j < kNumBins; ++j)
					{
						for (int y = 0; y < image.rows; ++y)
						{
							const uchar* src = m_images[i].ptr(y);
							uchar* dst = tmp.ptr(y);
							for (int x = 0; x < image.cols; ++x)
							{
								int bin = (int)(((float)*src/256)*kNumBins);
								*dst = (bin == j) ? 1 : 0;
								++src;
								++dst;
							}
						}

						integral(tmp, m_integralHistImages[i*kNumBins+j]);			
					}
				}
//			}
		}
		else
		{
			Mat tmp(image.rows, image.cols, CV_8UC1);
			tmp.setTo(0);
			for (int j = 0; j < kNumBins; ++j)
			{
				for (int y = 0; y < image.rows; ++y)
				{
					const uchar* src = m_grayimages.ptr(y);
					uchar* dst = tmp.ptr(y);
					for (int x = 0; x < image.cols; ++x)
					{
						int bin = (int)(((float)*src/256)*kNumBins);
						*dst = (bin == j) ? 1 : 0;
						++src;
						++dst;
					}
				}
				integral(tmp, m_integralHistImages[j]);			
			}
		}

	}
	makeRankImage();
	//makeGrayRankImage();
}
ImageRep::~ImageRep()
{
	for (int i=0;i<m_integralImages.size();i++)
	{
		m_integralImages[i].release();
	}	
	for (int i=0;i<m_integralHistImages.size();i++)
	{
		m_integralHistImages[i].release();
	}
	for (int i=0;i<m_images.size();i++)
	{
		m_images[i].release();
	}
	m_integralImages.clear();
	m_integralHistImages.clear();
	m_images.clear();
}
int ImageRep::Sum(const IntRect& rRect, int channel) const
{
	assert(rRect.XMin() >= 0 && rRect.YMin() >= 0 && rRect.XMax() <= m_images[0].cols && rRect.YMax() <= m_images[0].rows);
	return m_integralImages[channel].at<int>(rRect.YMin(), rRect.XMin()) +
		m_integralImages[channel].at<int>(rRect.YMax(), rRect.XMax()) -
		m_integralImages[channel].at<int>(rRect.YMax(), rRect.XMin()) -
			m_integralImages[channel].at<int>(rRect.YMin(), rRect.XMax());
}

void ImageRep::Hist(const IntRect& rRect, Eigen::VectorXd& h) const
{
	assert(rRect.XMin() >= 0 && rRect.YMin() >= 0 && rRect.XMax() <= m_images[0].cols && rRect.YMax() <= m_images[0].rows);
	int norm = rRect.Area();
	
	for (int i = 0; i < kNumBins*m_channels; ++i)
	{
		int sum = m_integralHistImages[i].at<int>(rRect.YMin(), rRect.XMin()) +
			m_integralHistImages[i].at<int>(rRect.YMax(), rRect.XMax()) -
			m_integralHistImages[i].at<int>(rRect.YMax(), rRect.XMin()) -
			m_integralHistImages[i].at<int>(rRect.YMin(), rRect.XMax());
		h[i] = (float)sum/norm;
	}
}

void ImageRep::Hist2(const IntRect& rRect, Eigen::VectorXd& h) const
{
	assert(rRect.XMin() >= 0 && rRect.YMin() >= 0 && rRect.XMax() <= m_images[0].cols && rRect.YMax() <= m_images[0].rows);
	int norm = rRect.Area();
		
 #pragma omp parallel
 	{
 #pragma omp for
		for (int i = 0; i < kNumBins*m_channels; ++i)
		{
			int ncounter = 2;
			int sum = 2*(m_integralHistImages[i].at<int>(rRect.YMin(), rRect.XMin()) +
				m_integralHistImages[i].at<int>(rRect.YMax(), rRect.XMax()) -
				m_integralHistImages[i].at<int>(rRect.YMax(), rRect.XMin()) -
				m_integralHistImages[i].at<int>(rRect.YMin(), rRect.XMax()));
			if (i%kNumBins!=0)
			{
				sum += m_integralHistImages[i-1].at<int>(rRect.YMin(), rRect.XMin()) +
					m_integralHistImages[i-1].at<int>(rRect.YMax(), rRect.XMax()) -
					m_integralHistImages[i-1].at<int>(rRect.YMax(), rRect.XMin()) -
					m_integralHistImages[i-1].at<int>(rRect.YMin(), rRect.XMax());
				ncounter++;
			}
			if (i%kNumBins!=(kNumBins-1))
			{
				sum += m_integralHistImages[i+1].at<int>(rRect.YMin(), rRect.XMin()) +
					m_integralHistImages[i+1].at<int>(rRect.YMax(), rRect.XMax()) -
					m_integralHistImages[i+1].at<int>(rRect.YMax(), rRect.XMin()) -
					m_integralHistImages[i+1].at<int>(rRect.YMin(), rRect.XMax());
				ncounter++;
			}
		
			h[i] = (float)sum/norm/ncounter;
		}
	}
}

Mat doWork(InputArray _src,Size ksize,int nbins)
{
	Mat src = _src.getMat();
	CV_Assert( src.type() == CV_8UC1 && nbins > 0 );

	vector<Mat> mv;
	Mat dst = Mat::zeros(src.size(),CV_8UC1);
	Mat mask = Mat::zeros(src.size(),CV_8UC1);

	int step = 256/nbins;
	for (int i = 0; i < nbins; i++)
	{
		Mat temp, temp_blr;
		inRange(src,Scalar(i*step),Scalar(i*step+step),temp);
		mask += temp;

		blur(temp, temp_blr, ksize, Point(-1,-1), BORDER_DEFAULT);
		dst += mask.mul(temp_blr,1/double(255));
	}
	return dst;
}
Mat ImageRep::mat2gray(const cv::Mat& src)
{
	Mat dst;
	normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

	return dst;
}
void ImageRep::makeRankImage()
{		
	int ddepth = CV_32F;
	Mat grayImg;
	blur( m_images[0], grayImg, Size(3,3) );
	Mat grad_x, grad_y;
	
	Sobel(grayImg, grad_x, ddepth, 1, 0, 3);	
	Sobel(grayImg, grad_y, ddepth, 0, 1, 3);
	
	Mat mag, ori, normMag,  normOri;
	magnitude(grad_x, grad_y, mag);
//	phase(grad_x, grad_y, ori, true);

	normalize(mag, normMag, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
//	normalize(ori, normOri, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
	
	Mat maghist, anglehist,norm_maghist,norm_anglehist;
	maghist =Mat::zeros(1,256,CV_32F);
//	anglehist =Mat::zeros(1,256,CV_32F);
	norm_maghist = Mat(1,256,CV_8UC1);
//	norm_anglehist = Mat(1,256,CV_8UC1);
   #pragma omp parallel
 	{
   #pragma omp for
	for(int i = 0; i < normMag.rows ; i++ ) 
	{
		for (int j=0;j<normMag.cols ;j++) 
		{ 
			int mag = (int)normMag.at<unsigned char>(i,j);
			maghist.at<int>(0,mag) += 1;
// 			int ang = (int)normOri.at<unsigned char>(i,j);
// 			anglehist.at<int>(0,ang) += 1;
		}
	}
	}

	for(int i = 0; i < maghist.rows ; i++ ) 
	{
		for (int j=1;j<maghist.cols ;j++) 
		{ 
			maghist.at<int>(i,j)+=maghist.at<int>(i,j-1);	
//			anglehist.at<int>(i,j)+=anglehist.at<int>(i,j-1);	
		}
	}
   #pragma omp parallel
 	{
   #pragma omp for
	for(int i = 0; i < maghist.rows ; i++ ) 
	{
		for (int j=1;j<maghist.cols ;j++) 
		{ 	
			norm_maghist.at<unsigned char>(i,j) = (unsigned char)(255*maghist.at<int>(i,j)/maghist.at<int>(0,255));
//			norm_anglehist.at<unsigned char>(i,j) = (unsigned char)(255*anglehist.at<int>(i,j)/anglehist.at<int>(0,255));
		}
	}
	}
	
	m_gradient = Mat(normMag.rows,normMag.cols,CV_8UC1);
//	m_angle = Mat(normOri.rows,normOri.cols,CV_8UC1);

 #pragma omp parallel
 {
 #pragma omp for
	for(int i = 0; i < normMag.rows ; i++ ) 
	{
		for (int j=0;j<normMag.cols ;j++) 
		{ 
			int magVal = (int)normMag.at<unsigned char>(i,j);
//			int angVal = (int)normOri.at<unsigned char>(i,j);

			m_gradient.at<unsigned char>(i,j) = norm_maghist.at<unsigned char>(0,magVal);
//			m_angle.at<unsigned char>(i,j) = norm_anglehist.at<unsigned char>(0,angVal);	
		}
	}
 }

	m_magIntImg=Mat(grayImg.rows+1, grayImg.cols+1, CV_32SC1);
//	m_angIntImg=Mat(grayImg.rows+1, grayImg.cols+1, CV_32SC1);

	integral(m_gradient, m_magIntImg);
//	integral(m_angle, m_angIntImg);
	
}
void ImageRep::makeGrayRankImage()
{		
	Size ksize(5,5);
	int nbins = 32;
	m_RankImg = Mat::zeros(m_images[0].size(),CV_8UC1);
	Mat mask = Mat::zeros(m_images[0].size(),CV_8UC1);

	int step = 256/nbins;
	for (int i = 0; i < nbins; i++)
	{
		Mat temp, temp_blr;
		inRange(m_images[0],Scalar(i*step),Scalar(i*step+step),temp);
		mask += temp;

		blur(temp, temp_blr, ksize, Point(-1,-1), BORDER_DEFAULT);
		m_RankImg += mask.mul(temp_blr,1/double(255));
	}
	imshow("RankImg",m_RankImg);
}

int ImageRep::gradSum(const IntRect& rRect, int ntype) const
{
	assert(rRect.XMin() >= 0 && rRect.YMin() >= 0 && rRect.XMax() <= m_images[0].cols && rRect.YMax() <= m_images[0].rows);
	if (ntype==0)
	{
		return m_magIntImg.at<int>(rRect.YMin(), rRect.XMin()) +
			m_magIntImg.at<int>(rRect.YMax(), rRect.XMax()) -
			m_magIntImg.at<int>(rRect.YMax(), rRect.XMin()) -
			m_magIntImg.at<int>(rRect.YMin(), rRect.XMax());
	}
	else
	{
		return m_angIntImg.at<int>(rRect.YMin(), rRect.XMin()) +
			m_angIntImg.at<int>(rRect.YMax(), rRect.XMax()) -
			m_angIntImg.at<int>(rRect.YMax(), rRect.XMin()) -
			m_angIntImg.at<int>(rRect.YMin(), rRect.XMax());
	}	
}