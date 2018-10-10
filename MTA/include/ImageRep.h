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

#ifndef IMAGE_REP_H
#define IMAGE_REP_H

#include "Rect.h"

#include <opencv/cv.h>
#include <vector>

#include <Eigen/Core>

class ImageRep
{
public:
	
	ImageRep(const cv::Mat& rImage, bool computeIntegral, bool computeIntegralHists, bool colour = false);	
	~ImageRep();
	int Sum(const IntRect& rRect, int channel = 0) const;
	void Hist(const IntRect& rRect, Eigen::VectorXd& h) const;
	void Hist2(const IntRect& rRect, Eigen::VectorXd& h) const;

	inline const cv::Mat& GetGrayImage() const { return m_grayimages; }
	inline const cv::Mat& GetImage(int channel = 0) const { return m_images[channel]; }
	inline const std::vector<cv::Mat>& GetImages(int channel = 0) const { return m_images; }
//	inline const cv::Mat& GetOriginImage() const { return m_ColorImage; }
	inline const IntRect& GetRect() const { return m_rect; }
	int m_channels;
	std::vector<cv::Mat> m_images;

	//////////////////////////////////////////////////////////////////////////
	//
	inline const cv::Mat& GetMagImg() const { return m_gradient; }
	inline const cv::Mat& GetAngImg() const { return m_angle; }
	int gradSum(const IntRect& rRect, int ntype) const;	
	inline const cv::Mat& GetRankImage() const { return m_RankImg; }
	//////////////////////////////////////////////////////////////////////////
private:
//	cv::Mat m_ColorImage;
	cv::Mat m_grayimages;
	
	std::vector<cv::Mat> m_integralImages;
	std::vector<cv::Mat> m_integralHistImages;

	
	IntRect m_rect;

	//////////////////////////////////////////////////////////////////////////
	//intensity invariant feature
	void makeRankImage();
	
	cv::Mat mat2gray(const cv::Mat& src);
	cv::Mat m_gradient;
	cv::Mat m_angle;
	cv::Mat m_magIntImg;
	cv::Mat m_angIntImg;

	void makeGrayRankImage();
	cv::Mat m_RankImg;
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//HOG feature
	std::vector<cv::Mat> m_HOGHistImg;
	void makeHOGImg();
	//////////////////////////////////////////////////////////////////////////
};

#endif
