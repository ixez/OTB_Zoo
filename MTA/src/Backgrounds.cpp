#include "Backgrounds.h"
#include <string>
#include <cstdio>
#include <Eigen/Core>
#include <float.h>


using namespace std;

Backgrounds::Backgrounds()
{	
	m_max_frame = 10;	
	m_HSV_Feature.clear();
}

Backgrounds::~Backgrounds()
{
	m_HSV_Feature.clear();
}

void Backgrounds::setMemory(int featdim)
{
	m_featdim = featdim;
	//	m_HSV_Feature.reserve(m_max_frame+1);
}


void Backgrounds::setHSVFeature(const Eigen::VectorXd& inFeat)
{

	m_HSV_Feature.push_back(inFeat);

	if (m_HSV_Feature.size()>(size_t)m_max_frame)
	{
		m_HSV_Feature.erase(m_HSV_Feature.begin());
	}	
}

float Backgrounds::getHSVDist(const Eigen::VectorXd& Feat)
{
	float minDist = FLT_MAX;
	float min2ndDist = FLT_MAX;
	int FeatIndex = -1;
	int Feat2ndIndex = -1;	

	for (int nf=0;nf<(int)m_HSV_Feature.size();nf++)
	{

		float tmp_dist = (float)(m_HSV_Feature[nf]-Feat).norm();
		if (minDist>tmp_dist)
		{
			minDist = tmp_dist;
			min2ndDist = minDist;
		}		
	}

	if (min2ndDist<FLT_MAX)
	{
		return (minDist+min2ndDist)/2.f;
	}
	return minDist;
}
void Backgrounds::copyHSVFeature(const Backgrounds& inBackgroundModel)
{
	m_HSV_Feature = inBackgroundModel.m_HSV_Feature;
}