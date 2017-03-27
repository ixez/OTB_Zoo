#ifndef HISTOGRAM_FEATURES_H
#define HISTOGRAM_FEATURES_H

#include "Features.h"

class Config;

class HistogramFeatures : public Features
{
public:
	HistogramFeatures(const Config& conf,const bool bcolor);

private:

	virtual void UpdateFeatureVector(const Sample& s);
	int m_nChannel;
};

#endif