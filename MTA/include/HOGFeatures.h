#ifndef HOG_FEATURES_H
#define HOG_FEATURES_H

#include "Features.h"

class Config;

class HOGFeatures : public Features
{
public:
	HOGFeatures(const Config& conf);

private:

	virtual void UpdateFeatureVector(const Sample& s);
	int m_nChannel;
};

#endif