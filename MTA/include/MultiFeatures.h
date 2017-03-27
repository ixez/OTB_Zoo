#ifndef MULTI_FEATURES_H
#define MULTI_FEATURES_H

#include "Features.h"

#include <vector>

class Config;

class MultiFeatures : public Features
{
public:
	MultiFeatures(const std::vector<Features*>& features);

private:
	std::vector<Features*> m_features;

	virtual void UpdateFeatureVector(const Sample& s);
};

#endif
