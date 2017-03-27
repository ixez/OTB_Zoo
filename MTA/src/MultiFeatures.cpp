#include "MultiFeatures.h"

using namespace Eigen;
using namespace std;

MultiFeatures::MultiFeatures(const vector<Features*>& features) :
m_features(features)
{
	int d = 0;
	for (int i = 0; i < (int)features.size(); ++i)
	{
		d += features[i]->GetCount();
	}
	SetCount(d);
}

void MultiFeatures::UpdateFeatureVector(const Sample& s)
{
	int start = 0;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		int n =  m_features[i]->GetCount();
		m_featVec.segment(start, n) = m_features[i]->Eval(s);
		start += n;
	}
}
