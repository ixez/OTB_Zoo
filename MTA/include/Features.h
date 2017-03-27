#ifndef FEATURES_H
#define FEATURES_H
	
#include "Sample.h"

#include <Eigen/Core>
#include <vector>

class Features
{
public:
	Features();

	inline const Eigen::VectorXd& Eval(const Sample& s) const
	{
		const_cast<Features*>(this)->UpdateFeatureVector(s);
		return m_featVec;
	}

	virtual void Eval(const MultiSample& s, std::vector<Eigen::VectorXd>& featVecs)
	{
		// default implementation
		featVecs.resize(s.GetRects().size());
		for (int i = 0; i < (int)featVecs.size(); ++i)
		{
			featVecs[i] = Eval(s.GetSample(i));
		}
	}

	inline int GetCount() const { return m_featureCount; }

protected:

	int m_featureCount;
	Eigen::VectorXd m_featVec;

	void SetCount(int c);
	virtual void UpdateFeatureVector(const Sample& s) = 0;

};
	

#endif