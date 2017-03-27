#include "Features.h"

using namespace Eigen;

Features::Features() :
m_featureCount(0)
{
}

void Features::SetCount(int c)
{
	m_featureCount = c;
	m_featVec = VectorXd::Zero(c);
}
