#ifndef KERNELS_H
#define KERNELS_H

#include <Eigen/Core>
#include <cmath>

class Kernel
{
public:
	virtual double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const = 0;
	virtual double Eval(const Eigen::VectorXd& x) const = 0;
};

class LinearKernel : public Kernel
{
public:
	inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
	{
		return x1.dot(x2);
	}

	inline double Eval(const Eigen::VectorXd& x) const
	{
		return x.squaredNorm();
	}
};

class GaussianKernel : public Kernel
{
public:
	GaussianKernel(double sigma) : m_sigma(sigma) {}
	inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
	{
		return exp(-m_sigma*(x1-x2).squaredNorm());
	}

	inline double Eval(const Eigen::VectorXd& x) const
	{
		return 1.0;
	}

private:
	double m_sigma;
};

class IntersectionKernel : public Kernel
{
public:
	inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
	{
		return x1.cwiseMin(x2).sum();
	}

	inline double Eval(const Eigen::VectorXd& x) const
	{
		return x.sum();
	}
};
class InterMulKernel : public Kernel
{
public:
	inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
	{
		return x1.cwiseProduct(x2).sum();
	}

	inline double Eval(const Eigen::VectorXd& x) const
	{
		return x.sum();
	}
};

class Chi2Kernel : public Kernel
{
public:
	inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
	{
		double result = 0.0;
		for (int i = 0; i < x1.size(); ++i)
		{
			double a = x1[i];
			double b = x2[i];
			result += (a-b)*(a-b)/(0.5*(a+b)+1e-8);
		}
		return 1.0 - result;
	}

	inline double Eval(const Eigen::VectorXd& x) const
	{
		return 1.0;
	}
};


// class SIGMOIDKernel : public Kernel
// {
// public:
// 	inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
// 	{
// 
// 		if (fApB >= 0)
// 			return exp(-fApB)/(1.0+exp(-fApB));
// 		else
// 			return 1.0/(1+exp(fApB)) ;
// 	}
// 
// };

class MultiKernel : public Kernel
{
public:
	MultiKernel(const std::vector<Kernel*>& kernels, const std::vector<int>& featureCounts) :
	  m_n(kernels.size()),
		  m_norm(1.0/kernels.size()),
		  m_kernels(kernels),
		  m_counts(featureCounts)
	  {
	  }

	  inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
	  {
		  double sum = 0.0;
		  int start = 0;
		  for (int i = 0; i < m_n; ++i)
		  {
			  int c = m_counts[i];
			  sum += m_norm*m_kernels[i]->Eval(x1.segment(start, c), x2.segment(start, c));
			  start += c;
		  }
		  return sum;	
	  }

	  inline double Eval(const Eigen::VectorXd& x) const
	  {
		  double sum = 0.0;
		  int start = 0;
		  for (int i = 0; i < m_n; ++i)
		  {
			  int c = m_counts[i];
			  sum += m_norm*m_kernels[i]->Eval(x.segment(start, c));
			  start += c;
		  }
		  return sum;	
	  }

private:
	int m_n;
	double m_norm;
	std::vector<Kernel*> m_kernels;
	std::vector<int> m_counts;	

};

#endif