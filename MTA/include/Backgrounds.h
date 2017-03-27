// #ifndef BACKGROUNDS_H
// #define BACKGROUNDS_H
// 
// #include<Eigen3.2/Core>
// 
// class Backgrounds
// {
// public:
// 	float m_HSV_Feature[15][4*48];		
// 
// 	Backgrounds();
// 	~Backgrounds();
// 
// 	void setHaarFeature(float* inFeature,int num_feature);
// 	void setHSVFeature(float* inFeature,int num_feature);
// 
// 	float getHSVDist(const Eigen::VectorXd& Feat);
// 	void printfHSVFeature();
// 	int m_frame_flag;
// 	int m_num_frame;
// 	int m_num_Feat[15];
// 	int m_max_frame;
// };
// #endif


#ifndef BACKGROUNDS_H
#define BACKGROUNDS_H

#include <Eigen3.2/Core>
#include <vector>

class Backgrounds
{
public:
	//float m_HSV_Feature[15][4*48];

	std::vector<Eigen::VectorXd> m_HSV_Feature;


	Backgrounds();
	~Backgrounds();

	void setHaarFeature(float* inFeature,int num_feature);

	void setHSVFeature(const Eigen::VectorXd& Feat);
	float getHSVDist(const Eigen::VectorXd& Feat);
	void printfHSVFeature();
	void setMemory(int featdim);
	void copyHSVFeature(const Backgrounds& inBackgroundModel);
	int m_max_frame;
	int m_featdim;
};

#endif

