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

#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>
#include <ostream>

#define VERBOSE (0)

class Config
{
public:
	Config() { SetDefaults(); }
	//Config(const std::string& path);
	void Config::setting(const std::string& path);
	void Config::setting(std::string& seqname,std::string& seqpath,
		int nSeed, int sRadius, double dsvmC, int svmBSize,std::string featName,std::string kernName, double dParam);

	enum FeatureType
	{
		kFeatureTypeHaar,
		kFeatureTypeRaw,
		kFeatureTypeHistogram
	};

	enum KernelType
	{
		kKernelTypeLinear,
		kKernelTypeGaussian,
		kKernelTypeIntersection,
		kKernelTypeChi2
	};

	struct FeatureKernelPair
	{
		FeatureType feature;
		KernelType kernel;
		std::vector<double> params;
	};
	
	bool							quietMode;
	bool							debugMode;
	
	std::string						sequenceBasePath;
	std::string						sequenceName;
	std::string						resultsPath;
	std::string						SelectRatioPath;
	std::string						timePath;

	int								frameWidth;
	int								frameHeight;
	int								channel;
	
	int								seed;
	int								searchRadius;
	double							svmC;
	int								svmBudgetSize;
	std::vector<FeatureKernelPair>	features;
	
	friend std::ostream& operator<< (std::ostream& out, const Config& conf);

	int sz_backblock_x_step;
	int sz_backblock_y_step;
	int sz_backblock_x;
	int sz_backblock_y;
	int nNum_backgrounds;
	int nNum_backgrounds_x;
	int nNum_backgrounds_y;
	
private:
	void SetBackgroudNumbers();

	void SetDefaults();
	static std::string FeatureName(FeatureType f);
	static std::string KernelName(KernelType k);
};

#endif