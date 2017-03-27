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

#ifndef SAMPLER_H
#define SAMPLER_H

#include "Rect.h"
#include "Config.h"
#include <vector>


class Sampler
{
public:	
	static std::vector<FloatRect> RadialSamples(FloatRect centre, int radius, int nr, int nt);
	static std::vector<FloatRect> PixelSamples(FloatRect centre, int radius, int SampleRatio=1);

	static std::vector<FloatRect> BackgroundSamples(Config conf);
	static std::vector<FloatRect> BackgroundSamples(Config conf, FloatRect bb, int nNum_backgrounds_x, int nNum_backgrounds_y);
	static std::vector<FloatRect> BackgroundSamples(Config conf, int nNum_backgrounds_x, int nNum_backgrounds_y,int nSZ_backgrounds_x, int nSZ_backgrounds_y);
	static std::vector<FloatRect> AllSamples(Config conf,FloatRect obj);
	static std::vector<FloatRect> LeftBottomSamples(FloatRect centre, int nSZ_block_x, int nSZ_block_y);
	static std::vector<FloatRect> BlockSamples(FloatRect centre, Config conf, int radius, int nSZ_backgrounds_x, int nSZ_backgrounds_y);
	static std::vector<FloatRect> HaarInputSamples(FloatRect centre, int radius);
	static std::vector<FloatRect> ObjWindowSamples(FloatRect centre, int nSZ_backgrounds_x, int nSZ_backgrounds_y);

	static std::vector<FloatRect> SaliencySamples(FloatRect centre, int radius_x,int radius_y, int nSZ_backgrounds_x, int nSZ_backgrounds_y,int framewidth, int frameheight);
	static std::vector<FloatRect> currentBackSamples(FloatRect centre, int nSZ_block_x, int nSZ_block_y);
	static std::vector<FloatRect> ObjWindowSamplesOverlap(FloatRect centre, int nSZ_backgrounds_x, int nSZ_backgrounds_y);
	static std::vector<FloatRect> ObjWindowAroundSamplesOverlap(FloatRect centre, int nSZ_backgrounds_x, int nSZ_backgrounds_y);
};

#endif
