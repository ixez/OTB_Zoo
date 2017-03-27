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

#include "Sampler.h"

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
	#define M_PI       3.14159265358979323846
#endif

using namespace std;

vector<FloatRect> Sampler::RadialSamples(FloatRect centre, int radius, int nr, int nt)
{
	vector<FloatRect> samples;
	
	FloatRect s(centre);
	float rstep = (float)radius/nr;
	float tstep = 2*(float)M_PI/nt;
	samples.push_back(centre);
	
	for (int ir = 1; ir <= nr; ++ir)
	{
		float phase = (ir % 2)*tstep/2;
		for (int it = 0; it < nt; ++it)
		{
			float dx = ir*rstep*cosf(it*tstep+phase);
			float dy = ir*rstep*sinf(it*tstep+phase);
			s.SetXMin(centre.XMin()+dx);
			s.SetYMin(centre.YMin()+dy);
			samples.push_back(s);
		}
	}
	
	return samples;
}

vector<FloatRect> Sampler::PixelSamples(FloatRect centre, int radius, int SampleRatio)
{
	vector<FloatRect> samples;
	
	IntRect s(centre);
	samples.push_back(s);
	
	int r2 = radius*radius;
	for (int iy = -radius; iy <= radius; ++iy)
	{
		for (int ix = -radius; ix <= radius; ++ix)
		{
			if (ix*ix+iy*iy > r2) continue;
			if (iy == 0 && ix == 0) continue; // already put this one at the start
			
			int x = (int)centre.XMin() + ix;
			int y = (int)centre.YMin() + iy;
			if (SampleRatio>1 && (ix % SampleRatio != 0 || iy % SampleRatio != 0)) continue;
			
			s.SetXMin(x);
			s.SetYMin(y);
			samples.push_back(s);
		}
	}
	
	return samples;
}

vector<FloatRect> Sampler::BackgroundSamples(Config conf, int nNum_backgrounds_x, int nNum_backgrounds_y,int nSZ_backgrounds_x, int nSZ_backgrounds_y)
{
	vector<FloatRect> samples;
	IntRect s;
	for (int iy = 0; iy < nNum_backgrounds_y; iy++)
	{
		int y = iy*(nSZ_backgrounds_y);
		if ((int)(y+nSZ_backgrounds_y)>=conf.frameHeight)
			y = (int)(conf.frameHeight-nSZ_backgrounds_y-1);

		for (int ix = 0; ix < nNum_backgrounds_x; ix++)
		{				
			int x = ix*(nSZ_backgrounds_y);

			if ((int)(x+nSZ_backgrounds_x)>=conf.frameWidth)
				x=(int)(conf.frameWidth-nSZ_backgrounds_x-1);		

			s.SetXMin(x);
			s.SetYMin(y);
			s.SetWidth(nSZ_backgrounds_x);
			s.SetHeight(nSZ_backgrounds_y);
			samples.push_back(s);			
		}				
	}

	return samples;
}

vector<FloatRect> Sampler::BlockSamples(FloatRect centre, Config conf, int radius, int nSZ_backgrounds_x, int nSZ_backgrounds_y)
{
	vector<FloatRect> samples;
	IntRect s;	
	int x_start = max((int)centre.XMin()-radius,0);
	int y_start = max((int)centre.YMin()-radius,0);
	int x_end = min((int)(centre.XMin()+centre.Width()+radius),conf.frameWidth);
	int y_end = min((int)(centre.YMin()+centre.Height()+radius),conf.frameHeight);
	int x_width = x_end-x_start;
	int y_Height = y_end-y_start;

	int x_counter = (int)ceilf((float)x_width/nSZ_backgrounds_x);
	int y_counter = (int)ceilf((float)y_Height/nSZ_backgrounds_y);

	for (int iy = 0; iy < y_counter; iy++)
	{
		int y = y_start+iy*(nSZ_backgrounds_y);
		if ((int)(y+nSZ_backgrounds_y)>=conf.frameHeight)
			y = (int)(conf.frameHeight-nSZ_backgrounds_y-1);

		for (int ix = 0; ix < x_counter; ix++)
		{				
			int x = x_start+ix*(nSZ_backgrounds_y);

			if ((int)(x+nSZ_backgrounds_x)>=conf.frameWidth)
				x=(int)(conf.frameWidth-nSZ_backgrounds_x-1);		

			s.SetXMin(x);
			s.SetYMin(y);
			s.SetWidth(nSZ_backgrounds_x);
			s.SetHeight(nSZ_backgrounds_y);
			samples.push_back(s);			
		}				
	}
	return samples;
}
vector<FloatRect> Sampler::ObjWindowSamples(FloatRect centre, int nSZ_backgrounds_x, int nSZ_backgrounds_y)
{
	vector<FloatRect> samples;
	IntRect s;	
	int x_start = (int)centre.XMin();
	int y_start = (int)centre.YMin();
	int x_end = (int)(centre.XMin()+centre.Width());
	int y_end = (int)(centre.YMin()+centre.Height());
	int x_width = x_end-x_start;
	int y_Height = y_end-y_start;

	int x_counter = (int)ceilf((float)x_width/nSZ_backgrounds_x);
	int y_counter = (int)ceilf((float)y_Height/nSZ_backgrounds_y);

	for (int iy = 0; iy < y_counter; iy++)
	{
		int y = y_start+iy*(nSZ_backgrounds_y);
		if ((int)(y+nSZ_backgrounds_y)>=y_end)
			y = (int)(y_end-nSZ_backgrounds_y-1);

		for (int ix = 0; ix < x_counter; ix++)
		{				
			int x = x_start+ix*(nSZ_backgrounds_y);

			if ((int)(x+nSZ_backgrounds_x)>=x_end)
				x=(int)(x_end-nSZ_backgrounds_x-1);		

			s.SetXMin(x);
			s.SetYMin(y);
			s.SetWidth(nSZ_backgrounds_x);
			s.SetHeight(nSZ_backgrounds_y);
			samples.push_back(s);			
		}				
	}
	return samples;
}
std::vector<FloatRect> Sampler::ObjWindowSamplesOverlap(FloatRect centre, int nSZ_backgrounds_x, int nSZ_backgrounds_y)
{
	vector<FloatRect> samples;
	IntRect s;	
	int x_start = (int)centre.XMin();
	int y_start = (int)centre.YMin();
	int x_end = (int)(centre.XMin()+centre.Width());
	int y_end = (int)(centre.YMin()+centre.Height());
	int x_width = x_end-x_start;
	int y_Height = y_end-y_start;

	int blsize_x =nSZ_backgrounds_x;
	int blsize_y =nSZ_backgrounds_y;

	int x_counter = (int)ceilf((float)x_width/blsize_x);
	int y_counter = (int)ceilf((float)y_Height/blsize_y);

// 	if (x_counter*y_counter<=100)
// 	{
// 		blsize_x =nSZ_backgrounds_x/2;
// 		blsize_y =nSZ_backgrounds_y/2;
// 
// 		x_counter = (int)ceilf((float)x_width/blsize_x);
// 		y_counter = (int)ceilf((float)y_Height/blsize_y);
// 	}

	for (int iy = 0; iy < y_counter; iy++)
	{
		int y = y_start+iy*(blsize_y);
		if ((int)(y+blsize_y)>=y_end)
			continue;

		for (int ix = 0; ix < x_counter; ix++)
		{				
			int x = x_start+ix*(blsize_y);

			if ((int)(x+blsize_x)>=x_end)
				continue;		

			s.SetXMin(x);
			s.SetYMin(y);
			s.SetWidth(blsize_x);
			s.SetHeight(blsize_y);
			samples.push_back(s);			
		}				
	}
	return samples;
}

std::vector<FloatRect> Sampler::ObjWindowAroundSamplesOverlap(FloatRect centre, int nSZ_backgrounds_x, int nSZ_backgrounds_y)
{
	vector<FloatRect> samples;
	IntRect s;	
	int x_start = (int)centre.XMin();
	int y_start = (int)centre.YMin();
	int x_end = (int)(centre.XMin()+centre.Width());
	int y_end = (int)(centre.YMin()+centre.Height());
	int x_width = x_end-x_start;
	int y_Height = y_end-y_start;

	int blsize_x =nSZ_backgrounds_x;
	int blsize_y =nSZ_backgrounds_y;

	int x_counter = (int)ceilf((float)x_width/blsize_x);
	int y_counter = (int)ceilf((float)y_Height/blsize_y);

	// 	if (x_counter*y_counter<=100)
	// 	{
	// 		blsize_x =nSZ_backgrounds_x/2;
	// 		blsize_y =nSZ_backgrounds_y/2;
	// 
	// 		x_counter = (int)ceilf((float)x_width/blsize_x);
	// 		y_counter = (int)ceilf((float)y_Height/blsize_y);
	// 	}

	for (int iy = -1; iy < y_counter+1; iy++)
	{
		int y = y_start+iy*(blsize_y);
		if ((int)(y+blsize_y)>=y_end)
			continue;

		for (int ix = -1; ix < x_counter+1; ix++)
		{				
			int x = x_start+ix*(blsize_y);

			if ((int)(x+blsize_x)>=x_end)
				continue;		

			s.SetXMin(x);
			s.SetYMin(y);
			s.SetWidth(blsize_x);
			s.SetHeight(blsize_y);
			samples.push_back(s);			
		}				
	}
	return samples;
}

vector<FloatRect> Sampler::HaarInputSamples(FloatRect centre, int radius)
{
	vector<FloatRect> samples;

	IntRect s(centre);
	samples.push_back(s);

	int r2 = radius*radius;
	for (int iy = -radius; iy <= radius; ++iy)
	{
		for (int ix = -radius; ix <= radius; ++ix)
		{			
			if (iy == 0 && ix == 0) continue; // already put this one at the start

			int x = (int)centre.XMin() + ix;
			int y = (int)centre.YMin() + iy;
			
			s.SetXMin(x);
			s.SetYMin(y);
			samples.push_back(s);
		}
	}

	return samples;
}

vector<FloatRect> Sampler::AllSamples(Config conf,FloatRect obj)
{
	vector<FloatRect> samples;
	IntRect s;
	for (int iy = 0; iy < conf.frameHeight-obj.Height(); iy++)
	{
		for (int ix = 0; ix < conf.frameWidth-obj.Width(); ix++)
		{				
	
			s.SetXMin(ix);
			s.SetYMin(iy);
			s.SetWidth((int)obj.Width());
			s.SetHeight((int)obj.Height());
			samples.push_back(s);
		}
	}

	return samples;
}

vector<FloatRect> Sampler::LeftBottomSamples(FloatRect centre, int nSZ_block_x, int nSZ_block_y)
{
	vector<FloatRect> samples;

	IntRect s(centre);
	samples.push_back(s);

	int x = (int)centre.XMin();
	int y = (int)centre.YMin();

	s.SetXMin(x+nSZ_block_x/2);
	s.SetYMin(y);
	samples.push_back(s);

	s.SetXMin(x);
	s.SetYMin(y+nSZ_block_y/2);
	samples.push_back(s);

	s.SetXMin(x+nSZ_block_x/2);
	s.SetYMin(y+nSZ_block_y/2);
	samples.push_back(s);

	return samples;
}
std::vector<FloatRect> Sampler::currentBackSamples(FloatRect centre, int nSZ_block_x, int nSZ_block_y)
{
	vector<FloatRect> samples;

	IntRect s(centre);
	samples.push_back(s);

	return samples;
}
vector<FloatRect> Sampler::SaliencySamples(FloatRect centre, int radius_x,int radius_y, int nSZ_backgrounds_x, int nSZ_backgrounds_y,int framewidth, int frameheight)
{
	vector<FloatRect> samples;
	IntRect s;	
	int x_start = max((int)centre.XMin()-radius_x,0);
	int y_start = max((int)centre.YMin()-radius_y,0);
	int x_end = min((int)(centre.XMin()+centre.Width()+radius_x),framewidth-1);
	int y_end = min((int)(centre.YMin()+centre.Height()+radius_y),frameheight-1);
	int x_width = x_end-x_start;
	int y_Height = y_end-y_start;

	int x_counter = (int)floorf((float)x_width/nSZ_backgrounds_x);
	int y_counter = (int)floorf((float)y_Height/nSZ_backgrounds_y);

	for (int iy = 0; iy < y_counter; iy++)
	{
		int y = y_start+iy*(nSZ_backgrounds_y);

		for (int ix = 0; ix < x_counter; ix++)
		{				
			int x = x_start+ix*(nSZ_backgrounds_y);

			s.SetXMin(x);
			s.SetYMin(y);
			s.SetWidth(nSZ_backgrounds_x);
			s.SetHeight(nSZ_backgrounds_y);
			samples.push_back(s);			
		}				
	}
	return samples;
}