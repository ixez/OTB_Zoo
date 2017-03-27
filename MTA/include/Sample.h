#ifndef SAMPLE_H
#define SAMPLE_H

#include "ImageRep.h"
#include "Rect.h"

#include <vector>

class Sample
{
public:
	Sample(const ImageRep& image, const FloatRect& roi) :
	  m_image(image),
		  m_roi(roi)
	  {
	  }

	  inline const ImageRep& GetImage() const { return m_image; }
	  inline const FloatRect& GetROI() const { return m_roi; }

private:
	const ImageRep& m_image;
	FloatRect m_roi;
};

class MultiSample
{
public:
	MultiSample(const ImageRep& image, const std::vector<FloatRect>& rects) :
	  m_image(image),
		  m_rects(rects)
	  {
	  }

	  inline const ImageRep& GetImage() const { return m_image; }
	  inline const std::vector<FloatRect>& GetRects() const { return m_rects; }
	  inline Sample GetSample(int i) const { return Sample(m_image, m_rects[i]); }

private:
	const ImageRep& m_image;
	std::vector<FloatRect> m_rects;
};

#endif