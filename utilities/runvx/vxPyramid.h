/* 
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#ifndef __VX_PYRAMID_H__
#define __VX_PYRAMID_H__

#include "vxParameter.h"
#include "vxParamHelper.h"
#include "vxUtils.h"

class CVxParamPyramid : public CVxParameter
{
public:
	CVxParamPyramid();
	virtual ~CVxParamPyramid();
	virtual int Initialize(vx_context context, vx_graph graph, const char * desc);
	virtual int InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params);
	virtual int Finalize();
	virtual int ReadFrame(int frameNumber);
	virtual int WriteFrame(int frameNumber);
	virtual int CompareFrame(int frameNumber);
	virtual int Shutdown();

private:
	// vx configuration
	vx_df_image m_format;
	vx_uint32 m_width;
	vx_uint32 m_height;
	vx_size m_numLevels;
	vx_float32 m_scale;
	// vx object
	vx_pyramid m_pyramid;
	// I/O configuration
	float m_comparePixelErrorMin;
	float m_comparePixelErrorMax;
	vx_rectangle_t m_rectCompare;        // rectangle used to save rectangular region used for compare
	vx_rectangle_t * m_rectFullLevel;    // rectangle for full image size for use by access/commit
	vx_rectangle_t * m_rectCompareLevel; // rectangle for image compate at each level
	vx_uint8 * m_bufForCompare;
	bool m_useCheckSumForCompare;
	bool m_generateCheckSumForCompare;
	int m_compareCountMatches;
	int m_compareCountMismatches;
	size_t m_pyramidFrameSize;
	size_t * m_imageFrameSize;
	FILE ** m_fpReadImage;
	FILE ** m_fpWriteImage;
	FILE ** m_fpCompareImage;
};

#endif /* __VX_PYRAMID_H__ */