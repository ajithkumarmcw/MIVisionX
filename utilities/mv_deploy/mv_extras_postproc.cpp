/*
MIT License

Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

/* This file is generated by nnir_to_clib.py on 2019-04-16T15:27:18.098033-07:00 */
#define ENABLE_MVDEPLOY 1
#include "mvdeploy.h"
#include <algorithm>
#include <numeric>
#include "mv_extras_postproc.h"

// sort indexes based on comparing values in v
template <typename T>
void sort_indexes(const std::vector<T> &v, std::vector<size_t> &idx) {
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
}


mv_status MIVID_CALLBACK mivid_add_postprocess_nodes_callback_fn(vx_context context, vx_graph graph, vx_tensor inp_tensor)
{
    return MV_ERROR_NOT_IMPLEMENTED;
}

// do any initializations for post processing
MIVID_API_ENTRY mv_status MIVID_API_CALL mv_postproc_init(mivid_session session, int classes, int blockwd, const float *BB_biases, int bias_size, float conf_thresh, float nms_thresh, int imgw, int imgh)
{
	mivid_handle hdl = (mivid_handle)session;
	hdl->postproc_data = (void *) new PostprocData;
	if (hdl->postproc_data == nullptr) {
		return MV_ERROR_NO_MEMORY;
	}
    PostprocDataPtr pPostproc = (PostprocDataPtr)hdl->postproc_data;
	pBBDetectAttributes bbAttr = &pPostproc->BBData;
	pPostproc->bb_biases = BB_biases;
	bbAttr->classes = classes;
	bbAttr->blockwd = blockwd;
	bbAttr->imgw = imgw;
	bbAttr->imgh = imgh;
	bbAttr->conf_thresh = conf_thresh;
	bbAttr->nms_thresh = nms_thresh;
	pPostproc->region = new CRegion(bbAttr);
	if (pPostproc->region == nullptr) {
		return MV_ERROR_NO_MEMORY;
	}
	return MV_SUCCESS;
}

// shutdown post proc
MIVID_API_ENTRY void MIVID_API_CALL mv_postproc_shutdown(mivid_session session)
{
	mivid_handle hdl = (mivid_handle)session;
	PostprocDataPtr pdata = (PostprocDataPtr) hdl->postproc_data;
	if (pdata->region) delete pdata->region;
	delete pdata;
}

MIVID_API_ENTRY mv_status MIVID_API_CALL mv_postproc_argmax(void *data, void *output, int topK, int n, int c, int h, int w)
{
    for (int b=0; b < n; b++) {
        float *out_data = (float*)data + b*(c*h*w);
        std::vector<float>  prob_vec(out_data, out_data + c);
        std::vector<size_t> idx(prob_vec.size());
        std::iota(idx.begin(), idx.end(), 0);		// index vector from 0 to #classes
        sort_indexes(prob_vec, idx);            // sort indeces based on prob
        int j=0;
        ClassLabel lb;
        for (auto i:idx) {
            lb.index = i;
            lb.probability =  prob_vec[i];
            memcpy(output, &lb, sizeof(lb));
            if (++j >= topK) break;
        }
    }
    return MV_SUCCESS;    
}

MIVID_API_ENTRY mv_status MIVID_API_CALL mv_postproc_getBB_detections(mivid_session session, void *data, int n, int c, int h, int w, std::vector<BBox> &outputBB)
{
	mivid_handle hdl = (mivid_handle)session;
	PostprocDataPtr pdata = (PostprocDataPtr) hdl->postproc_data;
	pBBDetectAttributes pBB = &pdata->BBData; 
	if (!pdata || !pdata->region || !pBB){
		printf("ERROR:: mv_postproc is not initialized\n");
		return MV_FAILURE;
	}
	pdata->region->GetObjectDetections(n, c, h, w, (float *)data, pdata->bb_biases, outputBB);
    return MV_SUCCESS;
}

// reshape transpose
void CRegion::Reshape(float *input, float *output, int numChannels, int n)
{
    int i, j, p;
    float *tmp = output;
    for(i = 0; i < n; ++i)
    {
        for(j = 0, p = i; j < numChannels; ++j, p += n)
        {
            *tmp++ = input[p];
        }
    }
}

// todo:: optimize
float CRegion::Sigmoid(float x)
{
    return 1./(1. + exp(-x));
}

void CRegion::SoftmaxRegion(float *input, int classes, float *output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for(i = 0; i < classes; i++){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < classes; i++){
        float e = exp(input[i] - largest);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < classes; i++){
        output[i] /= sum;
    }
}


inline float rect_overlap(rect &a, rect &b)
{
    float x_overlap = std::max(0.f, (std::min(a.right, b.right) - std::max(a.left, b.left)));
    float y_overlap = std::max(0.f, (std::min(a.bottom, b.bottom) - std::max(a.top, b.top)));
    return (x_overlap * y_overlap);
}

float CRegion::box_iou(BBox a, BBox b)
{
    float box_intersection, box_union;
    rect ra, rb;
    ra = {a.x-a.w/2, a.y-a.h/2, a.x+a.w/2, a.y+a.h/2};
    rb = {b.x-b.w/2, b.y-b.h/2, b.x+b.w/2, b.y+a.h/2};
    box_intersection = rect_overlap(ra, rb);
    box_union = a.w*a.h + b.w*b.h - box_intersection;

    return box_intersection/box_union;
}


int CRegion::argmax(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}



CRegion::CRegion(pBBDetectAttributes pBBAttr)
{
    initialized = false;
    outputSize = 0;
    frameNum = 0;
    if (pBBAttr) {
    	imgw = pBBAttr->imgw;
    	imgh = pBBAttr->imgh;
    	classes = pBBAttr->classes;
    	blockwd = pBBAttr->blockwd;
    	conf_thresh = pBBAttr->conf_thresh;
    	nms_thresh = pBBAttr->nms_thresh;
	}
}

CRegion::~CRegion()
{
    initialized = false;
    if (output)
        delete [] output;
    outputSize = 0;
}


void CRegion::Initialize(int c, int h, int w, int classes)
{
    int size = 4 + classes + 1;     // x,y,w,h,pc, c1...c20

    outputSize = c * h * w;
    totalObjectsPerClass = Nb * h * w;
    output = new float[outputSize];
    boxes.resize(totalObjectsPerClass);
    initialized = true;
}

// Same as doing inference for this layer
int CRegion::GetObjectDetections(int n, int c, int h, int w, float* in_data, const float *biases, std::vector<BBox> &objects)
{
    objects.clear();

    int size = 4 + classes + 1;
    Nb = 5;//biases.size();
    if(!initialized)
    {
        Initialize(c, h, w, classes);
    }

    if(!initialized)
    {
        printf("GetObjectDetections: initialization failed");
        return -1;
    }
    for (int m = 0; m < n; m++) {
        int i,j,k;
        float *input = in_data + outputSize*m;
        
        Reshape(input, output, size*Nb, w*h);        // reshape output
        // Initialize box, scale and probability
        for(i = 0; i < totalObjectsPerClass; ++i)
        {
            int index = i * size;
            //Box
            int n = i % Nb;
            int row = (i/Nb) / w;
            int col = (i/Nb) % w;

            boxes[i].x = (col + Sigmoid(output[index + 0])) / blockwd;      // box x location
            boxes[i].y = (row + Sigmoid(output[index + 1])) / blockwd;      //  box y location
            boxes[i].w = exp(output[index + 2]) * biases[n*2]/ blockwd; //w;
            boxes[i].h = exp(output[index + 3]) * biases[n*2+1] / blockwd; //h;

            //Scale
            output[index + 4] = Sigmoid(output[index + 4]);

            //Class Probability
            SoftmaxRegion(&output[index + 5], classes, &output[index + 5]);

            // remove the ones which has low confidance
            for(j = 0; j < classes; ++j)
            {
                output[index+5+j] *= output[index+4];
                if(output[index+5+j] < conf_thresh) output[index+5+j] = 0;
            }
        }

        //non_max_suppression using box_iou (intersection of union)
        for(k = 0; k < classes; ++k)
        {
            std::vector<float> class_prob_vec(totalObjectsPerClass);
            std::vector<size_t> s_idx(totalObjectsPerClass);
            for(i = 0; i < totalObjectsPerClass; ++i)
            {
                class_prob_vec[i] = output[i*size + k + 5];
                s_idx[i] = i;
            }
            //std::iota(idx.begin(), idx.end(), 0);         // todo::analyse for performance
            sort_indexes(class_prob_vec, s_idx);            // sort indeces based on prob
            for(i = 0; i < totalObjectsPerClass; ++i){
                if(output[s_idx[i] * size + k + 5] == 0) continue;
                BBox a = boxes[s_idx[i]];
                for(j = i+1; j < totalObjectsPerClass; ++j){
                    BBox b = boxes[s_idx[j]];
                    if (box_iou(a, b) > nms_thresh){
                        output[s_idx[j] * size + 5 + k] = 0;
                    }
                }
            }
        }

        // generate objects
        for(i = 0, j = 5; i < totalObjectsPerClass; ++i, j += size)
        {
            int iclass = argmax(&output[j], classes);

            float prob = output[j+iclass];

            if(prob > conf_thresh)
            {
                BBox b = boxes[i];
    #if 0
                // boundingbox to actual coordinates
                printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
                int left  = (b.x-b.w/2.)*imgw;
                int right = (b.x+b.w/2.)*imgw;
                int top   = (b.y-b.h/2.)*imgh;
                int bot   = (b.y+b.h/2.)*imgh;
                if(left < 0) left = 0;
                if(right > imgw-1) right = imgw-1;
                if(top < 0) top = 0;
                if(bot > imgh-1) bot = imgh-1;
    #endif
                BBox obj;
                obj.x = b.x;
                obj.y = b.y;
                obj.w = b.w;
                obj.h = b.h;
                obj.confidence = prob;
                obj.label = iclass;
                obj.imgnum = m;
                //std::cout << "BoundingBox(xywh): "<< i << " for frame: "<< frameNum << " (" << b.x << " " << b.y << " "<< b.w << " "<< b.h << ") " << "confidence: " << prob << " lablel: " << iclass << std::endl;
                objects.push_back(obj);
            }
        }
    }
    frameNum++;

    return 0;
}
