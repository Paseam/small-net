#ifndef __AFQ_CNN_CAFFE_H__
#define __AFQ_CNN_CAFFE_H__


#include "amcomdef.h"
#include "ammem.h"
#include "merror.h"


typedef float DATATYPE;

#define MOBILECV_OPT
//#define NOMAL_MUL

#define TEST_0_AND_1
//#define SQUEEZE_CHAR         //control how many bites to squeeze 

#ifdef TEST_0_AND_1
#ifdef SQUEEZE_CHAR
typedef unsigned char  SQZTYPE;
#else
typedef unsigned int SQZTYPE;
#endif

#endif

typedef enum CAFFECNN_POOLING_TYPE
{
	CAFFECNN_POOLING_CODE_AVE	= 0x0,
	CAFFECNN_POOLING_CODE_MAX	= 0x1
} CAFFECNN_POOLING_TYPE;

typedef enum CAFFECNN_LAYER_CODE
{
	CAFFECNN_LAYER_CODE_DATA			= 0x0,
	CAFFECNN_LAYER_CODE_CONV			= 0x1,
	CAFFECNN_LAYER_CODE_POOLING			= 0x2,
	CAFFECNN_LAYER_CODE_FC				= 0x3,
	CAFFECNN_LAYER_CODE_RELU			= 0x4,
	CAFFECNN_LAYER_CODE_SOFTMAX			= 0x5,
	CAFFECNN_LAYER_CODE_CATEGORY		= 0x6,
	CAFFECNN_LAYER_CODE_BATCHNORM		= 0x7,
	CAFFECNN_LAYER_CODE_BINARY			= 0x8
} CAFFECNN_LAYER_CODE;

typedef struct CAFFECNN_LAYER_DATA
{
	int channel;
	int height;
	int width;
	int num;

	float scale;
	
	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;
} CAFFECNN_LAYER_DATA;

typedef struct CAFFECNN_LAYER_CONV
{
	int kernel_h;
	int kernel_w;
	
	int stride_h;
	int stride_w;

	int pad_h;
	int pad_w;

	int input_channel;
	int input_height;
	int input_width;
	int input_num;

	int output_channel;
	int output_height;
	int output_width;
	int output_num;

	int K_;
	int M_;
	int N_;

	int bias_term;

	DATATYPE * weight;
	DATATYPE * bias;

	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;

	DATATYPE * col_data;

	int flag_after_data;
#ifdef TEST_0_AND_1
	SQZTYPE *data_ia_bin_ptr;
	SQZTYPE *weight_bin;
#endif
} CAFFECNN_LAYER_CONV;

typedef struct CAFFECNN_LAYER_FC
{
	int output_num;
	int input_num;
	
	int bias_term;

	DATATYPE * weight;
	DATATYPE * bias;

	int * weight_b;
	int * bias_b;

	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;

	int * data_ia_ptr_b;
	int * data_oa_b;

	int flag_after_data;
#ifdef TEST_0_AND_1
	SQZTYPE *data_ia_bin_ptr;//binary input
	SQZTYPE *weight_bin;//binary weight
#endif
} CAFFECNN_LAYER_FC;

typedef struct CAFFECNN_LAYER_POOLING
{
	int channel;

	int input_height;
	int input_width;
	int input_num;

	int output_height;
	int output_width;
	int output_num;

	int kernel_h;
	int kernel_w;

	int pad_h;
	int pad_w;

	int stride_h;
	int stride_w;

    int pooling_type;

	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;

	int * data_ia_ptr_b;
	int * data_oa_b;
} CAFFECNN_LAYER_POOLING;

typedef struct CAFFECNN_LAYER_RELU
{
	int channel;
	int height;
	int width;
	int num;

	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;
} CAFFECNN_LAYER_RELU;

typedef struct CAFFECNN_LAYER_SOFTMAX
{
	int channel;
	int height;
	int width;
	int num;

	DATATYPE * scale_data;

	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;
} CAFFECNN_LAYER_SOFTMAX;

typedef struct CAFFECNN_LAYER_CATEGORY
{
	int channel;
	int height;
	int width;
	int num;
	
	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;
	DATATYPE * data_oa_conf;
	MInt32 * idx_freq;
} CAFFECNN_LAYER_CATEGORY;

typedef struct CAFFECNN_LAYER_BATCHNORM
{
	int channel;
	int height;
	int width;
	int num;

	float eps;

	DATATYPE * mean;
	DATATYPE * variance;
	DATATYPE * scale;
	DATATYPE * bias;

	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;
} CAFFECNN_LAYER_BATCHNORM;

typedef struct CAFFECNN_LAYER_BINARY
{
	int channel;
	int height;
	int width;
	int num;

	DATATYPE * data_ia_ptr;
	DATATYPE * data_oa;

	int * data_ia_ptr_b;
	int * data_oa_b;
} CAFFECNN_LAYER_BINARY;

typedef struct CAFFECNN_NET
{
	void* * layers_ptr;
	int * layers_code;
	int layers_num;
	
	DATATYPE * inputdata;
} CAFFECNN_NET;




#ifdef __cplusplus
extern "C" {
#endif


int afq_caffecnn_load(CAFFECNN_NET ** _net, const float * pModelInfo, const float * pModelData, MHandle hMemMgr);

int afq_caffecnn_release(CAFFECNN_NET ** _net, MHandle hMemMgr);

int afq_caffecnn_predict_cls(CAFFECNN_NET * net, unsigned char * input_x, int width_x, int height_x, int linebytes_x, int channels,
							MInt32 * cls_result, MFloat * cls_conf, MHandle hMemMgr);


#ifdef __cplusplus
}
#endif


#endif//__AFQ_CNN_CAFFE_H__