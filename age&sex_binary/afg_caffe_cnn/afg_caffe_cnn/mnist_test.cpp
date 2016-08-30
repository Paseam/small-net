#include "afq_caffe_cnn.h"
#include "mobilecv.h"
#include <stdlib.h>
#include <stdio.h>
#include "opencv_std.h"

#define __LOGTIMEWIN32__

#ifdef __LOGTIMEWIN32__
#include "Windows.h"
#define LOGTIMEWIN32_DEFINITION \
	LARGE_INTEGER pCountStart, pCountFinish, pFreq; \
	double dfMinus, dfFreq,dMean=0;
#define LOGTIMEWIN32_BEGIN \
	QueryPerformanceFrequency(&pFreq); \
	QueryPerformanceCounter(&pCountStart);
#define LOGTIMEWIN32_END \
	QueryPerformanceCounter(&pCountFinish); \
	dfFreq = (double)pFreq.QuadPart; \
	dfMinus = (double)(pCountFinish.QuadPart - pCountStart.QuadPart); \
	printf("%f ms\n", dfMinus * 1000 / dfFreq), dMean = dMean + (dfMinus * 1000 / dfFreq);
#else
#define LOGTIMEWIN32_DEFINITION
#define LOGTIMEWIN32_BEGIN
#define LOGTIMEWIN32_END
#endif





static long getfilesize(FILE*stream)
{
	long curpos, length;
	curpos = ftell(stream);
	fseek(stream, 0L, SEEK_END);
	length = ftell(stream);
	fseek(stream, curpos, SEEK_SET);
	return length;
}


//model coefficients of mnist datasets
const float binarynet_model_info_60X60_age[] =
{
	27,

	CAFFECNN_LAYER_CODE_DATA, 3, 60, 60, 0.00390625,

	CAFFECNN_LAYER_CODE_CONV, 64, 5, 5, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_CONV, 64, 5, 5, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_CONV, 128, 5, 5, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_CONV, 128, 5, 5, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_CONV, 64, 5, 5, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_FC, 1024, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 1024, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 7, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_CATEGORY
};

const float binarynet_model_info_60X60_samllNet_age[] =
{
	19,

	CAFFECNN_LAYER_CODE_DATA, 3, 60, 60, 0.00390625,
	CAFFECNN_LAYER_CODE_CONV, 24, 5, 5, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_CONV, 50, 5, 5, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_CONV, 50, 5, 5, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 300, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 7, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_CATEGORY
};

//mnist≤‚ ‘
void main()
{
    //load model coefficients  for mnist
	LOGTIMEWIN32_DEFINITION
	FILE* fd_CNNModel = fopen("60X60_model/60X60_small_model.bin", "rb");
	if (!fd_CNNModel)
	{
		printf("fail to open model file\n");
		system("pause");
		return;
	}
	int lsize = getfilesize(fd_CNNModel);
	float * pCNNModel = (float*)malloc(lsize);
	if (!pCNNModel)
	{
		printf("no memory to load model file\n");
		system("pause");
		return;
	}
	fread(pCNNModel, 1, lsize, fd_CNNModel);
	fclose(fd_CNNModel);

	//load model configuration  for mnist
	CAFFECNN_NET * binnet = 0;
	afq_caffecnn_load(&binnet, binarynet_model_info_60X60_samllNet_age, pCNNModel, 0);


	MInt32 cls_result = 0;
	MFloat cls_conf = 0;
	//load the file including the all the name of test datasets
	FILE* flist = fopen("file_test_relative_name.txt", "r");
	char* orig_path = (char*)malloc(1024);
	std::string path;
	std::string path_test = "60X60_model/test/";
	std::string strLabel;
	int label = 0;
	int nNum =  8198;
	float err = 0;
	
	for (int i = 0; i < nNum; ++i)
	{
		if (fgets(orig_path, 1024, flist) != NULL)
		{
			path = strtok(orig_path, " ");
			path = path_test + path;
			strLabel = strtok(NULL, "\r");
			strLabel.erase(1, 1);
			label= atoi((strLabel.c_str()));
		}
		//∂¡»ÎÕº∆¨
		IplImage* img = cvLoadImage(path.c_str());
		LOGTIMEWIN32_BEGIN
		int flag = afq_caffecnn_predict_cls(binnet, (unsigned char *)img->imageData, 60, 60, img->widthStep, 3, &cls_result, &cls_conf, 0);
		//printf("image_path: %s   predict:%d  oringnal:%d\n", path.c_str(),cls_result, label);
		LOGTIMEWIN32_END
		printf("predict:%d  oringnal:%d\n", cls_result, label);
		if (cls_result != label)
			err++;
		
		cvReleaseImage(&img);
	}
	//delete[] input_x;
	printf("\n err=%f", err / nNum);
	printf("\n mean time=%f", dMean / nNum);
	free(pCNNModel);
	afq_caffecnn_release(&binnet, 0);

}
