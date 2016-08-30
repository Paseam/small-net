#include "afq_caffe_cnn.h"
#include "mobilecv.h"
#include <stdlib.h>
#include <stdio.h>


#define __LOGTIMEWIN32__

#ifdef __LOGTIMEWIN32__
#include "Windows.h"
#define LOGTIMEWIN32_DEFINITION \
	LARGE_INTEGER pCountStart, pCountFinish, pFreq; \
	double dfMinus, dfFreq;
#define LOGTIMEWIN32_BEGIN \
	QueryPerformanceFrequency(&pFreq); \
	QueryPerformanceCounter(&pCountStart);
#define LOGTIMEWIN32_END \
	QueryPerformanceCounter(&pCountFinish); \
	dfFreq = (double)pFreq.QuadPart; \
	dfMinus = (double)(pCountFinish.QuadPart - pCountStart.QuadPart); \
	printf("%f ms\n", dfMinus * 1000 / dfFreq);
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
const float binarynet_model_info_mnist[] =
{
	13,
	CAFFECNN_LAYER_CODE_DATA, 1, 28, 28, 0.00390625,
	CAFFECNN_LAYER_CODE_FC, 4096, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 4096, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 4096, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 10, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_CATEGORY
};


//model configuration of cifar10
const float binarynet_model_info_cifar10[] =
{
	31,

	CAFFECNN_LAYER_CODE_DATA, 3, 32, 32, 0.00390625,

	CAFFECNN_LAYER_CODE_CONV, 128, 3, 3, 1, 1, 1, 1, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_CONV, 128, 3, 3, 1, 1, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_CONV, 256, 3, 3, 1, 1, 1, 1, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_CONV, 256, 3, 3, 1, 1, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_CONV, 512, 3, 3, 1, 1, 1, 1, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_CONV, 512, 3, 3, 1, 1, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_FC, 1024, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 1024, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,
	CAFFECNN_LAYER_CODE_FC, 10, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_CATEGORY
};



//simple model configuration of cifar10
const float binarynet_model_info_cifar10_no_pad[] =
{
	15,

	CAFFECNN_LAYER_CODE_DATA, 3, 32, 32, 0.00390625,

	CAFFECNN_LAYER_CODE_CONV, 32, 3, 3, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_CONV, 32, 4, 4, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_POOLING, 2, 2, 0, 0, 2, 2, CAFFECNN_POOLING_CODE_MAX,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_CONV, 64, 3, 3, 0, 0, 1, 1, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_BINARY,

	CAFFECNN_LAYER_CODE_FC, 10, 1,
	CAFFECNN_LAYER_CODE_BATCHNORM,
	CAFFECNN_LAYER_CODE_CATEGORY
};

//cifar10≤‚ ‘
void main()
{
	//cvNamedWindow("src");
	LOGTIMEWIN32_DEFINITION
	//load the coefficients of model
	FILE* fd_CNNModel = fopen("cifar10/cifar10_no_pad.bin", "rb");
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

	//load model configuration
	CAFFECNN_NET * binnet = 0;
	afq_caffecnn_load(&binnet, binarynet_model_info_cifar10_no_pad, pCNNModel, 0);
	
	
	MInt32 cls_result = 0;
	MFloat cls_conf = 0;
	//open test samples
	FILE* flist = fopen("cifar10/test_batch.bin", "rb");
	int len = getfilesize(flist);
	int label_offset = 1;
	
	int size = 32 * 32 * 3;
	int nNum = len / (size + label_offset);
	unsigned char *label = new unsigned char;
	unsigned char * input_x = new unsigned char[size];
	LOGTIMEWIN32_BEGIN
	for (int i = 0; i < nNum; ++i)
	{
		
		fread(label, 1, label_offset, flist);
		fread(input_x, 1, size, flist);
		
		IplImage * src = cvCreateImage(cvSize(32, 32), 8, 3);
		for (int cc = 0; cc < 3; cc++)
		{
			for (int ii = 0; ii < 32; ii++)
			{
				for (int jj = 0; jj < 32; jj++)
				{
					((uchar*)src->imageData)[ii * 32 * 3 + jj * 3 + cc] = input_x[32 * 32 * cc + 32 * ii + jj];
				}
			}
		}
		int flag = afq_caffecnn_predict_cls(binnet, (uchar*)src->imageData, 32, 32, 32 * 3, 3, &cls_result, &cls_conf, 0);
		printf("label:%u  predict:%d\n", *label, cls_result);
		
		cvReleaseImage(&src);
		
	}
	LOGTIMEWIN32_END

	//release data
	delete[] input_x;
	free(pCNNModel);
	afq_caffecnn_release(&binnet, 0);

	//cvDestroyWindow("src");
}



////mnist≤‚ ‘
//void main()
//{
//    //load model coefficients  for mnist
//	FILE* fd_CNNModel = fopen("mnist/mnist_4.bin", "rb");
//	if (!fd_CNNModel)
//	{
//		printf("fail to open model file\n");
//		system("pause");
//		return;
//	}
//	int lsize = getfilesize(fd_CNNModel);
//	float * pCNNModel = (float*)malloc(lsize);
//	if (!pCNNModel)
//	{
//		printf("no memory to load model file\n");
//		system("pause");
//		return;
//	}
//	fread(pCNNModel, 1, lsize, fd_CNNModel);
//	fclose(fd_CNNModel);
//
//	//load model configuration  for mnist
//	CAFFECNN_NET * binnet = 0;
//	afq_caffecnn_load(&binnet, binarynet_model_info_mnist, pCNNModel, 0);
//
//
//	MInt32 cls_result = 0;
//	MFloat cls_conf = 0;
//	//load test datasets for mnist
//	FILE* flist = fopen("mnist/t10k-images.bin", "rb");
//	int len = getfilesize(flist);
//	int offset = 16;//16 bytes at head for other use
//	fseek(flist, offset, SEEK_SET);
//	int nNum = (len - offset) / (28 * 28);
//	int size = 28 * 28 * 1;
//	unsigned char * input_x = new unsigned char[lsize];
//	for (int i = 0; i < nNum; ++i)
//	{
//		fread(input_x, 1, size, flist);
//		int flag = afq_caffecnn_predict_cls(binnet, input_x, 28, 28, 28, 1, &cls_result, &cls_conf, 0);
//		printf("predict:%d\n",cls_result);
//	}
//	delete[] input_x;
//
//	free(pCNNModel);
//	afq_caffecnn_release(&binnet, 0);
//
//}
