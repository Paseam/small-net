#include "afq_caffe_cnn.h"
#include "math.h"
#include "stdlib.h"
#include "stdio.h"
#include "float.h"


#undef AFQ_MIN
#define AFQ_MIN(a,b)  ((a) > (b) ? (b) : (a))

#undef AFQ_MAX
#define AFQ_MAX(a,b)  ((a) < (b) ? (b) : (a))

#undef ASSERT_MEMORY
#define ASSERT_MEMORY(ptr) if (!(ptr))	return MERR_NO_MEMORY;

#undef ASSERT_EQ
#define ASSERT_EQ(x, y) if (x != y) return MERR_BAD_STATE;

//#define SHOWRESULT
#ifdef SHOWRESULT
#include "cv.h"
#include "highgui.h"
#pragma comment(lib,"cv.lib")
#pragma comment(lib,"highgui.lib")
#endif

//#define __LOGTIMEWIN32__
#ifdef __LOGTIMEWIN32__
#include "Windows.h"
#endif

// [row1*column1]*[column1*column2], float
static int afq_mcvMatrixMulMatrixRowMajor_f32(float *matrixOut,float *matrix1,float *matrix2,int row1,int column1,int column2)
{
    int i,j,k;
    float *p1,*p2,*p3,m1;
	float sum;

    if(matrixOut == 0 || matrix1 == 0 || matrix2 == 0)
    {
        return 1;
    }
	
    k = row1*column2;
    for(i = 0; i < k; i++)
    {
        matrixOut[i] = 0;
    }
	
#ifdef MOBILECV_OPT
	if (1 == row1)
	{
		for (i = 0; i < column1; i++)
		{
			p1 = matrixOut;
			m1 = matrix1[i];
			for (j = 0; j < column2; j++)
			{
				(*p1++) += m1 * (*matrix2++);
			}
		}
	}
	else if (1 == column2)
	{
		for(i = 0; i < row1; i++)
		{
			p1 = matrix2;
			sum = 0;
			for(k = 0; k < column1; k++)
			{
				sum += (*matrix1++)* (*p1++);
			} 
			(*matrixOut++) = sum;
		}
	}
	else
	{
		for(i = 0; i < row1; i++, matrix1 += column1,matrixOut += column2)
		{
			p1 = matrix2;
			for(k = 0; k < column1; k++,p1 += column2)
			{
				p2 = p1;
				p3 = matrixOut;
				m1 = matrix1[k];
				for(j = 0; j < column2; j++)
				{
					(*p3++) += m1 * (*p2++);
				}
			} 
		}
	}
#endif
	
#ifdef NOMAL_MUL
	if (1 == row1)
	{
		for (i = 0; i < column2; i++)
		{
			matrixOut[i] = 0;
			for (j = 0; j < column1; j++)
			{
				matrixOut[i] += matrix1[j] * matrix2[j*column2+i];
			}
		}
	}
	else if (1 == column2)
	{
		for (i = 0; i < row1; i++)
		{
			matrixOut[i] = 0;
			for (j = 0; j < column1; j++)
			{
				matrixOut[i] += matrix1[i*column1+j] * matrix2[j];
			}
		}
	}
	else
	{
		for (i = 0; i < row1; i++)
		{
			for (j = 0; j < column2; j++)
			{
				matrixOut[i*column2+j] = 0;
				for (k = 0; k < column1; k++)
				{
					matrixOut[i*column2+j] += matrix1[i*column1+k] * matrix2[k*column2+j];
				}
			}
		}
	}
#endif
	
    return 0;
}

static int afq_mcvMatrixMulMatrixRowMajor_f32_bin(float *matrixOut,float *matrix1,float *matrix2,int row1,int column1,int column2)
{
    int i,j,k;
    float *p1,*p2,*p3,m1;
	float sum;

    if(matrixOut == 0 || matrix1 == 0 || matrix2 == 0)
    {
        return 1;
    }
	
    k = row1*column2;
    for(i = 0; i < k; i++)
    {
        matrixOut[i] = 0;
    }
	
#ifdef MOBILECV_OPT
	if (1 == row1)
	{
		for (i = 0; i < column1; i++)
		{
			p1 = matrixOut;
			m1 = matrix1[i];
			for (j = 0; j < column2; j++)
			{
				(*p1++) += (m1 != (*matrix2++));
			}
		}
		for (i = 0; i < column2; i++)
		{
			*matrixOut = column1 - 2*(*matrixOut);
			matrixOut++;
		}
	}
	else if (1 == column2)
	{
		for(i = 0; i < row1; i++)
		{
			p1 = matrix2;
			sum = 0;
			for(k = 0; k < column1; k++)
			{
				sum += ((*matrix1++) != (*p1++));
			}
			(*matrixOut++) = column1 - sum*2;
		}
	}
	else
	{
		for(i = 0; i < row1; i++, matrix1 += column1,matrixOut += column2)
		{
			p1 = matrix2;
			for(k = 0; k < column1; k++,p1 += column2)
			{
				p2 = p1;
				p3 = matrixOut;
				m1 = matrix1[k];
				for(j = 0; j < column2; j++)
				{
					(*p3++) += (m1 != (*p2++));
				}
			}
			p3 = matrixOut;
			for (j = 0; j < column2; j++)
			{
				(*p3) = column1 - 2*(*p3);
				p3++;
			}
		}
	}
#endif
	
#ifdef NOMAL_MUL
	if (1 == row1)
	{
		for (i = 0; i < column2; i++)
		{
			matrixOut[i] = 0;
			for (j = 0; j < column1; j++)
			{
				matrixOut[i] += (matrix1[j] != matrix2[j*column2+i]);
			}
			matrixOut[i] = column1 - 2*matrixOut[i];
		}
	}
	else if (1 == column2)
	{
		for (i = 0; i < row1; i++)
		{
			matrixOut[i] = 0;
			for (j = 0; j < column1; j++)
			{
				matrixOut[i] += (matrix1[i*column1+j] != matrix2[j]);
			}
			matrixOut[i] = column1 - 2*matrixOut[i];
		}
	}
	else
	{
		for (i = 0; i < row1; i++)
		{
			for (j = 0; j < column2; j++)
			{
				matrixOut[i*column2+j] = 0;
				for (k = 0; k < column1; k++)
				{
					matrixOut[i*column2+j] += (matrix1[i*column1+k] != matrix2[k*column2+j]);
				}
				matrixOut[i*column2+j] = column1 - 2*matrixOut[i*column2+j];
			}
		}
	}
#endif
	
    return 0;
}

static int afq_mcvMatrixMulMatrixRowMajor_s32(int *matrixOut,int *matrix1,int *matrix2,int row1,int column1,int column2)
{
    int i,j,k;
    int *p1,*p2,*p3,m1;
	int sum;

    if(matrixOut == 0 || matrix1 == 0 || matrix2 == 0)
    {
        return 1;
    }
	
    k = row1*column2;
    for(i = 0; i < k; i++)
    {
        matrixOut[i] = 0;
    }
	
#ifdef MOBILECV_OPT
	if (1 == row1)
	{
		for (i = 0; i < column1; i++)
		{
			p1 = matrixOut;
			m1 = matrix1[i];
			for (j = 0; j < column2; j++)
			{
				(*p1++) += m1 * (*matrix2++);
			}
		}
	}
	else if (1 == column2)
	{
		for(i = 0; i < row1; i++)
		{
			p1 = matrix2;
			sum = 0;
			for(k = 0; k < column1; k++)
			{
				sum += (*matrix1++)* (*p1++);
			} 
			(*matrixOut++) = sum;
		}
	}
	else
	{
		for(i = 0; i < row1; i++, matrix1 += column1,matrixOut += column2)
		{
			p1 = matrix2;
			for(k = 0; k < column1; k++,p1 += column2)
			{
				p2 = p1;
				p3 = matrixOut;
				m1 = matrix1[k];
				for(j = 0; j < column2; j++)
				{
					(*p3++) += m1 * (*p2++);
				}
			} 
		}
	}
#endif
	
#ifdef NOMAL_MUL
	if (1 == row1)
	{
		for (i = 0; i < column2; i++)
		{
			matrixOut[i] = 0;
			for (j = 0; j < column1; j++)
			{
				matrixOut[i] += matrix1[j] * matrix2[j*column2+i];
			}
		}
	}
	else if (1 == column2)
	{
		for (i = 0; i < row1; i++)
		{
			matrixOut[i] = 0;
			for (j = 0; j < column1; j++)
			{
				matrixOut[i] += matrix1[i*column1+j] * matrix2[j];
			}
		}
	}
	else
	{
		for (i = 0; i < row1; i++)
		{
			for (j = 0; j < column2; j++)
			{
				matrixOut[i*column2+j] = 0;
				for (k = 0; k < column1; k++)
				{
					matrixOut[i*column2+j] += matrix1[i*column1+k] * matrix2[k*column2+j];
				}
			}
		}
	}
#endif
	
    return 0;
}

static int afq_mcvMatrixMulMatrixRowMajor_s32_bin(int *matrixOut,int *matrix1,int *matrix2,int row1,int column1,int column2)
{
    int i,j,k;
    int *p1,*p2,*p3,m1;
	int sum;

    if(matrixOut == 0 || matrix1 == 0 || matrix2 == 0)
    {
        return 1;
    }
	
    k = row1*column2;
    for(i = 0; i < k; i++)
    {
        matrixOut[i] = 0;
    }
	
#ifdef MOBILECV_OPT
	if (1 == row1)
	{
		for (i = 0; i < column1; i++)
		{
			p1 = matrixOut;
			m1 = matrix1[i];
			for (j = 0; j < column2; j++)
			{
				(*p1++) += (m1 != (*matrix2++));
			}
		}
		for (i = 0; i < column2; i++)
		{
			*matrixOut = column1 - 2*(*matrixOut);
			matrixOut++;
		}
	}
	else if (1 == column2)
	{
		for(i = 0; i < row1; i++)
		{
			p1 = matrix2;
			sum = 0;
			for(k = 0; k < column1; k++)
			{
				sum += ((*matrix1++) != (*p1++));
			}
			(*matrixOut++) = column1 - sum*2;
		}
	}
	else
	{
		for(i = 0; i < row1; i++, matrix1 += column1,matrixOut += column2)
		{
			p1 = matrix2;
			for(k = 0; k < column1; k++,p1 += column2)
			{
				p2 = p1;
				p3 = matrixOut;
				m1 = matrix1[k];
				for(j = 0; j < column2; j++)
				{
					(*p3++) += (m1 != (*p2++));
				}
			}
			p3 = matrixOut;
			for (j = 0; j < column2; j++)
			{
				(*p3) = column1 - 2*(*p3);
				p3++;
			}
		}
	}
#endif
	
#ifdef NOMAL_MUL
	if (1 == row1)
	{
		for (i = 0; i < column2; i++)
		{
			matrixOut[i] = 0;
			for (j = 0; j < column1; j++)
			{
				matrixOut[i] += (matrix1[j] != matrix2[j*column2+i]);
			}
			matrixOut[i] = column1 - 2*matrixOut[i];
		}
	}
	else if (1 == column2)
	{
		for (i = 0; i < row1; i++)
		{
			matrixOut[i] = 0;
			for (j = 0; j < column1; j++)
			{
				matrixOut[i] += (matrix1[i*column1+j] != matrix2[j]);
			}
			matrixOut[i] = column1 - 2*matrixOut[i];
		}
	}
	else
	{
		for (i = 0; i < row1; i++)
		{
			for (j = 0; j < column2; j++)
			{
				matrixOut[i*column2+j] = 0;
				for (k = 0; k < column1; k++)
				{
					matrixOut[i*column2+j] += (matrix1[i*column1+k] != matrix2[k*column2+j]);
				}
				matrixOut[i*column2+j] = column1 - 2*matrixOut[i*column2+j];
			}
		}
	}
#endif
	
    return 0;
}

static int caffecnn_forward_layer_data(CAFFECNN_LAYER_DATA * datalayer)
{
	DATATYPE * inputdata = datalayer->data_ia_ptr;
	DATATYPE * outputdata = datalayer->data_oa;
	DATATYPE scale = datalayer->scale;
	int num = datalayer->num;
	int i;

// 	{
// 		FILE* fd_testdata = fopen("test_1.bin", "rb");
// 		fread(datalayer->data_ia_ptr, sizeof(DATATYPE), num, fd_testdata);
// 		fclose(fd_testdata);
// 	}

	for (i = 0; i < num; i++)
	{
		(*outputdata++) = 2*(*inputdata++)*scale - 1;
	}
	
// 	{
// 		FILE* fd_testdata = fopen("data.txt", "r");
// 		for (i = 0; i < num; i++)
// 		{
// 			fscanf(fd_testdata, "%f", &datalayer->data_oa[i]);
// 		}
// 		fclose(fd_testdata);
// 	}
// 	for (i = 0; i < datalayer->height; i++)
// 	{
// 		int j;
// 		for (j = 0; j < datalayer->width; j++)
// 		{
// 			printf("%f ", datalayer->data_oa[i*datalayer->width+j]);
// 		}
// 		printf("\n");
// 	}
	return 0;
}

static void im2col(const DATATYPE* data_im, const int channels, const int height, const int width, 
				   const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, 
				   const int stride_h, const int stride_w, DATATYPE* data_col)
{
	int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int channels_col = channels * kernel_h * kernel_w;
	int c_col, h_col, w_col;

	for (c_col = 0; c_col < channels_col; ++c_col)
	{
		int w_offset = c_col % kernel_w;
		int h_offset = (c_col / kernel_w) % kernel_h;
		int c_im = c_col / kernel_h / kernel_w;
		for (h_col = 0; h_col < height_col; ++h_col) 
		{
			for (w_col = 0; w_col < width_col; ++w_col) 
			{
				int h_im = h_col * stride_h - pad_h + h_offset;
				int w_im = w_col * stride_w - pad_w + w_offset;
				data_col[(c_col * height_col + h_col) * width_col + w_col] = 
					(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? 
					data_im[(c_im * height + h_im) * width+ w_im] : 0;
			}
		}
	}
}

static int caffecnn_forward_layer_conv(CAFFECNN_LAYER_CONV * convlayer)
{
	int weight_offset = convlayer->M_ * convlayer->K_;
	int col_offset = convlayer->K_ * convlayer->N_;
	int top_offset = convlayer->M_ * convlayer->N_;
	DATATYPE * bias = convlayer->bias;
	DATATYPE * weight = convlayer->weight;
	DATATYPE * col_data = convlayer->col_data;
	DATATYPE * inputdata = convlayer->data_ia_ptr;
	DATATYPE * outputdata = convlayer->data_oa;
	int M_ = convlayer->M_;
	int N_ = convlayer->N_;
	int K_ = convlayer->K_;
	int input_channel = convlayer->input_channel;
	int output_channel = convlayer->output_channel;
	int input_height = convlayer->input_height;
	int input_width = convlayer->input_width;
	int kernel_h = convlayer->kernel_h;
	int kernel_w = convlayer->kernel_w;
	int pad_h = convlayer->pad_h;
	int pad_w = convlayer->pad_w;
	int stride_h = convlayer->stride_h;
	int stride_w = convlayer->stride_w;
	int ii, jj, kk, g;

	if (1==kernel_h && 1==kernel_w && 0==pad_h && 0==pad_w && 1==stride_h && 1==stride_w)
	{
		col_data = inputdata;
	}
	else
	{
		im2col(inputdata, input_channel, input_height, input_width, 
			kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, 
			col_data);
	}

	memset(outputdata, 0, sizeof(DATATYPE)*top_offset);
	
	if (convlayer->flag_after_data)
	{
		afq_mcvMatrixMulMatrixRowMajor_f32(outputdata, weight, col_data, M_, K_, N_);
	}
	else
	{
#ifndef TEST_0_AND_1
		afq_mcvMatrixMulMatrixRowMajor_f32_bin(outputdata, weight, col_data, M_, K_, N_);
#else//TEST_0_AND_1


#endif//TEST_0_AND_1
	}

	if (convlayer->bias_term)
	{
		for (ii = 0; ii < output_channel; ii++)
		{
			for (jj = 0; jj < N_; jj++)
			{
				outputdata[ii*N_+jj] += bias[ii];
			}
		}
	}

	return 0;
}

static int caffecnn_forward_layer_pooling(CAFFECNN_LAYER_POOLING * poolinglayer)
{
	DATATYPE * inputdata = poolinglayer->data_ia_ptr;
	DATATYPE * outputdata = poolinglayer->data_oa;
	int channel = poolinglayer->channel;
	int input_height = poolinglayer->input_height;
	int input_width = poolinglayer->input_width;
	int output_height = poolinglayer->output_height;
	int output_width = poolinglayer->output_width;
	int kernel_h = poolinglayer->kernel_h;
	int kernel_w = poolinglayer->kernel_w;
	int stride_h = poolinglayer->stride_h;
	int stride_w = poolinglayer->stride_w;
	int pad_h = poolinglayer->pad_h;
	int pad_w = poolinglayer->pad_w;
	int c, ph, pw;
	int h, w;

	switch (poolinglayer->pooling_type)
	{
	case CAFFECNN_POOLING_CODE_MAX:
		{
			for (c = 0; c < channel; ++c)
			{
				for (ph = 0; ph < output_height; ++ph)
				{
					for (pw = 0; pw < output_width; ++pw)
					{
						DATATYPE maxval = -FLT_MAX;

						int pool_index = ph * output_width + pw;

						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						
						int hend = AFQ_MIN(hstart + kernel_h, input_height);
						int wend = AFQ_MIN(wstart + kernel_w, input_width);
						
						hstart = AFQ_MAX(hstart, 0);
						wstart = AFQ_MAX(wstart, 0);

						for (h = hstart; h < hend; ++h)
						{
							for (w = wstart; w < wend; ++w)
							{
								int index = h * input_width + w;
								maxval = (inputdata[index] > maxval) ? inputdata[index] : maxval;
							}
						}
						outputdata[pool_index] = maxval;
					}
				}
				inputdata += input_height*input_width;
				outputdata += output_height*output_width;
			}
		}
		break;

	case CAFFECNN_POOLING_CODE_AVE:
		{
			for (c = 0; c < channel; ++c)
			{
				for (ph = 0; ph < output_height; ++ph)
				{
					for (pw = 0; pw < output_width; ++pw)
					{
						DATATYPE aveval = 0;
						
						int pool_index = ph * output_width + pw;

						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						
						int hend = AFQ_MIN(hstart + kernel_h, input_height + pad_h);
						int wend = AFQ_MIN(wstart + kernel_w, input_width + pad_w);
						
						hstart = AFQ_MAX(hstart, 0);
						wstart = AFQ_MAX(wstart, 0);

						hend = AFQ_MIN(hend, input_height);
						wend = AFQ_MIN(wend, input_width);
						
						for (h = hstart; h < hend; ++h)
						{
							for (w = wstart; w < wend; ++w)
							{
								aveval += inputdata[h * input_width + w];
							}
						}
						outputdata[pool_index] = aveval / ((hend - hstart) * (wend - wstart));
					}
				}
				inputdata += input_height*input_width;
				outputdata += output_height*output_width;
			}
		}
		break;
	}

	return 0;
}

static int caffecnn_forward_layer_fullconnected(CAFFECNN_LAYER_FC * fclayer)
{
	int i, j;
	int input_num = fclayer->input_num;
	int output_num = fclayer->output_num;
	DATATYPE * inputdata = fclayer->data_ia_ptr;
	DATATYPE * outputdata = fclayer->data_oa;
	DATATYPE * weight = fclayer->weight;
	DATATYPE * bias = fclayer->bias;

	// caffe style
// 	afq_mcvMatrixMulMatrixRowMajor_f32(outputdata, weight, inputdata, output_num, input_num, 1);

	// lasagne style
	if (fclayer->flag_after_data)//this case is like caffe style, but row-col order is changed.
	{
		afq_mcvMatrixMulMatrixRowMajor_f32(outputdata, inputdata, weight, 1, input_num, output_num);
	}
	else
	{
#ifndef TEST_0_AND_1
		afq_mcvMatrixMulMatrixRowMajor_f32_bin(outputdata, inputdata, weight, 1, input_num, output_num);
#else//TEST_0_AND_1

#endif//TEST_0_AND_1
	}

	if (fclayer->bias_term)
	{
		for (i = 0; i < output_num; i++)
		{
			outputdata[i] += bias[i];
		}
	}

	return 0;
}

static int caffecnn_forward_layer_relu(CAFFECNN_LAYER_RELU * relulayer)
{
	DATATYPE * inputdata = relulayer->data_ia_ptr;
	DATATYPE * outputdata = relulayer->data_oa;
	int num = relulayer->num;
	int i;

	for (i = 0; i < num; i++)
	{
		outputdata[i] = (inputdata[i] > 0) ? inputdata[i] : 0;
	}

	return 0;
}

static int caffecnn_forward_layer_softmax(CAFFECNN_LAYER_SOFTMAX * softmaxlayer)
{
	DATATYPE * inputdata = softmaxlayer->data_ia_ptr;
	DATATYPE * outputdata = softmaxlayer->data_oa;
	DATATYPE * scale_data = softmaxlayer->scale_data;
	int mapsize = softmaxlayer->height*softmaxlayer->width;
	int i, k;

	for (k = 0; k < mapsize; k++)
	{
		scale_data[k] = inputdata[k];

		for (i = 0; i < softmaxlayer->channel; i++)
		{
			if (inputdata[i*mapsize+k] > scale_data[k])
				scale_data[k] = inputdata[i*mapsize+k];
		}

		for (i = 0; i < softmaxlayer->channel; i++)
		{
			outputdata[i*mapsize+k] = (inputdata[i*mapsize+k] - scale_data[k]);
		}
		
		for (i = 0; i < softmaxlayer->channel; i++)
		{
			outputdata[i*mapsize+k] = (DATATYPE)(exp(outputdata[i*mapsize+k]));
		}
		
		scale_data[k] = 0;
		for (i = 0; i < softmaxlayer->channel; i++)
		{
			scale_data[k] += outputdata[i*mapsize+k];
		}
		for (i = 0; i < softmaxlayer->channel; i++)
		{
			outputdata[i*mapsize+k] /= scale_data[k];
		}

/*		printf("prob:%d:", softmaxlayer->channel);
		for (i = 0; i < softmaxlayer->channel; i++)
		{
			printf("%f, ", outputdata[i*mapsize+k]);
		}
		printf("\n");*/
	}

	return 0;
}

static int caffecnn_forward_layer_category(CAFFECNN_LAYER_CATEGORY * categorylayer)
{
	DATATYPE * inputdata = categorylayer->data_ia_ptr;
	DATATYPE * outputdata = categorylayer->data_oa;
	int mapsize = categorylayer->height*categorylayer->width;
	int i, j;
	
	for (i = 0; i < mapsize; i++)
	{
		DATATYPE max_val = inputdata[i];
		int max_id = 0;
		for (j = 0; j < categorylayer->channel; j++)
		{
			if (inputdata[j*mapsize+i] > max_val)
			{
				max_val = inputdata[j*mapsize+i];
				max_id = j;
			}
		}
		outputdata[i] = max_id;
		categorylayer->data_oa_conf[i] = max_val;
	}
	
	return 0;
}

static int caffecnn_forward_layer_batchnorm(CAFFECNN_LAYER_BATCHNORM * batchnormlayer)
{
	DATATYPE * inputdata = batchnormlayer->data_ia_ptr;
	DATATYPE * outputdata = batchnormlayer->data_oa;
	DATATYPE * meandata = batchnormlayer->mean;
	DATATYPE * vardata = batchnormlayer->variance;
	DATATYPE * scaledata = batchnormlayer->scale;
	DATATYPE * biasdata = batchnormlayer->bias;
	float eps = batchnormlayer->eps;
	int channel = batchnormlayer->channel;
	int height = batchnormlayer->height;
	int width = batchnormlayer->width;
	int num = batchnormlayer->num;
	int num_per_map = height*width;
	int i, k;
	
	for (k = 0; k < channel; k++)
	{
		DATATYPE mean = meandata[k];
		DATATYPE var = vardata[k];
		DATATYPE scale = scaledata[k];
		DATATYPE bias = biasdata[k];
		for (i = 0; i < num_per_map; i++)
		{
			//outputdata[i] = ((inputdata[i] - mean) / sqrt(var + eps)) * scale + bias;
			outputdata[i] = (inputdata[i] - mean) * var * scale + bias;
		}
		outputdata += num_per_map;
		inputdata += num_per_map;
	}
	
	return 0;
}

static int caffecnn_forward_layer_binary(CAFFECNN_LAYER_BINARY * binarylayer)
{
	DATATYPE * inputdata = binarylayer->data_ia_ptr;
	DATATYPE * outputdata = binarylayer->data_oa;
	int num = binarylayer->num;
	int i;

	for (i = 0; i < num; i++)
	{
#ifdef TEST_0_AND_1
		outputdata[i] = (inputdata[i] > 0) ? 1 : 0;
#else
		outputdata[i] = (inputdata[i] > 0) ? 1 : (-1);
#endif
	}
	
	return 0;
}

static int caffecnn_forward_net(CAFFECNN_NET * net)
{
	int callresult = MOK;
	int i;
#ifdef __LOGTIMEWIN32__
	LARGE_INTEGER pCountStart,pCountFinish,pFreq;
	double dfMinus,dfFreq;
#endif
	
	for (i = 0; i < net->layers_num; i++)
	{
		void * templayer = net->layers_ptr[i];
#ifdef __LOGTIMEWIN32__//模块运行前
		QueryPerformanceFrequency(&pFreq);
		QueryPerformanceCounter(&pCountStart);
#endif
		switch (net->layers_code[i])
		{
		case CAFFECNN_LAYER_CODE_DATA:
			{
				CAFFECNN_LAYER_DATA * datalayer = (CAFFECNN_LAYER_DATA*)templayer;
				caffecnn_forward_layer_data(datalayer);
			}
			break;
		case CAFFECNN_LAYER_CODE_CONV:
			{
				CAFFECNN_LAYER_CONV * convlayer = (CAFFECNN_LAYER_CONV*)templayer;
				caffecnn_forward_layer_conv(convlayer);
			}
			break;
		case CAFFECNN_LAYER_CODE_POOLING:
			{
				CAFFECNN_LAYER_POOLING * poolinglayer = (CAFFECNN_LAYER_POOLING*)templayer;
				caffecnn_forward_layer_pooling(poolinglayer);
			}
			break;
		case CAFFECNN_LAYER_CODE_FC:
			{
				CAFFECNN_LAYER_FC * fclayer = (CAFFECNN_LAYER_FC*)templayer;
				caffecnn_forward_layer_fullconnected(fclayer);
			}
			break;
		case CAFFECNN_LAYER_CODE_RELU:
			{
				CAFFECNN_LAYER_RELU * relulayer = (CAFFECNN_LAYER_RELU*)templayer;
				caffecnn_forward_layer_relu(relulayer);
			}
			break;
		case CAFFECNN_LAYER_CODE_SOFTMAX:
			{
				CAFFECNN_LAYER_SOFTMAX * softmaxlayer = (CAFFECNN_LAYER_SOFTMAX*)templayer;
				caffecnn_forward_layer_softmax(softmaxlayer);
			}
			break;
		case CAFFECNN_LAYER_CODE_CATEGORY:
			{
				CAFFECNN_LAYER_CATEGORY * categorylayer = (CAFFECNN_LAYER_CATEGORY*)templayer;
				caffecnn_forward_layer_category(categorylayer);
			}
			break;
		case CAFFECNN_LAYER_CODE_BATCHNORM:
			{
				CAFFECNN_LAYER_BATCHNORM * batchnormlayer = (CAFFECNN_LAYER_BATCHNORM*)templayer;
				caffecnn_forward_layer_batchnorm(batchnormlayer);
			}
			break;
		case CAFFECNN_LAYER_CODE_BINARY:
			{
				CAFFECNN_LAYER_BINARY * binarylayer = (CAFFECNN_LAYER_BINARY*)templayer;
				caffecnn_forward_layer_binary(binarylayer);
			}
			break;
		}
#ifdef __LOGTIMEWIN32__//模块运行完
		QueryPerformanceCounter(&pCountFinish);
		dfFreq = (double)pFreq.QuadPart;
		dfMinus = (double)(pCountFinish.QuadPart - pCountStart.QuadPart);
		printf("%d layer consume: %lf\n", i, dfMinus*1000/dfFreq);
#endif	
	}

	return MOK;
}

int afq_caffecnn_predict_cls(CAFFECNN_NET * net, unsigned char * input_x, int width_x, int height_x, int linebytes_x, int channels, 
							 MInt32 * cls_result, MFloat * cls_conf, MHandle hMemMgr)
{
	int i, j, c;
	int layeridx = 0;
	int clsnum = 0;
	int callresult = MOK;
	
	for (c = 0; c < channels; c++)
	{
		for (i = 0; i < height_x; i++)
		{
			for (j = 0; j < width_x; j++)
			{
				net->inputdata[width_x*height_x*c + width_x*i + j] = input_x[i*linebytes_x + j*channels + c];
			}
		}
	}
	
	callresult = caffecnn_forward_net(net);
	if (MOK == callresult)
	{
		for (layeridx = 0; layeridx < net->layers_num; layeridx++)
		{
			if (CAFFECNN_LAYER_CODE_CATEGORY == net->layers_code[layeridx])
			{
				CAFFECNN_LAYER_CATEGORY * categorylayer = net->layers_ptr[layeridx];
				//for (i = 0; i < categorylayer->num; i++)
				//{
				//	printf("%f,", categorylayer->data_ia_ptr[i]);
				//}
				//printf("\n");
				if (cls_result)
				{
					cls_result[clsnum] = (MInt32)categorylayer->data_oa[0];
				}
				if (cls_conf)
				{
					cls_conf[clsnum] = categorylayer->data_oa_conf[0];
				}
				clsnum++;
			}
		}
	}
	
	return callresult;
}

int afq_caffecnn_load(CAFFECNN_NET ** _net, const float * pModelInfo, const float * pModelData, MHandle hMemMgr)
{
	CAFFECNN_NET * net = 0;
	const float * pCNNModel = pModelData;
	int readptr = 0;
	int i, j, k;
	int outheight = 0, outwidth = 0, outchannel = 0, outnum = 0;
	DATATYPE * outdataptr = 0;
	int demand_memory = 0;
	int read_model_size = 0;

	net = (CAFFECNN_NET*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_NET));
	demand_memory += sizeof(CAFFECNN_NET);
	ASSERT_MEMORY(net);
	MMemSet(net, 0, sizeof(CAFFECNN_NET));
	
	net->layers_num = (int)(pModelInfo[readptr++]);

	net->layers_code = (int*)MMemAlloc(hMemMgr, sizeof(int)*net->layers_num);
	demand_memory += sizeof(int)*net->layers_num;
	ASSERT_MEMORY(net->layers_code);
	MMemSet(net->layers_code, 0, sizeof(int)*net->layers_num);

	net->layers_ptr = (void**)MMemAlloc(hMemMgr, sizeof(void*)*net->layers_num);
	demand_memory += sizeof(void*)*net->layers_num;
	ASSERT_MEMORY(net->layers_ptr);
	MMemSet(net->layers_ptr, 0, sizeof(void*)*net->layers_num);

	for (i = 0; i < net->layers_num; i++)
	{
		net->layers_code[i] = (int)(pModelInfo[readptr++]);

		switch(net->layers_code[i])
		{
		case CAFFECNN_LAYER_CODE_DATA:
			{
				CAFFECNN_LAYER_DATA * datalayer = 0;
				
				datalayer = (CAFFECNN_LAYER_DATA*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_DATA));
				demand_memory += sizeof(CAFFECNN_LAYER_DATA);
				ASSERT_MEMORY(datalayer);
				MMemSet(datalayer, 0, sizeof(CAFFECNN_LAYER_DATA));
				
				datalayer->channel = (int)(pModelInfo[readptr++]);
				datalayer->height = (int)(pModelInfo[readptr++]);
				datalayer->width = (int)(pModelInfo[readptr++]);
				datalayer->num = datalayer->channel*datalayer->height*datalayer->width;
				datalayer->scale = pModelInfo[readptr++];
				
				net->layers_ptr[i] = (void*)datalayer;
				
				net->inputdata = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*(datalayer->height*datalayer->width*datalayer->channel));
				demand_memory += sizeof(DATATYPE)*(datalayer->height*datalayer->width*datalayer->channel);
				ASSERT_MEMORY(net->inputdata);
				MMemSet(net->inputdata, 0, sizeof(DATATYPE)*(datalayer->height*datalayer->width*datalayer->channel));
			}
			break;

		case CAFFECNN_LAYER_CODE_CONV:
			{
				CAFFECNN_LAYER_CONV * templayer = 0;
				
				templayer = (CAFFECNN_LAYER_CONV*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_CONV));
				demand_memory += sizeof(CAFFECNN_LAYER_CONV);
				ASSERT_MEMORY(templayer);
				MMemSet(templayer, 0, sizeof(CAFFECNN_LAYER_CONV));

				templayer->output_channel = (int)(pModelInfo[readptr++]);
				templayer->kernel_h = (int)(pModelInfo[readptr++]);
				templayer->kernel_w = (int)(pModelInfo[readptr++]);
				templayer->pad_h = (int)(pModelInfo[readptr++]);
				templayer->pad_w = (int)(pModelInfo[readptr++]);
				templayer->stride_h = (int)(pModelInfo[readptr++]);
				templayer->stride_w = (int)(pModelInfo[readptr++]);
				templayer->bias_term = (int)(pModelInfo[readptr++]);

				if (net->layers_code[i-1] == CAFFECNN_LAYER_CODE_DATA)
					templayer->flag_after_data = 1;

				net->layers_ptr[i] = (void*)templayer;
			}
			break;

		case CAFFECNN_LAYER_CODE_POOLING:
			{
				CAFFECNN_LAYER_POOLING * templayer = 0;
				
				templayer = (CAFFECNN_LAYER_POOLING*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_POOLING));
				demand_memory += sizeof(CAFFECNN_LAYER_POOLING);
				ASSERT_MEMORY(templayer);
				MMemSet(templayer, 0, sizeof(CAFFECNN_LAYER_POOLING));
				
				templayer->kernel_h = (int)(pModelInfo[readptr++]);
				templayer->kernel_w = (int)(pModelInfo[readptr++]);
				templayer->pad_h = (int)(pModelInfo[readptr++]);
				templayer->pad_w = (int)(pModelInfo[readptr++]);
				templayer->stride_h = (int)(pModelInfo[readptr++]);
				templayer->stride_w = (int)(pModelInfo[readptr++]);
				templayer->pooling_type = (int)(pModelInfo[readptr++]);
				
				net->layers_ptr[i] = (void*)templayer;
			}
			break;

		case CAFFECNN_LAYER_CODE_FC:
			{
				CAFFECNN_LAYER_FC * templayer = 0;
				
				templayer = (CAFFECNN_LAYER_FC*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_FC));
				demand_memory += sizeof(CAFFECNN_LAYER_FC);
				ASSERT_MEMORY(templayer);
				MMemSet(templayer, 0, sizeof(CAFFECNN_LAYER_FC));
				
				templayer->output_num = (int)(pModelInfo[readptr++]);
				templayer->bias_term = (int)(pModelInfo[readptr++]);

				if (net->layers_code[i-1] == CAFFECNN_LAYER_CODE_DATA)
					templayer->flag_after_data = 1;
				
				net->layers_ptr[i] = (void*)templayer;
			}
			break;

		case CAFFECNN_LAYER_CODE_RELU:
			{
				CAFFECNN_LAYER_RELU * templayer = 0;
				
				templayer = (CAFFECNN_LAYER_RELU*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_RELU));
				demand_memory += sizeof(CAFFECNN_LAYER_RELU);
				ASSERT_MEMORY(templayer);
				MMemSet(templayer, 0, sizeof(CAFFECNN_LAYER_RELU));
				
				net->layers_ptr[i] = (void*)templayer;
			}
			break;

		case CAFFECNN_LAYER_CODE_SOFTMAX:
			{
				CAFFECNN_LAYER_SOFTMAX * templayer = 0;
				
				templayer = (CAFFECNN_LAYER_SOFTMAX*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_SOFTMAX));
				demand_memory += sizeof(CAFFECNN_LAYER_SOFTMAX);
				ASSERT_MEMORY(templayer);
				MMemSet(templayer, 0, sizeof(CAFFECNN_LAYER_SOFTMAX));
				
				net->layers_ptr[i] = (void*)templayer;				
			}
			break;

		case CAFFECNN_LAYER_CODE_CATEGORY:
			{
				CAFFECNN_LAYER_CATEGORY * templayer = 0;
				
				templayer = (CAFFECNN_LAYER_CATEGORY*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_CATEGORY));
				demand_memory += sizeof(CAFFECNN_LAYER_CATEGORY);
				ASSERT_MEMORY(templayer);
				MMemSet(templayer, 0, sizeof(CAFFECNN_LAYER_CATEGORY));
				
				net->layers_ptr[i] = (void*)templayer;
			}
			break;

		case CAFFECNN_LAYER_CODE_BATCHNORM:
			{
				CAFFECNN_LAYER_BATCHNORM * templayer = 0;
				
				templayer = (CAFFECNN_LAYER_BATCHNORM*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_BATCHNORM));
				demand_memory += sizeof(CAFFECNN_LAYER_BATCHNORM);
				ASSERT_MEMORY(templayer);
				MMemSet(templayer, 0, sizeof(CAFFECNN_LAYER_BATCHNORM));
				
				//templayer->eps = pModelInfo[readptr++];

				net->layers_ptr[i] = (void*)templayer;
			}
			break;

		case CAFFECNN_LAYER_CODE_BINARY:
			{
				CAFFECNN_LAYER_BINARY * templayer = 0;
				
				templayer = (CAFFECNN_LAYER_BINARY*)MMemAlloc(hMemMgr, sizeof(CAFFECNN_LAYER_BINARY));
				demand_memory += sizeof(CAFFECNN_LAYER_BINARY);
				ASSERT_MEMORY(templayer);
				MMemSet(templayer, 0, sizeof(CAFFECNN_LAYER_BINARY));
				
				net->layers_ptr[i] = (void*)templayer;
			}
			break;
		};
	}


	for (i = 0; i < net->layers_num; i++)
	{
		void * templayer = net->layers_ptr[i];

		switch (net->layers_code[i])
		{
		case CAFFECNN_LAYER_CODE_DATA:
			{
				CAFFECNN_LAYER_DATA * datalayer = (CAFFECNN_LAYER_DATA *)templayer;

				datalayer->data_ia_ptr = net->inputdata;

				datalayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*datalayer->num);
				demand_memory += sizeof(DATATYPE)*datalayer->num;
				ASSERT_MEMORY(datalayer->data_oa);
				MMemSet(datalayer->data_oa, 0, sizeof(DATATYPE)*datalayer->num);
				
				outchannel = datalayer->channel;
				outheight = datalayer->height;
				outwidth = datalayer->width;
				outnum = datalayer->num;
				outdataptr = datalayer->data_oa;
			}
			break;

		case CAFFECNN_LAYER_CODE_CONV:
			{
				CAFFECNN_LAYER_CONV * convlayer = (CAFFECNN_LAYER_CONV *)templayer;

				convlayer->data_ia_ptr = outdataptr;
				
				convlayer->input_channel = outchannel;
				convlayer->input_height = outheight;
				convlayer->input_width = outwidth;
				convlayer->input_num = convlayer->input_channel*convlayer->input_height*convlayer->input_width;
				
				convlayer->output_height = (convlayer->input_height + 2 * convlayer->pad_h - convlayer->kernel_h) / convlayer->stride_h + 1;
				convlayer->output_width = (convlayer->input_width + 2 * convlayer->pad_w - convlayer->kernel_w) / convlayer->stride_w + 1;
				convlayer->output_num = convlayer->output_channel*convlayer->output_height*convlayer->output_width;
				
				convlayer->M_ = convlayer->output_channel;
				convlayer->K_ = convlayer->input_channel * convlayer->kernel_h * convlayer->kernel_w;
				convlayer->N_ = convlayer->output_height * convlayer->output_width;

				convlayer->weight = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*(convlayer->M_*convlayer->K_));
				demand_memory += sizeof(DATATYPE)*(convlayer->M_ * convlayer->K_);
				ASSERT_MEMORY(convlayer->weight);
				MMemSet(convlayer->weight, 0, sizeof(DATATYPE)*(convlayer->M_ * convlayer->K_));
				
				convlayer->bias = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*convlayer->output_channel);
				demand_memory += sizeof(DATATYPE)*convlayer->output_channel;
				ASSERT_MEMORY(convlayer->bias);
				MMemSet(convlayer->bias, 0, sizeof(DATATYPE)*convlayer->output_channel);
				
				convlayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*convlayer->output_num);
				demand_memory += sizeof(DATATYPE)*convlayer->output_num;
				ASSERT_MEMORY(convlayer->data_oa);
				MMemSet(convlayer->data_oa, 0, sizeof(DATATYPE)*convlayer->output_num);
				
				convlayer->col_data = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*(convlayer->input_channel*convlayer->kernel_h*convlayer->kernel_w*convlayer->output_height*convlayer->output_width));
				demand_memory += (sizeof(DATATYPE)*(convlayer->input_channel*convlayer->kernel_h*convlayer->kernel_w*convlayer->output_height*convlayer->output_width));
				ASSERT_MEMORY(convlayer->col_data);
				MMemSet(convlayer->col_data, 0, sizeof(DATATYPE)*(convlayer->input_channel*convlayer->kernel_h*convlayer->kernel_w*convlayer->output_height*convlayer->output_width));
				
				outchannel = convlayer->output_channel;
				outheight = convlayer->output_height;
				outwidth = convlayer->output_width;
				outnum = convlayer->output_num;
				outdataptr = convlayer->data_oa;
			}
			break;

		case CAFFECNN_LAYER_CODE_POOLING:
			{
				CAFFECNN_LAYER_POOLING * poolinglayer = (CAFFECNN_LAYER_POOLING *)templayer;
				
				poolinglayer->data_ia_ptr = outdataptr;

				poolinglayer->channel = outchannel;
				
				poolinglayer->input_height = outheight;
				poolinglayer->input_width = outwidth;
				poolinglayer->input_num = poolinglayer->channel*poolinglayer->input_height*poolinglayer->input_width;
				
				poolinglayer->output_height = ceil(((float)(poolinglayer->input_height + 2 * poolinglayer->pad_h - poolinglayer->kernel_h)) / poolinglayer->stride_h) + 1;
				poolinglayer->output_width = ceil(((float)(poolinglayer->input_width + 2 * poolinglayer->pad_w - poolinglayer->kernel_w)) / poolinglayer->stride_w) + 1;
				
				if (poolinglayer->pad_h || poolinglayer->pad_w)
				{
					// If we have padding, ensure that the last pooling starts strictly
					// inside the image (instead of at the padding); otherwise clip the last.
					if ((poolinglayer->output_height - 1) * poolinglayer->stride_h >= poolinglayer->input_height + poolinglayer->pad_h)
					{
						--poolinglayer->output_height;
					}
					if ((poolinglayer->output_width - 1) * poolinglayer->stride_w >= poolinglayer->input_width + poolinglayer->pad_w)
					{
						--poolinglayer->output_width;
					}
				}
				poolinglayer->output_num = poolinglayer->channel*poolinglayer->output_height*poolinglayer->output_width;
				
				poolinglayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*poolinglayer->output_num);
				demand_memory += sizeof(DATATYPE)*poolinglayer->output_num;
				ASSERT_MEMORY(poolinglayer->data_oa);
				MMemSet(poolinglayer->data_oa, 0, sizeof(DATATYPE)*poolinglayer->output_num);
				
				outheight = poolinglayer->output_height;
				outwidth = poolinglayer->output_width;
				outnum = poolinglayer->output_num;
				outdataptr = poolinglayer->data_oa;
			}
			break;

		case CAFFECNN_LAYER_CODE_FC:
			{
				CAFFECNN_LAYER_FC * fclayer = (CAFFECNN_LAYER_FC *)templayer;
				
				fclayer->data_ia_ptr = outdataptr;
				fclayer->input_num = outnum;
				
				fclayer->weight = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*(fclayer->output_num * fclayer->input_num));
				demand_memory += sizeof(DATATYPE)*(fclayer->output_num * fclayer->input_num);
				ASSERT_MEMORY(fclayer->weight);
				MMemSet(fclayer->weight, 0, sizeof(DATATYPE)*(fclayer->output_num * fclayer->input_num));
				
				fclayer->bias = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*fclayer->output_num);
				demand_memory += sizeof(DATATYPE)*fclayer->output_num;
				ASSERT_MEMORY(fclayer->bias);
				MMemSet(fclayer->bias, 0, sizeof(DATATYPE)*fclayer->output_num);
				
				fclayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*fclayer->output_num);
				demand_memory += sizeof(DATATYPE)*fclayer->output_num;
				ASSERT_MEMORY(fclayer->data_oa);
				MMemSet(fclayer->data_oa, 0, sizeof(DATATYPE)*fclayer->output_num);
				
				outchannel = fclayer->output_num;
				outheight = 1;
				outwidth = 1;
				outnum = fclayer->output_num;
				outdataptr = fclayer->data_oa;
			}
			break;

		case CAFFECNN_LAYER_CODE_RELU:
			{
				CAFFECNN_LAYER_RELU * relulayer = (CAFFECNN_LAYER_RELU *)templayer;
				
				relulayer->data_ia_ptr = outdataptr;
				relulayer->channel = outchannel;
				relulayer->height = outheight;
				relulayer->width = outwidth;
				relulayer->num = outnum;
				
				relulayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*relulayer->num);
				demand_memory += sizeof(DATATYPE)*relulayer->num;
				ASSERT_MEMORY(relulayer->data_oa);
				MMemSet(relulayer->data_oa, 0, sizeof(DATATYPE)*relulayer->num);
				
				outdataptr = relulayer->data_oa;
			}
			break;

		case CAFFECNN_LAYER_CODE_SOFTMAX:
			{
				CAFFECNN_LAYER_SOFTMAX * softmaxlayer = (CAFFECNN_LAYER_SOFTMAX *)templayer;
				
				softmaxlayer->data_ia_ptr = outdataptr;
				softmaxlayer->channel = outchannel;
				softmaxlayer->height = outheight;
				softmaxlayer->width = outwidth;
				softmaxlayer->num = outnum;
				
				softmaxlayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*softmaxlayer->num);
				demand_memory += sizeof(DATATYPE)*softmaxlayer->num;
				ASSERT_MEMORY(softmaxlayer->data_oa);
				MMemSet(softmaxlayer->data_oa, 0, sizeof(DATATYPE)*softmaxlayer->num);
				
				softmaxlayer->scale_data = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*softmaxlayer->height*softmaxlayer->width);
				demand_memory += sizeof(DATATYPE)*softmaxlayer->height*softmaxlayer->width;
				ASSERT_MEMORY(softmaxlayer->scale_data);
				MMemSet(softmaxlayer->scale_data, 0, sizeof(DATATYPE)*softmaxlayer->height*softmaxlayer->width);

				outdataptr = softmaxlayer->data_oa;
			}
			break;

		case CAFFECNN_LAYER_CODE_CATEGORY:
			{
				CAFFECNN_LAYER_CATEGORY * categorylayer = (CAFFECNN_LAYER_CATEGORY *)templayer;
				
				categorylayer->data_ia_ptr = outdataptr;
				categorylayer->channel = outchannel;
				categorylayer->height = outheight;
				categorylayer->width = outwidth;
				categorylayer->num = outnum;
				
				categorylayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*categorylayer->height*categorylayer->width);
				demand_memory += sizeof(DATATYPE)*categorylayer->height*categorylayer->width;
				ASSERT_MEMORY(categorylayer->data_oa);
				MMemSet(categorylayer->data_oa, 0, sizeof(DATATYPE)*categorylayer->height*categorylayer->width);
				
				categorylayer->data_oa_conf = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*categorylayer->height*categorylayer->width);
				demand_memory += sizeof(DATATYPE)*categorylayer->height*categorylayer->width;
				ASSERT_MEMORY(categorylayer->data_oa_conf);
				MMemSet(categorylayer->data_oa_conf, 0, sizeof(DATATYPE)*categorylayer->height*categorylayer->width);
				
				categorylayer->idx_freq = (MInt32*)MMemAlloc(hMemMgr, sizeof(MInt32)*categorylayer->channel);
				demand_memory += sizeof(MInt32)*categorylayer->channel;
				ASSERT_MEMORY(categorylayer->idx_freq);
				MMemSet(categorylayer->idx_freq, 0, sizeof(MInt32)*categorylayer->channel);
				
				outdataptr = categorylayer->data_oa;
			}
			break;

		case CAFFECNN_LAYER_CODE_BATCHNORM:
			{
				CAFFECNN_LAYER_BATCHNORM * batchnormlayer = (CAFFECNN_LAYER_BATCHNORM *)templayer;
				
				batchnormlayer->data_ia_ptr = outdataptr;
				batchnormlayer->channel = outchannel;
				batchnormlayer->height = outheight;
				batchnormlayer->width = outwidth;
				batchnormlayer->num = outnum;
				
				batchnormlayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*batchnormlayer->num);
				demand_memory += sizeof(DATATYPE)*batchnormlayer->num;
				ASSERT_MEMORY(batchnormlayer->data_oa);
				MMemSet(batchnormlayer->data_oa, 0, sizeof(DATATYPE)*batchnormlayer->num);
				
				batchnormlayer->mean = (DATATYPE *)MMemAlloc(hMemMgr, sizeof(DATATYPE)*batchnormlayer->channel);
				demand_memory += sizeof(DATATYPE)*batchnormlayer->channel;
				ASSERT_MEMORY(batchnormlayer->mean);
				MMemSet(batchnormlayer->mean, 0, sizeof(DATATYPE)*batchnormlayer->channel);

				batchnormlayer->variance = (DATATYPE *)MMemAlloc(hMemMgr, sizeof(DATATYPE)*batchnormlayer->channel);
				demand_memory += sizeof(DATATYPE)*batchnormlayer->channel;
				ASSERT_MEMORY(batchnormlayer->variance);
				MMemSet(batchnormlayer->variance, 0, sizeof(DATATYPE)*batchnormlayer->channel);

				batchnormlayer->scale = (DATATYPE *)MMemAlloc(hMemMgr, sizeof(DATATYPE)*batchnormlayer->channel);
				demand_memory += sizeof(DATATYPE)*batchnormlayer->channel;
				ASSERT_MEMORY(batchnormlayer->scale);
				MMemSet(batchnormlayer->scale, 0, sizeof(DATATYPE)*batchnormlayer->channel);
				
				batchnormlayer->bias = (DATATYPE *)MMemAlloc(hMemMgr, sizeof(DATATYPE)*batchnormlayer->channel);
				demand_memory += sizeof(DATATYPE)*batchnormlayer->channel;
				ASSERT_MEMORY(batchnormlayer->bias);
				MMemSet(batchnormlayer->bias, 0, sizeof(DATATYPE)*batchnormlayer->channel);

				outdataptr = batchnormlayer->data_oa;
			}
			break;

		case CAFFECNN_LAYER_CODE_BINARY:
			{
				CAFFECNN_LAYER_BINARY * binarylayer = (CAFFECNN_LAYER_BINARY *)templayer;
				
				binarylayer->data_ia_ptr = outdataptr;
				binarylayer->channel = outchannel;
				binarylayer->height = outheight;
				binarylayer->width = outwidth;
				binarylayer->num = outnum;
				
				binarylayer->data_oa = (DATATYPE*)MMemAlloc(hMemMgr, sizeof(DATATYPE)*binarylayer->num);
				demand_memory += sizeof(DATATYPE)*binarylayer->num;
				ASSERT_MEMORY(binarylayer->data_oa);
				MMemSet(binarylayer->data_oa, 0, sizeof(DATATYPE)*binarylayer->num);
				
				outdataptr = binarylayer->data_oa;
			}
			break;
		}
	}

	{// load model data
		readptr = 0;
		
		for (i = 0; i < net->layers_num; i++)
		{
			if (CAFFECNN_LAYER_CODE_BATCHNORM == net->layers_code[i])
			{
				CAFFECNN_LAYER_BATCHNORM * batchnormlayer = (CAFFECNN_LAYER_BATCHNORM *)net->layers_ptr[i];
				for (j = 0; j < batchnormlayer->channel; j++)
				{
					batchnormlayer->bias[j] = pCNNModel[readptr++];
				}
				for (j = 0; j < batchnormlayer->channel; j++)
				{
					batchnormlayer->scale[j] = pCNNModel[readptr++];
				}
				for (j = 0; j < batchnormlayer->channel; j++)
				{
					batchnormlayer->mean[j] = pCNNModel[readptr++];
				}
				for (j = 0; j < batchnormlayer->channel; j++)
				{
					batchnormlayer->variance[j] = pCNNModel[readptr++];
				}

				read_model_size += batchnormlayer->channel*4;
			}
			else if (CAFFECNN_LAYER_CODE_CONV == net->layers_code[i])
			{
				CAFFECNN_LAYER_CONV * convlayer = (CAFFECNN_LAYER_CONV *)net->layers_ptr[i];
				for (j = 0; j < convlayer->output_channel * convlayer->K_; j++)
				{
					convlayer->weight[j] = pCNNModel[readptr++];
#ifdef TEST_0_AND_1
					if (convlayer->flag_after_data)
#endif
					{
						convlayer->weight[j] = convlayer->weight[j] > 0 ? 1 : -1;
					}
#ifdef TEST_0_AND_1
					else
					{
						convlayer->weight[j] = convlayer->weight[j] > 0 ? 1 : 0;
					}
#endif
				}
				if (convlayer->bias_term)
				{
					for (j = 0; j < convlayer->output_channel; j++)
					{
						convlayer->bias[j] = pCNNModel[readptr++];
					}
				}

				read_model_size += convlayer->output_channel * convlayer->K_ + convlayer->output_channel;
			}
			else if (CAFFECNN_LAYER_CODE_FC == net->layers_code[i])
			{
				CAFFECNN_LAYER_FC * fclayer = (CAFFECNN_LAYER_FC *)net->layers_ptr[i];
				for (j = 0; j < fclayer->output_num * fclayer->input_num; j++)
				{
					fclayer->weight[j] = pCNNModel[readptr++];
#ifdef TEST_0_AND_1
					if (fclayer->flag_after_data)
#endif
					{
						fclayer->weight[j] = fclayer->weight[j] > 0 ? 1 : -1;
					}
#ifdef TEST_0_AND_1
					else
					{
						fclayer->weight[j] = fclayer->weight[j] > 0 ? 1 : 0;
					}
#endif
				}
				if (fclayer->bias_term)
				{
					for (j = 0; j < fclayer->output_num; j++)
					{
						fclayer->bias[j] = pCNNModel[readptr++];
					}
				}

				read_model_size += fclayer->output_num * fclayer->input_num + fclayer->output_num;
			}
		}
	}

	*_net = net;
	//printf("demand_memory: %dK\n", demand_memory/1024);

	return MOK;
}


int afq_caffecnn_release(CAFFECNN_NET ** _net, MHandle hMemMgr)
{
	if (_net && *_net)
	{
		CAFFECNN_NET * net = *_net;
		int i;
		
		for (i = 0; i < net->layers_num; i++)
		{
			void * templayer = net->layers_ptr[i];

			switch (net->layers_code[i])
			{
			case CAFFECNN_LAYER_CODE_DATA:
				{
					CAFFECNN_LAYER_DATA * datalayer = (CAFFECNN_LAYER_DATA *)templayer;
					
					if (datalayer)
					{
						if (datalayer->data_oa)
							MMemFree(hMemMgr, datalayer->data_oa);
						MMemFree(hMemMgr, datalayer);
					}
				}
				break;
				
			case CAFFECNN_LAYER_CODE_CONV:
				{
					CAFFECNN_LAYER_CONV * convlayer = (CAFFECNN_LAYER_CONV *)templayer;

					if (convlayer)
					{
						if (convlayer->data_oa)
							MMemFree(hMemMgr, convlayer->data_oa);
						if (convlayer->col_data)
							MMemFree(hMemMgr, convlayer->col_data);
						if (convlayer->weight)
							MMemFree(hMemMgr, convlayer->weight);
						if (convlayer->bias)
							MMemFree(hMemMgr, convlayer->bias);
						MMemFree(hMemMgr, convlayer);
					}
				}
				break;

			case CAFFECNN_LAYER_CODE_POOLING:
				{
					CAFFECNN_LAYER_POOLING * poolinglayer = (CAFFECNN_LAYER_POOLING *)templayer;
					
					if (poolinglayer)
					{
						if (poolinglayer->data_oa)
							MMemFree(hMemMgr, poolinglayer->data_oa);
						MMemFree(hMemMgr, poolinglayer);
					}
				}
				break;
				
			case CAFFECNN_LAYER_CODE_FC:
				{
					CAFFECNN_LAYER_FC * fclayer = (CAFFECNN_LAYER_FC *)templayer;
					
					if (fclayer)
					{
						if (fclayer->data_oa)
							MMemFree(hMemMgr, fclayer->data_oa);
						if (fclayer->weight)
							MMemFree(hMemMgr, fclayer->weight);
						if (fclayer->bias)
							MMemFree(hMemMgr, fclayer->bias);
						MMemFree(hMemMgr, fclayer);
					}
				}
				break;
				
			case CAFFECNN_LAYER_CODE_RELU:
				{
					CAFFECNN_LAYER_RELU * relulayer = (CAFFECNN_LAYER_RELU *)templayer;
					
					if (relulayer)
					{
						if (relulayer->data_oa)
							MMemFree(hMemMgr, relulayer->data_oa);
						MMemFree(hMemMgr, relulayer);
					}
				}
				break;
				
			case CAFFECNN_LAYER_CODE_SOFTMAX:
				{
					CAFFECNN_LAYER_SOFTMAX * softmaxlayer = (CAFFECNN_LAYER_SOFTMAX *)templayer;
					
					if (softmaxlayer)
					{
						if (softmaxlayer->scale_data)
							MMemFree(hMemMgr, softmaxlayer->scale_data);
						if (softmaxlayer->data_oa)
							MMemFree(hMemMgr, softmaxlayer->data_oa);
						MMemFree(hMemMgr, softmaxlayer);
					}
				}
				break;

			case CAFFECNN_LAYER_CODE_CATEGORY:
				{
					CAFFECNN_LAYER_CATEGORY * categorylayer = (CAFFECNN_LAYER_CATEGORY *)templayer;
					
					if (categorylayer)
					{
						if (categorylayer->data_oa)
							MMemFree(hMemMgr, categorylayer->data_oa);
						if (categorylayer->data_oa_conf)
							MMemFree(hMemMgr, categorylayer->data_oa_conf);
						if (categorylayer->idx_freq)
							MMemFree(hMemMgr, categorylayer->idx_freq);
						MMemFree(hMemMgr, categorylayer);
					}
				}
				break;

			case CAFFECNN_LAYER_CODE_BATCHNORM:
				{
					CAFFECNN_LAYER_BATCHNORM * batchnormlayer = (CAFFECNN_LAYER_BATCHNORM *)templayer;
					
					if (batchnormlayer)
					{
						if (batchnormlayer->data_oa)
							MMemFree(hMemMgr, batchnormlayer->data_oa);
						if (batchnormlayer->mean)
							MMemFree(hMemMgr, batchnormlayer->mean);
						if (batchnormlayer->variance)
							MMemFree(hMemMgr, batchnormlayer->variance);
						if (batchnormlayer->scale)
							MMemFree(hMemMgr, batchnormlayer->scale);
						if (batchnormlayer->bias)
							MMemFree(hMemMgr, batchnormlayer->bias);
						MMemFree(hMemMgr, batchnormlayer);
					}
				}
				break;

			case CAFFECNN_LAYER_CODE_BINARY:
				{
					CAFFECNN_LAYER_BINARY * binarylayer = (CAFFECNN_LAYER_BINARY *)templayer;
					
					if (binarylayer)
					{
						if (binarylayer->data_oa)
							MMemFree(hMemMgr, binarylayer->data_oa);
						MMemFree(hMemMgr, binarylayer);
					}
				}
				break;
			}
		}

		if (net->inputdata)
			MMemFree(hMemMgr, net->inputdata);
		if (net->layers_code)
			MMemFree(hMemMgr, net->layers_code);
		if (net->layers_ptr)
			MMemFree(hMemMgr, net->layers_ptr);

		MMemFree(hMemMgr, *_net);
		*_net = 0;
	}
	
	return 0;
}

