#ifndef __OPENCV_STD_H__
#define __OPENCV_STD_H__

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
//#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#else
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#endif

// platform base
#include "amcomdef.h"
#include "merror.h"
#include "ammem.h"
#pragma comment(lib, "mpbase.lib")





#endif	//__OPENCV_STD_H__