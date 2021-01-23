#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#define SHAPE 10 //要检测的多边形边数shape 检测形状 3为三角形，4矩形，5为五边形……

using namespace cv;
using namespace std;

double angle(CvPoint* pt1, CvPoint* pt2, CvPoint* pt0);
CvSeq* findSquares4(IplImage* img, CvMemStorage* storage, int minarea, int maxarea, int minangle, int maxangle);
void drawSquares(IplImage* img, CvSeq* squares, const char* wndname);

int main()
{

#pragma region 识别圆

	//载入原始图、Mat变量定义   
	Mat srcImage = imread("..//5.jpg");//目录下加载图片
	Mat midImage, dstImage;//定义临时变量和目标图

	//显示原始图
	imshow("【原始图】", srcImage);

	//转为灰度图并进行图像平滑
	cvtColor(srcImage, midImage, COLOR_BGR2GRAY);//转化图为灰度图
	GaussianBlur(midImage, midImage, Size(9, 9), 2, 2);//高斯滤波
	//blur(midImage, midImage, Size(3, 3));//模糊降噪

	//进行霍夫圆变换
	vector<Vec3f> circles;//定义一个矢量结构circles用于存放得到的圆矢量集合，Vec3f表示的是3通道float类型的Vect
	HoughCircles(midImage, circles, HOUGH_GRADIENT, 1.5, 80, 100, 100, 0, 0);//霍夫圆变换

	//依次在图中绘制出圆
	for (size_t i = 0; i < circles.size(); i++)
	{
		//参数定义
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//定义圆心点坐标，cvRound取整
		int radius = cvRound(circles[i][2]);//半径
		//绘制圆心
		circle(srcImage, center, 5, Scalar(255, 0, 0), -1, 8, 0);
		//绘制圆轮廓
		circle(srcImage, center, radius, Scalar(0, 0, 255), 4, 8, 0);
	}
	
	for (int i = 0; i< 10; i++)circles.push_back(i);
	Mat circl  = Mat(circles);
	cout << format(circl,Formatter::FMT_PYTHON) << endl;

	//显示效果图  
	imshow("【效果图】", srcImage);

#pragma endregion

/**********************************************************************************************/

#pragma region 识别多边形
	Mat srcImage1 = imread("..//6.jpg");//读入图像
	IplImage* img0 = &IplImage(srcImage1);//图像类型转换

	CvMemStorage* storage = 0;//创建一个内存存储器，来统一管理各种动态对象的内存
	int c;
	//const char* wndname = "多边形检测"; //窗口名称
	storage = cvCreateMemStorage(0); //创建一个内存存储器，来统一管理各种动态对象的内存
	cvNamedWindow("多边形检测", 1);//窗口
	while (true)
	{
		drawSquares(img0, findSquares4(img0, storage, 1000, 12000, 10, 180), "多边形检测");//画图
		cvClearMemStorage(storage);  //清空存储
		c = cvWaitKey(10);//停顿10
		//if (c == 27)
		//	break;
	}

	//cvReleaseImage(&img0);
	//cvClearMemStorage(storage);
	//cvDestroyWindow(wndname);

#pragma endregion

	//等待按键按下
	waitKey(0);

	return 0;
}

#pragma region 调用函数
//////////////////////////////////////////////////////////////////
//函数功能：用向量来做COSα=两向量之积/两向量模的乘积求两条线段夹角
//输入：   线段3个点坐标pt1,pt2,pt0,最后一个参数为公共点
//输出：   线段夹角，单位为角度
//////////////////////////////////////////////////////////////////
double angle(CvPoint* pt1, CvPoint* pt2, CvPoint* pt0)
{
	double dx1 = pt1->x - pt0->x;
	double dy1 = pt1->y - pt0->y;
	double dx2 = pt2->x - pt0->x;
	double dy2 = pt2->y - pt0->y;
	double angle_line = (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);//余弦值
	return acos(angle_line) * 180 / 3.141592653;
}


//////////////////////////////////////////////////////////////////
//函数功能：采用多边形逼近检测，通过约束条件寻找多边形
//输入：    img 原图像
//          storage 存储
//          minarea，maxarea 检测多边形的最小/最大面积
//          minangle,maxangle 检测多边形边夹角范围，单位为角度  
//输出：   多边形序列
//////////////////////////////////////////////////////////////////
CvSeq* findSquares4(IplImage* img, CvMemStorage* storage, int minarea, int maxarea, int minangle, int maxangle)
{
	CvSeq* contours;//边缘，轮廓容器，类似于C++中的vector
	int N = 6;  //阈值分级
	CvSize sz = cvSize(img->width & -2, img->height & -2);//创建图像sz金字塔处理偶数
	IplImage* timg = cvCloneImage(img);//拷贝一次img
	IplImage* gray = cvCreateImage(sz, 8, 1); //创建img灰度图
	IplImage* pyr = cvCreateImage(cvSize(sz.width / 2, sz.height / 2), 8, 3);  //金字塔滤波3通道图像中间变量
	IplImage* tgray = cvCreateImage(sz, 8, 1);//创建img灰度图
	CvSeq* result;//转轮廓向量
	double s, t;
	CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);//创建轮廓容器序列

	//cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));//截取rio区域
	//金字塔滤波 
	cvPyrDown(timg, pyr, 7);//图像向下采样
	cvPyrUp(pyr, timg, 7);//图像向上采样
	//在3个通道中寻找多边形 
	for (int c = 0; c < 3; c++) //对3个通道分别进行处理 
	{
		cvSetImageCOI(timg, c + 1);//选择感兴趣通道
		cvCopy(timg, tgray, 0);  //依次将BGR通道送入tgray         
		for (int l = 0; l < N; l++)
		{	//把255均分成N个等级
			//不同阈值下二值化
			cvThreshold(tgray, gray, (l + 1) * 255 / N, 255, CV_THRESH_BINARY);

			cvFindContours(gray, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));//轮廓检测
			while (contours)
			{ //多边形逼近             
				result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
				//如果是凸多边形并且面积在范围内
				if (result->total == SHAPE && fabs(cvContourArea(result, CV_WHOLE_SEQ)) > minarea  && fabs(cvContourArea(result, CV_WHOLE_SEQ)) < maxarea &&  cvCheckContourConvexity(result))
				{//result容器中total表示稠密序列的元素个数，或者稀疏序列被分配的节点数。fabs求绝对值。cvContourArea计算轮廓面积， cvCheckContourConvexity函数用于判断轮廓是否为凸(凸返回1，凹返回0
					s = 0;
					//判断每一条边
					for (int i = 0; i < SHAPE + 1; i++)
					{
						if (i >= 2)//边数大于2
						{   //角度            
							t = fabs(angle((CvPoint*)cvGetSeqElem(result, i), (CvPoint*)cvGetSeqElem(result, i - 2), (CvPoint*)cvGetSeqElem(result, i - 1)));
							s = s > t ? s : t;//返回大的，(CvPoint*)cvGetSeqElem(result, i)对应返回直线端点
						}//返回最大的角度
					}
					//这里的S为直角判定条件 单位为角度
					if (s > minangle && s < maxangle)
					for (int i = 0; i < SHAPE; i++)
						cvSeqPush(squares, (CvPoint*)cvGetSeqElem(result, i));// 把找到的轮廓顶点序列放进去
				}
				contours = contours->h_next;//指向下一个
			}
		}
	}
	cvReleaseImage(&gray);
	cvReleaseImage(&pyr);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);
	return squares;
}


//////////////////////////////////////////////////////////////////
//函数功能：画出所有矩形
//输入：   img 原图像
//          squares 多边形序列
//          wndname 窗口名称
//输出：   图像中标记多边形
//////////////////////////////////////////////////////////////////
void drawSquares(IplImage* img, CvSeq* squares, const char* wndname)
{
	CvSeqReader reader;
	IplImage* cpy = cvCloneImage(img);
	CvPoint pt[SHAPE];
	int i;
	cvStartReadSeq(squares, &reader, 0);
	for (i = 0; i < squares->total; i += SHAPE)
	{
		CvPoint* rect = pt;
		int count = SHAPE;
		for (int j = 0; j < count; j++)
		{
			memcpy(pt + j, reader.ptr, squares->elem_size);
			CV_NEXT_SEQ_ELEM(squares->elem_size, reader);
		}
		cvPolyLine(cpy, &rect, &count, 1, 1, CV_RGB(/*rand() & 255, rand() & 255, rand() & 255*/255, 0, 0), 0, CV_AA, 0);//彩色绘制改成了红色
	}
	cvShowImage(wndname, cpy);
	cvReleaseImage(&cpy);
}


#pragma endregion

