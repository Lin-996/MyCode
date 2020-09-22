#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
using namespace std;
using namespace cv;
int main() {
	
	VideoCapture capture;
	capture.open("炮台素材蓝车前对角-ev--3.MOV");

	while (1) {

		Mat img;
		capture >> img;
		vector<Mat>channels;
		split(img, channels);//分离色彩通道
		//预处理删除己方装甲板颜色
		Mat _grayImg = channels.at(0) - channels.at(2);//Get blue-red image;
		Mat binBrightImg;
		//阈值化
		threshold(_grayImg, binBrightImg, 110
			, 255, cv::THRESH_BINARY);
		Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		//膨胀
		dilate(binBrightImg, binBrightImg, element);

		//轮廓数组
		vector<vector<Point>> lightContours;
		//找轮廓
		findContours(binBrightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		Mat showContours = Mat(binBrightImg.size(),binBrightImg.type());
		showContours.setTo(Scalar(0));

		for (const auto& contour : lightContours)
		{
			//得到面积
			float lightContourArea = contourArea(contour);
			//面积太小的不要
			if (contour.size() <= 5 ||
				lightContourArea < _param.light_min_area) continue;
			//椭圆拟合区域得到外接矩形
			RotatedRect lightRec = fitEllipse(contour);
			//矫正灯条
			adjustRec(lightRec, ANGLE_TO_UP);
			//宽高比、凸度筛选灯条
			if (lightRec.size.width / lightRec.size.height >
				_param.light_max_ratio ||
				lightContourArea / lightRec.size.area() <
				_param.light_contour_min_solidity
				)continue;
			//对灯条范围适当扩大
			lightRec.size.width *= _param.light_color_detect_extend_ratio;
			lightRec.size.height *= _param.light_color_detect_extend_ratio;
			Rect lightRect = lightRec.boundingRect();
			const Rect srcBound(Point(0, 0), _roiImg.size());
			lightRect &= srcBound;
			//因为颜色通道相减后己方灯条直接过滤，不需要判断颜色了,可以直接将灯条保存
			lightInfos.push_back(LightDescriptor(lightRec));
		}
		//没找到灯条就返回没找到
		if (lightInfos.empty())
		{
			return _flag = ARMOR_NO;
		}

		

		
//		imshow("1111", showContours);
//		imshow("0000", img);
		waitKey(1);

	}
	waitKey(1);

	
	


	return 0;
}
