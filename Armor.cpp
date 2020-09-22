#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
using namespace std;
using namespace cv;
int main() {
	
	VideoCapture capture;
	capture.open("��̨�ز�����ǰ�Խ�-ev--3.MOV");

	while (1) {

		Mat img;
		capture >> img;
		vector<Mat>channels;
		split(img, channels);//����ɫ��ͨ��
		//Ԥ����ɾ������װ�װ���ɫ
		Mat _grayImg = channels.at(0) - channels.at(2);//Get blue-red image;
		Mat binBrightImg;
		//��ֵ��
		threshold(_grayImg, binBrightImg, 110
			, 255, cv::THRESH_BINARY);
		Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		//����
		dilate(binBrightImg, binBrightImg, element);

		//��������
		vector<vector<Point>> lightContours;
		//������
		findContours(binBrightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		Mat showContours = Mat(binBrightImg.size(),binBrightImg.type());
		showContours.setTo(Scalar(0));

		for (const auto& contour : lightContours)
		{
			//�õ����
			float lightContourArea = contourArea(contour);
			//���̫С�Ĳ�Ҫ
			if (contour.size() <= 5 ||
				lightContourArea < _param.light_min_area) continue;
			//��Բ�������õ���Ӿ���
			RotatedRect lightRec = fitEllipse(contour);
			//��������
			adjustRec(lightRec, ANGLE_TO_UP);
			//��߱ȡ�͹��ɸѡ����
			if (lightRec.size.width / lightRec.size.height >
				_param.light_max_ratio ||
				lightContourArea / lightRec.size.area() <
				_param.light_contour_min_solidity
				)continue;
			//�Ե�����Χ�ʵ�����
			lightRec.size.width *= _param.light_color_detect_extend_ratio;
			lightRec.size.height *= _param.light_color_detect_extend_ratio;
			Rect lightRect = lightRec.boundingRect();
			const Rect srcBound(Point(0, 0), _roiImg.size());
			lightRect &= srcBound;
			//��Ϊ��ɫͨ������󼺷�����ֱ�ӹ��ˣ�����Ҫ�ж���ɫ��,����ֱ�ӽ���������
			lightInfos.push_back(LightDescriptor(lightRec));
		}
		//û�ҵ������ͷ���û�ҵ�
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
