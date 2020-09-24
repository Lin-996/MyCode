#include"Armor.h"
int main() {
	
	VideoCapture capture;
	capture.open("炮台素材蓝车前对角-ev--3.MOV");

	// 各种参数 直接套用
	ArmorParam _param;
	// 识别函数的总类
	ArmorFun _fun;
	// 描述光的信息的容器
	vector<LightDescriptor>lightInfos;
	// 特征的容器
	vector<ArmorDescriptor> _armors;
	// 敌人信息
	int _flag;
	while (1) {

		Mat img;
		capture >> img;
		vector<Mat>channels;

		split(img, channels);//分离色彩通道
		//预处理删除己方装甲板颜色
		Mat _grayImg = channels.at(0) - channels.at(2);//Get blue-red image;
		Mat binBrightImg;
		//cvtColor(_roiImg, _grayImg, COLOR_BGR2GRAY, 1);
		//阈值化
		threshold(_grayImg, binBrightImg, _param.brightness_threshold, 255, cv::THRESH_BINARY);
		Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		//中值滤波
		// medianBlur(binBrightImg, binBrightImg, 3);
		//膨胀
		//试一下 先闭运算后开
		//dilate(binBrightImg, binBrightImg, element);

		//找轮廓
		vector<vector<Point>> lightContours;
		cv::findContours(binBrightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//过滤轮廓
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
			_fun.adjustRec(lightRec, ANGLE_TO_UP);
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
			const Rect srcBound(Point(0, 0), img.size());
			lightRect &= srcBound;
			//因为颜色通道相减后己方灯条直接过滤，不需要判断颜色了,可以直接将灯条保存
			lightInfos.push_back(LightDescriptor(lightRec));
		}
		//没找到灯条就返回没找到
		if (lightInfos.empty())
		{
			cout << "当前无对象" << endl;
		}
		else {
			//按灯条中心x从小到大排序
			sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2)
				{//Lambda函数,作为sort的cmp函数
					return ld1.center.x < ld2.center.x;
				});
			for (size_t i = 0; i < lightInfos.size(); i++)
			{//遍历所有灯条进行匹配
				for (size_t j = i + 1; (j < lightInfos.size()); j++)
				{
					const LightDescriptor& leftLight = lightInfos[i];
					const LightDescriptor& rightLight = lightInfos[j];
					/*
					*	Works for 2-3 meters situation
					*	morphologically similar: // parallel
									 // similar height
					*/
					//角差
					float angleDiff_ = abs(leftLight.angle - rightLight.angle);
					//长度差比率
					float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
					//筛选
					if (angleDiff_ > _param.light_max_angle_diff_ ||
						LenDiff_ratio > _param.light_max_height_diff_ratio_)
					{
						continue;
					}

					/*
					*	proper location:  y value of light bar close enough
					*			  ratio of length and width is proper
					*/
					//左右灯条相距距离
					float dis = distance(leftLight.center, rightLight.center);
					//左右灯条长度的平均值
					float meanLen = (leftLight.length + rightLight.length) / 2;
					//左右灯条中心点y的差值
					float yDiff = abs(leftLight.center.y - rightLight.center.y);
					//y差比率
					float yDiff_ratio = yDiff / meanLen;
					//左右灯条中心点x的差值
					float xDiff = abs(leftLight.center.x - rightLight.center.x);
					//x差比率
					float xDiff_ratio = xDiff / meanLen;
					//相距距离与灯条长度比值
					float ratio = dis / meanLen;
					//筛选
					if (yDiff_ratio > _param.light_max_y_diff_ratio_ ||
						xDiff_ratio < _param.light_min_x_diff_ratio_ ||
						ratio > _param.armor_max_aspect_ratio_ ||
						ratio < _param.armor_min_aspect_ratio_)
					{
						continue;
					}

					// calculate pairs' info 
						  //按比值来确定大小装甲
					int armorType = ratio > _param.armor_big_armor_ratio ? BIG_ARMOR : SMALL_ARMOR;
					// calculate the rotation score
					float ratiOff = (armorType == BIG_ARMOR) ? max(_param.armor_big_armor_ratio - ratio, float(0)) : max(_param.armor_small_armor_ratio - ratio, float(0));
					float yOff = yDiff / meanLen;
					float rotationScore = -(ratiOff * ratiOff + yOff * yOff);
					//得到匹配的装甲板
					ArmorDescriptor armor(leftLight, rightLight, armorType, channels.at(1), rotationScore, _param);

					_armors.emplace_back(armor);
					break;
				}
			}
			//没匹配到装甲板则返回没找到
			if (_armors.empty())
			{
				cout << "找不到装甲板" << endl;
			}
			else {
				//calculate the final score 计算最终得分
				for (auto& armor : _armors)
				{
					armor.finalScore = armor.sizeScore + armor.distScore + armor.rotationScore;
				}
				for (auto& armor : _armors) {
				//	if (armor.finalScore > 100) {
						_fun.drawRect(img, armor);
				//	}
				}

			}
		}
		_armors.clear();
		lightInfos.clear();

		imshow("1222", binBrightImg);
		imshow("0000", img);
		waitKey(1);

	}
	waitKey(1);
	return 0;
}
