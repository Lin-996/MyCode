#include"Armor.h"
int main() {
	
	VideoCapture capture;
	capture.open("��̨�ز�����ǰ�Խ�-ev--3.MOV");

	// ���ֲ��� ֱ������
	ArmorParam _param;
	// ʶ����������
	ArmorFun _fun;
	// ���������Ϣ������
	vector<LightDescriptor>lightInfos;
	// ����������
	vector<ArmorDescriptor> _armors;
	// ������Ϣ
	int _flag;
	while (1) {

		Mat img;
		capture >> img;
		vector<Mat>channels;

		split(img, channels);//����ɫ��ͨ��
		//Ԥ����ɾ������װ�װ���ɫ
		Mat _grayImg = channels.at(0) - channels.at(2);//Get blue-red image;
		Mat binBrightImg;
		//cvtColor(_roiImg, _grayImg, COLOR_BGR2GRAY, 1);
		//��ֵ��
		threshold(_grayImg, binBrightImg, _param.brightness_threshold, 255, cv::THRESH_BINARY);
		Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		//��ֵ�˲�
		// medianBlur(binBrightImg, binBrightImg, 3);
		//����
		//��һ�� �ȱ������
		//dilate(binBrightImg, binBrightImg, element);

		//������
		vector<vector<Point>> lightContours;
		cv::findContours(binBrightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//��������
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
			_fun.adjustRec(lightRec, ANGLE_TO_UP);
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
			const Rect srcBound(Point(0, 0), img.size());
			lightRect &= srcBound;
			//��Ϊ��ɫͨ������󼺷�����ֱ�ӹ��ˣ�����Ҫ�ж���ɫ��,����ֱ�ӽ���������
			lightInfos.push_back(LightDescriptor(lightRec));
		}
		//û�ҵ������ͷ���û�ҵ�
		if (lightInfos.empty())
		{
			cout << "��ǰ�޶���" << endl;
		}
		else {
			//����������x��С��������
			sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2)
				{//Lambda����,��Ϊsort��cmp����
					return ld1.center.x < ld2.center.x;
				});
			for (size_t i = 0; i < lightInfos.size(); i++)
			{//�������е�������ƥ��
				for (size_t j = i + 1; (j < lightInfos.size()); j++)
				{
					const LightDescriptor& leftLight = lightInfos[i];
					const LightDescriptor& rightLight = lightInfos[j];
					/*
					*	Works for 2-3 meters situation
					*	morphologically similar: // parallel
									 // similar height
					*/
					//�ǲ�
					float angleDiff_ = abs(leftLight.angle - rightLight.angle);
					//���Ȳ����
					float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
					//ɸѡ
					if (angleDiff_ > _param.light_max_angle_diff_ ||
						LenDiff_ratio > _param.light_max_height_diff_ratio_)
					{
						continue;
					}

					/*
					*	proper location:  y value of light bar close enough
					*			  ratio of length and width is proper
					*/
					//���ҵ���������
					float dis = distance(leftLight.center, rightLight.center);
					//���ҵ������ȵ�ƽ��ֵ
					float meanLen = (leftLight.length + rightLight.length) / 2;
					//���ҵ������ĵ�y�Ĳ�ֵ
					float yDiff = abs(leftLight.center.y - rightLight.center.y);
					//y�����
					float yDiff_ratio = yDiff / meanLen;
					//���ҵ������ĵ�x�Ĳ�ֵ
					float xDiff = abs(leftLight.center.x - rightLight.center.x);
					//x�����
					float xDiff_ratio = xDiff / meanLen;
					//��������������ȱ�ֵ
					float ratio = dis / meanLen;
					//ɸѡ
					if (yDiff_ratio > _param.light_max_y_diff_ratio_ ||
						xDiff_ratio < _param.light_min_x_diff_ratio_ ||
						ratio > _param.armor_max_aspect_ratio_ ||
						ratio < _param.armor_min_aspect_ratio_)
					{
						continue;
					}

					// calculate pairs' info 
						  //����ֵ��ȷ����Сװ��
					int armorType = ratio > _param.armor_big_armor_ratio ? BIG_ARMOR : SMALL_ARMOR;
					// calculate the rotation score
					float ratiOff = (armorType == BIG_ARMOR) ? max(_param.armor_big_armor_ratio - ratio, float(0)) : max(_param.armor_small_armor_ratio - ratio, float(0));
					float yOff = yDiff / meanLen;
					float rotationScore = -(ratiOff * ratiOff + yOff * yOff);
					//�õ�ƥ���װ�װ�
					ArmorDescriptor armor(leftLight, rightLight, armorType, channels.at(1), rotationScore, _param);

					_armors.emplace_back(armor);
					break;
				}
			}
			//ûƥ�䵽װ�װ��򷵻�û�ҵ�
			if (_armors.empty())
			{
				cout << "�Ҳ���װ�װ�" << endl;
			}
			else {
				//calculate the final score �������յ÷�
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
