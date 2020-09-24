#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<array>
#include <map>
using namespace std;
using namespace cv;

template<typename T>
float distance(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2);

template<typename ValType>
const cv::Point2f crossPointOf(const std::array<cv::Point_<ValType>, 2>& line1, const std::array<cv::Point_<ValType>, 2>& line2);

double TemplateMatch(Mat orginImg, Mat tmpImg, Point& finalLoc, int mode);



enum ColorChannels
{
	BLUE = 0,
	GREEN = 1,
	RED = 2
};

enum
{
	WIDTH_GREATER_THAN_HEIGHT,
	ANGLE_TO_UP
};

enum ArmorFlag
{
	ARMOR_NO = 0,		// not found
	ARMOR_LOST = 1,		// lose tracking
	ARMOR_GLOBAL = 2,	// armor found globally 在全球范围内
	ARMOR_LOCAL = 3		// armor found locally(in tracking mode) 在跟踪范围内
};

enum ObjectType
{
	UNKNOWN_ARMOR = 0,
	SMALL_ARMOR = 1,
	BIG_ARMOR = 2,
	MINI_RUNE = 3,
	GREAT_RUNE = 4
};
class LightDescriptor
{
public:
	LightDescriptor() {};
	LightDescriptor(const cv::RotatedRect& light)
	{
		width = light.size.width;
		length = light.size.height;
		center = light.center;
		angle = light.angle;
		area = light.size.area();
	}
	const LightDescriptor& operator =(const LightDescriptor& ld)
	{
		this->width = ld.width;
		this->length = ld.length;
		this->center = ld.center;
		this->angle = ld.angle;
		this->area = ld.area;
		return *this;
	}

	/*
	*	@Brief: return the light as a cv::RotatedRect object
	*/
	cv::RotatedRect rec() const
	{
		return cv::RotatedRect(center, cv::Size2f(width, length), angle);
	}

public:
	float width;
	float length;
	cv::Point2f center;
	float angle;
	float area;
};
struct ArmorParam
{
	//Pre-treatment
	int brightness_threshold;
	int color_threshold;
	float light_color_detect_extend_ratio;

	//Filter lights
	float light_min_area;
	float light_max_angle;
	float light_min_size;
	float light_contour_min_solidity;
	float light_max_ratio;

	//Filter pairs
	float light_max_angle_diff_;
	float light_max_height_diff_ratio_; // hdiff / max(r.length, l.length)
	float light_max_y_diff_ratio_;  // ydiff / max(r.length, l.length)
	float light_min_x_diff_ratio_;

	//Filter armor
	float armor_big_armor_ratio;
	float armor_small_armor_ratio;
	float armor_min_aspect_ratio_;
	float armor_max_aspect_ratio_;

	//other params
	float sight_offset_normalized_base;
	float area_normalized_base;
	int enemy_color;
	int max_track_num = 3000;

	/*
	*	@Brief: 为各项参数赋默认值
	*/
	ArmorParam()
	{
		//pre-treatment
		brightness_threshold = 110;
		color_threshold = 40;
		light_color_detect_extend_ratio = 1.2; //光色检测扩展比


		// Filter lights
		light_min_area = 10;	//亮度最小区域
		light_max_angle = 45.0;	//亮度最大角
		light_min_size = 5.0;	//亮度最小尺寸
		light_contour_min_solidity = 0.5;	//亮度轮廓最小可靠性
		light_max_ratio = 1.0;	//亮度最大比率

	// Filter pairs
		light_max_angle_diff_ = 7.0; //20	亮度最大角度微分
		light_max_height_diff_ratio_ = 0.2; //0.5  亮度最大高度微分率
		light_max_y_diff_ratio_ = 2.0; //100  y轴亮度最大高度微分率
		light_min_x_diff_ratio_ = 0.5; //100  x轴亮度最大高度微分率

		// Filter armor
		armor_big_armor_ratio = 3.2;	//装甲大装甲比率
		armor_small_armor_ratio = 2;	//装甲小装甲比率
		//armor_max_height_ = 100.0;
		//armor_max_angle_ = 30.0;
		armor_min_aspect_ratio_ = 1.0;	//装甲最小长宽比率
		armor_max_aspect_ratio_ = 5.0;	//装甲最大长宽比率

		//other params
		sight_offset_normalized_base = 200;	//瞄准偏差归一化基线
		area_normalized_base = 1000;	//区域正常基线
		enemy_color = BLUE;	//敌人颜色
	}
};
class ArmorDescriptor
{
public:
	/*
	*	@Brief: Initialize with all 0
	*/
	ArmorDescriptor() {
		rotationScore = 0;
		sizeScore = 0;
		vertex.resize(4);
		for (int i = 0; i < 4; i++)
		{
			vertex[i] = cv::Point2f(0, 0);
		}
		type = UNKNOWN_ARMOR;
	}

	/*
	*	@Brief: calculate the rest of information(except for match&final score)of ArmroDescriptor based on:
			l&r light, part of members in ArmorDetector, and the armortype(for the sake of saving time)
			计算ArmroDescriptor的剩余信息(比赛和最终得分除外)基于:
			l&r灯，盔甲探测器的部分成员，盔甲型(节省时间)
	*	@Calls: ArmorDescriptor::getFrontImg()
	*/
	ArmorDescriptor(const LightDescriptor& lLight, const LightDescriptor& rLight, const int armorType, const cv::Mat& grayImg, float rotaScore, ArmorParam _param)
	{
		//handle two lights
		lightPairs[0] = lLight.rec();
		lightPairs[1] = rLight.rec();

		/*
		为啥乘以2
		*/
		cv::Size exLSize(int(lightPairs[0].size.width), int(lightPairs[0].size.height * 2));
		cv::Size exRSize(int(lightPairs[1].size.width), int(lightPairs[1].size.height * 2));
		cv::RotatedRect exLLight(lightPairs[0].center, exLSize, lightPairs[0].angle);
		cv::RotatedRect exRLight(lightPairs[1].center, exRSize, lightPairs[1].angle);

		cv::Point2f pts_l[4];
		exLLight.points(pts_l);
		cv::Point2f upper_l = pts_l[2];
		cv::Point2f lower_l = pts_l[3];

		cv::Point2f pts_r[4];
		exRLight.points(pts_r);
		cv::Point2f upper_r = pts_r[1];
		cv::Point2f lower_r = pts_r[0];

		vertex.resize(4);
		vertex[0] = upper_l;
		vertex[1] = upper_r;
		vertex[2] = lower_r;
		vertex[3] = lower_l;

		//set armor type
		type = armorType;

		//get front view
		getFrontImg(grayImg);
		rotationScore = rotaScore;

		// calculate the size score
		float normalized_area = contourArea(vertex) / _param.area_normalized_base;
		sizeScore = exp(normalized_area);

		// calculate the distance score
		Point2f srcImgCenter(grayImg.cols / 2, grayImg.rows / 2);
		float sightOffset = distance(srcImgCenter, crossPointOf(array<Point2f, 2>{vertex[0], vertex[2]}, array<Point2f, 2>{vertex[1], vertex[3]}));
		distScore = exp(-sightOffset / _param.sight_offset_normalized_base);
	}

	/*
	*	@Brief: empty the object
	*	@Called :ArmorDetection._targetArmor
	*/
	void clear()
	{
		rotationScore = 0;
		sizeScore = 0;
		distScore = 0;
		finalScore = 0;
		for (int i = 0; i < 4; i++)
		{
			vertex[i] = cv::Point2f(0, 0);
		}
		type = UNKNOWN_ARMOR;
	}

	void setType() {
		type = BIG_ARMOR;
	}

	/*
	*	@Brief: get the front img(prespective transformation) of armor(if big, return the middle part)
	* 获得装甲的前变形(如果大，返回中间部分)
	*	@Inputs: grayImg of roi
	* 感兴趣的部分
	*	@Outputs: store the front img to ArmorDescriptor's public
	* 将前面的img存储到ArmorDescriptor的public
	*/
	void getFrontImg(const cv::Mat& grayImg) {
		const Point2f&
			tl = vertex[0],
			tr = vertex[1],
			br = vertex[2],
			bl = vertex[3];

		int width, height;
		if (type == BIG_ARMOR)
		{
			width = 92;
			height = 50;
		}
		else
		{
			width = 50;
			height = 50;
		}

		Point2f src[4]{ Vec2f(tl), Vec2f(tr), Vec2f(br), Vec2f(bl) };
		Point2f dst[4]{ Point2f(0.0, 0.0), Point2f(width, 0.0), Point2f(width, height), Point2f(0.0, height) };
		const Mat perspMat = getPerspectiveTransform(src, dst);
		warpPerspective(grayImg, frontImg, perspMat, Size(width, height));
	}

	bool isArmorPattern() {
		return true;
	}
public:
	std::array<cv::RotatedRect, 2> lightPairs; //0 left, 1 right
	float sizeScore;		//S1 = e^(size)
	float distScore;		//S2 = e^(-offset)
	float rotationScore;		//S3 = -(ratio^2 + yDiff^2) 
	float finalScore;
	//	0 -> small
	//	1 -> big
	//	-1 -> unkown
	int type;
	// 敌人的号码
	int enemy_num;

	std::vector<cv::Point2f> vertex; //four vertex of armor area, lihgt bar area exclued!!	
	cv::Mat frontImg; //front img after prespective transformation from vertex,1 channel gray img


};

class ArmorFun {
public:
	cv::RotatedRect& adjustRec(cv::RotatedRect& rec, const int mode)
	{
		using std::swap;

		float& width = rec.size.width;
		float& height = rec.size.height;
		// OpenCV中，坐标的原点在左上角，与x轴平行的方向为角度为0，逆时针旋转角度为负，顺时针旋转角度为正
		float& angle = rec.angle;



		if (mode == WIDTH_GREATER_THAN_HEIGHT)
		{
			if (width < height)
			{
				swap(width, height);
				angle += 90.0;
			}
		}
		// 使角度始终保持在[-90,90]度
		while (angle >= 90.0) angle -= 180.0;
		while (angle < -90.0) angle += 180.0;

		if (mode == ANGLE_TO_UP)
		{
			if (angle >= 45.0)
			{
				swap(width, height);
				angle -= 90.0;
			}
			else if (angle < -45.0)
			{
				swap(width, height);
				angle += 90.0;
			}
		}

		return rec;
	}
	void drawRect(Mat img,ArmorDescriptor a1) {
		vector<Point2f>rectPoint;
		Point2f oneRectPoint[4];
		a1.lightPairs[0].points(oneRectPoint);
		for (int i = 0; i < 4; i++) {
			rectPoint.push_back(oneRectPoint[i]);
		}
		a1.lightPairs[1].points(oneRectPoint);
		for (int i = 0; i < 4; i++) {
			rectPoint.push_back(oneRectPoint[i]);
		}
		RotatedRect ret = minAreaRect((Mat)rectPoint);
		
		ret.points(oneRectPoint);
		for (int i = 0; i < 4; i++)
		{
			cv::line(img, oneRectPoint[i], oneRectPoint[(i + 1) % 4], cv::Scalar(0, 255, 0), 2, LINE_AA);
		}
	}

};




template<typename T>
float distance(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2)
{
	return std::sqrt(std::pow((pt1.x - pt2.x), 2) + std::pow((pt1.y - pt2.y), 2));
}

template<typename ValType>
const cv::Point2f crossPointOf(const std::array<cv::Point_<ValType>, 2>& line1, const std::array<cv::Point_<ValType>, 2>& line2)
{
	ValType a1 = line1[0].y - line1[1].y;
	ValType b1 = line1[1].x - line1[0].x;
	ValType c1 = line1[0].x * line1[1].y - line1[1].x * line1[0].y;

	ValType a2 = line2[0].y - line2[1].y;
	ValType b2 = line2[1].x - line2[0].x;
	ValType c2 = line2[0].x * line2[1].y - line2[1].x * line2[0].y;

	ValType d = a1 * b2 - a2 * b1;

	if (d == 0.0)
	{
		return cv::Point2f(FLT_MAX, FLT_MAX);
	}
	else
	{
		return cv::Point2f(float(b1 * c2 - b2 * c1) / d, float(c1 * a2 - c2 * a1) / d);
	}
}
double TemplateMatch(Mat orginImg, Mat tmpImg, Point& finalLoc, int mode) {
	Mat result;

	matchTemplate(orginImg, tmpImg, result, mode);
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc); //找到最佳匹配点
	// 我们的用的是越大越好

	finalLoc = maxLoc;
	return maxVal;


}