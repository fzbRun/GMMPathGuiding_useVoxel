#pragma once

#include "structSet.h"

#include <vector>

#ifndef MY_RAY
#define MY_RAY

class myRay {

public:

	//分辨率
	uint32_t width;
	uint32_t height;

	myRay(uint32_t width, uint32_t hegiht);
	//碰撞检测的函数要放到计算着色器中去
	void rayStart(VkImageView targetTextureView, std::vector<VkBuffer> sceneBuffers);

};

#endif // !MY_RAY
