#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <array>
#include <iostream>

#include "structSet.h"

#ifndef MY_DISCRIPTOR
#define MY_DISCRIPTOR

class myDescriptor {

public:

	VkDevice logicalDevice;
	uint32_t frameSize;

	//uint32_t uniformBufferNum;
	//uint32_t textureNum;

	VkDescriptorPool discriptorPool;
	std::vector<DescriptorObject> descriptorObjects;

	myDescriptor(VkDevice logicalDevice, uint32_t frameSize);

	void createDescriptorPool(uint32_t uniformBufferNumAllLayout, uint32_t storageBufferNumAllLayout, std::vector<VkDescriptorType> types, std::vector<uint32_t> textureNumAllLayout);

	DescriptorObject createDescriptorObject(uint32_t uniformBufferNum, uint32_t textureNum, std::vector<VkShaderStageFlagBits>* uniformBufferUsages, std::vector<VkDescriptorType>* textureDescriptorType,
		uint32_t descriptorSetSize, std::vector<std::vector<VkBuffer>>* uniformBuffers, std::vector<VkDeviceSize> bufferSize, std::vector < std::vector<VkImageView>>* textureViews, std::vector<std::vector<VkSampler>>* textureSamplers);
	VkDescriptorSetLayout createDescriptorSetLayout(uint32_t uniformBufferNum, uint32_t textureNum, std::vector<VkShaderStageFlagBits>* uniformBufferUsages, std::vector<VkDescriptorType>* textureDescriptorType);
	VkDescriptorSet createDescriptorSet(DescriptorObject descriptorObject, std::vector<VkBuffer>* uniformBuffers, std::vector<VkDeviceSize> bufferSize, std::vector<VkDescriptorType>* textureDescriptorType, std::vector<VkImageView>* textureViews, std::vector<VkSampler>* textureSamplers);

	void clean();

};

#endif // !MY_DISCRIPTOR
