#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include<iostream>
#include<vector>
#include<map>
#include<set>

#include "structSet.h"

#ifndef MY_DEVICE
#define MY_DEVICE

class myDevice {

public:

	VkInstance instance;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;

	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue computeQueue;

	SwapChainSupportDetails swapChainSupportDetails;
	QueueFamilyIndices queueFamilyIndices;

	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	//构造函数
	myDevice(VkInstance instance, VkSurfaceKHR surface);

	//vulkan函数
	void pickPhysicalDevice();
	void createLogicalDevice(bool enableValidationLayers, std::vector<const char*> validationLayers);

	//辅助函数
	int rateDeviceSuitability(VkSurfaceKHR surface, VkPhysicalDevice device);
	SwapChainSupportDetails querySwapChainSupport(VkSurfaceKHR surface, VkPhysicalDevice device);
	QueueFamilyIndices findQueueFamilies(VkSurfaceKHR surface, VkPhysicalDevice device);
	bool checkDeviceExtensionSupport(VkPhysicalDevice device);
	VkSampleCountFlagBits getMaxUsableSampleCount();
	//uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	void clean();

};


#endif