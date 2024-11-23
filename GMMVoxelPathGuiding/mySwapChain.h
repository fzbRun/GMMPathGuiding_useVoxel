#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include<algorithm>
#include<iostream>

#include "structSet.h"

#ifndef MY_SWAPCHAIN
#define MY_SWAPCHAIN
class mySwapChain {

public:

	GLFWwindow* window;
	VkSurfaceKHR surface;
	VkDevice logicalDevice;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	VkSurfaceFormatKHR surfaceFormat;
	VkExtent2D extent;

	mySwapChain(GLFWwindow* window, VkSurfaceKHR surface, VkDevice logicalDevice, SwapChainSupportDetails swapChainSupport, QueueFamilyIndices indices);
	void createSwapChain(SwapChainSupportDetails swapChainSupport, QueueFamilyIndices indices);
	void createSwapChainImageViews();

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

};
#endif
