#pragma once

#include "myBuffer.h"

#ifndef MY_IMAGE
#define MY_IMAGE
class myImage {

public:

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;

	VkImage image;
	VkImageView imageView;
	VkDeviceMemory imageMemory;
	VkSampler textureSampler;
	uint32_t mipLevels = 1;

	myImage(std::string path, VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, bool mipmapEnable);
	myImage(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t width, uint32_t height, uint32_t mipLevel, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImageAspectFlags aspectFlags);

	void createTextureImage(const char* texturePath, VkQueue queue, VkCommandPool commandPool, bool mipmapEnable);
	VkImageView createTextureImageView(VkImage textureImage, uint32_t mipLevels);
	VkSampler createTextureSampler();
	
	
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevel, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
	void transitionImageLayout(VkQueue queue, VkCommandPool commandPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
	void copyBufferToImage(VkQueue queue, VkCommandPool commandPool, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
	void generateMipmaps(VkQueue queue, VkCommandPool commandPool, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

	static VkFormat findDepthFormat(VkPhysicalDevice physicalDevice);
	static VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	static bool hasStencilComponent(VkFormat format);

	void clean();

};
#endif

