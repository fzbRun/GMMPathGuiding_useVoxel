#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include<glm/glm.hpp>

#include <iostream>
#include<array>
#include <vector>

#include "structSet.h"

#ifndef MY_BUFFER
#define MY_BUFFER

class myBuffer {

public:

	VkCommandPool commandPool;
	std::vector<std::vector<VkCommandBuffer>> commandBuffers;

	//一个大的全塞进去，然后用offset
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;

	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	std::vector<VkBuffer> buffersStatic;
	std::vector<VkDeviceMemory> buffersMemorysStatic;

	std::vector<std::vector<VkBuffer>> uniformBuffers;
	std::vector<std::vector<VkDeviceMemory>> uniformBuffersMemorys;
	std::vector<std::vector<void*>> uniformBuffersMappeds;

	std::vector<VkBuffer> uniformBuffersStatic;
	std::vector<VkDeviceMemory> uniformBuffersMemorysStatic;
	std::vector<void*> uniformBuffersMappedsStatic;

	std::vector<std::vector<VkFramebuffer>> swapChainFramebuffers;

	void createCommandPool(VkDevice logicalDevice, QueueFamilyIndices queueFamilyIndices);
	//void createVertexBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, uint32_t verticeSize, std::vector<Vertex>* vertices);
	//void createIndexBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, uint32_t indiceSize, std::vector<uint32_t>* indices);
	template<typename T>	//模板函数需要在头文件中定义
	void createStaticBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, uint32_t bufferSize, std::vector<T>* bufferData) {
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, bufferData->data(), (size_t)bufferSize);
		vkUnmapMemory(logicalDevice, stagingBufferMemory);

		VkBuffer staticBuffer;
		VkDeviceMemory staticBufferMemory;
		createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, staticBuffer, staticBufferMemory);

		copyBuffer(logicalDevice, queue, this->commandPool, stagingBuffer, staticBuffer, bufferSize);

		this->buffersStatic.push_back(staticBuffer);
		this->buffersMemorysStatic.push_back(staticBufferMemory);

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

	}
	void createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t frameSize, VkDeviceSize bufferSize, bool uniformBufferStatic);
	void createCommandBuffers(VkDevice logicalDevice, uint32_t frameSize);
	void createFramebuffers(uint32_t swapChainImageViewsSize, std::vector<VkImageView> swapChainImageViews, VkExtent2D swapChainExtent, std::vector<VkImageView> imageViews, VkImageView depthImageView, VkRenderPass renderPass, VkDevice logicalDevice);

	void clean(VkDevice logicalDevice, int frameSize);

	static void createBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	static void copyBuffer(VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
	static VkCommandBuffer beginSingleTimeCommands(VkDevice logicalDevice, VkCommandPool commandPool);
	static void endSingleTimeCommands(VkDevice logicalDevice, VkQueue queue, VkCommandBuffer commandBuffer, VkCommandPool commandPool);
	static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

};

#endif
