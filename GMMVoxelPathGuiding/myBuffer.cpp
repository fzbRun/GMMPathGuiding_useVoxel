#include "myBuffer.h"

void myBuffer::createCommandPool(VkDevice logicalDevice, QueueFamilyIndices queueFamilyIndices) {

	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	//VK_COMMAND_POOL_CREATE_TRANSIENT_BIT：提示命令缓冲区经常会重新记录新命令（可能会改变内存分配行为）
	//VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT：允许单独重新记录命令缓冲区，如果没有此标志，则必须一起重置所有命令缓冲区
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();
	if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &this->commandPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create command pool!");
	}

}

/*
void myBuffer::createVertexBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, uint32_t verticeSize, std::vector<Vertex>* vertices) {

	//sizeof不能查指针所指向的类型的大小
	VkDeviceSize bufferSize = verticeSize;// vertices->size();

	//GPU访问最快的缓冲区是VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT，这种缓冲区GPU无法访问
	//而现在顶点数据在GPU中，那么我们需要先将数据传入暂存缓冲区，再由暂存缓冲区复制到GPU访问的缓冲区中（为啥我也不是很清楚，好像和数据在不同端的格式相关，也好像和传输的带宽有关）
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* data;
	//可以将一块设备缓冲区（不是GPU独占的)与一个内存数据指针相映射
	vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, vertices->data(), (size_t)bufferSize);	//该函数只能用于都是CPU端缓冲区的时候
	vkUnmapMemory(logicalDevice, stagingBufferMemory);

	createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, this->vertexBuffer, this->vertexBufferMemory);

	copyBuffer(logicalDevice, queue, this->commandPool, stagingBuffer, this->vertexBuffer, bufferSize);
	vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

}

void myBuffer::createIndexBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, uint32_t indiceSize, std::vector<uint32_t>* indices) {

	VkDeviceSize bufferSize = indiceSize;// *indices->size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, indices->data(), (size_t)bufferSize);
	vkUnmapMemory(logicalDevice, stagingBufferMemory);

	createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, this->indexBuffer, this->indexBufferMemory);

	copyBuffer(logicalDevice, queue, this->commandPool, stagingBuffer, indexBuffer, bufferSize);

	vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

}
*/
/*
template<typename T>
void myBuffer::createStaticBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, uint32_t bufferSize, std::vector<T>* bufferData) {

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, bufferData->data(), (size_t)bufferSize);
	vkUnmapMemory(logicalDevice, stagingBufferMemory);

	VkBuffer staticBuffer;
	VkDeviceMemory staticBufferMemory;
	createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, staticBuffer, staticBufferMemory);

	copyBuffer(logicalDevice, queue, this->commandPool, stagingBuffer, staticBuffer, bufferSize);

	this->buffersStatic.push_back(staticBuffer);
	this->buffersMemorysStatic.push_back(staticBufferMemory);

	vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

}
*/

//静态场景顶点数据每帧复用，但是uniform数据每帧不同
void myBuffer::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t frameSize, VkDeviceSize bufferSize, bool uniformBufferStatic) {

	if (uniformBufferStatic) {

		VkBuffer uniformBuffer;
		VkDeviceMemory uniformBuffersMemory;
		void* uniformBuffersMapped;

		createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBuffersMemory);
		//持久映射，获得缓冲区指针以直接操作而不需要通过api
		vkMapMemory(logicalDevice, uniformBuffersMemory, 0, bufferSize, 0, &uniformBuffersMapped);

		this->uniformBuffersStatic.push_back(uniformBuffer);
		this->uniformBuffersMemorysStatic.push_back(uniformBuffersMemory);
		this->uniformBuffersMappedsStatic.push_back(uniformBuffersMapped);

		return;

	}

	std::vector<VkBuffer> uniformBufferAllFrame;
	std::vector<VkDeviceMemory> uniformBuffesMemoryAllFrame;
	std::vector<void*> uniformBuffersMappedAllFrame;
	for (size_t i = 0; i < frameSize; i++) {

		VkBuffer uniformBuffer;
		VkDeviceMemory uniformBuffersMemory;
		void* uniformBuffersMapped;
		createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBuffersMemory);
		vkMapMemory(logicalDevice, uniformBuffersMemory, 0, bufferSize, 0, &uniformBuffersMapped);

		uniformBufferAllFrame.push_back(uniformBuffer);
		uniformBuffesMemoryAllFrame.push_back(uniformBuffersMemory);
		uniformBuffersMappedAllFrame.push_back(uniformBuffersMapped);

	}

	//createBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBuffersMemory);
	////持久映射，获得缓冲区指针以直接操作而不需要通过api
	//vkMapMemory(logicalDevice, uniformBuffersMemory, 0, bufferSize, 0, &uniformBuffersMapped);

	this->uniformBuffers.push_back(uniformBufferAllFrame);
	this->uniformBuffersMemorys.push_back(uniformBuffesMemoryAllFrame);
	this->uniformBuffersMappeds.push_back(uniformBuffersMappedAllFrame);

}

void myBuffer::createCommandBuffers(VkDevice logicalDevice, uint32_t frameSize) {

	//我们现在想像流水线一样绘制，所以需要多个指令缓冲区

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = this->commandPool;
	//VK_COMMAND_BUFFER_LEVEL_PRIMARY：可以提交到队列执行，但不能从其他命令缓冲区调用。
	//VK_COMMAND_BUFFER_LEVEL_SECONDARY：不能直接提交，但可以从主命令缓冲区调用
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = frameSize;	//指定分配的命令缓冲区是主命令缓冲区还是辅助命令缓冲区的大小

	this->commandBuffers.resize(3);
	//这个函数的第三个参数可以是单个命令缓冲区指针也可以是数组
	this->commandBuffers[0].resize(frameSize);
	if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, this->commandBuffers[0].data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate shadow command buffers!");
	}

	this->commandBuffers[1].resize(frameSize);
	if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, this->commandBuffers[1].data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate compute command buffers!");
	}

	this->commandBuffers[2].resize(frameSize);
	if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, this->commandBuffers[2].data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffers!");
	}

}

//一个交换链图像视图代表一个展示缓冲
//一个renderPass代表一帧中所有的输出流程，其中最后的输出图像是一个交换链图像
//一个frameBuffer是renderPass的一种实例，将renderPass中规定的输出图像进行填充
void myBuffer::createFramebuffers(uint32_t swapChainImageViewsSize, std::vector<VkImageView> swapChainImageViews, VkExtent2D swapChainExtent, std::vector<VkImageView> imageViews, VkImageView depthImageView, VkRenderPass renderPass, VkDevice logicalDevice) {

	std::vector<VkFramebuffer> frameBuffers;
	frameBuffers.resize(swapChainImageViewsSize);
	for (size_t i = 0; i < swapChainImageViewsSize; i++) {

		std::vector<VkImageView> attachments = imageViews;
		if (swapChainImageViews.size() > 0) {
			attachments.push_back(swapChainImageViews[i]);
		}
		if (depthImageView) {
			attachments.push_back(depthImageView);
		}

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = swapChainExtent.width;
		framebufferInfo.height = swapChainExtent.height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &frameBuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}

	}

	this->swapChainFramebuffers.push_back(frameBuffers);

}

//frameBuffer和reSwapChain相关，所以放在别处clean
void myBuffer::clean(VkDevice logicalDevice, int frameSize) {

	for (int i = 0; i < buffersStatic.size(); i++) {
		vkDestroyBuffer(logicalDevice, buffersStatic[i], nullptr);
		vkFreeMemory(logicalDevice, buffersMemorysStatic[i], nullptr);
	}

	for (int i = 0; i < uniformBuffers.size(); i++) {
		for (size_t j = 0; j < frameSize; j++) {
			vkDestroyBuffer(logicalDevice, uniformBuffers[i][j], nullptr);
			vkFreeMemory(logicalDevice, uniformBuffersMemorys[i][j], nullptr);
		}
	}

	for (int i = 0; i < uniformBuffersStatic.size(); i++) {
		vkDestroyBuffer(logicalDevice, uniformBuffersStatic[i], nullptr);
		vkFreeMemory(logicalDevice, uniformBuffersMemorysStatic[i], nullptr);
	}


	vkDestroyBuffer(logicalDevice, indexBuffer, nullptr);
	vkFreeMemory(logicalDevice, indexBufferMemory, nullptr);

	vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
	vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);

	vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

}


void myBuffer::createBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to create vertex buffer!");
	}

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);
	if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate vertex buffer memory!");
	}

	vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);

}

void myBuffer::copyBuffer(VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {

	VkCommandBuffer commandBuffer = beginSingleTimeCommands(logicalDevice, commandPool);

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);	//该函数可以用于任意端的缓冲区，不像memcpy

	endSingleTimeCommands(logicalDevice, queue, commandBuffer, commandPool);

}

VkCommandBuffer myBuffer::beginSingleTimeCommands(VkDevice logicalDevice, VkCommandPool commandPool) {

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = commandPool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;

}

void myBuffer::endSingleTimeCommands(VkDevice logicalDevice, VkQueue queue, VkCommandBuffer commandBuffer, VkCommandPool commandPool) {

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);
	vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);

}

uint32_t myBuffer::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {

	VkPhysicalDeviceMemoryProperties memProperties;
	//找到当前物理设备（显卡）所支持的显存类型（是否支持纹理存储，顶点数据存储，通用数据存储等）
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		//这里的type八成是通过one-hot编码的，如0100表示三号,1000表示四号
		if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");

}