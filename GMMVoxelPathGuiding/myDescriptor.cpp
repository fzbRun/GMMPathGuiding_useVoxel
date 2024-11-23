#include "myDescriptor.h"

myDescriptor::myDescriptor(VkDevice logicalDevice, uint32_t frameSize) {
	this->logicalDevice = logicalDevice;
	this->frameSize = frameSize;
	//createDescriptorPool();
}

//这里其实挺奇怪的，因为描述符池是将不同的描述符分开来记录的，而描述符集合是将不同描述符配对记录的
//如描述符池记录有几种描述符，每种描述符有几个；而描述符集合记录每个集合由哪些描述符组成，一共有几个集合
//相当于描述符池从自己的各个分池中拿出描述符组成一个描述符集合
void myDescriptor::createDescriptorPool(uint32_t uniformBufferNumAllLayout, uint32_t storageBufferNumAllLayout, std::vector<VkDescriptorType> types, std::vector<uint32_t> textureNumAllLayout) {

	std::vector<VkDescriptorPoolSize> poolSizes{};
	VkDescriptorPoolSize poolSize;
	if (uniformBufferNumAllLayout > 0) {
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(this->frameSize * uniformBufferNumAllLayout);
		poolSizes.push_back(poolSize);
	}
	if (storageBufferNumAllLayout > 0) {
		poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(storageBufferNumAllLayout);
		poolSizes.push_back(poolSize);
	}
	for (int i = 0; i < textureNumAllLayout.size(); i++) {

		poolSize.type = types[i];
		poolSize.descriptorCount = static_cast<uint32_t>(this->frameSize * textureNumAllLayout[i]);
		poolSizes.push_back(poolSize);

	}

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = static_cast<uint32_t>(this->frameSize * 16);

	if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &this->discriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}

}

//通用缓冲区数——纹理数——通用缓冲区着色器阶段——纹理描述符类型——描述符集合数量——通用缓冲区——纹理视图——纹理采样器
//为啥要vector中套vector的原因是外面的vector是对应几个set，而里面的vector表示一个set里有几个
DescriptorObject myDescriptor::createDescriptorObject(uint32_t uniformBufferNum, uint32_t textureNum, std::vector<VkShaderStageFlagBits>* uniformBufferUsages, std::vector<VkDescriptorType>* textureDescriptorType,
	uint32_t descriptorSetSize, std::vector<std::vector<VkBuffer>>* uniformBuffers, std::vector<VkDeviceSize> bufferSize, std::vector<std::vector<VkImageView>>* textureViews, std::vector<std::vector<VkSampler>>* textureSamplers) {

	DescriptorObject descriptorObject;

	VkDescriptorSetLayout discriptorLayout = createDescriptorSetLayout(uniformBufferNum, textureNum, uniformBufferUsages, textureDescriptorType);

	descriptorObject.discriptorLayout = discriptorLayout;
	descriptorObject.uniformBufferNum = uniformBufferNum;
	descriptorObject.textureNum = textureNum;

	for (int i = 0; i < frameSize; i++) {

		if (uniformBuffers->at(0).size() == 1 && i == 1) {
			break;
		}

		//传进来的uniformBuffer都是两个（帧数）一样的为一组，我们传入createSet的要不一样的
		std::vector<VkBuffer> bufferComb;
		for (int j = 0; j < uniformBuffers->size(); j++) {
			bufferComb.push_back(uniformBuffers->at(j)[i]);
		}

		for (int u = 0; u < descriptorSetSize; u++) {
			descriptorObject.descriptorSets.push_back(createDescriptorSet(descriptorObject, uniformBuffers == nullptr ? nullptr : &bufferComb, bufferSize, textureDescriptorType,
				textureViews == nullptr ? nullptr : &(textureViews->at(u)),
				textureSamplers == nullptr ? nullptr : &(textureSamplers->at(u))));
		}
	}


	return descriptorObject;
}

VkDescriptorSetLayout myDescriptor::createDescriptorSetLayout(uint32_t uniformBufferNum, uint32_t textureNum, std::vector<VkShaderStageFlagBits>* uniformBufferUsages, std::vector<VkDescriptorType>* textureDescriptorType) {

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	for (int i = 0; i < uniformBufferNum; i++) {

		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = i;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		//这里的count表示该绑定点有几个描述符，即需要传递一个数组给shader，那么该数组容量多少，但是一般就是一个
		//因为我们的mvp矩阵是通过一个结构体一起传的，所以这里为1
		uboLayoutBinding.descriptorCount = 1;
		//VK_SHADER_STAGE_ALL_GRAPHICS可以放在任意着色器使用
		uboLayoutBinding.stageFlags = uniformBufferUsages->at(i);

		bindings.push_back(uboLayoutBinding);

	}

	for (int i = 0; i < textureNum; i++) {

		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = uniformBufferNum + i;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = textureDescriptorType->at(i);
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings.push_back(samplerLayoutBinding);

	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	VkDescriptorSetLayout layout;
	if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout!");
	}

	return layout;

}

VkDescriptorSet myDescriptor::createDescriptorSet(DescriptorObject descriptorObject, std::vector<VkBuffer>* uniformBuffers, std::vector<VkDeviceSize> bufferSize, std::vector<VkDescriptorType>* textureDescriptorType, std::vector<VkImageView>* textureViews, std::vector<VkSampler>* textureSamplers) {

	VkDescriptorSetLayout layout = descriptorObject.discriptorLayout;
	uint32_t uniformBufferNum = descriptorObject.uniformBufferNum;
	uint32_t textureNum = descriptorObject.textureNum;

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = this->discriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;

	VkDescriptorSet descriptorSet;
	//返回空的描述符集合
	if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, &descriptorSet) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	std::vector<VkWriteDescriptorSet> descriptorWrites;
	descriptorWrites.resize(uniformBufferNum + textureNum);

	std::vector<VkDescriptorBufferInfo> bufferInfos;
	bufferInfos.resize(uniformBufferNum);
	for (int j = 0; j < uniformBufferNum; j++) {

		bufferInfos[j].buffer = uniformBuffers->at(j);
		bufferInfos[j].offset = 0;
		bufferInfos[j].range = bufferSize[j];

		descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[j].dstSet = descriptorSet;
		descriptorWrites[j].dstBinding = j;
		descriptorWrites[j].dstArrayElement = 0;
		descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[j].descriptorCount = 1;
		descriptorWrites[j].pBufferInfo = &bufferInfos[j];

	}

	std::vector<VkDescriptorImageInfo> imageInfos;
	imageInfos.resize(textureNum);
	for (int j = 0; j < textureNum; j++) {

		imageInfos[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfos[j].imageView = textureViews->at(j);
		if (textureDescriptorType->at(j) != VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
			imageInfos[j].sampler = textureSamplers->at(j);
		}

		descriptorWrites[j + uniformBufferNum].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[j + uniformBufferNum].dstSet = descriptorSet;
		descriptorWrites[j + uniformBufferNum].dstBinding = uniformBufferNum + j;
		descriptorWrites[j + uniformBufferNum].dstArrayElement = 0;
		descriptorWrites[j + uniformBufferNum].descriptorType = textureDescriptorType->at(j);
		descriptorWrites[j + uniformBufferNum].descriptorCount = 1;
		descriptorWrites[j + uniformBufferNum].pImageInfo = &imageInfos[j];	//imageInfo.data();

	}

	vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

	return descriptorSet;

}




void myDescriptor::clean() {
	for (int i = 0; i < this->descriptorObjects.size(); i++) {
		vkDestroyDescriptorSetLayout(logicalDevice, this->descriptorObjects[i].discriptorLayout, nullptr);
	}
}