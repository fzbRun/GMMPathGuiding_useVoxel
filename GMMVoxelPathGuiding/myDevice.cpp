#include "myDevice.h"

myDevice::myDevice(VkInstance instance, VkSurfaceKHR surface) {
	this->instance = instance;
	this->surface = surface;
}


//我们需要找到一个（也可以多个）物理设备（主要是显卡，以下都用现在来代替）来帮助我们渲染
//不同的显卡有不同的功能，比如有的显卡可以计算而不能渲染
//这里的功能指的是队列族，是大的功能，而feature是更细的
void myDevice::pickPhysicalDevice() {

	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
	if (deviceCount == 0) {
		throw std::runtime_error("failed to find GPUS with Vulkan support");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());	//不是按优先级排的

	//按显卡能力进行排序，妈的，不排序默认用的是intel的集成显卡，我的3070只能吃灰
	std::multimap<int, VkPhysicalDevice> candidates;
	for (const auto& device : devices) {
		int score = rateDeviceSuitability(surface, device);
		candidates.insert(std::make_pair(score, device));
	}

	if (candidates.rbegin()->first > 0) {
		this->physicalDevice = candidates.rbegin()->second;
		this->msaaSamples = getMaxUsableSampleCount();
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		std::cout << deviceProperties.deviceName << std::endl;

	}
	else {
		throw std::runtime_error("failed to find a suitable GPU!");
	}

}

//操作系统中就有逻辑设备和物理设备之分，这是为了去除掉物理设备的差异性，将麻烦交给操作系统而不是用户
//但是这里好像是不一样的，一块物理设备对应多块逻辑设备，这是因为这里的物理和逻辑的概念不同
//操作系统中物理和逻辑是实际与抽象，而这里的物理是功能的集合，而逻辑是对功能的划分的子集
//即逻辑设备是某一些功能实现的句柄，而物理设备可以有多个功能，将这些功能划分为一个一个的逻辑设备（子集）
//实际上逻辑设备还可以再次划分，即队列Queue才是一次实现的具体句柄
void myDevice::createLogicalDevice(bool enableValidationLayers, std::vector<const char*> validationLayers) {

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { queueFamilyIndices.graphicsAndComputeFamily.value(), queueFamilyIndices.presentFamily.value() };

	//我们选取的物理设备拥有一定的队列族（功能），但没有创建，现在需要将之创建出来
	//这里的物理设备对应一个逻辑设备，而一个逻辑设备对应两个队列
	float queuePriority = 1.0f;
	for (uint32_t queueFamily : uniqueQueueFamilies) {
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	//deviceFeatures.sampleRateShading = VK_TRUE;

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();

	// 为设备指定和实例相同的校验层
	// 实际上，新版本的Vulkan已经不再区分二者的校验层，
	// 会自动忽略设备中关于校验层的字段。但是设置一下的话，可以旧版本兼容
	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}

	if (vkCreateDevice(this->physicalDevice, &createInfo, nullptr, &this->logicalDevice) != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device!");
	}

	vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &this->graphicsQueue);
	vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &this->computeQueue);
	vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.presentFamily.value(), 0, &this->presentQueue);

}

int myDevice::rateDeviceSuitability(VkSurfaceKHR surface, VkPhysicalDevice device) {

	//VkPhysicalDeviceProperties deviceProperties;
	//VkPhysicalDeviceFeatures deviceFeatures;
	//vkGetPhysicalDeviceProperties(device, &deviceProperties);	//设备信息
	//vkGetPhysicalDeviceFeatures(device, &deviceFeatures);		//设备功能
		
	this->queueFamilyIndices = findQueueFamilies(surface, device);
	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(device, &deviceProperties);
	//std::cout << deviceProperties.limits.maxPerStageDescriptorStorageImages << std::endl;

	//检查设备是否支持交换链扩展
	bool extensionsSupport = checkDeviceExtensionSupport(device);
	bool swapChainAdequate = false;
	if (extensionsSupport) {
		//判断物理设备的图像和展示功能是否支持
		this->swapChainSupportDetails = querySwapChainSupport(surface, device);
		swapChainAdequate = !swapChainSupportDetails.formats.empty() && !swapChainSupportDetails.presentModes.empty();
	}

	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
	if (queueFamilyIndices.isComplete() && extensionsSupport && swapChainAdequate && supportedFeatures.samplerAnisotropy) {
		int score = 0;
		if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			score += 1000;
		}
		score += deviceProperties.limits.maxImageDimension2D;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		if (!deviceFeatures.geometryShader) {	//我可以只要可以支持几何着色器的显卡
			return -1;
		}
		return score;
	}

	return -1;

}

SwapChainSupportDetails myDevice::querySwapChainSupport(VkSurfaceKHR surface, VkPhysicalDevice device) {

	SwapChainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
	if (formatCount != 0) {
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;

}

QueueFamilyIndices myDevice::findQueueFamilies(VkSurfaceKHR surface, VkPhysicalDevice device) {

	QueueFamilyIndices indices;
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());	//获得队列系列的详细信息

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		//这里的图像队列是不是说显卡有专门对渲染的优化
		//因为VK_QUEUE_COMPUTE_BIT是说显卡可以通用计算(计算着色器)，而渲染实际上也是一种计算，那么分开两者的原因应该就是是否有专门优化
		//注意支持VK_QUEUE_GRAPHICS_BIT与VK_QUEUE_COMPUTE_BIT的设备默认支持VK_QUEUE_TRANSFER_BIT（用来传递缓冲区数据）
		if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			indices.graphicsAndComputeFamily = i;
		}

		VkBool32 presentSupport = false;
		//判断i族群是否也支持展示，这里展示的意思是能否将GPU渲染出来的画面传到显示器上，有些显卡可能并未连接到显示器
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

		if (presentSupport) {
			indices.presentFamily = i;
		}

		if (indices.isComplete()) {
			break;
		}
		i++;
	}

	return indices;

}

bool myDevice::checkDeviceExtensionSupport(VkPhysicalDevice device) {

	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	//若requiredExtensions空了，说明需要的拓展全有
	return requiredExtensions.empty();

}

VkSampleCountFlagBits myDevice::getMaxUsableSampleCount() {
	VkPhysicalDeviceProperties physicalDeviceProperties;
	vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

	VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
	if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
	if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
	if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
	if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
	if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
	if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

	return VK_SAMPLE_COUNT_1_BIT;
}

/*
uint32_t myDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

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
*/

void myDevice::clean() {
	vkDestroyDevice(logicalDevice, nullptr);
}
