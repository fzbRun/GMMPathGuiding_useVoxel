#include "mySwapChain.h"

mySwapChain::mySwapChain(GLFWwindow* window, VkSurfaceKHR surface, VkDevice logicalDevice, SwapChainSupportDetails swapChainSupport, QueueFamilyIndices indices) {
	this->window = window;
	this->surface = surface;
	this->logicalDevice = logicalDevice;
	createSwapChain(swapChainSupport, indices);
}

void mySwapChain::createSwapChain(SwapChainSupportDetails swapChainSupport, QueueFamilyIndices indices) {

	this->surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);	//主要是surface所展示的纹理的通道数、精度以及色彩空间
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	this->extent = chooseSwapExtent(swapChainSupport.capabilities);

	//如果交换链最小和最大的图像数相等，则确定可支持的图象数就是现在支持的图象数，否则是最小图象数+1
	//如果maxImageCount=0，则表示没有限制（但可能其他地方会限制，无法做到）
	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;	//我要传到哪
	createInfo.minImageCount = imageCount;	//规定了交换缓冲区中纹理的数量，如2就是双缓冲
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;	//纹理数组的z，1就表示2D纹理
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	uint32_t queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

	//图形队列族负责渲染功能，然后交给交换链；交换链再交给展示队列族呈现到surface上
	if (indices.graphicsAndComputeFamily != indices.presentFamily) {
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else {
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		//createInfo.queueFamilyIndexCount = 0;
		//createInfo.pQueueFamilyIndices = nullptr;
	}

	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;	//指明是否需要提前旋转或反转
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
		throw std::runtime_error("failed to create swap chain!");
	}

	//std::vector<VkImage> swapChainImagesTemp;
	vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
	this->swapChainImages.resize(imageCount);
	vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, this->swapChainImages.data());

	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;

	createSwapChainImageViews();

}

//对于交换链中的渲染结果（纹理），我们可以通过imageView作为句柄来操作
void mySwapChain::createSwapChainImageViews() {

	//imageViews和交换链中的image数量相同
	this->swapChainImageViews.resize(this->swapChainImages.size());
	for (size_t i = 0; i < this->swapChainImages.size(); i++) {

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = swapChainImages[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = this->swapChainImageFormat;
		viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &this->swapChainImageViews[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image views!");
		}

	}

}


VkSurfaceFormatKHR mySwapChain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return availableFormat;
		}
	}
	return availableFormats[0];

}

VkPresentModeKHR mySwapChain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
	for (const auto& availablePresentMode : availablePresentModes) {
		//交换链如何呈现画面，比如是直接展示还是双缓冲，等等
		//VK_PRESENT_MODE_IMMEDIATE_KHR 渲染完成后立即展示，每帧呈现后都需要等待下一帧渲染完成才能替换，如果下一帧渲染的时快时慢，就会出现卡顿
		//VK_PRESENT_MODE_FIFO_KHR 多缓冲，渲染完成后提交画面到后面的缓冲，固定时间（显示器刷新时间）后呈现到显示器上。若缓冲区满了，渲染就会停止（阻塞）
		//VK_PRESENT_MODE_FIFO_RELAXED_KHR 渲染完成后提交画面到后面的缓冲，但是如果这一帧渲染的较慢，导致上一帧在刷新后仍存在，则当前帧提交后立刻呈现，那么就可能导致割裂
		//VK_PRESENT_MODE_MAILBOX_KHR 渲染完成后提交画面到后面的缓冲，固定时间（显示器刷新时间）后呈现到显示器上。若缓冲区满了，则画面会替换最后的缓冲区，不会阻塞
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return availablePresentMode;
		}
	}
	return VK_PRESENT_MODE_FIFO_KHR;
}

//一般来说，Vulkan会设置交换链的图像分辨率跟窗口分辨率相等。但是，有些窗口管理器则另辟蹊径，把长和宽的分辨率均设置为uint32_t这个数据类型所支持的最大值
	//这样做的目的，是希望我们可以自己设置一些其他的分辨率，不被窗口分辨率所束缚。就是说我们不想让画面和窗口等大，比如可以将多个渲染结果放到一个窗口
	//如果出现了这种情况，我们需要手动设置一下交换链图像分辨率。
VkExtent2D mySwapChain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {

	if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
		return capabilities.currentExtent;
	}
	else {		//某些窗口管理器允许我们在此处使用不同的值，这通过将currentExtent的宽度和高度设置为最大值来表示，我们不想要这样，可以将之重新设置为窗口大小
		int width, height;
		//查询窗口分辨率
		glfwGetFramebufferSize(window, &width, &height);
		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;

	}
}


