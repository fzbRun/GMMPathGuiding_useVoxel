#include <chrono>
#include<stdexcept>
#include<functional>
#include<cstdlib>
#include<cstdint>
#include<limits>
#include<fstream>
#include <random>

#include "structSet.h"
#include "myDevice.h"
#include "myBuffer.h"
#include "myImage.h"
#include "mySwapChain.h"
#include "myModel.h"
#include "myCamera.h"
#include "myDescriptor.h"
#include "myScene.h"
#include "myRay.h"

const uint32_t WIDTH = 512;
const uint32_t HEIGHT = 512;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//如果不调试，则关闭校验层
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebygMessenger) {
	//由于是扩展函数，所以需要通过vkGetInstanceProcAddr获得该函数指针
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebygMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

myCamera camera(glm::vec3(0.0f, 1.0f, 4.0f));
float lastTime = 0.0f;
float deltaTime = 0.0f;
bool firstMouse = true;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

// 创建一个随机数生成器  
std::random_device rd;  // 用于获取随机数种子  
std::mt19937 gen(rd()); // 以rd()作为种子的Mersenne Twister生成器  
// 创建一个0到99之间均匀分布的整数分布  
std::uniform_real_distribution<> dis(0, 99);

class pathTracing {

public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:

	GLFWwindow* window;		//窗口

	VkInstance instance;	//vulkan实例
	VkDebugUtilsMessengerEXT debugMessenger;	//消息传递者
	VkSurfaceKHR surface;

	//Device
	std::unique_ptr<myDevice> my_device;

	//SwapChain
	std::unique_ptr<mySwapChain> my_swapChain;

	std::unique_ptr<myBuffer> my_buffer;

	std::unique_ptr<myDescriptor> my_descriptor;

	VkRenderPass shadowRenderPass;
	VkRenderPass finalRenderPass;
	VkPipelineLayout computePipelineLayout;
	VkPipeline computePipeline;
	VkPipelineLayout graphicsPipelineLayout;
	VkPipeline graphicsPipeline;
	VkPipeline shadowGraphicsPipeline;
	VkPipelineLayout shadowGraphicsPipelineLayout;

	std::unique_ptr<myImage> pathTracingResult;
	std::unique_ptr<myImage> shadowMap;

	std::unique_ptr<myModel> my_model;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<ComputeInputMesh> computeInputMeshs;
	std::vector<uint32_t> meshIndexInIndices;
	std::vector<ComputeVertex> computeVertices;

	std::unique_ptr<myScene> my_scene;

	std::vector<VkSemaphore> shadowRenderFinishedSemaphores;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkSemaphore> computeFinishedSemaphores;
	std::vector<VkFence> shadowInFlightFences;
	std::vector<VkFence> inFlightFences;
	std::vector<VkFence> computeInFlightFences;
	uint32_t currentFrame = 0;
	int frameNum = 0;

	bool framebufferResized = false;

	VkPipelineLayout gmmLightTrainPipelineLayout;
	VkPipeline gmmLightTrainPipeline;
	float GMMVoxelSize = 0.25f;
	glm::ivec3 GMMVoxelNum;
	uint32_t GMMVoxelNumSum;
	uint32_t GMMMaxSize;
	std::vector<GMMPara> gmmParas;
	std::vector<PhotonTracingResult> photonTracingResults;
	uint32_t photonNumPerFrame = 32 * 32;	//每帧发射若干个光子用于训练

	void initWindow() {

		glfwInit();

		//阻止GLFW自动创建OpenGL上下文
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		//是否禁止改变窗口大小
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		//glfwSetFramebufferSizeCallback函数在回调时，需要为我们设置framebufferResized，但他不知道我是谁
		//所以通过对window设置我是谁，从而让回调函数知道我是谁
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetScrollCallback(window, scroll_callback);

	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {

		auto app = reinterpret_cast<pathTracing*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;

	}

	void initVulkan() {

		createInstance();
		setupDebugMessenger();
		createSurface();
		createMyDevice();
		createMySwapChain();
		createMyBuffer();
		createTargetTextureResources();
		loadModel();
		createBuffers();
		createShadowRenderPass();
		createFinalRenderPass();
		createFramebuffers();
		createMyDescriptor();
		createGraphicsPipeline();
		createComputePipeline();
		createSyncObjects();

	}

	void createInstance() {

		//检测layer
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		//扩展就是Vulkan本身没有实现，但被程序员封装后的功能函数，如跨平台的各种函数，把它当成普通函数即可，别被名字唬到了
		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();	//将扩展的具体信息的指针存储在该结构体中

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();	//将校验层的具体信息的指针存储在该结构体中

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;

		}
		else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}


		//VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}

	}

	bool checkValidationLayerSupport() {

		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);	//返回可用的层数
		std::vector<VkLayerProperties> availableLayers(layerCount);	//VkLayerProperties是一个结构体，记录层的名字、描述等
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {

			bool layerFound = false;
			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}

		}

		return true;
	}

	std::vector<const char*> getRequiredExtensions() {

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);	//得到glfw所需的扩展数
		//参数1是指针起始位置，参数2是指针终止位置
		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);	//这个扩展是为了打印校验层反映的错误，所以需要知道是否需要校验层
		}

		return extensions;
	}

	void setupDebugMessenger() {

		if (!enableValidationLayers)
			return;
		VkDebugUtilsMessengerCreateInfoEXT  createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		//通过func的构造函数给debugMessenger赋值
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}

	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface");
		}
	}

	void createMyDevice() {
		my_device = std::make_unique<myDevice>(instance, surface);
		my_device->pickPhysicalDevice();
		my_device->createLogicalDevice(enableValidationLayers, validationLayers);
	}

	void createMySwapChain() {
		my_swapChain = std::make_unique<mySwapChain>(window, surface, my_device->logicalDevice, my_device->swapChainSupportDetails, my_device->queueFamilyIndices);
	}

	void createMyBuffer() {
		my_buffer = std::make_unique<myBuffer>();
		my_buffer->createCommandPool(my_device->logicalDevice, my_device->queueFamilyIndices);
		my_buffer->createCommandBuffers(my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT);
	}

	void createTargetTextureResources() {
		//VK_FORMAT_R8G8B8A8_SRGB 不支持VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT，所以不支持VK_IMAGE_USAGE_STORAGE_BIT
		pathTracingResult = std::make_unique<myImage>(my_device->physicalDevice, my_device->logicalDevice, my_swapChain->swapChainExtent.width, my_swapChain->swapChainExtent.height,
			1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
		pathTracingResult->transitionImageLayout(my_device->computeQueue, my_buffer->commandPool, pathTracingResult->image, VK_FORMAT_R8G8B8A8_UINT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1);

		shadowMap = std::make_unique<myImage>(my_device->physicalDevice, my_device->logicalDevice, my_swapChain->swapChainExtent.width, my_swapChain->swapChainExtent.height,
			1, VK_SAMPLE_COUNT_1_BIT, myImage::findDepthFormat(my_device->physicalDevice), VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
		//shadowMap->transitionImageLayout(my_device->computeQueue, my_buffer->commandPool, shadowMap->image, myImage::findDepthFormat(my_device->physicalDevice), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 1);
	}

	void loadModel() {

		//使用assimp
		uint32_t index = 0;
		//my_model = std::make_unique<myModel>("models/CornellBox-Original.obj");	//给绝对路径读不到，给相对路径能读到
		my_model = std::make_unique<myModel>("models/bear_box.obj");
		if (my_model->meshs.size() == 0) {
			throw std::runtime_error("failed to create model!");
		}

		//将所有mesh的顶点合并
		for (uint32_t i = 0; i < my_model->meshs.size(); i++) {

			//this->materials.push_back(my_model->meshs[i].material);

			this->vertices.insert(this->vertices.end(), my_model->meshs[i].vertices.begin(), my_model->meshs[i].vertices.end());

			//因为assimp是按一个mesh一个mesh的存，所以每个indices都是相对一个mesh的，当我们将每个mesh的顶点存到一起时，indices就会出错，我们需要增加索引
			for (uint32_t j = 0; j < my_model->meshs[i].indices.size(); j++) {
				my_model->meshs[i].indices[j] += index;
			}
			meshIndexInIndices.push_back(this->indices.size());
			index += my_model->meshs[i].vertices.size();
			this->indices.insert(this->indices.end(), my_model->meshs[i].indices.begin(), my_model->meshs[i].indices.end());
		}
		meshIndexInIndices.push_back(this->indices.size());
		my_scene = std::make_unique<myScene>(&my_model->meshs);

	}

	//简单一点，静态场景
	void createBuffers() {

		//bvhArray
		VkDeviceSize bvhArraySize = sizeof(BvhArrayNode) * my_scene->bvhArray.size();
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, bvhArraySize, &(my_scene->bvhArray));

		//Mesh
		//其实只要顶点Position就好了，法线可以通过面法线得到，texCoord没啥用，但是放着吧，以后用到texture了就有用了
		for (int i = 0; i < this->vertices.size(); i++) {
			ComputeVertex computeVertex;
			computeVertex.pos = glm::vec4(this->vertices[i].pos, 1.0f);
			computeVertex.normal = glm::vec4(this->vertices[i].normal, 1.0f);
			computeVertices.push_back(computeVertex);
		}
		VkDeviceSize verticesSize = sizeof(ComputeVertex) * computeVertices.size();
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, verticesSize, &computeVertices);

		VkDeviceSize indicesSize = sizeof(uint32_t) * this->indices.size();
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, indicesSize, &indices);

		VkDeviceSize meshSize = sizeof(ComputeInputMesh) * my_model->meshs.size();
		for (int i = 0; i < my_model->meshs.size(); i++) {
			ComputeInputMesh computeInputMesh;
			computeInputMesh.AABB = my_model->meshs[i].AABB;
			computeInputMesh.indexInIndicesArray = glm::ivec2(meshIndexInIndices[i], meshIndexInIndices[i + 1]);
			computeInputMesh.material = my_model->meshs[i].material;
			computeInputMeshs.push_back(computeInputMesh);
		}
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, meshSize, &computeInputMeshs);

		//uniform
		//光源MVP
		my_buffer->createUniformBuffers(my_device->physicalDevice, my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT, sizeof(UniformLightBufferObject), true);
		UniformLightBufferObject lightUniform;
		lightUniform.model = glm::mat4(1.0f);
		lightUniform.view = glm::lookAt(glm::vec3(0.0f, 1.94f, -0.03f), glm::vec3(0.0f, 1.95f, -0.03f) + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		lightUniform.proj = glm::perspective(glm::radians(90.0f), my_swapChain->swapChainExtent.width / (float)my_swapChain->swapChainExtent.height, 0.1f, 100.0f);
		lightUniform.proj[1][1] *= -1;
		lightUniform.lightPos_strength = glm::vec4(-0.24f, 1.95f, -0.22f, 50.0f);
		lightUniform.normal = glm::vec4(0.0f, -1.0f, 0.0f, dis(gen));
		lightUniform.size = glm::vec4(0.47f, 0.0f, 0.38f, 0.0f);
		memcpy(my_buffer->uniformBuffersMappedsStatic[0], &lightUniform, sizeof(UniformLightBufferObject));

		//相机MVP
		my_buffer->createUniformBuffers(my_device->physicalDevice, my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT, sizeof(UniformBufferObject), false);

		my_buffer->createUniformBuffers(my_device->physicalDevice, my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT, sizeof(GMMConstant), true);
		GMMConstant gmmConstant;
		gmmConstant.voxelSize = GMMVoxelSize;

		AABBBox aabb = my_scene->bvhArray[0].AABB;
		gmmConstant.voxelStartPos = glm::vec4(aabb.leftX, aabb.leftY, aabb.leftZ, 0.0f);
		GMMVoxelNum = glm::ivec3(std::ceil((aabb.rightX - aabb.leftX) / GMMVoxelSize), std::ceil((aabb.rightY - aabb.leftY) / GMMVoxelSize), std::ceil((aabb.rightZ - aabb.leftZ) / GMMVoxelSize));
		gmmConstant.voxelNum = glm::ivec4(GMMVoxelNum, 0.0f);
		gmmConstant.photonTracingNum = photonNumPerFrame;
		memcpy(my_buffer->uniformBuffersMappedsStatic[1], &gmmConstant, sizeof(GMMConstant));

		//GMM参数的存储buffer
		GMMVoxelNumSum = GMMVoxelNum.x * GMMVoxelNum.y * GMMVoxelNum.z;

		for (int i = 0; i < photonNumPerFrame; i++) {
			photonTracingResults.push_back(PhotonTracingResult());
		}
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, sizeof(PhotonTracingResult) * photonNumPerFrame, &photonTracingResults);

		uint32_t gmmPerVoxel = 1;
		for (int i = 0; i < GMMVoxelNumSum * gmmPerVoxel; i++) {
			gmmParas.push_back(GMMPara());
		}
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, sizeof(GMMPara) * GMMVoxelNumSum * gmmPerVoxel, &gmmParas);


	}

	void createShadowRenderPass() {

		//shadowMap
		VkAttachmentDescription shadowMapAttachment{};
		shadowMapAttachment.format = shadowMap->findDepthFormat(my_device->physicalDevice);
		shadowMapAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		shadowMapAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		shadowMapAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		shadowMapAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		shadowMapAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		shadowMapAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		shadowMapAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

		VkAttachmentReference shadowMapAttachmentResolveRef{};
		shadowMapAttachmentResolveRef.attachment = 0;
		shadowMapAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription shadow_subpass{};
		shadow_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		shadow_subpass.colorAttachmentCount = 0;
		shadow_subpass.pDepthStencilAttachment = &shadowMapAttachmentResolveRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &shadowMapAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &shadow_subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(my_device->logicalDevice, &renderPassInfo, nullptr, &shadowRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

	}

	void createFinalRenderPass() {

		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = my_swapChain->swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;;

		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 0;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription final_subpass{};
		final_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		final_subpass.colorAttachmentCount = 1;
		final_subpass.pColorAttachments = &colorAttachmentResolveRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachmentResolve;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &final_subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(my_device->logicalDevice, &renderPassInfo, nullptr, &finalRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

	}

	void createFramebuffers() {
		std::vector<VkImageView> imageViews;
		my_buffer->createFramebuffers(my_swapChain->swapChainImageViews.size(), imageViews, my_swapChain->extent, imageViews, shadowMap->imageView, shadowRenderPass, my_device->logicalDevice);
		my_buffer->createFramebuffers(my_swapChain->swapChainImageViews.size(), my_swapChain->swapChainImageViews, my_swapChain->extent, imageViews, nullptr, finalRenderPass, my_device->logicalDevice);
	}

	void createMyDescriptor() {

		my_descriptor = std::make_unique<myDescriptor>(my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT);

		uint32_t uniformBufferNumAllLayout = 3;
		uint32_t storageBufferNumAllLayout = 7;
		std::vector<uint32_t> textureNumAllLayout = { 1, 2 };
		std::vector<VkDescriptorType> types = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER };
		my_descriptor->createDescriptorPool(uniformBufferNumAllLayout, storageBufferNumAllLayout, types, textureNumAllLayout);

		//创造光源uniformDescriptorObject
		std::vector<VkShaderStageFlagBits> usages = { VK_SHADER_STAGE_ALL };
		std::vector<VkBuffer> uniformBuffers = { my_buffer->uniformBuffersStatic[0] };
		std::vector<std::vector<VkBuffer>> uniformBuffersAllSet = { uniformBuffers };
		std::vector<VkDeviceSize> bufferSize = { sizeof(UniformLightBufferObject) };
		std::vector<uint32_t> uniformDescriptorCount = { 1 };
		my_descriptor->descriptorObjects.push_back(my_descriptor->createDescriptorObject(1, 0, &usages, nullptr, 1, &uniformBuffersAllSet, bufferSize, nullptr, nullptr));

		usages = { VK_SHADER_STAGE_COMPUTE_BIT };
		// 创造相机uniformDescriptorObject
		uniformBuffers = { my_buffer->uniformBuffers[0] };
		uniformBuffersAllSet = { uniformBuffers };
		bufferSize = { sizeof(UniformBufferObject) };
		my_descriptor->descriptorObjects.push_back(my_descriptor->createDescriptorObject(1, 0, &usages, nullptr, 1, &uniformBuffersAllSet, bufferSize, nullptr, nullptr));

		//创建computeDescriptorObject
		DescriptorObject computeDescriptorObject{};
		VkDescriptorSetLayout computeDescriptorSetLayout;
		std::array<VkDescriptorSetLayoutBinding, 9> layoutBindings{};
		layoutBindings[0].binding = 0;
		layoutBindings[0].descriptorCount = 1;
		layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[0].pImmutableSamplers = nullptr;
		layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		layoutBindings[1].binding = 1;
		layoutBindings[1].descriptorCount = 1;
		layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[1].pImmutableSamplers = nullptr;
		layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		layoutBindings[2].binding = 2;
		layoutBindings[2].descriptorCount = 1;
		layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[2].pImmutableSamplers = nullptr;
		layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		layoutBindings[3].binding = 3;
		layoutBindings[3].descriptorCount = 1;
		layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[3].pImmutableSamplers = nullptr;
		layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		layoutBindings[4].binding = 4;
		layoutBindings[4].descriptorCount = 1;
		layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		layoutBindings[4].pImmutableSamplers = nullptr;
		layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		layoutBindings[5].binding = 5;
		layoutBindings[5].descriptorCount = 1;
		layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		layoutBindings[5].pImmutableSamplers = nullptr;
		layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		//GMMPConstant
		layoutBindings[6].binding = 6;
		layoutBindings[6].descriptorCount = 1;
		layoutBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings[6].pImmutableSamplers = nullptr;
		layoutBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		//PhotonTracingResults
		layoutBindings[7].binding = 7;
		layoutBindings[7].descriptorCount = 1;
		layoutBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[7].pImmutableSamplers = nullptr;
		layoutBindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		//GMMParaBuffer
		layoutBindings[8].binding = 8;
		layoutBindings[8].descriptorCount = 1;
		layoutBindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[8].pImmutableSamplers = nullptr;
		layoutBindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = layoutBindings.size();
		layoutInfo.pBindings = layoutBindings.data();
		if (vkCreateDescriptorSetLayout(my_device->logicalDevice, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute descriptor set layout!");
		}

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = my_descriptor->discriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &computeDescriptorSetLayout;

		VkDescriptorSet computeDescriptorSet;
		if (vkAllocateDescriptorSets(my_device->logicalDevice, &allocInfo, &computeDescriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		std::array<VkWriteDescriptorSet, 9> descriptorWrites{};

		VkDescriptorBufferInfo bvhArrayNodeBufferInfo{};
		bvhArrayNodeBufferInfo.buffer = my_buffer->buffersStatic[0];
		bvhArrayNodeBufferInfo.offset = 0;
		bvhArrayNodeBufferInfo.range = sizeof(BvhArrayNode) * my_scene->bvhArray.size();
		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = computeDescriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bvhArrayNodeBufferInfo;

		VkDescriptorBufferInfo verticesBufferInfo{};
		verticesBufferInfo.buffer = my_buffer->buffersStatic[1];
		verticesBufferInfo.offset = 0;
		verticesBufferInfo.range = sizeof(ComputeVertex) * this->vertices.size();
		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = computeDescriptorSet;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &verticesBufferInfo;

		VkDescriptorBufferInfo indicesBufferInfo{};
		indicesBufferInfo.buffer = my_buffer->buffersStatic[2];
		indicesBufferInfo.offset = 0;
		indicesBufferInfo.range = sizeof(uint32_t) * this->indices.size();
		descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[2].dstSet = computeDescriptorSet;
		descriptorWrites[2].dstBinding = 2;
		descriptorWrites[2].dstArrayElement = 0;
		descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[2].descriptorCount = 1;
		descriptorWrites[2].pBufferInfo = &indicesBufferInfo;

		VkDescriptorBufferInfo meshBufferInfo{};
		meshBufferInfo.buffer = my_buffer->buffersStatic[3];
		meshBufferInfo.offset = 0;
		meshBufferInfo.range = sizeof(ComputeInputMesh) * this->computeInputMeshs.size();
		descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[3].dstSet = computeDescriptorSet;
		descriptorWrites[3].dstBinding = 3;
		descriptorWrites[3].dstArrayElement = 0;
		descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[3].descriptorCount = 1;
		descriptorWrites[3].pBufferInfo = &meshBufferInfo;

		VkDescriptorImageInfo computeShadowImageInfos;
		computeShadowImageInfos.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		computeShadowImageInfos.imageView = shadowMap->imageView;
		computeShadowImageInfos.sampler = shadowMap->textureSampler;
		descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[4].dstSet = computeDescriptorSet;
		descriptorWrites[4].dstBinding = 4;
		descriptorWrites[4].dstArrayElement = 0;
		descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrites[4].descriptorCount = 1;
		descriptorWrites[4].pImageInfo = &computeShadowImageInfos;

		VkDescriptorImageInfo computeImageInfos;
		computeImageInfos.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		computeImageInfos.imageView = pathTracingResult->imageView;
		descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[5].dstSet = computeDescriptorSet;
		descriptorWrites[5].dstBinding = 5;
		descriptorWrites[5].dstArrayElement = 0;
		descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorWrites[5].descriptorCount = 1;
		descriptorWrites[5].pImageInfo = &computeImageInfos;

		//GMMConstant
		VkDescriptorBufferInfo GMMConstantBufferInfo{};
		GMMConstantBufferInfo.buffer = my_buffer->uniformBuffersStatic[1];
		GMMConstantBufferInfo.offset = 0;
		GMMConstantBufferInfo.range = sizeof(GMMConstant);
		descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[6].dstSet = computeDescriptorSet;
		descriptorWrites[6].dstBinding = 6;
		descriptorWrites[6].dstArrayElement = 0;
		descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[6].descriptorCount = 1;
		descriptorWrites[6].pBufferInfo = &GMMConstantBufferInfo;

		//PhotonTracingResults
		VkDescriptorBufferInfo PhotonTracingResultsBufferInfo{};
		PhotonTracingResultsBufferInfo.buffer = my_buffer->buffersStatic[4];
		PhotonTracingResultsBufferInfo.offset = 0;
		PhotonTracingResultsBufferInfo.range = sizeof(PhotonTracingResult) * photonNumPerFrame;
		descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[7].dstSet = computeDescriptorSet;
		descriptorWrites[7].dstBinding = 7;
		descriptorWrites[7].dstArrayElement = 0;
		descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[7].descriptorCount = 1;
		descriptorWrites[7].pBufferInfo = &PhotonTracingResultsBufferInfo;

		//GMMParaBuffer
		VkDescriptorBufferInfo GMMParaBufferInfo{};
		GMMParaBufferInfo.buffer = my_buffer->buffersStatic[5];
		GMMParaBufferInfo.offset = 0;
		GMMParaBufferInfo.range = sizeof(GMMPara) * GMMVoxelNumSum * 3;
		descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[8].dstSet = computeDescriptorSet;
		descriptorWrites[8].dstBinding = 8;
		descriptorWrites[8].dstArrayElement = 0;
		descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[8].descriptorCount = 1;
		descriptorWrites[8].pBufferInfo = &GMMParaBufferInfo;

		vkUpdateDescriptorSets(my_device->logicalDevice, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);

		computeDescriptorObject.discriptorLayout = computeDescriptorSetLayout;
		computeDescriptorObject.descriptorSets.push_back(computeDescriptorSet);

		my_descriptor->descriptorObjects.push_back(computeDescriptorObject);

		//创建fragDescriptorObject
		DescriptorObject fragDescriptorObject{};
		VkDescriptorSetLayout fragDescriptorSetLayout{};
		VkDescriptorSetLayoutBinding fraglayoutBindings{};
		fraglayoutBindings.binding = 0;
		fraglayoutBindings.descriptorCount = 1;
		fraglayoutBindings.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		fraglayoutBindings.pImmutableSamplers = nullptr;
		fraglayoutBindings.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo fragLayoutInfo{};
		fragLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		fragLayoutInfo.bindingCount = 1;
		fragLayoutInfo.pBindings = &fraglayoutBindings;
		if (vkCreateDescriptorSetLayout(my_device->logicalDevice, &fragLayoutInfo, nullptr, &fragDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create frag descriptor set layout!");
		}

		VkDescriptorSetAllocateInfo fragAllocInfo{};
		fragAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		fragAllocInfo.descriptorPool = my_descriptor->discriptorPool;
		fragAllocInfo.descriptorSetCount = 1;
		fragAllocInfo.pSetLayouts = &fragDescriptorSetLayout;

		VkDescriptorSet fragDescriptorSet;
		if (vkAllocateDescriptorSets(my_device->logicalDevice, &fragAllocInfo, &fragDescriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		VkWriteDescriptorSet descriptorWrite{};
		VkDescriptorImageInfo fragImageInfos{};
		fragImageInfos.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		fragImageInfos.imageView = pathTracingResult->imageView;
		fragImageInfos.sampler = pathTracingResult->textureSampler;

		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = fragDescriptorSet;
		descriptorWrite.dstBinding = 0;
		descriptorWrite.dstArrayElement = 0;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pImageInfo = &fragImageInfos;

		vkUpdateDescriptorSets(my_device->logicalDevice, 1, &descriptorWrite, 0, nullptr);

		fragDescriptorObject.discriptorLayout = fragDescriptorSetLayout;
		fragDescriptorObject.descriptorSets.push_back(fragDescriptorSet);

		my_descriptor->descriptorObjects.push_back(fragDescriptorObject);

	}

	void createGraphicsPipeline() {

		auto shadowVertShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/rayTracing/GMMVoxelPathGuiding/GMMVoxelPathGuiding/shaders/shadowVert.spv");
		auto shadowFragShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/rayTracing/GMMVoxelPathGuiding/GMMVoxelPathGuiding/shaders/shadowFrag.spv");

		VkShaderModule shadowVertShaderModule = createShaderModule(shadowVertShaderCode);
		VkShaderModule shadowFragShaderModule = createShaderModule(shadowFragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = shadowVertShaderModule;
		vertShaderStageInfo.pName = "main";
		//允许指定着色器常量的值，比起在渲染时指定变量配置更加有效，因为可以通过编译器优化（没搞懂）
		vertShaderStageInfo.pSpecializationInfo = nullptr;

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = shadowFragShaderModule;
		fragShaderStageInfo.pName = "main";
		fragShaderStageInfo.pSpecializationInfo = nullptr;

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		//VAO
		VkPipelineVertexInputStateCreateInfo shadowVertexInputInfo{};
		shadowVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		auto bindingDescription = ComputeVertex::getBindingDescription();
		auto attributeDescriptions = ComputeVertex::getAttributeDescriptions();
		shadowVertexInputInfo.vertexBindingDescriptionCount = 1;
		shadowVertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		shadowVertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		shadowVertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		//设置渲染图元方式
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		//设置视口
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		//设置光栅化器，主要是深度测试等的开关、面剔除等
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		//多采样
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;// .2f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optiona
		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; // Optional
		colorBlending.blendConstants[1] = 0.0f; // Optional
		colorBlending.blendConstants[2] = 0.0f; // Optional
		colorBlending.blendConstants[3] = 0.0f; // Optional

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f; // Optional
		depthStencil.maxDepthBounds = 1.0f; // Optional
		depthStencil.stencilTestEnable = VK_FALSE;
		depthStencil.front = {}; // Optional
		depthStencil.back = {}; // Optional

		//一般渲染管道状态都是固定的，不能渲染循环中修改，但是某些状态可以，如视口，长宽和混合常数
		//同样通过宏来确定可动态修改的状态
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		//pipeline布局
		VkPipelineLayoutCreateInfo uniformPipelineLayoutInfo{};
		uniformPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		uniformPipelineLayoutInfo.setLayoutCount = 1;
		uniformPipelineLayoutInfo.pSetLayouts = &my_descriptor->descriptorObjects[0].discriptorLayout;
		uniformPipelineLayoutInfo.pushConstantRangeCount = 0;
		uniformPipelineLayoutInfo.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(my_device->logicalDevice, &uniformPipelineLayoutInfo, nullptr, &shadowGraphicsPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;	//顶点和片元两个着色器
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &shadowVertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = shadowGraphicsPipelineLayout;
		pipelineInfo.renderPass = shadowRenderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = 0;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(my_device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowGraphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(my_device->logicalDevice, shadowVertShaderModule, nullptr);
		vkDestroyShaderModule(my_device->logicalDevice, shadowFragShaderModule, nullptr);

		auto sceneVertShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/rayTracing/GMMVoxelPathGuiding/GMMVoxelPathGuiding/shaders/sceneVert.spv");
		auto sceneFragShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/rayTracing/GMMVoxelPathGuiding/GMMVoxelPathGuiding/shaders/sceneFrag.spv");

		VkShaderModule sceneVertShaderModule = createShaderModule(sceneVertShaderCode);
		VkShaderModule sceneFragShaderModule = createShaderModule(sceneFragShaderCode);

		vertShaderStageInfo.module = sceneVertShaderModule;
		fragShaderStageInfo.module = sceneFragShaderModule;

		shaderStages[0] = vertShaderStageInfo;
		shaderStages[1] = fragShaderStageInfo;

		//VAO
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr;

		//pipeline布局
		uniformPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		uniformPipelineLayoutInfo.setLayoutCount = 1;
		uniformPipelineLayoutInfo.pSetLayouts = &(my_descriptor->descriptorObjects[3].discriptorLayout);
		uniformPipelineLayoutInfo.pushConstantRangeCount = 0;
		uniformPipelineLayoutInfo.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(my_device->logicalDevice, &uniformPipelineLayoutInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		rasterizer.cullMode = VK_CULL_MODE_NONE;

		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.layout = graphicsPipelineLayout;
		pipelineInfo.renderPass = finalRenderPass;
		pipelineInfo.subpass = 0;

		if (vkCreateGraphicsPipelines(my_device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(my_device->logicalDevice, sceneVertShaderModule, nullptr);
		vkDestroyShaderModule(my_device->logicalDevice, sceneFragShaderModule, nullptr);

	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(my_device->logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;

	}

	static std::vector<char> readFile(const std::string& filename) {

		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();
		return buffer;

	}

	void createComputePipeline() {
		auto computeShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/rayTracing/GMMVoxelPathGuiding/GMMVoxelPathGuiding/shaders/GMMPathGuidingVoxel.spv");
		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

		VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		computeShaderStageInfo.module = computeShaderModule;
		computeShaderStageInfo.pName = "main";

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 3;
		std::array<VkDescriptorSetLayout, 3> computeDescriptorSetLayouts = { my_descriptor->descriptorObjects[0].discriptorLayout, my_descriptor->descriptorObjects[1].discriptorLayout, my_descriptor->descriptorObjects[2].discriptorLayout };
		pipelineLayoutInfo.pSetLayouts = computeDescriptorSetLayouts.data();
		if (vkCreatePipelineLayout(my_device->logicalDevice, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.layout = computePipelineLayout;
		pipelineInfo.stage = computeShaderStageInfo;
		if (vkCreateComputePipelines(my_device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(my_device->logicalDevice, computeShaderModule, nullptr);

	}

	void createSyncObjects() {

		//信号量主要用于Queue之间的同步
		shadowRenderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		//用于CPU和GPU之间的同步
		shadowInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		//第一帧可以直接获得信号，而不会阻塞
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		//每一帧都需要一定的信号量和栏栅
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (
				vkCreateSemaphore(my_device->logicalDevice, &semaphoreInfo, nullptr, &shadowRenderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(my_device->logicalDevice, &fenceInfo, nullptr, &shadowInFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create shadow graphics semaphores!");
			}
			if (vkCreateSemaphore(my_device->logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(my_device->logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(my_device->logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create graphics semaphores!");
			}
			if (vkCreateSemaphore(my_device->logicalDevice, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(my_device->logicalDevice, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create compute semaphores!");
			}
		}


	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			processInput(window);
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(my_device->logicalDevice);

	}

	void processInput(GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			camera.ProcessKeyboard(FORWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			camera.ProcessKeyboard(BACKWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			camera.ProcessKeyboard(LEFT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			camera.ProcessKeyboard(RIGHT, deltaTime);
	}

	//从这个函数中看出来，fence主要在循环中进行阻塞，而semaphore主要在每个循环中的各个阶段进行阻塞，实现串行
	void drawFrame() {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		//渲染阴影图
		vkWaitForFences(my_device->logicalDevice, 1, &shadowInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(my_device->logicalDevice, my_swapChain->swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		//VK_ERROR_OUT_OF_DATE_KHR：交换链与表面不兼容，无法再用于渲染。通常在调整窗口大小后发生。
		//VK_SUBOPTIMAL_KHR：交换链仍可用于成功呈现到表面，但表面属性不再完全匹配。
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		updateUniformBuffer(currentFrame);

		vkResetFences(my_device->logicalDevice, 1, &shadowInFlightFences[currentFrame]);
		vkResetCommandBuffer(my_buffer->commandBuffers[0][currentFrame], 0);
		recordShadowCommandBuffer(my_buffer->commandBuffers[0][currentFrame], imageIndex);

		VkSemaphore shadowWaitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags shadowWaitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = shadowWaitSemaphores;
		submitInfo.pWaitDstStageMask = shadowWaitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &my_buffer->commandBuffers[0][currentFrame];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &shadowRenderFinishedSemaphores[currentFrame];

		if (vkQueueSubmit(my_device->graphicsQueue, 1, &submitInfo, shadowInFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit shadow command buffer!");
		};

		//计算着色器
		vkWaitForFences(my_device->logicalDevice, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkResetFences(my_device->logicalDevice, 1, &computeInFlightFences[currentFrame]);
		vkResetCommandBuffer(my_buffer->commandBuffers[1][currentFrame], 0);
		recordComputeCommandBuffer(my_buffer->commandBuffers[1][currentFrame]);

		VkSemaphore computeWaitSemaphores[] = { shadowRenderFinishedSemaphores[currentFrame] };
		VkPipelineStageFlags computeWaitStages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
		submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = computeWaitSemaphores;
		submitInfo.pWaitDstStageMask = computeWaitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &my_buffer->commandBuffers[1][currentFrame];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];

		if (vkQueueSubmit(my_device->computeQueue, 1, &submitInfo, computeInFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit compute command buffer!");
		};

		//第2个参数为是否等待栏栅的数量，第四个参数为是否等待所有栏栅信号化
		vkWaitForFences(my_device->logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkResetFences(my_device->logicalDevice, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(my_buffer->commandBuffers[2][currentFrame], 0);
		recordCommandBuffer(my_buffer->commandBuffers[2][currentFrame], imageIndex);

		//只有两个信号量都有时，Queue才开始执行
		VkSemaphore waitSemaphores[] = { computeFinishedSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };	//片元着色器必须等待计算着色器将pathTracing的结果放到纹理中才能开始执行
		submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &my_buffer->commandBuffers[2][currentFrame];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];

		if (vkQueueSubmit(my_device->graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];

		VkSwapchainKHR swapChains[] = { my_swapChain->swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		result = vkQueuePresentKHR(my_device->presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		frameNum++;

	}

	void updateUniformBuffer(uint32_t currentImage) {

		float currentTime = static_cast<float>(glfwGetTime());;
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
		lastTime = currentTime;

		UniformBufferObject ubo{};
		ubo.model = glm::mat4(1.0f);
		ubo.view = camera.GetViewMatrix();
		ubo.proj = glm::perspective(glm::radians(45.0f), my_swapChain->swapChainExtent.width / (float)my_swapChain->swapChainExtent.height, 0.1f, 100.0f);
		//怪不得，我从obj文件中看到场景的顶点是顺时针的，但是在shader中得是逆时针才对，原来是这里proj[1][1]1 *= -1搞的鬼
		//那我们在计算着色器中处理顶点数据似乎不需要这个啊
		ubo.proj[1][1] *= -1;
		ubo.cameraPos = glm::vec4(camera.Position, dis(gen));
		ubo.randomNumber = glm::vec4(dis(gen), dis(gen), dis(gen), float(frameNum % 1000));

		memcpy(my_buffer->uniformBuffersMappeds[0][currentFrame], &ubo, sizeof(ubo));

		//标量必须按 N 对齐（= 32 位浮点数为 4 个字节）。
		//Avec2必须按 2N（ = 8 个字节）对齐
		//Avec3或vec4必须按 4N（ = 16 字节）对齐
		//嵌套结构必须按其成员的基本对齐方式对齐，并向上四舍五入为 16 的倍数。
		//矩阵mat4必须具有与之相同的对齐vec4。

	}

	void recreateSwapChain() {

		int width = 0, height = 0;
		//获得当前window的大小
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(my_device->logicalDevice);
		cleanupSwapChain();
		createMySwapChain();
		createTargetTextureResources();
		createFramebuffers();
	}

	void recordShadowCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(my_swapChain->swapChainExtent.width);
		viewport.height = static_cast<float>(my_swapChain->swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = my_swapChain->swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = shadowRenderPass;
		renderPassInfo.framebuffer = my_buffer->swapChainFramebuffers[0][imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = my_swapChain->swapChainExtent;

		std::array<VkClearValue, 1> clearValues{};
		clearValues[0].depthStencil = { 1.0f, 0 };
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkBuffer vertexBuffers[] = { my_buffer->buffersStatic[1] };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, my_buffer->buffersStatic[2], 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowGraphicsPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowGraphicsPipelineLayout, 0, 1, &my_descriptor->descriptorObjects[0].descriptorSets[0], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->indices.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	}

	void recordComputeCommandBuffer(VkCommandBuffer commandBuffer) {

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording compute command buffer!");
		}

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &my_descriptor->descriptorObjects[0].descriptorSets[0], 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 1, 1, &my_descriptor->descriptorObjects[1].descriptorSets[currentFrame], 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 2, 1, &my_descriptor->descriptorObjects[2].descriptorSets[0], 0, nullptr);

		vkCmdDispatch(commandBuffer, WIDTH / 16, HEIGHT / 16, 1);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record compute command buffer!");
		}

	}

	//这个函数记录渲染的命令，并指定渲染结果所在的纹理索引
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(my_swapChain->swapChainExtent.width);
		viewport.height = static_cast<float>(my_swapChain->swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = my_swapChain->swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = finalRenderPass;
		renderPassInfo.framebuffer = my_buffer->swapChainFramebuffers[1][imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = my_swapChain->swapChainExtent;

		std::array<VkClearValue, 1> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		//VkBuffer vertexBuffers[] = { my_buffer->vertexBuffer };
		//VkDeviceSize offsets[] = { 0 };
		//vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		//vkCmdBindIndexBuffer(commandBuffer, my_buffer->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
		//
		//vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
		//VkDescriptorSet uniformDescriptorSet = my_descriptor->descriptorObjects[0].descriptorSets[currentFrame];
		//vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &uniformDescriptorSet, 0, nullptr);
		//uint32_t index = 0;
		//for (uint32_t i = 0; i < my_model->meshs.size(); i++) {
		//
		//	VkDescriptorSet uniformMaterialDescriptorSet = my_descriptor->descriptorObjects[1].descriptorSets[i];
		//	vkCmdBindDescriptorSets(my_buffer->graphicsCommandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 1, 1, &uniformMaterialDescriptorSet, 0, nullptr);
		//
		//	vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(my_model->meshs[i].indices.size()), 1, index, 0, 0);
		//	index += my_model->meshs[i].indices.size();
		//
		//}

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &my_descriptor->descriptorObjects[3].descriptorSets[0], 0, nullptr);
		vkCmdDraw(commandBuffer, 3, 1, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	}

	void cleanup() {

		cleanupSwapChain();

		vkDestroyPipeline(my_device->logicalDevice, shadowGraphicsPipeline, nullptr);
		vkDestroyPipelineLayout(my_device->logicalDevice, shadowGraphicsPipelineLayout, nullptr);
		vkDestroyPipeline(my_device->logicalDevice, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(my_device->logicalDevice, graphicsPipelineLayout, nullptr);
		vkDestroyPipeline(my_device->logicalDevice, computePipeline, nullptr);
		vkDestroyPipelineLayout(my_device->logicalDevice, computePipelineLayout, nullptr);
		vkDestroyRenderPass(my_device->logicalDevice, shadowRenderPass, nullptr);
		vkDestroyRenderPass(my_device->logicalDevice, finalRenderPass, nullptr);

		vkDestroyDescriptorPool(my_device->logicalDevice, my_descriptor->discriptorPool, nullptr);

		my_descriptor->clean();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(my_device->logicalDevice, shadowRenderFinishedSemaphores[i], nullptr);
			vkDestroyFence(my_device->logicalDevice, shadowInFlightFences[i], nullptr);
			vkDestroySemaphore(my_device->logicalDevice, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(my_device->logicalDevice, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(my_device->logicalDevice, inFlightFences[i], nullptr);
			vkDestroySemaphore(my_device->logicalDevice, computeFinishedSemaphores[i], nullptr);
			vkDestroyFence(my_device->logicalDevice, computeInFlightFences[i], nullptr);
		}
		my_buffer->clean(my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT);

		my_device->clean();

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();

	}

	void cleanupSwapChain() {

		pathTracingResult->clean();
		shadowMap->clean();
		for (size_t i = 0; i < my_buffer->swapChainFramebuffers.size(); i++) {
			for (int j = 0; j < my_buffer->swapChainFramebuffers[i].size(); j++) {
				vkDestroyFramebuffer(my_device->logicalDevice, my_buffer->swapChainFramebuffers[i][j], nullptr);
			}
		}
		for (size_t i = 0; i < my_swapChain->swapChainImageViews.size(); i++) {
			vkDestroyImageView(my_device->logicalDevice, my_swapChain->swapChainImageViews[i], nullptr);
		}
		vkDestroySwapchainKHR(my_device->logicalDevice, my_swapChain->swapChain, nullptr);
	}

};

int main() {

	pathTracing app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		system("pause");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

}