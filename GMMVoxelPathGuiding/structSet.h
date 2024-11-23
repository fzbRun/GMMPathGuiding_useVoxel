#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include<glm/glm.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>

#ifndef STRUCT_SET
#define STRUCT_SET
//所有命令都要被提交到队列中执行，而每个队列属于不同的队列族，不同的队列族支持不同的指令，比如有些队列族只负责数值计算，有些队列族只负责内存操作等
//而队列族是物理设备提供的，即一个物理设备提供某些队列族，如显卡能提供计算、渲染的功能，而不能提供造宇宙飞船的功能
struct QueueFamilyIndices {

	//std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> graphicsAndComputeFamily;

	bool isComplete() {
		return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
	}

};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
	glm::vec3 pos;
	glm::vec2 texCoord;
	glm::vec3 normal;
	glm::vec3 tangent;


	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {

		//VAO
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);	//找pos在Vertex中的偏移

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, normal);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, tangent);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {
		return pos == other.pos && texCoord == other.texCoord && normal == other.normal && tangent == other.tangent;
	}

};

struct ComputeVertex {

	glm::vec4 pos;
	glm::vec4 normal;

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(ComputeVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {

		//VAO
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(ComputeVertex, pos);	//找pos在Vertex中的偏移

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(ComputeVertex, normal);

		return attributeDescriptions;
	}

	bool operator==(const ComputeVertex& other) const {
		return pos == other.pos;
	}

};

struct Material {
	glm::vec4 bxdfPara;	//这里本来是ka的，但是我修改mtl文件，将粗糙度放到了ka.x，金属值放到了ka.y，而折射率放到了ka.z，粗糙度放到
	glm::vec4 kd;
	glm::vec4 ks;
	glm::vec4 ke;
};

struct Texture {
	//uint32_t id;
	std::string type;
	std::string path;
};

struct AABBBox {

	float leftX;
	float rightX;
	float leftY;
	float rightY;
	float leftZ;
	float rightZ;

	float getAxis(int k) {
		if (k == 0) {
			return leftX;
		}
		else if (k == 1) {
			return rightX;
		}
		else if (k == 2) {
			return leftY;
		}
		else if (k == 3) {
			return rightY;
		}
		else if (k == 4) {
			return leftZ;
		}
		else if (k == 5) {
			return rightZ;
		}
	}

};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<Texture> textures;
	Material material;
	AABBBox AABB;

	Mesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices, std::vector<Texture> textures, Material material) {
		this->vertices = vertices;
		this->indices = indices;
		this->textures = textures;
		this->material = material;
	}

};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 cameraPos;
	glm::vec4 randomNumber;
};

struct UniformLightBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 lightPos_strength;
	glm::vec4 normal;
	glm::vec4 size;
};

struct DescriptorObject {

	VkDescriptorSetLayout discriptorLayout;
	uint32_t uniformBufferNum;
	uint32_t textureNum;
	std::vector<VkDescriptorSet> descriptorSets;
};

struct BvhTreeNode {

	BvhTreeNode* leftNode;
	BvhTreeNode* rightNode;
	AABBBox AABB;
	std::vector<uint32_t> meshIndex;

};

struct BvhArrayNode {

	int32_t leftNodeIndex;
	int32_t rightNodeIndex;
	AABBBox AABB;
	//我们只关注叶子节点的mesh，因为我们光线先与scene碰撞，直到bvhTree的叶子节点，再与节点中的mesh碰撞，所以我们只需要记录叶子节点的meshIndex即可
	//而叶子节点最多只有2个mesh
	//我真是天才
	int32_t meshIndex;
	//int32_t meshIndex2;

};

struct ComputeInputMesh {

	Material material;
	glm::ivec2 indexInIndicesArray;
	AABBBox AABB;
	//char padding[4];	//因为material中是vec4，是16字节，所以需要当前ComputeInputMesh是16字节的倍数，这样下一个ComputeInputMesh才能得到正确的值

};

struct GMMConstant {

	glm::vec4 voxelStartPos;
	glm::ivec4 voxelNum;
	float voxelSize;
	uint32_t photonTracingNum;

};

struct GaussianPara {

	glm::vec2 mean;
	float mixWeight;
	float padding;
	glm::mat2 covarianceMatrix;

	GaussianPara() {
		mixWeight = 0.125f;
		mean = glm::vec2(0.0f);
		covarianceMatrix = glm::mat2(1.0f);
		padding = 0.0;
	}

};

struct Sufficient_Statistic {

	glm::vec2 ss2;	//高斯分布对该样本的贡献比例乘以样本的位置的乘积之和
	float ss1;		//高斯分布对每个样本的贡献比例之和
	float padding;
	glm::mat2 ss3;	//样本的位置向量与其转置的乘积乘以高斯分布对该样本的贡献比例的乘积之和

	Sufficient_Statistic() {
		ss1 = 0.0f;
		ss2 = glm::vec2(0.0f);
		ss3 = glm::mat2(0.0f);
		padding = 0.0;
	}

};

const int gmmNum = 6;

struct GMMPara {

	float photonAvgWeight;
	uint32_t photonNum;
	float padding1;
	float padding2;
	glm::vec4 pos;		//因为std430中vec3单独（不是数据的元素）传输时，对齐边界为vec4，所以我们在这里传vec4，但是shader中用vec3接收
	GaussianPara gaussianParas[gmmNum];
	Sufficient_Statistic SSs[gmmNum];


	GMMPara() {
		pos = glm::vec4(0.0f);
		for (int i = 0; i < gmmNum; i++) {
			gaussianParas[i] = GaussianPara();
			SSs[i] = Sufficient_Statistic();
		}
		photonAvgWeight = 0.0f;
		photonNum = 0.0f;
		padding1 = 0.0f;
		padding2 = 0.0f;
	}

};

struct Photon {

	glm::vec2 direction[gmmNum];	//将三维方向转为2维方向，同心圆映射
	float weight;	//亮度 = 0.299f * R + 0.587f * G + 0.114f * B
	float padding1;
	float padding2;
	float padding3;
	glm::vec4 direction_3D;
	glm::vec4 hitPos;
	glm::vec4 startPos;

	Photon() {
		padding1 = 0.0f;
		padding2 = 0.0f;
		padding3 = 0.0f;
		hitPos = glm::vec4(0.0f);
		startPos = glm::vec4(0.0f);
		direction_3D = glm::vec4(0.0f);
		for (int i = 0; i < gmmNum; i++) {
			direction[i] = glm::vec2(0.0f);
		}
		weight = 0.0f;
	}

};

//其实不需要关心布局，因为我们传的都是0，具体数据是在shader中修改的
struct PhotonTracingResult {
	uint32_t photonNum;
	float padding1;
	float padding2;
	float padding3;
	Photon photons[10];

	PhotonTracingResult() {
		this->photonNum = 0;
		for (int i = 0; i < 10; i++) {
			photons[0] = Photon();
		}
		padding1 = 0.0f;
		padding2 = 0.0f;
		padding3 = 0.0f;
	}

};

#endif