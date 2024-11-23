#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "structSet.h"
#include "myImage.h"

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#ifndef  MY_MODEL
#define MY_MODEL

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

class myModel {

public:

	std::vector<Mesh> meshs;
	//std::vector<Material> materials;
	std::vector<Texture> textures_loaded;
	//std::vector<std::vector<Texture>> textures_loaded;

	myModel(std::string path);

private:

	std::string directory;

	void loadModel(std::string path);
	void processNode(aiNode* node, const aiScene* scene);
	Mesh processMesh(aiMesh* mesh, const aiScene* scene);
	std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);
	void simplify();	//先将复杂物体剔除
	void optimize();
	void modelChange();
	void makeAABB();

};

#endif // ! MY_MODEL
