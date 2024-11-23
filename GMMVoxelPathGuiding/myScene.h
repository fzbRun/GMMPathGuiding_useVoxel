#pragma once

#include "structSet.h"
#include <vector>
#include <memory>

#ifndef MY_SCENE
#define MY_SCENE

class myScene {

public:

	std::vector<Mesh>* sceneMeshs;
	BvhTreeNode* bvhTree;
	std::vector<BvhArrayNode> bvhArray;

	myScene(std::vector<Mesh>* meshs);
	bool splitSpace(int k, float splitPoint, std::vector<uint32_t> meshsIndex, std::vector<uint32_t>& meshsLeftIndex, std::vector<uint32_t>& meshsRightIndex);
	void makeBvhTree(BvhTreeNode* bvhTree, std::vector<uint32_t> meshsIndex);
	//最后的要传入buffer的数据需要固定大小，且是数组
	//那么我们可以通过设置一个数据结构数据，每个元素存放子场景（bvhNode）所包含的mesh的索引，以及左右子树在数组中的索引
	void bvhTreeToArray(BvhTreeNode* node, uint32_t& childNum, int32_t nodeIndex);

	~myScene();
	void deleteBvhTree(BvhTreeNode* bvhTreeNode);

};

#endif // !MY_SCENE

