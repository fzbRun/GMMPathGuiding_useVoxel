#include "myScene.h"

myScene::myScene(std::vector<Mesh>* meshs) {

	this->sceneMeshs = meshs;
	this->bvhTree = new BvhTreeNode();

	std::vector<uint32_t> meshsIndex;
	for (int i = 0; i < meshs->size(); i++) {
		meshsIndex.push_back(i);
	}
	makeBvhTree(this->bvhTree, meshsIndex);
	uint32_t childNum = 0;
	bvhTreeToArray(this->bvhTree, childNum, 0);

}

void myScene::deleteBvhTree(BvhTreeNode* node) {

	if (node->leftNode) {
		deleteBvhTree(node->leftNode);
	}
	if (node->rightNode) {
		deleteBvhTree(node->rightNode);
	}

	delete node;

}

myScene:: ~myScene() {
	deleteBvhTree(this->bvhTree);
}

bool myScene::splitSpace(int k, float splitPoint, std::vector<uint32_t> meshsIndex, std::vector<uint32_t> &meshsLeftIndex, std::vector<uint32_t> &meshsRightIndex) {

	for (int i = 0; i < meshsIndex.size(); i++) {
		AABBBox meshAABB = this->sceneMeshs->at(meshsIndex[i]).AABB;
		float axisCenter = (meshAABB.getAxis(k) + meshAABB.getAxis(k + 1)) / 2;
		if (axisCenter < splitPoint) {
			meshsLeftIndex.push_back(meshsIndex[i]);
		}
		else {
			meshsRightIndex.push_back(meshsIndex[i]);
		}
	}

	//按左边框算后一般不会出现分不开的情况，但保险起见
	if (meshsLeftIndex.size() == 0 || meshsRightIndex.size() == 0) {
		return false;
	}

	return true;

}

void myScene::makeBvhTree(BvhTreeNode* node, std::vector<uint32_t> meshsIndex) {

	AABBBox AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	glm::vec3 AABBMean = glm::vec3(0.0f);
	for (int i = 0; i < meshsIndex.size(); i++) {

		AABBBox meshAABB = this->sceneMeshs->at(meshsIndex[i]).AABB;
		AABB.leftX  = meshAABB.leftX  < AABB.leftX  ? meshAABB.leftX  : AABB.leftX;
		AABB.rightX = meshAABB.rightX > AABB.rightX ? meshAABB.rightX : AABB.rightX;
		AABB.leftY  = meshAABB.leftY  < AABB.leftY  ? meshAABB.leftY  : AABB.leftY;
		AABB.rightY = meshAABB.rightY > AABB.rightY ? meshAABB.rightY : AABB.rightY;
		AABB.leftZ  = meshAABB.leftZ  < AABB.leftZ  ? meshAABB.leftZ  : AABB.leftZ;
		AABB.rightZ = meshAABB.rightZ > AABB.rightZ ? meshAABB.rightZ : AABB.rightZ;

		AABBMean += glm::vec3((meshAABB.leftX + meshAABB.rightX) / 2,
								(meshAABB.leftY + meshAABB.rightY) / 2,
								(meshAABB.leftZ + meshAABB.rightZ) / 2);

	}
	AABBMean /= meshsIndex.size();	//获得所有mesh的AABB的中点的平均值
	node->AABB = AABB;
	node->meshIndex = meshsIndex;

	//如果子场景的mesh数等于1，则停止划分
	if (meshsIndex.size() == 1) {
		return;
	}

	//std::array<float, 3> maxAxis;
	//maxAxis[0]= AABB.rightX - AABB.leftX;
	//maxAxis[1]= AABB.rightY - AABB.leftY;
	//maxAxis[2]= AABB.rightZ - AABB.leftZ;
	//int k = maxAxis[0] > maxAxis[1] ? maxAxis[0] > maxAxis[2] ? 0 : 4 : maxAxis[1] > maxAxis[2] ? 2 : 4;
	//float splitPoint = (AABB.getAxis(k) + maxAxis[k / 2]) / 2;	//最长轴的中点
	std::array<float, 3> splitPoint = { AABBMean.x, AABBMean.y, AABBMean.z };
	std::vector<uint32_t> meshsLeftIndex;
	std::vector<uint32_t> meshsRightIndex;
	std::vector<uint32_t> meshsLeftIndexTemp;
	std::vector<uint32_t> meshsRightIndexTemp;
	bool splitEnable = false;
	bool splitEnableAll = false;
	int difference = meshsIndex.size();
	for (int i = 0; i < 3; i++) {

		splitEnable = splitSpace(i * 2, splitPoint[i], meshsIndex, meshsLeftIndexTemp, meshsRightIndexTemp);
		if (splitEnable) {
			splitEnableAll = true;
			int diff = meshsLeftIndexTemp.size() - meshsRightIndexTemp.size() < 0 ? meshsRightIndexTemp.size() - meshsLeftIndexTemp.size() : meshsLeftIndexTemp.size() - meshsRightIndexTemp.size();
			if (diff < difference) {
				meshsLeftIndex = meshsLeftIndexTemp;
				meshsRightIndex = meshsRightIndexTemp;
				difference = diff;
			}
		}
		meshsLeftIndexTemp.resize(0);
		meshsRightIndexTemp.resize(0);

	}
	if (!splitEnableAll) {
		return;
	}

	BvhTreeNode* leftNode = new BvhTreeNode();
	node->leftNode = leftNode;

	BvhTreeNode* rightNode = new BvhTreeNode();
	node->rightNode = rightNode;

	makeBvhTree(leftNode, meshsLeftIndex);
	makeBvhTree(rightNode, meshsRightIndex);


}

void myScene::bvhTreeToArray(BvhTreeNode* node, uint32_t& childNum, int32_t nodeIndex) {

	BvhTreeNode* tNode = node;
	BvhArrayNode aNode;
	aNode.AABB = tNode->AABB;
	aNode.leftNodeIndex = -1;
	aNode.rightNodeIndex = -1;
	int currentIndex = this->bvhArray.size();
	this->bvhArray.push_back(aNode);

	uint32_t lcn = 0;
	uint32_t rcn = 0;
	if (tNode->leftNode) {

		aNode.meshIndex = -1;
		//aNode.meshIndex2 = -1;

		aNode.leftNodeIndex = nodeIndex + 1;
		bvhTreeToArray(tNode->leftNode, lcn, aNode.leftNodeIndex);

		aNode.rightNodeIndex = nodeIndex + lcn + 1;
		bvhTreeToArray(tNode->rightNode, rcn, aNode.rightNodeIndex);

	}
	else {
		aNode.meshIndex = tNode->meshIndex[0];
		//if (tNode->meshIndex.size() > 1) {
		//	aNode.meshIndex2 = tNode->meshIndex[1];
		//}
		//else {
		//	aNode.meshIndex2 = -1;
		//}
	}

	childNum = childNum + lcn + rcn + 1;
	this->bvhArray[currentIndex] = aNode;

}