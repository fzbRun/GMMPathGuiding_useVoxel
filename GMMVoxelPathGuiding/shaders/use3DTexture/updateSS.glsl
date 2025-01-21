#version 450

#include "common.glsl"

struct Sufficient_Statistic_uint {
	uint ss2;	//将两个float大包为一个uint
	uint ss1_ss3;	//将2个float打包为一个uint，第二个是ss3的x的方差
	uint ss3;	//两个float，协方差和y的标准差
};

struct Sufficient_Statistic_vec2 {
	vec2 ss2;
	vec2 ss1_ss3;
	vec2 ss3;
};

 struct Voxel_AVG_Sufficient_Statistic{
	uint photonAvgWeight_photonNum;	//将两个float大包为一个uint
	Sufficient_Statistic_uint SSs[K];
};

 layout(set = 3, binding = 0, std430) coherent volatile buffer AVG_Sufficient_Statistic {
	 Voxel_AVG_Sufficient_Statistic vgss[];
 };

layout(set = 3, binding = 2) uniform sampler3D GMMParaTexture[9];

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;


//---------------------------------------------function---------------------------------------------------
GMMPara sampleGMMPara(ivec3 voxelIndexXYZ) {

	vec3 voxelIndex = voxelIndexXYZ / gmmConstant.voxelNum.xyz;

	GMMPara gmmPara;
	vec4 gmmPara1 = texture(GMMParaTexture[0], voxelIndexXYZ);
	vec4 gmmPara2 = texture(GMMParaTexture[1], voxelIndexXYZ);
	vec4 gmmPara3 = texture(GMMParaTexture[2], voxelIndexXYZ);
	vec4 gmmPara4 = texture(GMMParaTexture[3], voxelIndexXYZ);
	vec4 gmmPara5 = texture(GMMParaTexture[4], voxelIndexXYZ);
	vec4 gmmPara6 = texture(GMMParaTexture[5], voxelIndexXYZ);
	vec4 gmmPara7 = texture(GMMParaTexture[6], voxelIndexXYZ);
	vec4 gmmPara8 = texture(GMMParaTexture[7], voxelIndexXYZ);
	vec4 gmmPara9 = texture(GMMParaTexture[8], voxelIndexXYZ);

	GaussianPara gp;
	gp.mixWeight = gmmPara1.x;
	gp.mean = gmmPara1.yz;
	gp.covarianceMatrix = mat2(vec2(gmmPara1.w, gmmPara2.x), vec2(gmmPara2.x, gmmPara2.y));
	gmmPara.gaussianParas[0] = gp;

	gp.mixWeight = gmmPara2.z;
	gp.mean = vec2(gmmPara2.w, gmmPara3.x);
	gp.covarianceMatrix = mat2(vec2(gmmPara3.y, gmmPara3.z), vec2(gmmPara3.z, gmmPara3.w));
	gmmPara.gaussianParas[1] = gp;

	gp.mixWeight = gmmPara4.x;
	gp.mean = gmmPara4.yz;
	gp.covarianceMatrix = mat2(vec2(gmmPara4.w, gmmPara5.x), vec2(gmmPara5.x, gmmPara5.y));
	gmmPara.gaussianParas[2] = gp;

	gp.mixWeight = gmmPara5.z;
	gp.mean = vec2(gmmPara5.w, gmmPara6.x);
	gp.covarianceMatrix = mat2(vec2(gmmPara6.y, gmmPara6.z), vec2(gmmPara6.z, gmmPara6.w));
	gmmPara.gaussianParas[3] = gp;

	gp.mixWeight = gmmPara7.x;
	gp.mean = gmmPara7.yz;
	gp.covarianceMatrix = mat2(vec2(gmmPara7.w, gmmPara8.x), vec2(gmmPara8.x, gmmPara8.y));
	gmmPara.gaussianParas[4] = gp;

	gp.mixWeight = gmmPara8.z;
	gp.mean = vec2(gmmPara8.w, gmmPara9.x);
	gp.covarianceMatrix = mat2(vec2(gmmPara9.y, gmmPara9.z), vec2(gmmPara9.z, gmmPara9.w));
	gmmPara.gaussianParas[5] = gp;

	return gmmPara;

}

void updateSS(Photon photon) {

	ivec3 voxelIndexXYZ = ivec3((photon.hitPos.xyz - gmmConstant.voxelStartPos.xyz) / gmmConstant.voxelSize);
	uint voxelIndex = voxelIndexXYZ.z * gmmConstant.voxelNum.x * gmmConstant.voxelNum.y + voxelIndexXYZ.y * gmmConstant.voxelNum.x + voxelIndexXYZ.x;
	GMMPara gmmPara = sampleGMMPara(voxelIndexXYZ);

	float gamma[K];
	float gammaSum = 0.0f;
	for (int i = 0; i < K; i++) {
		GaussianPara gp = gmmPara.gaussianParas[i];
		gamma[i] = photon.direction[i] == vec2(-1.0f) ? 0.0f : GaussianPdf(gp, photon.direction[i]);
		gammaSum += gamma[i];
	}

	//gammaSum = gammaSum == 0.0f ? 1.0f : gammaSum;
	if (gammaSum == 0.0f) {
		return;
	}

	Sufficient_Statistic_vec2 SS_vec2[K];
	Sufficient_Statistic_uint SS_uint[K];
	for (int i = 0; i < K; i++) {
		float ss1 = photon.weight * gamma[i];
		mat2 ss3 = (photon.weight * gamma[i] / gammaSum * outerProduct(photon.direction[i], photon.direction[i]));
		vec3 ss3_simple = vec3(ss3[0][0], ss3[0][1], ss3[1][1]);
		SS_vec2[i].ss1_ss3 = vec2(ss1, ss3_simple.x);
		SS_vec2[i].ss3 = vec2(ss3_simple.y, ss3_simple.z);
		SS_vec2[i].ss2 = (photon.weight * gamma[i] / gammaSum * photon.direction[i]);

		SS_uint[i].ss1_ss3 = myPackUnorm2x16(SS_vec2[i].ss1_ss3);
		SS_uint[i].ss3 = myPackUnorm2x16(SS_vec2[i].ss3);
		SS_uint[i].ss2 = myPackUnorm2x16(SS_vec2[i].ss2);
	}
	vec2 photonWeight_photonNum_vec2 = vec2(photon.weight, 1.0f);

	//其实Sufficient_Statistic_uint中各个数据都需要独立的进行判断和原子添加，但是为了节约时间，我们可以放在同一个while中
	//即，我们认为如果一个数据可以存入了，就代表所有数据都存入了
	uint photonWeight_photonNum_uint = myPackUnorm2x16(photonWeight_photonNum_vec2);
	uint prevStoredVal = 0;
	uint curStoredVal = atomicCompSwap(vgss[voxelIndex].photonAvgWeight_photonNum, prevStoredVal, photonWeight_photonNum_uint);
	uint cur_ss1_ss3[K];
	uint cur_ss2[K];
	uint cur_ss3[K];
	for (int i = 0; i < K; i++) {
		cur_ss1_ss3[i] = atomicCompSwap(vgss[voxelIndex].SSs[i].ss1_ss3, 0, SS_uint[i].ss1_ss3);
		cur_ss2[i] = atomicCompSwap(vgss[voxelIndex].SSs[i].ss2, 0, SS_uint[i].ss2);
		cur_ss3[i] = atomicCompSwap(vgss[voxelIndex].SSs[i].ss3, 0, SS_uint[i].ss3);
	}

	while (curStoredVal != prevStoredVal) {
		prevStoredVal = curStoredVal;
		vec2 old_photonWeight_photonNum = myUnPackUnorm2x16(curStoredVal);
		float old_photonNum = old_photonWeight_photonNum.y;
		float old_photonWeight = old_photonWeight_photonNum.x;
		old_photonWeight = old_photonWeight * old_photonNum;	//算出当前存入光子的总权重
		vec2 new_photonWeight_photonNum = vec2(old_photonWeight, old_photonNum) + photonWeight_photonNum_vec2;
		new_photonWeight_photonNum.x /= new_photonWeight_photonNum.y;
		photonWeight_photonNum_uint = myPackUnorm2x16(new_photonWeight_photonNum);

		for (int i = 0; i < K; i++) {
			SS_uint[i].ss1_ss3 = myPackUnorm2x16((myUnPackUnorm2x16(cur_ss1_ss3[i]) * old_photonNum + SS_vec2[i].ss1_ss3) / (old_photonNum + 1));
			SS_uint[i].ss2 = myPackUnorm2x16((myUnPackUnorm2x16(cur_ss2[i]) * uint(old_photonNum) + SS_vec2[i].ss2) / (old_photonNum + 1));
			SS_uint[i].ss3 = myPackUnorm2x16((myUnPackUnorm2x16(cur_ss3[i]) * uint(old_photonNum) + SS_vec2[i].ss3) / (old_photonNum + 1));
		}

		curStoredVal = atomicCompSwap(vgss[voxelIndex].photonAvgWeight_photonNum, prevStoredVal, photonWeight_photonNum_uint);
		if (curStoredVal == prevStoredVal) {
			for (int i = 0; i < K; i++) {
				cur_ss1_ss3[i] = atomicCompSwap(vgss[voxelIndex].SSs[i].ss1_ss3, 0, SS_uint[i].ss1_ss3);
				cur_ss2[i] = atomicCompSwap(vgss[voxelIndex].SSs[i].ss2, 0, SS_uint[i].ss2);
				cur_ss3[i] = atomicCompSwap(vgss[voxelIndex].SSs[i].ss3, 0, SS_uint[i].ss3);
			}
		}
	}

}

void photonTracing(Ray ray, inout uint randomNumberSeed) {

	vec3 normal;
	mat3 TBN;
	Material material;
	float RR;
	int lossNum = 0;
	int maxLossNum = 50;
	vec3 lastHitPos;
	float startWeight = 0.299f * ray.radiance.x + 0.587f * ray.radiance.g + 0.114f * ray.radiance.b;
	float pdf = 1.0;

	ivec2 result = hitScene(ray);
	while (lossNum < maxLossNum && result.x == -1) {
		randomNumberSeed++;
		ray = makeStartPhoton(randomNumberSeed, pdf);
		result = hitScene(ray);
		lossNum++;
	}
	if (lossNum == maxLossNum) {
		return;
	}

	int meshIndex = result.x;
	int vertexIndex = result.y;
	vec3 hitPos = ray.startPos + ray.depth * ray.direction;
	getHitPointNormalAndTBN(vertexIndex, normal, TBN);
	material = meshs[meshIndex].material;

	//hitTest(hitPos, normal);

	Photon photon;
	photon.hitPos = vec4(hitPos, 0.0f);
	photon.startPos = vec4(ray.startPos, 0.0f);
	photon.direction_3D = vec4(-ray.direction, 0.0f);
	for (int i = 0; i < K; i++) {		//0: +x		1: -x以此类推
		vec3 axisNormal = axisNormals[i];
		mat3 axisTBN = createTBN(axisNormal);
		photon.direction[i] = concentricMapping_hemisphere_3DTo2D(-ray.direction, axisTBN);
	}
	photon.weight = (0.299f * ray.radiance.x + 0.587f * ray.radiance.g + 0.114f * ray.radiance.b) * dot(ray.normal, ray.direction) * dot(-ray.direction, normal) / pdf;
	
	updateSS(photon);

	if (material.bxdfPara.x > 0.75f) {	//当前是漫反射平面，结束弹射
		return;
	}

	RR = min(photon.weight / startWeight / 0.00001f, 1.0f);
	if (rand(randomNumberSeed) > RR) {
		return;
	}

	//获得出射方向
	ray.radiance = ray.radiance * dot(ray.normal, ray.direction) * dot(-ray.direction, normal) / RR;	//需要打到下个顶点时才可以计算bxdf
	lastHitPos = ray.startPos;
	ray.startPos = hitPos;
	ray.direction = getRayFromBxdf(material, TBN, -ray.direction, normal, randomNumberSeed, pdf);
	ray.normal = normal;
	ray.depth = 100.0f;

	int maxIterationNum = 9;	//一共是10
	while (maxIterationNum > 0) {

		result = hitScene(ray);
		lossNum = 0;
		while (result.x == -1 && lossNum < maxLossNum) {
			ray.direction = getRayFromBxdf(material, TBN, normalize(lastHitPos - ray.startPos), normal, randomNumberSeed, pdf);
			result = hitScene(ray);
			lossNum++;
		}
		if (lossNum == maxLossNum) {
			return;
		}

		int meshIndex = result.x;
		int vertexIndex = result.y;
		vec3 hitPos = ray.startPos + ray.depth * ray.direction;
		getHitPointNormalAndTBN(vertexIndex, normal, TBN);

		//hitTest(hitPos, normal);

		//计算光子强度的衰减fr项
		ray.radiance *= getFr(hitPos, ray.startPos, ray.normal, lastHitPos, material);	//现在的材质还是上一个的
		material = meshs[meshIndex].material;

		photon.hitPos = vec4(hitPos, 0.0f);
		photon.startPos = vec4(ray.startPos, 0.0f);
		photon.direction_3D = vec4(-ray.direction, 0.0f);
		for (int i = 0; i < K; i++) {		//0: +x		1: -x以此类推
			vec3 axisNormal = axisNormals[i];
			mat3 axisTBN = createTBN(axisNormal);
			photon.direction[i] = concentricMapping_hemisphere_3DTo2D(-ray.direction, axisTBN);
		}
		photon.weight = (0.299f * ray.radiance.x + 0.587f * ray.radiance.g + 0.114f * ray.radiance.b) * dot(-ray.direction, normal) / pdf / RR;

		updateSS(photon);

		if (material.bxdfPara.x > 0.75f) {	//当前是漫反射平面，结束弹射
			return;
		}

		RR = min(photon.weight / startWeight / 0.00001f, 1.0f);
		if (rand(randomNumberSeed) > RR) {
			return;
		}

		//获得出射方向
		ray.radiance = ray.radiance * dot(-ray.direction, normal) / pdf / RR;	//需要打到下个顶点时才可以计算bxdf
		lastHitPos = ray.startPos;
		ray.startPos = hitPos;
		ray.direction = getRayFromBxdf(material, TBN, -ray.direction, normal, randomNumberSeed, pdf);
		ray.normal = normal;
		ray.depth = 100.0f;

		maxIterationNum--;

	}

}

void main() {

	uvec2 seed2 = pcg2d(ivec2(gl_GlobalInvocationID) * (uint(cubo.randomNumber.w + cubo.randomNumber.x) + 1));
	uint seed = seed2.x + seed2.y;

	float pdf;
	Ray ray = makeStartPhoton(seed, pdf);
	photonTracing(ray, seed);

}

