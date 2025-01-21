//����
const float PI = 3.1415926535f;
const uint K = 6;	//��˹�ֲ�������
const vec3 axisNormals[6] = vec3[6](vec3(1.0f, 0.0f, 0.0f), vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 0.0f, -1.0f));
const float lightA = 25.0f;
const float sqrtLightA = 5.0f;
const vec3 lightStrength = vec3(27.0f, 22.0f, 14.0f);
const uint SSDigit_16 = 65536;	//2^16
const uint SSDigit_8 = 65536;	//2^16

//struct
//���ھ�һ��pos�������Ժ���Ҫ����textureʱ���߱��ʱ������ټ�
struct Vertex {
	vec4 pos;
	vec4 normal;
};

struct AABBBox {
	float leftX;
	float rightX;
	float leftY;
	float rightY;
	float leftZ;
	float rightZ;
};

struct BvhArrayNode {
	int leftNodeIndex;
	int rightNodeIndex;
	AABBBox AABB;
	int meshIndex;
};

struct Material {
	//bxdfPara.x��ʾroughness��y��ʾmetallic��z��ʾrefractivity����z = 1��ʾ����������
	vec4 bxdfPara;
	vec4 kd;
	vec4 ks;
	vec4 ke;
};

struct Mesh {
	Material material;
	ivec2 indexInIndicesArray;
	AABBBox AABB;
};

struct Ray {
	vec3 startPos;
	vec3 direction;
	vec3 normal;	//�����Ҫ�Ǽ�¼��һ����ɫ��ķ���
	vec3 radiance;
	float depth;
};

struct GaussianPara {
	vec2 mean;
	float mixWeight;
	mat2 covarianceMatrix;
};

struct Sufficient_Statistic {
	vec2 ss2;	//��˹�ֲ��Ը������Ĺ��ױ�������������λ�õĳ˻�֮��
	float ss1;		//��˹�ֲ���ÿ�������Ĺ��ױ���֮��
	mat2 ss3;	//������λ����������ת�õĳ˻����Ը�˹�ֲ��Ը������Ĺ��ױ����ĳ˻�֮��
};

struct Photon {
	vec2 direction[K];	//����ά����תΪ2ά����ͬ��Բӳ��
	float weight;	//���� = 0.299f * R + 0.587f * G + 0.114f * B
	vec4 direction_3D;
	vec4 hitPos;
	vec4 startPos;
};

struct GMMPara {
	GaussianPara gaussianParas[K];
};

layout(set = 0, binding = 0) uniform LightUniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 lightPos;
	vec4 normal;
	vec4 size;
	vec4 ex, ey;
} lubo;

layout(set = 1, binding = 0) uniform cameraUniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 randomNumber;	//xyz�����������w��֡��
} cubo;

layout(set = 2, binding = 0, std430) readonly buffer BvhArray {
	BvhArrayNode bvhArrayNode[];
};
layout(set = 2, binding = 1, std430) readonly buffer Vertices {
	Vertex vertices[];
};
layout(set = 2, binding = 2, std430) readonly buffer Indices {
	uint indices[];
};
layout(set = 2, binding = 3, std430) readonly buffer Meshs {
	Mesh meshs[];
};

layout(set = 2, binding = 6) uniform GMMConstant{
	vec4 voxelStartPos;
	ivec4 voxelNum;
	float voxelSize;
} gmmConstant;

//function
//----------------------------randomNumber-------------------------------------------

uint pcg(inout uint state)
{
	uint prev = state * 747796405u + 2891336453u;
	uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
	state = prev;
	return (word >> 22u) ^ word;
}

uvec2 pcg2d(uvec2 v)
{
	v = v * 1664525u + 1013904223u;
	v.x += v.y * 1664525u;
	v.y += v.x * 1664525u;
	v = v ^ (v >> 16u);
	v.x += v.y * 1664525u;
	v.y += v.x * 1664525u;
	v = v ^ (v >> 16u);
	return v;
}

float rand(inout uint seed)
{
	uint val = pcg(seed);
	return (float(val) * (1.0 / float(0xffffffffu)));
}

//�Ͳ�������
float RadicalInverse_VdC(uint bits)
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
vec2 Hammersley(uint i, uint N)
{
	return vec2(float(i) / float(N), RadicalInverse_VdC(i));
}

//-----------------------------Pack----------------------------------------------
uint myPackUnorm2x16(vec2 value) {
	uvec2 packed = uvec2(clamp(value * 32768.0f, 0.0f, 65535.0f));
	return uint(uint(packed.y) & 0x0000FFFF) << 16 | (uint(packed.x) & 0x0000FFFF);
}

vec2 myUnPackUnorm2x16(uint value) {
	vec2 result = vec2(float(value & 0x0000FFFF), float((value & 0x0000FFFF) >> 16));
	return result / 32768.0f;
}

//----------------------------createRay------------------------------------------
mat3 createTBN(vec3 normal) {

	vec3 tangent;
	if (abs(normal.x) > abs(normal.y))
		tangent = vec3(normal.z, 0, -normal.x);
	else
		tangent = vec3(0, -normal.z, normal.y);
	tangent = normalize(tangent);
	vec3 bitangent = normalize(cross(normal, tangent));
	return mat3(tangent, bitangent, normal);

}

//������ʼ����
Ray makeStartPhoton(inout uint randomNumberSeed, inout float pdf) {

	Ray ray;
	//vec2 randomNumber = Hammersley(uint(rand(randomNumberSeed) * 100), 100) * 0.5f + rand(randomNumberSeed) * 0.5f;
	//ray.startPos = lubo.lightPos.xyz + lubo.size.xyz * vec3(randomNumber.x, rand(randomNumberSeed), randomNumber.y);
	ray.startPos = lubo.lightPos.xyz + lubo.size.xyz * vec3(rand(randomNumberSeed), rand(randomNumberSeed), rand(randomNumberSeed));
	ray.normal = lubo.normal.xyz;

	//cos��Ȩ
	randomNumberSeed++;
	vec2 randomNumberH = vec2(rand(randomNumberSeed), rand(randomNumberSeed));
	float phi = 2.0 * PI * randomNumberH.x;
	float cosTheta = sqrt(1.0 - randomNumberH.y);	// 1�����Ǿ��ȷֲ�����ƽ�����Ǿ��ȷֲ������Կ���ֱ����
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	ray.direction = normalize(vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta));

	ray.direction = normalize(createTBN(ray.normal) * ray.direction);
	ray.depth = 100.0f;
	ray.radiance = lightStrength / lightA;

	pdf = cosTheta / PI;

	return ray;

}

//--------------------------------------EnvironmentMapping------------------------------------

vec2 concentricMapping_hemisphere_3DTo2D(vec3 direction, mat3 TBN) {

	direction = transpose(TBN) * direction;
	if (direction.z < 0.0f) {
		return vec2(-1.0f);
	}

	float x = max(abs(direction.x), abs(direction.y));
	float y = min(abs(direction.x), abs(direction.y));
	float r = sqrt(1.0f - direction.z);

	float alpha = x == 0.0f ? 0.0f : y / x;
	float phi_2DivPI = (0.00000406531 + 0.636227 * alpha +
		0.00615523 * alpha * alpha -
		0.247326 * alpha * alpha * alpha +
		0.0881627 * alpha * alpha * alpha * alpha +
		0.0419157 * alpha * alpha * alpha * alpha * alpha -
		0.0251427 * alpha * alpha * alpha * alpha * alpha * alpha);

	float u = r;
	float v = 2.0f * phi_2DivPI * u;
	if (abs(direction.x) < abs(direction.y)) {
		float temp = u;
		u = v;
		v = temp;
	}

	u *= sign(direction.x);
	v *= sign(direction.y);

	return vec2(u, v);	// * 0.5f + 0.5f;

}

vec3 concentricMapping_hemisphere_2DTo3D(vec2 direction, mat3 TBN) {

	direction = direction;	// * 2.0f - 1.0f;
	float u = max(abs(direction.x), abs(direction.y));
	float v = min(abs(direction.x), abs(direction.y));

	float r = u;
	float phi = PI / 4 * v / u;

	float x = cos(phi) * r * sqrt(2.0f - r * r);
	float y = sin(phi) * r * sqrt(2.0f - r * r);
	float z = 1.0f - r * r;

	if (abs(direction.x) < abs(direction.y)) {
		float temp = x;
		x = y;
		y = temp;
	}

	x *= sign(direction.x);
	y *= sign(direction.y);

	return normalize(TBN * vec3(x, y, z));

}

void getHitPointNormalAndTBN(uint vertexIndex, inout vec3 normal, inout mat3 TBN) {

	//�����淨��
	vec3 P0 = vertices[vertexIndex].pos.xyz;
	vec3 P1 = vertices[vertexIndex + 1].pos.xyz;
	vec3 P2 = vertices[vertexIndex + 2].pos.xyz;

	vec3 tangent = normalize(P1 - P0);
	vec3 bitangent = normalize(P2 - P0);
	normal = normalize(cross(tangent, bitangent));
	bitangent = normalize(cross(normal, tangent));
	TBN = mat3(tangent, bitangent, normal);

}

//-------------------------------------------------------probability distribution---------------------------------------------------------

//��Ԫ��˹�ֲ�
float GaussianPdf(GaussianPara gp, vec2 pos) {

	float x = pos.x;
	float y = pos.y;
	float mu1 = gp.mean.x;	//��ֵ
	float mu2 = gp.mean.y;
	float d1 = sqrt(gp.covarianceMatrix[0].x);	//��׼��
	float d2 = sqrt(gp.covarianceMatrix[1].y);
	float d3 = gp.covarianceMatrix[0].y;	//Э����
	float rho = d3 / d1 / d2;	//���ϵ��
	float pdf = 1 / (2.0f * PI * d1 * d2 * sqrt(1.0f - rho * rho)) * exp(-0.5f / (1.0f - rho * rho) * ((x - mu1) * (x - mu1) / d1 / d1 - 2.0f * rho * (x - mu1) / d1 * (y - mu2) / d2 + (y - mu2) * (y - mu2) / d2 / d2));

	return clamp(pdf, 0.000001f, 1.0f);

}

//----------------------------getDirection-------------------------------------------

//��Ҫ���ʡ�TBN
vec3 getRayFromBxdf(Material material, mat3 TBN, vec3 i, vec3 normal, inout uint randomNumberSeed, inout float pdf) {

	//vec2 randomNumberH = Hammersley(uint(randomNumber * 100), 100);
	vec2 randomNumberH = vec2(rand(randomNumberSeed), rand(randomNumberSeed));

	float roughness = max(material.bxdfPara.x, 0.1f);	//���ֲڶ�Ϊ0��D����0/0����������������Ƿ�������a4�������ֻᵼ�²���������0ʱ��ֵ�Ĵ���
	float a2 = roughness * roughness * roughness * roughness;
	float phi = 2.0 * PI * randomNumberH.x;
	//��roughness��1����cosTheta��sqrt(1-randomNumberH.y)��������ȫ�ֲڵ�����£�����cos��Ȩ����pdfҲ����cosTheta / PI
	float cosTheta = sqrt((1.0 - randomNumberH.y) / (1.0 + (a2 - 1.0) * randomNumberH.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	vec3 h = normalize(vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta));

	float d = (a2 - 1) * cosTheta * cosTheta + 1;
	float D = a2 / (PI * d * d);
	pdf = D * cosTheta;	//h��pdf

	//h = normalize(TBN * h);
	//vec3 rayDirection = normalize(2.0f * dot(h, i) * h - i);
	//
	//if(dot(rayDirection, normal) > 0.0f){
	//	pdf = pdf / (4 * dot(h, i));	//�����pdf
	//}else{
	//	rayDirection = h;	//��h������������
	//}

	vec3 rayDirection = normalize(createTBN(reflect(-i, normal)) * h);
	int k = 0;
	while (dot(rayDirection, normal) <= 0.0f && k < 5) {
		rayDirection = normalize(rayDirection + reflect(-i, normal));
		k++;
	}
	if (k == 5) {
		rayDirection = reflect(-i, normal);
	}

	pdf = clamp(pdf, 0.01f, 1.0f);
	return rayDirection;

}

//-----------------------------------hitTest-----------------------------------------------

//AABB��ײ��⣬-1��ʾû����ײ����0��ʾ��ײ�����Ҳ����ڳ����ڲ���1��ʾ�ڳ����ڲ�
bool hitAABB(AABBBox AABB, Ray ray) {

	//�жϹ����ǲ����ڳ����ڲ������ģ�����ǻ�����ֱ��������ĳ���
	//��Ȼ�ᵼ��ÿ�ζ�����������AABB�ټ��һ��hitMesh
	if (ray.startPos.x > AABB.leftX && ray.startPos.x < AABB.rightX &&
		ray.startPos.y > AABB.leftY && ray.startPos.y < AABB.rightY &&
		ray.startPos.z > AABB.leftZ && ray.startPos.z < AABB.rightZ) {
		return true;
	}

	float maxInTime = 0.0f;
	float minOutTime = 1000000.0f;	//������˵

	if (ray.direction.x != 0) {	//ֱ�����涼������
		float leftX = (AABB.leftX - ray.startPos.x) / ray.direction.x;
		float rightX = (AABB.rightX - ray.startPos.x) / ray.direction.x;
		maxInTime = max(min(leftX, rightX), maxInTime);
		minOutTime = min(max(leftX, rightX), minOutTime);
	}

	if (ray.direction.y != 0) {
		float leftY = (AABB.leftY - ray.startPos.y) / ray.direction.y;
		float rightY = (AABB.rightY - ray.startPos.y) / ray.direction.y;
		maxInTime = max(min(leftY, rightY), maxInTime);
		minOutTime = min(max(leftY, rightY), minOutTime);
	}

	if (ray.direction.z != 0) {
		float leftZ = (AABB.leftZ - ray.startPos.z) / ray.direction.z;
		float rightZ = (AABB.rightZ - ray.startPos.z) / ray.direction.z;
		maxInTime = max(min(leftZ, rightZ), maxInTime);
		minOutTime = min(max(leftZ, rightZ), minOutTime);
	}

	if (minOutTime < maxInTime) {
		return false;
	}

	//ֱ���ð�Χ���е�����ܻᵼ��ǰ���mesh��AABB��depth�����Ⱥ���Ĵ󣬵��±��޳�
	if (maxInTime > ray.depth) {
		return false;	//��Ȳ��Բ�ͨ��
	}

	return true;


}

//������ײ���mesh��������������Ƭ�ĵ�һ��indicis������û��ײ���򲻶�
void hitMesh(inout Ray ray, uint meshIndex, inout ivec2 result) {

	Mesh mesh = meshs[meshIndex];

	uint startVertexIndex = mesh.indexInIndicesArray.x;
	uint endVertexIndex = mesh.indexInIndicesArray.y;

	for (uint i = startVertexIndex; i < endVertexIndex; i += 3) {
		vec3 P0 = vertices[indices[i]].pos.xyz;
		vec3 P1 = vertices[indices[i + 1]].pos.xyz;
		vec3 P2 = vertices[indices[i + 2]].pos.xyz;

		vec3 tangent = normalize(P1 - P0);
		vec3 bitangent = normalize(P2 - P0);
		vec3 normal = normalize(cross(tangent, bitangent));
		if (dot(normal, -ray.direction) <= 0) {
			continue;
		}

		vec3 E1 = P1 - P0;
		vec3 E2 = P2 - P0;
		vec3 S = ray.startPos - P0;
		vec3 S1 = cross(ray.direction, E2);
		vec3 S2 = cross(S, E1);

		vec3 tbb = 1 / dot(S1, E1) * vec3(dot(S2, E2), dot(S1, S), dot(S2, ray.direction));
		if (tbb.x > 0 && (1.0f - tbb.y - tbb.z) > 0 && tbb.y > 0 && tbb.z > 0) {	//����
			if (tbb.x > ray.depth) {
				continue;	//��Ȳ���ûͨ��
			}
			result = ivec2(meshIndex, indices[i]);
			ray.depth = tbb.x;
			return;
		}
	}

}

//���ڲ���ʹ�õݹ飬������Ҫ����ջ�ķ�ʽѭ����ȡ
ivec2 hitScene(inout Ray ray) {

	ivec2 result = ivec2(-1, -1);
	//ջ�Ĵ�С��Ҫ�ͺ�bvh���ڵ�������ͬ����������Ӧ�ô�CPU��uniform�����ģ���������д�ˣ�ֱ���ø���СΪ10�����飬�����������С����Ӧ�ù�����
	//��һ����ʾsceneIndex���ڶ������������ĸ��������������Ǹ�����Ƿ�Ҫȥ��
	ivec3 sceneStack[15] = ivec3[15](ivec3(0, 1, 1), ivec3(-1), ivec3(-1), ivec3(-1), ivec3(-1), ivec3(-1),
		ivec3(-1), ivec3(-1), ivec3(-1), ivec3(-1), ivec3(-1), ivec3(-1), ivec3(-1), ivec3(-1), ivec3(-1));
	int stackTop = 0;
	while (stackTop >= 0) {

		if (sceneStack[stackTop].z == -1) {
			int isRight = sceneStack[stackTop].y;
			sceneStack[stackTop] = ivec3(-1, -1, -1);
			stackTop -= 1;
			if (isRight == 1) {
				sceneStack[stackTop].z = -1;
			}
			continue;
		}

		BvhArrayNode scene = bvhArrayNode[sceneStack[stackTop].x];
		if (!hitAABB(scene.AABB, ray)) {
			int isRight = sceneStack[stackTop].y;
			sceneStack[stackTop] = ivec3(-1, -1, -1);
			stackTop -= 1;
			if (isRight == 1) {
				sceneStack[stackTop].z = -1;
			}
			continue;
		}

		//����Ҷ�ӽڵ㣬��ֱ�ӽ���mesh��ײ
		if (scene.leftNodeIndex == -1) {
			hitMesh(ray, scene.meshIndex, result);
			int isRight = sceneStack[stackTop].y;
			sceneStack[stackTop] = ivec3(-1, -1, -1);
			stackTop -= 1;
			if (isRight == 1) {
				sceneStack[stackTop].z = -1;
			}
			continue;
		}

		//�Ƚ���������ѹջ���ȱ�����������������
		stackTop += 1;
		sceneStack[stackTop] = ivec3(scene.rightNodeIndex, 1, 1);
		stackTop += 1;
		sceneStack[stackTop] = ivec3(scene.leftNodeIndex, 0, 1);

	}

	return result;

}

//-------------------------------------------bxdf---------------------------------------------------------

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

//����ǰhitPoint��i����ôcurrentPos��i-1��startPos��i-2��Ҳ����˵ֻ�е�i >= 2ʱ�ż���weight
vec3 getFr(vec3 startPos, vec3 currentPos, vec3 currentNormal, vec3 hitPos, Material material) {

	vec3 albedo = material.kd.rgb;
	float roughness = clamp(material.bxdfPara.x, 0.1f, 1.0f);
	float metallic = material.bxdfPara.y;
	float refractivity = material.bxdfPara.z;

	float diff_fr = 1 / (2 * PI);

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);

	vec3 i = normalize(hitPos - currentPos);
	vec3 o = normalize(startPos - currentPos);
	vec3 h = normalize(i + o);
	float NDF = DistributionGGX(currentNormal, h, roughness);
	float G = GeometrySmith(currentNormal, o, i, roughness);
	vec3 F = fresnelSchlick(max(dot(h, o), 0.0), F0);

	vec3 nominator = NDF * G * F;
	float denominator = 4.0 * max(dot(currentNormal, o), 0.0) * max(dot(currentNormal, i), 0.0) + 0.001;
	vec3 spec_fr = nominator / denominator;

	vec3 ks = F;
	vec3 kd = vec3(1.0) - ks;
	kd *= (1.0 - metallic) * material.bxdfPara.x;

	return kd * albedo * diff_fr + spec_fr;

}

//----------------------------------------------TestFunction---------------------------------------------------
/*
void hitTest(vec3 hitPos, vec3 normal) {

	Ray ray;
	ray.startPos = cubo.cameraPos.xyz;
	ray.direction = normalize(hitPos - cubo.cameraPos.xyz);
	ray.depth = 100.0f;
	ray.normal = ray.direction;
	ivec2 result = hitScene(ray);
	if (result.x == -1 || abs(ray.depth - length(hitPos - cubo.cameraPos.xyz)) > 0.1f) {
		return;
	}

	vec4 clipPos = cubo.proj * cubo.view * vec4(hitPos, 1.0f);
	vec4 ndcPos = clipPos / clipPos.w;
	ivec2 texelUV = ivec2((ndcPos.xy * 0.5f + 0.5f) * (gl_WorkGroupSize * gl_NumWorkGroups).xy);
	imageStore(pathTracingResult, texelUV, vec4(10.0f));
}
*/