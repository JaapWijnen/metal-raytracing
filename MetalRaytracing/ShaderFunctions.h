//
//  ShaderFunctions.h
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 19/04/2022.
//

#ifndef ShaderFunctions_h
#define ShaderFunctions_h

#include "ShaderTypes.h"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;
using namespace raytracing;

struct Resource
{
    device float3 *normals;
    device int *indices;
    device Material *material;
};

float halton(int i, short d);
float3 alignHemisphereWithNormal(float3 sample, float3 normal);
float3 sampleCosineWeightedHemisphere(float2 u);
void sampleAreaLight(Light light,
                     float2 u,
                     float3 position,
                     thread float3 & lightDirection,
                     thread float3 & lightColor,
                     thread float & lightDistance);

// Interpolates vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection struct
template<typename T>
T interpolateVertexAttribute(device T *attributes, intersector<triangle_data, instancing>::result_type intersection, device int *vertexIndices) {
    float3 uvw;
    uvw.xy = intersection.triangle_barycentric_coord;
    uvw.z = 1.0 - uvw.x - uvw.y;
    unsigned int triangleIndex = intersection.primitive_id;
    unsigned int index1 = vertexIndices[triangleIndex * 3 + 1];
    unsigned int index2 = vertexIndices[triangleIndex * 3 + 2];
    unsigned int index3 = vertexIndices[triangleIndex * 3 + 0];
    T T0 = attributes[index1];
    T T1 = attributes[index2];
    T T2 = attributes[index3];
    return uvw.x * T0 + uvw.y * T1 + uvw.z * T2;
}

// Interpolates vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection struct
template<typename T>
T interpolateVertexAttribute(device T *attributes, Intersection intersection, device int *vertexIndices) {
    float3 uvw;
    uvw.xy = intersection.coordinates;
    uvw.z = 1.0 - uvw.x - uvw.y;
    unsigned int triangleIndex = intersection.primitiveID;
    unsigned int index1 = vertexIndices[triangleIndex * 3 + 1];
    unsigned int index2 = vertexIndices[triangleIndex * 3 + 2];
    unsigned int index3 = vertexIndices[triangleIndex * 3 + 0];
    T T0 = attributes[index1];
    T T1 = attributes[index2];
    T T2 = attributes[index3];
    return uvw.x * T0 + uvw.y * T1 + uvw.z * T2;
}

#endif /* ShaderFunctions_h */
