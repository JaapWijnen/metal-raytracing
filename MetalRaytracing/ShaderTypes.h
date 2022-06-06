//
//  ShaderTypes.h
//  Raytracing Shared
//
//  Created by Jaap Wijnen on 21/11/2021.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_LIGHT    2

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE)

#define RAY_MASK_PRIMARY   (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT)
#define RAY_MASK_SHADOW    GEOMETRY_MASK_GEOMETRY
#define RAY_MASK_SECONDARY GEOMETRY_MASK_GEOMETRY


typedef NS_ENUM(NSInteger, BufferIndex)
{
    BufferIndexUniforms                                 = 0,
    BufferIndexRandom                                   = 2,
    BufferIndexVertexColor                              = 3,
    BufferIndexVertexNormals                            = 4,
    BufferIndexResources                                = 5,
    BufferIndexLights                                   = 6,
    BufferIndexInstances                                = 7,
    BufferIndexAccelerationStructure                    = 8,
    BufferIndexInstanceDescriptors                      = 9,
    BufferIndexRays                                     = 10,
    BufferIndexShadowRays                               = 11,
    BufferIndexIntersections                            = 12,
    BufferIndexRayColors                                = 13,
    BufferIndexLightColors                              = 14,
    BufferIndexAccumulatedColors                        = 15,
    BufferIndexBounce                                   = 16,
    BufferIndexRayOrigins                               = 17,
    BufferIndexRayDirections                            = 18,
    BufferIndexRayMaxDistances                          = 19,
    BufferIndexShadowRayOrigins                         = 20,
    BufferIndexShadowRayDirections                      = 21,
    BufferIndexShadowRayMaxDistances                    = 22,
    BufferIndexIntersectionDistances                    = 23,
    BufferIndexIntersectionInstanceIDs                  = 24,
    BufferIndexIntersectionGeometryIDs                  = 25,
    BufferIndexIntersectionPrimitiveIDs                 = 26,
    BufferIndexIntersectionTriangleCoordinates          = 27,
    BufferIndexIntersectionWorldSpaceIntersectionPoints = 28,
    BufferIndexRayBinningCoordinates                    = 29,
    BufferIndexRaysPerTile                              = 30,
};

typedef NS_ENUM(NSInteger, TextureIndex)
{
    TextureIndexAccumulationTarget      = 0,
    TextureIndexPreviousAccumulation    = 1,
    TextureIndexRandom                  = 2,
    TextureIndexRayColor                = 3,
    TextureIndexRenderTarget            = 4,
    TextureIndexSurfaceColor            = 5,
    TextureIndexWorldSpaceSurfaceNormal = 6,
    TextureIndexRayOrigins              = 7,
    TextureIndexRayDirections           = 8,
    TextureIndexShadowRayOrigins        = 9,
    TextureIndexShadowRayDirections     = 10,
};

typedef NS_ENUM(NSInteger, VertexAttribute)
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
};

struct Camera {
    vector_float3 position;
    vector_float3 right;
    vector_float3 up;
    vector_float3 forward;
};

typedef NS_ENUM(NSInteger, LightType)
{
    LightTypeUnused = 0,
    LightTypeSunlight = 1,
    LightTypeSpotlight = 2,
    LightTypePointlight = 3,
    LightTypeAreaLight = 4
};

struct Light {
    LightType type;
    vector_float3 position;
    vector_float3 color;
    // area light
    vector_float3 forward;
    vector_float3 right;
    vector_float3 up;
    // spot light
    float coneAngle;
    vector_float3 direction;
};

struct Uniforms
{
    int width;
    int height;
    int blocksWide;
    unsigned int frameIndex;
    int lightCount;
    struct Camera camera;
};

struct Material
{
    vector_float3 baseColor;
    vector_float3 specular;
    vector_float3 emission;
    float specularExponent;
    float refractionIndex;
    float dissolve;
};

struct Intersection
{
    float distance;
    int instanceID;
    int geometryID;
    int primitiveID;
    vector_float2 coordinates;
    vector_float3 worldSpaceIntersectionPoint;
};

struct Ray
{
    vector_float3 origin;
    vector_float3 direction;
    float min_distance;
    float max_distance;
};

#endif /* ShaderTypes_h */

