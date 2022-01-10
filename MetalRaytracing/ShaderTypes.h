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
    BufferIndexUniforms                         = 0,
    BufferIndexInstanceAccelerationStructure    = 1,
    BufferIndexRandom                           = 2,
    BufferIndexVertexColor                      = 3,
    BufferIndexVertexNormals                    = 4,
    BufferIndexResources                        = 5,
    BufferIndexLights                           = 6,
    BufferIndexInstances                        = 7,
    BufferIndexAccelerationStructure            = 8,
    BufferIndexInstanceDescriptors              = 9,
};

typedef NS_ENUM(NSInteger, TextureIndex)
{
    TextureIndexAccumulation            = 0,
    TextureIndexPreviousAccumulation    = 1,
    TextureIndexRandom                  = 2
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

#endif /* ShaderTypes_h */

