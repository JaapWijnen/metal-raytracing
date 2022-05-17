//
//  RTShaders.metal
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 20/04/2022.
//

#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

#include "ShaderFunctions.h"

kernel void sd_primary_ray_kernel(uint2 tid [[ thread_position_in_grid ]],
                                  texture2d<float, access::write> rayOrigins [[ texture(TextureIndexRayOrigins) ]],
                                  texture2d<float, access::write> rayDirections [[ texture(TextureIndexRayDirections) ]],
                                  device float *rayMaxDistances [[ buffer(BufferIndexRayMaxDistances) ]],
                                  constant Uniforms &uniforms [[ buffer(BufferIndexUniforms) ]],
                                  texture2d<unsigned int, access::read> randomTexture [[ texture(TextureIndexRandom) ]])
{
    // TODO see if there's a difference between accessing these in a tile same as the threadgroup. ie store them either as tiles in the buffer as well or store a tile in successive memory.
    //unsigned int rayIdx = tid.y * uniforms.width + tid.x;
        
    // TODO look into bounding boxes acceleration structure improvements (that should be in the algo itself right?)
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        unsigned int random = randomTexture.read(tid).x;
        float2 offset = float2(halton(random + uniforms.frameIndex, 0),
                               halton(random + uniforms.frameIndex, 1));
        
        float2 pixel = (float2)tid + offset;
        
        float2 uv = pixel / float2(uniforms.width, uniforms.height);
        uv = uv * 2.0 - 1.0;
        uv.y = -uv.y; // transform to metal normalised device coordinates (otherwise intermediate textures are flipped in metal graph).
        
        constant Camera &camera = uniforms.camera;
        
        unsigned int rayIndex = tid.y * uniforms.width + tid.x;
        
        rayOrigins.write(float4(camera.position, 1.0), tid);
        rayDirections.write(float4(normalize(uv.x * camera.right + uv.y * camera.up + camera.forward), 1.0), tid);
        rayMaxDistances[rayIndex] = INFINITY;
    }
}

kernel void sd_intersection_kernel(uint2 tid [[ thread_position_in_grid ]],
                                   texture2d<float, access::read> rayOrigins [[ texture(TextureIndexRayOrigins) ]],
                                   texture2d<float, access::read> rayDirections [[ texture(TextureIndexRayDirections) ]],
                                   constant float *rayMaxDistances [[ buffer(BufferIndexRayMaxDistances) ]],
                                   device float *intersectionDistances [[ buffer(BufferIndexIntersectionDistances) ]],
                                   device uint8_t *intersectionInstanceIDs [[ buffer(BufferIndexIntersectionInstanceIDs) ]],
                                   device uint8_t *intersectionGeometryIDs [[ buffer(BufferIndexIntersectionGeometryIDs) ]],
                                   device unsigned int *intersectionPrimitiveIDs [[ buffer(BufferIndexIntersectionPrimitiveIDs) ]],
                                   device half2 *intersectionTriangleCoordinates [[ buffer(BufferIndexIntersectionTriangleCoordinates) ]],
                                   device float3 *intersectionWorldSpaceIntersectionPoints [[ buffer(BufferIndexIntersectionWorldSpaceIntersectionPoints) ]],
                                   constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                                   acceleration_structure<instancing> accelerationStructure [[ buffer(BufferIndexAccelerationStructure) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        unsigned int rayIndex = tid.y * uniforms.width + tid.x;

        float maxDistance = rayMaxDistances[rayIndex];
        
        // TODO: should be a bool instead of max distance (since always infinite)
        if (maxDistance < 0) {
            return;
        }
        
        intersector<triangle_data, instancing> intersector;
                
        intersector.assume_geometry_type(geometry_type::triangle);
        intersector.force_opacity(forced_opacity::opaque);
        
        ray ray;
        ray.origin = rayOrigins.read(tid).xyz;
        ray.direction = rayDirections.read(tid).xyz;
        ray.min_distance = 0;
        ray.max_distance = INFINITY;
        
        // Check for intersection between the ray and the acceleration structure.
        //intersection = i.intersect(ray, accelerationStructure, bounce == 0 ? RAY_MASK_PRIMARY : RAY_MASK_SECONDARY);
        typename ::intersector<triangle_data, instancing>::result_type i;
        i = intersector.intersect(ray, accelerationStructure);
        // Stop if the ray didn't hit anything and has bounced out of the scene.
        
        if (i.type == intersection_type::none) {
            intersectionDistances[rayIndex] = -1.0;
            return;
        }
        
        intersectionDistances[rayIndex] = i.distance;
        intersectionInstanceIDs[rayIndex] = i.instance_id;
        intersectionGeometryIDs[rayIndex] = i.geometry_id;
        intersectionPrimitiveIDs[rayIndex] = i.primitive_id;
        intersectionTriangleCoordinates[rayIndex] = (half2)i.triangle_barycentric_coord;
        intersectionWorldSpaceIntersectionPoints[rayIndex] = ray.origin + ray.direction * i.distance;
    }
}

// Interpolates vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection struct
template<typename T>
T interpolateVertexAttribute2(device T *attributes, half2 coordinates, int primitiveID, device int *vertexIndices) {
    half3 uvw;
    uvw.xy = coordinates;
    uvw.z = 1.0 - uvw.x - uvw.y;
    unsigned int triangleIndex = primitiveID;
    unsigned int index1 = vertexIndices[triangleIndex * 3 + 1];
    unsigned int index2 = vertexIndices[triangleIndex * 3 + 2];
    unsigned int index3 = vertexIndices[triangleIndex * 3 + 0];
    T T0 = attributes[index1];
    T T1 = attributes[index2];
    T T2 = attributes[index3];
    return uvw.x * T0 + uvw.y * T1 + uvw.z * T2;
}

constant int resourcesStride [[function_constant(0)]];
constant int maxSubmeshes [[function_constant(1)]];

kernel void sd_shade_kernel(uint2 tid [[ thread_position_in_grid ]],
                            constant float *intersectionDistances [[ buffer(BufferIndexIntersectionDistances) ]],
                            constant uint8_t *intersectionInstanceIDs [[ buffer(BufferIndexIntersectionInstanceIDs) ]],
                            constant uint8_t *intersectionGeometryIDs [[ buffer(BufferIndexIntersectionGeometryIDs) ]],
                            constant unsigned int *intersectionPrimitiveIDs [[ buffer(BufferIndexIntersectionPrimitiveIDs) ]],
                            constant half2 *intersectionTriangleCoordinatesB [[ buffer(BufferIndexIntersectionTriangleCoordinates) ]],
                            constant Uniforms &uniforms [[ buffer(BufferIndexUniforms) ]],
                            constant MTLAccelerationStructureInstanceDescriptor *instances [[ buffer(BufferIndexInstanceDescriptors) ]],
                            constant Resource *resources [[ buffer(BufferIndexResources) ]],
                            //device float3 *rayColors [[ buffer(BufferIndexRayColors) ]],
                            device float *rayMaxDistances [[ buffer(BufferIndexRayMaxDistances) ]],
                            device float *shadowRayMaxDistances [[ buffer(BufferIndexShadowRayMaxDistances) ]],
                            texture2d<float, access::write> surfaceColorTexture [[ texture(TextureIndexSurfaceColor) ]],
                            texture2d<float, access::write> worldSpaceSurfaceNormalTexture [[ texture(TextureIndexWorldSpaceSurfaceNormal) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        unsigned int rayIndex = tid.y * uniforms.width + tid.x;
        
        float intersectionDistance = intersectionDistances[rayIndex];
        
        if (rayMaxDistances[rayIndex] < 0 || intersectionDistance < 0) {
            rayMaxDistances[rayIndex] = -1.0;
            shadowRayMaxDistances[rayIndex] = -1.0;
            return;
        }
            
        // The ray hit something. Look up the transformation matrix for this instance.
        float4x4 objectToWorldSpaceTransform(1.0f);
        
        int instanceID = intersectionInstanceIDs[rayIndex];
        int geometryID = intersectionGeometryIDs[rayIndex];
        int primitiveID = intersectionPrimitiveIDs[rayIndex];
        half2 intersectionTriangleCoordinates = intersectionTriangleCoordinatesB[rayIndex];
        
        MTLAccelerationStructureInstanceDescriptor instance = instances[instanceID];

        for (int column = 0; column < 4; column++)
            for (int row = 0; row < 3; row++)
                objectToWorldSpaceTransform[column][row] = instance.transformationMatrix[column][row];
        
        // Compute intersection point in world space.
        int resourceIndex = instanceID * maxSubmeshes + geometryID;
        Resource resource = resources[resourceIndex];
        
        float3 objectSpaceSurfaceNormal = interpolateVertexAttribute2(resource.normals, intersectionTriangleCoordinates, primitiveID, resource.indices);
        float3 worldSpaceSurfaceNormal = (objectToWorldSpaceTransform * float4(objectSpaceSurfaceNormal, 0)).xyz;
        
        worldSpaceSurfaceNormal = normalize(worldSpaceSurfaceNormal);
        worldSpaceSurfaceNormalTexture.write(float4(worldSpaceSurfaceNormal, 1.0), tid);
        
        surfaceColorTexture.write(float4(resource.material->baseColor, 1.0), tid);
    }
}

kernel void sd_shade2_kernel(uint2 tid [[ thread_position_in_grid ]],
                             constant Uniforms &uniforms [[ buffer(BufferIndexUniforms) ]],
                             constant int &bounce [[ buffer(BufferIndexBounce) ]],
                             constant Light *lights [[ buffer(BufferIndexLights) ]],
                             device float3 *lightColors [[ buffer(BufferIndexLightColors) ]],
                             constant float3 *intersectionWorldSpaceIntersectionPoints [[ buffer(BufferIndexIntersectionWorldSpaceIntersectionPoints) ]],
                             texture2d<unsigned int, access::read> randomTexture [[ texture(TextureIndexRandom) ]],
                             texture2d<float, access::read_write> rayColors [[ texture(TextureIndexRayColor) ]],
                             texture2d<float, access::read> surfaceColorTexture [[ texture(TextureIndexSurfaceColor) ]],
                             texture2d<float, access::read> worldSpaceSurfaceNormalTexture [[ texture(TextureIndexWorldSpaceSurfaceNormal) ]],
                             texture2d<float, access::write> shadowRayOrigins [[ texture(TextureIndexShadowRayOrigins) ]],
                             texture2d<float, access::write> shadowRayDirections [[ texture(TextureIndexShadowRayDirections) ]],
                             device float *shadowRayMaxDistances [[ buffer(BufferIndexShadowRayMaxDistances) ]],
                             constant float *rayMaxDistances [[ buffer(BufferIndexRayMaxDistances) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        unsigned int rayIndex = tid.y * uniforms.width + tid.x;
                
        if (rayMaxDistances[rayIndex] < 0) {
            return;
        }

        // Choose a random light source to sample.
        //int bounce = 0;
        unsigned int offset = randomTexture.read(tid).x;
        float lightSample = halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 0);
        int lightIndex = min((int)(lightSample * uniforms.lightCount), uniforms.lightCount - 1);
        
        Light light = lights[lightIndex];
        
        float3 worldSpaceLightDirection;
        float lightDistance;
        float3 lightColor;
        
        float3 worldSpaceIntersectionPoint = intersectionWorldSpaceIntersectionPoints[rayIndex];
        
        if (light.type == LightTypeAreaLight) {
            
            // Choose a random point to sample on the light source.
            float2 r = float2(halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 1),
                              halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 2));

            // Sample the lighting between the intersection point and the point on the area light.
            sampleAreaLight(light, r, worldSpaceIntersectionPoint, worldSpaceLightDirection,
                            lightColor, lightDistance);
                        
        } else if (light.type == LightTypeSpotlight) {
            // Compute vector from sample point on light source to intersection point
            worldSpaceLightDirection = light.position - worldSpaceIntersectionPoint;
                    
            lightDistance = length(worldSpaceLightDirection);
            
            float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
            
            // Normalize the light direction
            worldSpaceLightDirection *= inverseLightDistance;

            lightColor = 0.0;
            
            float3 coneDirection = normalize(light.direction);
            float spotResult = dot(-worldSpaceLightDirection, coneDirection);
            
            if (spotResult > cos(light.coneAngle)) {
                // Light falls off with the inverse square of the distance to the intersection point
                lightColor = light.color * inverseLightDistance * inverseLightDistance;
            }
        } else if (light.type == LightTypePointlight) {
            worldSpaceLightDirection = light.position - worldSpaceIntersectionPoint;
            lightDistance = length(worldSpaceLightDirection);
            float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
            worldSpaceLightDirection *= inverseLightDistance;
            lightColor = light.color * inverseLightDistance * inverseLightDistance;
        } else  { // light.type == LightTypeSunlight
            worldSpaceLightDirection = -normalize(light.direction);
            lightDistance = INFINITY;
            lightColor = light.color;
        }
        
        float3 worldSpaceSurfaceNormal = worldSpaceSurfaceNormalTexture.read(tid).xyz;
        
        // Scale the light color by the cosine of the angle between the light direction and
        // surface normal.
        lightColor *= saturate(dot(worldSpaceSurfaceNormal, worldSpaceLightDirection));

        // Scale the light color by the number of lights to compensate for the fact that
        // the sample only samples one light source at random.
        lightColor *= uniforms.lightCount;
        
        float3 surfaceColor = surfaceColorTexture.read(tid).xyz;

        // Scale the ray color by the color of the surface. This simulates light being absorbed into
        // the surface.
        float3 rayColor = surfaceColor;
        rayColor *= rayColors.read(tid).xyz;
        rayColors.write(float4(rayColor, 1.0), tid);
        
        // TODO: Could be split out if light direction and distance are written to buffer
        
        if (length(lightColor) > 0.0001) {
            // Compute the shadow ray. The shadow ray checks if the sample position on the
            // light source is visible from the current intersection point.
            // If it is, the lighting contribution is added to the output image.
            

            // Add a small offset to the intersection point to avoid intersecting the same
            // triangle again.
            shadowRayOrigins.write(float4(worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f, 1.0), tid);

            // Travel towards the light source.
            shadowRayDirections.write(float4(worldSpaceLightDirection, 1.0), tid);

            // Don't overshoot the light source.
            shadowRayMaxDistances[rayIndex] = lightDistance - 1e-3f;
            
            lightColors[rayIndex] = lightColor;
        } else {
            shadowRayMaxDistances[rayIndex] = -1.0;
        }
    }
}

kernel void sd_update_rays_kernel(uint2 tid [[ thread_position_in_grid ]],
                                  constant Uniforms &uniforms [[ buffer(BufferIndexUniforms) ]],
                                  constant int &bounce [[ buffer(BufferIndexBounce) ]],
                                  constant float3 *intersectionWorldSpaceIntersectionPoints [[ buffer(BufferIndexIntersectionWorldSpaceIntersectionPoints) ]],
                                  texture2d<unsigned int, access::read> randomTexture [[ texture(TextureIndexRandom) ]],
                                  texture2d<float, access::read> worldSpaceSurfaceNormalTexture [[ texture(TextureIndexWorldSpaceSurfaceNormal) ]],
                                  texture2d<float, access::write> rayOrigins [[ texture(TextureIndexRayOrigins) ]],
                                  texture2d<float, access::write> rayDirections [[ texture(TextureIndexRayDirections) ]],
                                  constant float *rayMaxDistances [[ buffer(BufferIndexRayMaxDistances) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        unsigned int rayIndex = tid.y * uniforms.width + tid.x;
                        
        if (rayMaxDistances[rayIndex] < 0) {
            return;
        }
        
        // TODO: Can be split out into separate pass to setup next rays! (maybe pass along data in memmoryless texture?)
        
        // Next choose a random direction to continue the path of the ray. This will
        // cause light to bounce between surfaces. The sample could apply a fair bit of math
        // to compute the fraction of light reflected by the current intersection point to the
        // previous point from the next point. However, by choosing a random direction with
        // probability proportional to the cosine (dot product) of the angle between the
        // sample direction and surface normal, the math entirely cancels out except for
        // multiplying by the surface color. This sampling strategy also reduces the amount
        // of noise in the output image.
        
        unsigned int offset = randomTexture.read(tid).x;
        float2 r = float2(halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 3),
                   halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 4));

        float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere(r);
        
        float3 worldSpaceSurfaceNormal = worldSpaceSurfaceNormalTexture.read(tid).xyz;
        worldSpaceSampleDirection = alignHemisphereWithNormal(worldSpaceSampleDirection, worldSpaceSurfaceNormal);
        
        float3 worldSpaceIntersectionPoint = intersectionWorldSpaceIntersectionPoints[rayIndex];
        
        rayOrigins.write(float4(worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f, 1.0), tid);
        rayDirections.write(float4(worldSpaceSampleDirection, 1.0), tid);
    }
}

kernel void sd_shadow_intersection_kernel(uint2 tid [[ thread_position_in_grid ]],
                                          texture2d<float, access::read> shadowRayOrigins [[ texture(TextureIndexShadowRayOrigins) ]],
                                          texture2d<float, access::read> shadowRayDirections [[ texture(TextureIndexShadowRayDirections) ]],
                                          constant float *shadowRayMaxDistances [[ buffer(BufferIndexShadowRayMaxDistances) ]],
                                          constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                                          acceleration_structure<instancing> accelerationStructure [[ buffer(BufferIndexAccelerationStructure) ]],
                                          //device float3 *rayColors [[ buffer(BufferIndexRayColors) ]],
                                          texture2d<float, access::read> rayColors [[ texture(TextureIndexRayColor) ]],
                                          constant float3 *lightColors [[ buffer(BufferIndexLightColors) ]],
                                          texture2d<float, access::read_write> accumulatedColors [[ texture(TextureIndexAccumulationTarget) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        intersector<triangle_data, instancing> intersector;
                
        intersector.assume_geometry_type(geometry_type::triangle);
        intersector.force_opacity(forced_opacity::opaque);
        // Shadow rays check only whether there is an object between the intersection point
        // and the light source. Tell Metal to return after finding any intersection.
        intersector.accept_any_intersection(true);

                    
        // Get the closest intersection, not the first intersection. This is the default, but
        // the sample adjusts this property below when it casts shadow rays.
        
        unsigned int rayIndex = tid.y * uniforms.width + tid.x;
        
        float maxDistance = shadowRayMaxDistances[rayIndex];
        
        if (maxDistance < 0) {
            return;
        }
            
        ray shadowRay;
        shadowRay.origin = shadowRayOrigins.read(tid).xyz;
        shadowRay.direction = shadowRayDirections.read(tid).xyz;
        shadowRay.min_distance = 0;
        shadowRay.max_distance = maxDistance;
        
        // Check for intersection between the ray and the acceleration structure.
        //intersection = i.intersect(ray, accelerationStructure, bounce == 0 ? RAY_MASK_PRIMARY : RAY_MASK_SECONDARY);
        typename ::intersector<triangle_data, instancing>::result_type i;
        /*if (useIntersectionFunctions)
            intersection = i.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW, intersectionFunctionTable);
        else
            intersection = i.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW);
         */
    //    intersection = i.intersect(shadowRay, accelerationStructure);
        
        i = intersector.intersect(shadowRay, accelerationStructure);

        // If there was no intersection, then the light source is visible from the original
        // intersection  point. Add the light's contribution to the image.
        if (i.type == intersection_type::none) {
            float3 rayColor = rayColors.read(tid).xyz;
            float3 lightColor = lightColors[rayIndex];
            float3 accumulatedColor = rayColor * lightColor;
            accumulatedColor += accumulatedColors.read(tid).xyz;
            accumulatedColors.write(float4(accumulatedColor, 1.0), tid);
        }
    }
}
