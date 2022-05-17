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

kernel void reset_ray_color_kernel(uint2 tid [[ thread_position_in_grid ]],
                                   texture2d<float, access::write> rayColor [[ texture(TextureIndexRayColor) ]],
                                   constant Uniforms &uniforms [[ buffer(BufferIndexUniforms) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        rayColor.write(float4(1.0), tid);
    }
}

kernel void reset_accumulation_texture_kernel(uint2 tid [[ thread_position_in_grid ]],
                                   constant Uniforms &uniforms [[ buffer(BufferIndexUniforms) ]],
                                   texture2d<float, access::write> accumulatedColor [[ texture(TextureIndexAccumulationTarget) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        accumulatedColor.write(float4(0.0), tid);
    }
}

kernel void primary_ray_kernel(uint2 tid [[ thread_position_in_grid ]],
                               device Ray *rays [[ buffer(BufferIndexRays) ]],
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
        device Ray &ray = rays[rayIndex];
        
        ray.origin = camera.position;
        ray.direction = normalize(uv.x * camera.right + uv.y * camera.up + camera.forward);
        ray.min_distance = 0;
        ray.max_distance = INFINITY;
    }
}

kernel void intersection_kernel(uint2 tid [[ thread_position_in_grid ]],
                                constant Ray *rays [[ buffer(BufferIndexRays) ]],
                                device Intersection *intersections [[ buffer(BufferIndexIntersections) ]],
                                constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                                acceleration_structure<instancing> accelerationStructure [[ buffer(BufferIndexAccelerationStructure) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        intersector<triangle_data, instancing> intersector;
                
        intersector.assume_geometry_type(geometry_type::triangle);
        intersector.force_opacity(forced_opacity::opaque);
        intersector.accept_any_intersection(false);
                    
        // Get the closest intersection, not the first intersection. This is the default, but
        // the sample adjusts this property below when it casts shadow rays.
        
        unsigned int rayIndex = tid.y * uniforms.width + tid.x;
        Ray tmpRay = rays[rayIndex];
        
        if (tmpRay.max_distance >= 0) {
            ray ray;
            ray.origin = tmpRay.origin;
            ray.direction = tmpRay.direction;
            ray.min_distance = tmpRay.min_distance;
            ray.max_distance = tmpRay.max_distance;
            
            // Check for intersection between the ray and the acceleration structure.
            //intersection = i.intersect(ray, accelerationStructure, bounce == 0 ? RAY_MASK_PRIMARY : RAY_MASK_SECONDARY);
            typename ::intersector<triangle_data, instancing>::result_type i;
            i = intersector.intersect(ray, accelerationStructure);
            // Stop if the ray didn't hit anything and has bounced out of the scene.
            
            device Intersection &intersection = intersections[rayIndex];
            intersection.distance = i.distance;
            intersection.instanceID = i.instance_id;
            intersection.primitiveID = i.primitive_id;
            intersection.geometryID = i.geometry_id;
            intersection.coordinates = i.triangle_barycentric_coord;
            intersection.worldSpaceIntersectionPoint = ray.origin + ray.direction * i.distance;
            
            if (i.type == intersection_type::none) {
                intersection.distance = -1.0;
            }
        }
    }
}

// Interpolates vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection struct
template<typename T>
T interpolateVertexAttribute2(device T *attributes, Intersection intersection, device int *vertexIndices) {
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

constant int resourcesStride [[function_constant(0)]];
constant int maxSubmeshes [[function_constant(1)]];

kernel void shade_kernel(uint2 tid [[ thread_position_in_grid ]],
                         constant Intersection *intersections [[ buffer(BufferIndexIntersections) ]],
                         constant Uniforms &uniforms [[ buffer(BufferIndexUniforms) ]],
                         constant int &bounce [[ buffer(BufferIndexBounce) ]],
                         constant MTLAccelerationStructureInstanceDescriptor *instances [[ buffer(BufferIndexInstanceDescriptors) ]],
                         constant Resource *resources [[ buffer(BufferIndexResources) ]],
                         constant Light *lights [[ buffer(BufferIndexLights) ]],
                         texture2d<float, access::read_write> rayColors [[ texture(TextureIndexRayColor) ]],
                         //device float3 *rayColors [[ buffer(BufferIndexRayColors) ]],
                         device float3 *lightColors [[ buffer(BufferIndexLightColors) ]],
                         device Ray *rays [[ buffer(BufferIndexRays) ]],
                         device Ray *shadowRays [[ buffer(BufferIndexShadowRays) ]],
                         texture2d<unsigned int, access::read> randomTexture [[ texture(TextureIndexRandom) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        unsigned int rayIndex = tid.y * uniforms.width + tid.x;
        Intersection intersection = intersections[rayIndex];
        device Ray &ray = rays[rayIndex];
        device Ray &shadowRay = shadowRays[rayIndex];
        
        if (ray.max_distance < 0 || intersection.distance < 0) {
            ray.max_distance = -1.0;
            shadowRay.max_distance = -1.0;
            return;
        }
            
        // The ray hit something. Look up the transformation matrix for this instance.
        float4x4 objectToWorldSpaceTransform(1.0f);

        for (int column = 0; column < 4; column++)
            for (int row = 0; row < 3; row++)
                objectToWorldSpaceTransform[column][row] = instances[intersection.instanceID].transformationMatrix[column][row];
        
        // Compute intersection point in world space.
        int resourceIndex = intersection.instanceID * maxSubmeshes + intersection.geometryID;
        Resource resource = resources[resourceIndex];
        
        float3 objectSpaceSurfaceNormal = interpolateVertexAttribute2(resource.normals, intersection, resource.indices);
        float3 worldSpaceSurfaceNormal = (objectToWorldSpaceTransform * float4(objectSpaceSurfaceNormal, 0)).xyz;
        worldSpaceSurfaceNormal = normalize(worldSpaceSurfaceNormal);
        float3 surfaceColor = resource.material->baseColor;
        
        // Choose a random light source to sample.
        //int bounce = 0;
        unsigned int offset = randomTexture.read(tid).x;
        float lightSample = halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 0);
        int lightIndex = min((int)(lightSample * uniforms.lightCount), uniforms.lightCount - 1);
        
        Light light = lights[lightIndex];
        
        float3 worldSpaceLightDirection;
        float lightDistance;
        float3 lightColor;
        
        if (light.type == LightTypeAreaLight) {
            
            // Choose a random point to sample on the light source.
            float2 r = float2(halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 1),
                              halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 2));

            // Sample the lighting between the intersection point and the point on the area light.
            sampleAreaLight(light, r, intersection.worldSpaceIntersectionPoint, worldSpaceLightDirection,
                            lightColor, lightDistance);
            
            
        } else if (light.type == LightTypeSpotlight) {
            // Compute vector from sample point on light source to intersection point
            worldSpaceLightDirection = light.position - intersection.worldSpaceIntersectionPoint;
                    
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
            worldSpaceLightDirection = light.position - intersection.worldSpaceIntersectionPoint;
            lightDistance = length(worldSpaceLightDirection);
            float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
            worldSpaceLightDirection *= inverseLightDistance;
            lightColor = light.color * inverseLightDistance * inverseLightDistance;
        } else  { // light.type == LightTypeSunlight
            worldSpaceLightDirection = -normalize(light.direction);
            lightDistance = INFINITY;
            lightColor = light.color;
        }
        
        // Scale the light color by the cosine of the angle between the light direction and
        // surface normal.
        lightColor *= saturate(dot(worldSpaceSurfaceNormal, worldSpaceLightDirection));

        // Scale the light color by the number of lights to compensate for the fact that
        // the sample only samples one light source at random.
        lightColor *= uniforms.lightCount;

        // Scale the ray color by the color of the surface. This simulates light being absorbed into
        // the surface.
        float3 rayColor = surfaceColor;
        rayColor *= rayColors.read(tid).xyz;
        rayColors.write(float4(rayColor, 1.0), tid);
        
        if (length(lightColor) > 0.0001) {
            // Compute the shadow ray. The shadow ray checks if the sample position on the
            // light source is visible from the current intersection point.
            // If it is, the lighting contribution is added to the output image.
            

            // Add a small offset to the intersection point to avoid intersecting the same
            // triangle again.
            shadowRay.origin = intersection.worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;

            // Travel towards the light source.
            shadowRay.direction = worldSpaceLightDirection;

            // Don't overshoot the light source.
            shadowRay.max_distance = lightDistance - 1e-3f;
            
            lightColors[rayIndex] = lightColor;
        } else {
            shadowRay.max_distance = -1.0;
        }
        
        // Next choose a random direction to continue the path of the ray. This will
        // cause light to bounce between surfaces. The sample could apply a fair bit of math
        // to compute the fraction of light reflected by the current intersection point to the
        // previous point from the next point. However, by choosing a random direction with
        // probability proportional to the cosine (dot product) of the angle between the
        // sample direction and surface normal, the math entirely cancels out except for
        // multiplying by the surface color. This sampling strategy also reduces the amount
        // of noise in the output image.
        float2 r = float2(halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 3),
                   halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 4));

        float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere(r);
        worldSpaceSampleDirection = alignHemisphereWithNormal(worldSpaceSampleDirection, worldSpaceSurfaceNormal);
        
        ray.origin = intersection.worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;
        ray.direction = worldSpaceSampleDirection;
    }
}

kernel void shadow_intersection_kernel(uint2 tid [[ thread_position_in_grid ]],
                                       constant Ray *shadowRays [[ buffer(BufferIndexShadowRays) ]],
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
        Ray tmpRay = shadowRays[rayIndex];
        
        if (tmpRay.max_distance >= 0) {
            ray shadowRay;
            shadowRay.origin = tmpRay.origin;
            shadowRay.direction = tmpRay.direction;
            shadowRay.min_distance = tmpRay.min_distance;
            shadowRay.max_distance = tmpRay.max_distance;
            
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
}

struct Vertex {
    float4 position [[position]];
    float2 uv;
};

constant float2 quadVertices[] = {
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

kernel void accumulation_kernel(uint2 tid [[ thread_position_in_grid ]],
                                constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                                texture2d<float, access::read> accumulationTarget [[ texture(TextureIndexAccumulationTarget) ]],
                                texture2d<float, access::read_write> renderTarget [[ texture(TextureIndexRenderTarget) ]])
{
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        float3 color = accumulationTarget.read(tid).xyz;
        if (uniforms.frameIndex > 0) {
          float3 prevColor = renderTarget.read(tid).xyz;
          prevColor *= uniforms.frameIndex;
          color += prevColor;
          color /= (uniforms.frameIndex + 1);
        }
        renderTarget.write(float4(color, 1.0), tid);
      }
}

vertex Vertex vertex_split_rt(unsigned short vid [[vertex_id]])
{
    float2 position = quadVertices[vid];
    Vertex out;
    out.position = float4(position, 0, 1);
    out.uv = position * 0.5 + 0.5;
    out.uv.y = 1 - out.uv.y; // flip from metal ndc coordinates to texture coordinate space.
    return out;
}

fragment float4 fragment_split_rt(Vertex in [[stage_in]],
                                  constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                                  texture2d<float, access::sample> renderTarget [[ texture(TextureIndexRenderTarget) ]])
{
    constexpr sampler s(min_filter::nearest,
                        mag_filter::nearest,
                        mip_filter::none);
    float3 color = renderTarget.sample(s, in.uv).xyz;
    
    // Apply a very simple tonemapping function to reduce the dynamic range of the
    // input image into a range which the screen can display.
    color = color / (1.0f + color);
    
    return float4(color, 1.0);
}
