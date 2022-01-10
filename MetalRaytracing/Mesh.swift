//
//  Mesh.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

struct Mesh {
    let mtkMesh: MTKMesh
    let transform: matrix_float4x4
    let submeshes: [Submesh]
    
    var vertexBuffer: MTLBuffer { mtkMesh.vertexBuffers[0].buffer }
    var normalBuffer: MTLBuffer { mtkMesh.vertexBuffers[1].buffer }
    
    init(modelName: String, mdlMesh: MDLMesh, mtkMesh: MTKMesh, position: SIMD3<Float>, rotation: SIMD3<Float>, scale: Float, on device: MTLDevice) {
        self.mtkMesh = mtkMesh
        
        let rotationMatrix = matrix_float4x4.rotate(rotation)
        let scaleMatrix = matrix_float4x4.scale(scale)
        let translationMatrix = matrix_float4x4.translate(position)
        self.transform = translationMatrix * rotationMatrix * scaleMatrix
        
        var submeshes: [Submesh] = []
        let normalBuffer = mtkMesh.vertexBuffers[1].buffer
        for mesh in zip(mdlMesh.submeshes!, mtkMesh.submeshes) {
            submeshes.append(Submesh(modelName: modelName, mdlSubmesh: mesh.0 as! MDLSubmesh, mtkSubmesh: mesh.1, normalBuffer: normalBuffer, mask: GEOMETRY_MASK_TRIANGLE, on: device))
        }

        self.submeshes = submeshes
    }
    
    var resources: [MTLResource] {
        submeshes.flatMap { $0.resources }
    }
    
    var geometryDescriptors: [MTLAccelerationStructureTriangleGeometryDescriptor] {
        submeshes.map { submesh in
            let descriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
            descriptor.vertexBuffer = self.vertexBuffer
            descriptor.indexBuffer = submesh.mtkSubmesh.indexBuffer.buffer
            descriptor.vertexStride = MemoryLayout<SIMD3<Float>>.stride
            descriptor.triangleCount = submesh.mtkSubmesh.indexCount / 3
            return descriptor
        }
    }
}
