//
//  Model.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

class Model {
    var meshes: [Mesh]
    
    init(name: String, position: SIMD3<Float>, rotation: SIMD3<Float> = [0, 0, 0], scale: Float, on device: MTLDevice) {
        let assetURL = Bundle.main.url(forResource: "Resources/\(name)", withExtension: "obj")!
        let allocator = MTKMeshBufferAllocator(device: device)
        let asset = MDLAsset(url: assetURL, vertexDescriptor: Self.vertexDescriptor, bufferAllocator: allocator)
        
        let mdlMeshes = asset.childObjects(of: MDLMesh.self) as! [MDLMesh]
        
        self.meshes = mdlMeshes.map { mdlMesh -> Mesh in
            let mtkMesh = try! MTKMesh(mesh: mdlMesh, device: device)
            return Mesh(modelName: name, mdlMesh: mdlMesh, mtkMesh: mtkMesh, position: position, rotation: rotation, scale: scale, on: device)
        }
    }
    
    static var vertexDescriptor: MDLVertexDescriptor = {
        let vertexDescriptor = MDLVertexDescriptor()
        vertexDescriptor.attributes[0] =
        MDLVertexAttribute(name: MDLVertexAttributePosition,
                           format: .float3,
                           offset: 0, bufferIndex: 0)
        vertexDescriptor.attributes[1] =
        MDLVertexAttribute(name: MDLVertexAttributeNormal,
                           format: .float3,
                           offset: 0, bufferIndex: 1)
        vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
        vertexDescriptor.layouts[1] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
        return vertexDescriptor
    }()
}
