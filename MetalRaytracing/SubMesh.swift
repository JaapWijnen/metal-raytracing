//
//  SubMesh.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

class Submesh {
    let mtkSubmesh: MTKSubmesh
    var material: Material
    
    let normalBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let materialBuffer: MTLBuffer
    let mask: Int32
    
    var resources: [MTLResource] {
        [normalBuffer, indexBuffer, materialBuffer]
    }
    
    init(modelName: String, mdlSubmesh: MDLSubmesh, mtkSubmesh: MTKSubmesh, normalBuffer: MTLBuffer, mask: Int32, on device: MTLDevice) {
        self.mtkSubmesh = mtkSubmesh
        self.material = Material(material: mdlSubmesh.material)
        self.normalBuffer = normalBuffer
        self.normalBuffer.label = "\(modelName) Normals"
        self.indexBuffer = mtkSubmesh.indexBuffer.buffer
        self.indexBuffer.label = "\(modelName) Indices"
        self.materialBuffer = device.makeBuffer(bytes: &self.material, length: MemoryLayout<Material>.stride, options: .storageModeManaged)!
        self.materialBuffer.label = "\(modelName) Material"
        self.mask = mask
    }
}

private extension Material {
    init(material: MDLMaterial?) {
        self.init()
        if let baseColor = material?.property(with: .baseColor), baseColor.type == .float3 {
            self.baseColor = baseColor.float3Value
        }
        if let emission = material?.property(with: .emission), emission.type == .float3 {
            self.emission = emission.float3Value
        }
        if let specular = material?.property(with: .specular), specular.type == .float3 {
            self.specular = specular.float3Value
        }
        if let specularExponent = material?.property(with: .specularExponent), specularExponent.type == .float3 {
            self.specularExponent = specularExponent.floatValue
        }
        if let refractionIndex = material?.property(with: .materialIndexOfRefraction), refractionIndex.type == .float {
            self.refractionIndex = refractionIndex.floatValue
        }
    }
}
