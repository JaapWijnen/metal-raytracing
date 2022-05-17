//
//  Scene.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

class Scene {
    var models: [Model]
    
    var camera: Camera
    
    var lights: [Light]
    var lightBuffer: Buffer<Light>!
    
    var resourceStride = 0
    var maxSubmeshes = 0
    
    var resourcesBuffer: MTLBuffer!
    
    var instanceDescriptorBuffer: MTLBuffer!

    var instancedAccelarationStructure: MTLAccelerationStructure!
    var primitiveAccelerationStructures: [MTLAccelerationStructure] = []
    
    init(size: CGSize, device: MTLDevice) {
        self.camera = Scene.setupCamera(size: size)
        self.models = []
        let light1 = Scene.setupLight()
        var light2 = Scene.setupLight()
        light2.position = [2.0, 1.98, 3.0]
        light2.forward = [0, -0.5, 0.0]
        light2.right = [0.1, 0.0, 0.0]
        light2.up = [0.0, 0.0, 0.1]
        let light3 = Light.spotLight(position: [2, 1, 4], direction: [-1.5, -0.5, -1.5], coneAngle: 25 / 180 * .pi, color: [4,4,4])
        //let sunLight = Light.sunLight(direction: [-1, -2, 0], color: [1,1,1])
        //let pointLight = Light.pointLight(position: [1,1,1], color: [1, 1, 1])
        self.lights = [light1, light3]
        
        self.lightBuffer = device.makeBuffer(bytes: &lights, count: lights.count, index: .lights, label: "Lights Buffer", options: .storageModeShared)!
    }
    
    func updateUniforms(size: CGSize) {
        camera = Self.setupCamera(size: size)
    }
    
    static func setupCamera(size: CGSize) -> Camera {
        var camera = Camera(
            position: SIMD3<Float>(0.0, 1.0, 5.38),
            right: SIMD3<Float>(1.0, 0.0, 0.0),
            up: SIMD3<Float>(0.0, 1.0, 0.0),
            forward: SIMD3<Float>(0.0, 0.0, -1.0)
        )
        
        let fieldOfView = 45.0 * (Float.pi / 180.0)
        let aspectRatio = Float(size.width) / Float(size.height)
        let imagePlaneHeight = tanf(fieldOfView / 2.0)
        let imagePlaneWidth = aspectRatio * imagePlaneHeight
        
        camera.right *= imagePlaneWidth
        camera.up *= imagePlaneHeight
        
        return camera
    }
    
    static func setupLight() -> Light {
        return .areaLight(
            position: [0.0, 1.98, 0.0],
            forward: [0.0, -1.0, 0.0],
            right: [0.25, 0.0, 0.0],
            up: [0.0, 0.0, 0.25],
            color: [4.0, 4.0, 4.0]
        )
    }
}

extension Light {
    static func areaLight(position: SIMD3<Float>, forward: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>, color: SIMD3<Float>) -> Light {
        var light = Light()
        light.type = .areaLight
        light.position = position
        light.forward = forward
        light.right = right
        light.up = up
        light.color = color
        return light
    }
    
    static func sunLight(direction: SIMD3<Float>, color: SIMD3<Float>) -> Light {
        var light = Light()
        light.type = .sunlight
        light.direction = direction
        light.color = color
        return light
    }
    
    static func pointLight(position: SIMD3<Float>, color: SIMD3<Float>) -> Light {
        var light = Light()
        light.type = .pointlight
        light.position = position
        light.color = color
        return light
    }
    
    static func spotLight(position: SIMD3<Float>, direction: SIMD3<Float>, coneAngle: Float, color: SIMD3<Float>) -> Light {
        var light = Light()
        light.type = .spotlight
        light.position = position
        light.direction = direction
        light.coneAngle = coneAngle
        light.color = color
        return light
    }
}

extension Scene {
    func createAccelerationStructures(on commandQueue: MTLCommandQueue) {
        let primitiveAccelerationStructureDescriptors = models.flatMap(\.meshes).map { mesh -> MTLPrimitiveAccelerationStructureDescriptor in
            let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
            descriptor.geometryDescriptors = mesh.geometryDescriptors
            return descriptor
        }
        
        self.primitiveAccelerationStructures = commandQueue.buildCompactedAccelerationStructures(for: primitiveAccelerationStructureDescriptors)
        
        var instanceDescriptors = models.flatMap(\.meshes).enumerated().map { index, mesh -> MTLAccelerationStructureInstanceDescriptor in
            var descriptor = MTLAccelerationStructureInstanceDescriptor()
            descriptor.accelerationStructureIndex = UInt32(index)
            descriptor.mask = 0xFF
            descriptor.options = []
            descriptor.transformationMatrix = .matrix4x4_drop_last_row(mesh.transform)
            return descriptor
        }
        
        self.instanceDescriptorBuffer = commandQueue.device.makeBuffer(bytes: &instanceDescriptors, length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * models.flatMap(\.meshes).count, options: .storageModeManaged)
        self.instanceDescriptorBuffer?.label = "Instance Descriptor Buffer"
        
        let instanceAccelerationStructureDescriptor = MTLInstanceAccelerationStructureDescriptor()
        let instanceCount = models.reduce(0) { result, model in
            return result + model.meshes.count
        }
        instanceAccelerationStructureDescriptor.instanceCount = instanceCount
        instanceAccelerationStructureDescriptor.instancedAccelerationStructures = primitiveAccelerationStructures
        instanceAccelerationStructureDescriptor.instanceDescriptorBuffer = instanceDescriptorBuffer
        
        instancedAccelarationStructure = commandQueue.buildCompactedAccelerationStructure(with: instanceAccelerationStructureDescriptor)!
    }
    
    func createResourceBuffer(on device: MTLDevice) {
        self.resourceStride = 0
        for model in models {
            for mesh in model.meshes {
                for submesh in mesh.submeshes {
                    let encoder = device.makeArgumentEncoder(for: submesh.resources)!
                    if encoder.encodedLength > resourceStride {
                        resourceStride = encoder.encodedLength
                    }
                }
            }
        }
        self.maxSubmeshes = models.flatMap(\.meshes).map { $0.submeshes.count }.max()!

        let resourceCount = maxSubmeshes * models.flatMap(\.meshes).count
        
        resourcesBuffer = device.makeBuffer(length: resourceStride * resourceCount/*scene.geometries.count*/, options: .storageModeShared)!
        resourcesBuffer.label = "Resources Buffer"
        
        for (i, mesh) in models.flatMap(\.meshes).enumerated() {
            for (j, submesh) in mesh.submeshes.enumerated() {
                let index = i * maxSubmeshes + j
                let encoder = device.makeArgumentEncoder(for: submesh.resources)!
                encoder.setArgumentBuffer(resourcesBuffer, offset: resourceStride * index)
                
                for (resourceIndex, resource) in submesh.resources.enumerated() {
                    if let buffer = resource as? MTLBuffer {
                        encoder.setBuffer(buffer, offset: 0, index: resourceIndex)
                    } else if let texture = resource as? MTLTexture {
                        encoder.setTexture(texture, index: resourceIndex)
                    }
                }
            }
        }
        //resourcesBuffer.didModifyRange(0..<resourcesBuffer.length)
    }
}
