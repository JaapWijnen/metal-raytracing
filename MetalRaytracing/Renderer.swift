//
//  Renderer.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import Metal
import MetalKit
import simd

class Renderer: NSObject {
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    
    var scene: Scene
    
    var raytracingPipelineState: MTLComputePipelineState!
    var fragmentRaytracingPipelineState: MTLRenderPipelineState!
    var renderPipelineState: MTLRenderPipelineState!
        
    var accumulationTargets: [MTLTexture] = []
    var randomTexture: MTLTexture!
    
    var uniformBuffer: MTLBuffer!
    var resourcesBuffer: MTLBuffer!
    var instanceDescriptorBuffer: MTLBuffer!
    
    var instancedAccelarationStructure: MTLAccelerationStructure!
    var primitiveAccelerationStructures: [MTLAccelerationStructure] = []
    
    let maxFramesInFlight = 3
    var semaphore: DispatchSemaphore!
    
    var renderMode: RenderMode = .fragment

    var uniformBufferIndex = 0
    var uniformBufferOffset: Int {
        uniformBufferIndex * MemoryLayout<Uniforms>.stride
    }
    
    var frameIndex: UInt32 = 0
    var resourceStride = 0
    var maxSubmeshes = 0
    
    init?(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("GPU not available")
        }
        
        let size = metalView.frame.size
        
        metalView.device = device
        metalView.colorPixelFormat = .rgba16Float // .rgba16Float
        metalView.sampleCount = 1
        metalView.drawableSize = size
        
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.library = device.makeDefaultLibrary()!
        
        self.scene = DragonScene(size: size, device: device)
        
        super.init()
        metalView.delegate = self
        mtkView(metalView, drawableSizeWillChange: size)
        semaphore = DispatchSemaphore.init(value: maxFramesInFlight)
        
        createBuffers()
        createPipelineStates(metalView: metalView)
        createAccelerationStructures()
    }
    
    func createPipelineStates(metalView: MTKView) {
        let vertexFunction = library.makeFunction(name: "vertexShader")
        let fragmentFunction = library.makeFunction(name: "fragmentShader")
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.sampleCount = metalView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineDescriptor.label = "Render PipelineState"
        
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print(error.localizedDescription)
        }
        
        let functionConstants = MTLFunctionConstantValues()
        var resourceStride = Int32(self.resourceStride)
        functionConstants.setConstantValue(&resourceStride, type: .int, index: 0)
        var maxSubmeshes = Int32(self.maxSubmeshes)
        functionConstants.setConstantValue(&maxSubmeshes, type: .int, index: 1)
        
        let rtVertexFunction = library.makeFunction(name: "raytracingVertex")
        
        pipelineDescriptor.vertexFunction = rtVertexFunction
        pipelineDescriptor.label = "Fragment Raytracing PipelineState"
        
        do {
            let rtFragmentFunction = try library.makeFunction(name: "raytracingFragment", constantValues: functionConstants)
            pipelineDescriptor.fragmentFunction = rtFragmentFunction
            fragmentRaytracingPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print(error.localizedDescription)
        }
        
        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        computeDescriptor.label = "Raytracing PipelineState"
        
        do {
            computeDescriptor.computeFunction = try library.makeFunction(name: "raytracingKernel", constantValues: functionConstants)
            raytracingPipelineState = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
        } catch {
            print(error.localizedDescription)
        }
        
        print(raytracingPipelineState.threadExecutionWidth,
        raytracingPipelineState.maxTotalThreadsPerThreadgroup,
        raytracingPipelineState.staticThreadgroupMemoryLength)
    }

    func createBuffers() {
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.stride * maxFramesInFlight, options: .storageModeShared)!
        uniformBuffer.label = "Uniform Buffer"
        
        self.resourceStride = 0
        for model in scene.models {
            for mesh in model.meshes {
                for submesh in mesh.submeshes {
                    let encoder = device.makeArgumentEncoder(for: submesh.resources)!
                    if encoder.encodedLength > resourceStride {
                        resourceStride = encoder.encodedLength
                    }
                }
            }
        }
//        for geometry in scene.geometries {
//            let encoder = device.makeArgumentEncoder(for: geometry.resources)!
//            if encoder.encodedLength > resourceStride {
//                resourceStride = encoder.encodedLength
//            }
//        }
        self.maxSubmeshes = scene.models.flatMap(\.meshes).map { $0.submeshes.count }.max()!
//        let resourceCount = scene.models.flatMap { $0.meshes }.reduce(0) { result, mesh in
//            result + mesh.submeshes.count
//        }
        let resourceCount = maxSubmeshes * scene.models.flatMap(\.meshes).count
        
        resourcesBuffer = device.makeBuffer(length: resourceStride * resourceCount/*scene.geometries.count*/, options: .storageModeManaged)!
        resourcesBuffer.label = "Resources Buffer"
        
        for (i, mesh) in scene.models.flatMap(\.meshes).enumerated() {
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
        
//        let submeshes = scene.models.flatMap(\.meshes).flatMap { mesh in mesh.submeshes }
//
//        for (submeshIndex, submesh) in submeshes.enumerated() {
//            let encoder = device.makeArgumentEncoder(for: submesh.resources)!
//            encoder.setArgumentBuffer(resourcesBuffer, offset: resourceStride * submeshIndex)
//
//            for (resourceIndex, resource) in submesh.resources.enumerated() {
//                if let buffer = resource as? MTLBuffer {
//                    encoder.setBuffer(buffer, offset: 0, index: resourceIndex)
//                } else if let texture = resource as? MTLTexture {
//                    encoder.setTexture(texture, index: resourceIndex)
//                }
//            }
//        }
        
        
//        for (geometryIndex, geometry) in scene.geometries.enumerated() {
//            let encoder = device.makeArgumentEncoder(for: geometry.resources)!
//            encoder.setArgumentBuffer(resourcesBuffer, offset: resourceStride * geometryIndex)
//
//            for (resourceIndex, resource) in geometry.resources.enumerated() {
//                if let buffer = resource as? MTLBuffer {
//                    encoder.setBuffer(buffer, offset: 0, index: resourceIndex)
//                } else if let texture = resource as? MTLTexture {
//                    encoder.setTexture(texture, index: resourceIndex)
//                }
//            }
//        }
        resourcesBuffer.didModifyRange(0..<resourcesBuffer.length)
    }

    func createAccelerationStructures() {
        let primitiveAccelerationStructureDescriptors = scene.models.flatMap(\.meshes).map { mesh -> MTLPrimitiveAccelerationStructureDescriptor in
            let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
            descriptor.geometryDescriptors = mesh.geometryDescriptors
            return descriptor
        }
        
        self.primitiveAccelerationStructures = commandQueue.buildCompactedAccelerationStructures(for: primitiveAccelerationStructureDescriptors)
        
        var instanceDescriptors = scene.models.flatMap(\.meshes).enumerated().map { index, mesh -> MTLAccelerationStructureInstanceDescriptor in
            var descriptor = MTLAccelerationStructureInstanceDescriptor()
            descriptor.accelerationStructureIndex = UInt32(index)
            descriptor.mask = 0xFF
            descriptor.options = []
            descriptor.transformationMatrix = .matrix4x4_drop_last_row(mesh.transform)
            return descriptor
        }
        
        self.instanceDescriptorBuffer = device.makeBuffer(bytes: &instanceDescriptors, length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * scene.models.flatMap(\.meshes).count, options: .storageModeManaged)
        self.instanceDescriptorBuffer?.label = "Instance Descriptor Buffer"
        
        let instanceAccelerationStructureDescriptor = MTLInstanceAccelerationStructureDescriptor()
        let instanceCount = scene.models.reduce(0) { result, model in
            return result + model.meshes.count
        }
        instanceAccelerationStructureDescriptor.instanceCount = instanceCount
        instanceAccelerationStructureDescriptor.instancedAccelerationStructures = primitiveAccelerationStructures
        instanceAccelerationStructureDescriptor.instanceDescriptorBuffer = instanceDescriptorBuffer
        
        instancedAccelarationStructure = commandQueue.buildCompactedAccelerationStructure(with: instanceAccelerationStructureDescriptor)!
    }
    
    func updateUniforms(size: CGSize) {
        scene.updateUniforms(size: size)
        
        let pointer = uniformBuffer.contents().advanced(by: uniformBufferOffset)
        let uniforms = pointer.bindMemory(to: Uniforms.self, capacity: 1)
        
        uniforms.pointee.camera = scene.camera
        uniforms.pointee.lightCount = Int32(scene.lights.count)
        uniforms.pointee.width = Int32(size.width)
        uniforms.pointee.height = Int32(size.height)
        uniforms.pointee.blocksWide = ((uniforms.pointee.width) + 15) / 16
        uniforms.pointee.frameIndex = frameIndex
        frameIndex += 1
    }
    
    func createTextures(size: CGSize) {
        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = .rgba32Float
        descriptor.textureType = .type2D
        descriptor.width = Int(size.width)
        descriptor.height = Int(size.height)

        // Stored in private memory because only the GPU will read or write this texture.
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        accumulationTargets = [device.makeTexture(descriptor: descriptor)!, device.makeTexture(descriptor: descriptor)!]
        accumulationTargets[0].label = "Accumulation Texture 1"
        accumulationTargets[1].label = "Accumulation Texture 2"

        // Create a texture containing a random integer value for each pixel. the sample
        // uses these values to decorrelate pixels while drawing pseudorandom numbers from the
        // Halton sequence.
        descriptor.pixelFormat = .r32Uint
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared

        randomTexture = device.makeTexture(descriptor: descriptor)
        randomTexture.label = "Random Texture"
        
        // Initialize random values.
        let numbers = Array<UInt32>.init(unsafeUninitializedCapacity: randomTexture.width * randomTexture.height) { buffer, initializedCount in
            for i in 0..<randomTexture.width * randomTexture.height {
                buffer[i] = arc4random() % (1024 * 1024)
            }
            initializedCount = randomTexture.width * randomTexture.height
        }
        
        numbers.withUnsafeBufferPointer { bufferPointer in
            randomTexture.replace(
                region: .init(
                    origin: .init(x: 0, y: 0, z: 0),
                    size: .init(width: randomTexture.width, height: randomTexture.height, depth: 1)
                ),
                mipmapLevel: 0,
                withBytes: bufferPointer.baseAddress!,
                bytesPerRow: MemoryLayout<UInt32>.stride * randomTexture.width
            )
        }
    }
    
    func update(size: CGSize) {
        if InputController.shared.keysPressed.contains(.one) {
            renderMode = .compute
        }
        if InputController.shared.keysPressed.contains(.two) {
            renderMode = .fragment
        }
        updateUniforms(size: size)
        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }
}

enum RenderMode {
    case compute
    case fragment
}

extension Renderer: MTKViewDelegate {
    func draw(in view: MTKView) {
        semaphore.wait()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.addCompletedHandler { thing in self.semaphore.signal() }
        
        let size = view.drawableSize
        update(size: size)
        
        if renderMode == .compute {
            let width = Int(size.width)
            let height = Int(size.height)
            // process rays in 8x8 tiles
            let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
            let threadGroups = MTLSize(
                width: (width + threadsPerGroup.width - 1) / threadsPerGroup.width,
                height: (height + threadsPerGroup.height - 1) / threadsPerGroup.height,
                depth: 1
            )
            
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            computeEncoder.label = "Raytracing Pass"
            computeEncoder.setComputePipelineState(raytracingPipelineState)
        
            computeEncoder.setBuffer(uniformBuffer,            offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
            computeEncoder.setBuffer(resourcesBuffer,          offset: 0,                   index: BufferIndex.resources.rawValue)
            computeEncoder.setBuffer(instanceDescriptorBuffer, offset: 0,                   index: BufferIndex.instanceDescriptors.rawValue)
            computeEncoder.setBuffer(scene.lightBuffer,        offset: 0,                   index: BufferIndex.lights.rawValue)
        
            computeEncoder.setAccelerationStructure(instancedAccelarationStructure, bufferIndex: BufferIndex.accelerationStructure.rawValue)
        
            computeEncoder.setTexture(randomTexture,          index: TextureIndex.random.rawValue)
            computeEncoder.setTexture(accumulationTargets[0], index: TextureIndex.accumulation.rawValue)
            computeEncoder.setTexture(accumulationTargets[1], index: TextureIndex.previousAccumulation.rawValue)
            
            for model in scene.models {
                for mesh in model.meshes {
                    for resource in mesh.resources {
                        computeEncoder.useResource(resource, usage: .read)
                    }
                }
            }
            
            for accelerationStructure in primitiveAccelerationStructures {
                computeEncoder.useResource(accelerationStructure, usage: .read)
            }
            
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            computeEncoder.endEncoding()
            
            let tmp = accumulationTargets[0]
            accumulationTargets[0] = accumulationTargets[1]
            accumulationTargets[1] = tmp
            
            guard let descriptor = view.currentRenderPassDescriptor,
                  let renderEncoder = commandBuffer.makeRenderCommandEncoder(
                    descriptor: descriptor) else {
                        return
                    }
            renderEncoder.setRenderPipelineState(renderPipelineState)
            
            // MARK: draw call
            renderEncoder.setFragmentTexture(accumulationTargets[0], index: 0)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            renderEncoder.endEncoding()
        } else { // fragment mode
            guard let descriptor = view.currentRenderPassDescriptor else { return }
            guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
            renderEncoder.label = "Fragment Raytracing Pass"
            renderEncoder.setRenderPipelineState(fragmentRaytracingPipelineState)
                    
            renderEncoder.setFragmentBuffer(uniformBuffer,            offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
            renderEncoder.setFragmentBuffer(resourcesBuffer,          offset: 0,                   index: BufferIndex.resources.rawValue)
            renderEncoder.setFragmentBuffer(instanceDescriptorBuffer, offset: 0,                   index: BufferIndex.instanceDescriptors.rawValue)
            renderEncoder.setFragmentBuffer(scene.lightBuffer,        offset: 0,                   index: BufferIndex.lights.rawValue)
        
            renderEncoder.setFragmentAccelerationStructure(instancedAccelarationStructure, bufferIndex: BufferIndex.accelerationStructure.rawValue)
        
            renderEncoder.setFragmentTexture(randomTexture,          index: TextureIndex.random.rawValue)
            renderEncoder.setFragmentTexture(accumulationTargets[0], index: TextureIndex.accumulation.rawValue)
            renderEncoder.setFragmentTexture(accumulationTargets[1], index: TextureIndex.previousAccumulation.rawValue)
            
            for model in scene.models {
                for mesh in model.meshes {
                    for resource in mesh.resources {
                        renderEncoder.useResource(resource, usage: .read)
                    }
                }
            }
            
            for accelerationStructure in primitiveAccelerationStructures {
                renderEncoder.useResource(accelerationStructure, usage: .read)
            }
            
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            renderEncoder.endEncoding()
            
            let tmp = accumulationTargets[0]
            accumulationTargets[0] = accumulationTargets[1]
            accumulationTargets[1] = tmp
        }
        
        guard let drawable = view.currentDrawable else { return }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        createTextures(size: size)
        frameIndex = 0
    }
}
