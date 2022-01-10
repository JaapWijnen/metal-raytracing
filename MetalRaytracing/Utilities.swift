//
//  Utilities.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

extension MTLDevice {
    func makeArgumentEncoder(for resources: [MTLResource]) -> MTLArgumentEncoder? {
        let argumentDescriptors = resources.enumerated().map { i, resource -> MTLArgumentDescriptor in
            let argumentDescriptor = MTLArgumentDescriptor()
            argumentDescriptor.index = i
            argumentDescriptor.access = .readOnly
            if resource is MTLBuffer {
                argumentDescriptor.dataType = .pointer
            } else if let texture = resource as? MTLTexture {
                argumentDescriptor.textureType = texture.textureType
                argumentDescriptor.dataType = .texture
            }
            return argumentDescriptor
        }
        return self.makeArgumentEncoder(arguments: argumentDescriptors)
    }
}

extension MTLCommandQueue {
    func buildCompactedAccelerationStructures(for descriptors: [MTLAccelerationStructureDescriptor]) -> [MTLAccelerationStructure] {
        guard
            !descriptors.isEmpty,
            let commandBuffer = self.makeCommandBuffer(),
            let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else { return [] }
        commandBuffer.label = "CommandBuffer BuildAccelerationStructures"
        encoder.label = "CommandEncoder BuildAccelerationStructures"
        
        let descriptorsAndSizes = descriptors.map { descriptor -> (MTLAccelerationStructureDescriptor, MTLAccelerationStructureSizes) in
            let sizes: MTLAccelerationStructureSizes = self.device.accelerationStructureSizes(descriptor: descriptor)
            return (descriptor, sizes)
        }
        
        let scratchBufferSize = descriptorsAndSizes.map(\.1.buildScratchBufferSize).max()!
        guard
            let scratchBuffer = self.device.makeBuffer(length: scratchBufferSize, options: .storageModePrivate),
            let compactedSizesBuffer = self.device.makeBuffer(length: MemoryLayout<UInt32>.stride * descriptors.count, options: .storageModeManaged)
        else { return [] }
        scratchBuffer.label = "Scratch Buffer"
        compactedSizesBuffer.label = "Compacted Sizes Buffer"
        
        let accelerationStructures = descriptorsAndSizes.enumerated().map { index, descriptorAndSizes -> MTLAccelerationStructure in
            let (descriptor, sizes) = descriptorAndSizes
            let accelerationStructure = self.device.makeAccelerationStructure(size: sizes.accelerationStructureSize)!
            accelerationStructure.label = "AccelerationStructure \(index)"
            encoder.build(accelerationStructure: accelerationStructure, descriptor: descriptor, scratchBuffer: scratchBuffer, scratchBufferOffset: 0)
            encoder.writeCompactedSize(accelerationStructure: accelerationStructure, buffer: compactedSizesBuffer, offset: MemoryLayout<UInt32>.stride * index)
            return accelerationStructure
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
        
        commandBuffer.waitUntilCompleted()
        
        guard
            let commandBuffer = self.makeCommandBuffer(),
            let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else { return [] }
        commandBuffer.label = "CommandBuffer BuildCompactedAccelerationStructures"
        encoder.label = "CommandEncoder BuildCompactedAccelerationStructures"
        
        let compactedSizes = compactedSizesBuffer.contents().bindMemory(to: UInt32.self, capacity: descriptors.count)
        
        let compactedAccelerationStructures = accelerationStructures.enumerated().map { index, accelerationStructure -> MTLAccelerationStructure in
            let compactedAccelerationStructure = self.device.makeAccelerationStructure(size: Int(compactedSizes.advanced(by: index).pointee))!
            compactedAccelerationStructure.label = "CompactedAccelerationStructure \(index)"
            encoder.copyAndCompact(sourceAccelerationStructure: accelerationStructure, destinationAccelerationStructure: compactedAccelerationStructure)
            return compactedAccelerationStructure
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return compactedAccelerationStructures
    }
    
    func buildCompactedAccelerationStructure(with descriptor: MTLAccelerationStructureDescriptor) -> MTLAccelerationStructure? {
        return buildCompactedAccelerationStructures(for: [descriptor]).first
    }
}

extension MTLPackedFloat4x3 {
    static func matrix4x4_drop_last_row(_ m: matrix_float4x4) -> MTLPackedFloat4x3 {
        return MTLPackedFloat4x3.init(columns: (
            MTLPackedFloat3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
            MTLPackedFloat3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
            MTLPackedFloat3(m.columns.2.x, m.columns.2.y, m.columns.2.z),
            MTLPackedFloat3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        ))
    }
}

extension MTLPackedFloat3 {
    init(_ x: Float, _ y: Float, _ z: Float) {
        var p = MTLPackedFloat3()
        p.x = x
        p.y = y
        p.z = z
        self = p
    }
}

extension matrix_float4x4 {
    static func translate(_ t: vector_float3) -> matrix_float4x4 {
        return .init(columns: (
            [  1,   0,   0, 0],
            [  0,   1,   0, 0],
            [  0,   0,   1, 0],
            [t.x, t.y, t.z, 1]
        ))
    }
    
    static func rotate(radians: Float, axis: vector_float3) -> matrix_float4x4 {
        let axis = normalize(axis)
        let ct = cosf(radians)
        let st = sinf(radians)
        let ci = 1 - ct
        let x = axis.x, y = axis.y, z = axis.z
        
        return .init(columns: (
            [ ct + x * x * ci,     y * x * ci + z * st, z * x * ci - y * st, 0],
            [ x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0],
            [ x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0],
            [                   0,                   0,                   0, 1]
        ))
    }
    
    static func rotateX(_ radians: Float) -> matrix_float4x4 {
        rotate(radians: radians, axis: [1,0,0])
    }
    
    static func rotateY(_ radians: Float) -> matrix_float4x4 {
        rotate(radians: radians, axis: [0,1,0])
    }
    
    static func rotateZ(_ radians: Float) -> matrix_float4x4 {
        rotate(radians: radians, axis: [0,0,1])
    }
    
    static func rotate(_ r: vector_float3) -> matrix_float4x4 {
        rotateX(r.x) * rotateY(r.y) * rotateZ(r.z)
    }
    
    static func scale(_ s: vector_float3) -> matrix_float4x4 {
        return .init(columns: (
            [s.x,   0,   0, 0],
            [0,   s.y,   0, 0],
            [0,     0, s.z, 0],
            [0,     0,   0, 1]
        ))
    }
    
    static func scale(_ s: Float) -> matrix_float4x4 {
        scale([s, s, s])
    }
}

extension SIMD4 {
    var xyz: SIMD3<Scalar> {
        return [self.x, self.y, self.z]
    }
}
