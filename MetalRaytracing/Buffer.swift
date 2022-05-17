import MetalKit

struct Buffer<T> {
    let index: Int
    let buffer: MTLBuffer
}

struct Texture {
    let index: Int
    let texture: MTLTexture
}

extension MTLDevice {
    func makeBuffer<T>(count: Int, index: BufferIndex, label: String? = nil, options: MTLResourceOptions = []) -> Buffer<T>? {
        let mtlBuffer = self.makeBuffer(length: count * MemoryLayout<T>.stride, options: options)
        mtlBuffer?.label = label
        return mtlBuffer.map { Buffer(index: index.rawValue, buffer: $0) }
    }
    
    func makeBuffer<T>(bytes: UnsafeRawPointer, count: Int, index: BufferIndex, label: String? = nil, options: MTLResourceOptions = []) -> Buffer<T>? {
        let mtlBuffer = self.makeBuffer(bytes: bytes, length: count * MemoryLayout<T>.stride, options: options)
        mtlBuffer?.label = label
        return mtlBuffer.map { Buffer(index: index.rawValue, buffer: $0)}
    }
    
    func makeTexture(descriptor: MTLTextureDescriptor, index: TextureIndex, label: String?) -> Texture? {
        let texture = self.makeTexture(descriptor: descriptor)
        texture?.label = label
        return texture.map { Texture(index: index.rawValue, texture: $0) }
    }
}

extension MTLComputeCommandEncoder {
    func setBuffer<T>(_ buffer: Buffer<T>, offset: Int = 0) {
        self.setBuffer(buffer.buffer, offset: offset, index: buffer.index)
    }
    
    func setTexture(_ texture: Texture) {
        self.setTexture(texture.texture, index: texture.index)
    }
}

extension MTLRenderCommandEncoder {
    func setFragmentBuffer<T>(_ buffer: Buffer<T>, offset: Int = 0) {
        self.setFragmentBuffer(buffer.buffer, offset: offset, index: buffer.index)
    }
    
    func setFragmentTexture(_ texture: Texture) {
        self.setFragmentTexture(texture.texture, index: texture.index)
    }
}
