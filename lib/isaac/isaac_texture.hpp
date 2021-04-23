/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include "isaac_helper.hpp"
#include "isaac_types.hpp"


namespace isaac
{
    enum class FilterType
    {
        NEAREST,
        LINEAR
    };

    enum class BorderType
    {
        CLAMP,
        REPEAT,
        VALUE
    };

    enum class IndexType
    {
        SWEEP,
        MORTON
    };


    /**
     * @brief Software texture implementation
     *
     * @tparam T_Type Type of the buffer values
     * @tparam T_textureDim Dimension of the Texture
     */
    template<typename T_Type, int T_textureDim, IndexType T_indexType = IndexType::SWEEP>
    class Texture
    {
    public:
        Texture() = default;

        /**
         * @brief Initialize texture
         *
         * @param bufferPtr Valid pointer to free memory
         * @param size Size of the texture in T_TextureDim dimensions
         * @param guardSize Size of the memory access guard, default = 0
         */
        Texture(
            T_Type* bufferPtr,
            const isaac_size_dim<T_textureDim>& size,
            ISAAC_IDX_TYPE guardSize = 0,
            FilterType filter = FilterType::NEAREST,
            BorderType border = BorderType::CLAMP)
            : bufferPtr(bufferPtr)
            , size(size)
            , sizeWithGuard(size + ISAAC_IDX_TYPE(2) * guardSize)
            , guardSize(guardSize)
            , filter(filter)
            , border(border)
        {
        }


        ISAAC_HOST_DEVICE_INLINE T_Type
        sample(const isaac_float_dim<T_textureDim>& coord, const T_Type& borderValue = T_Type(0)) const
        {
            T_Type result;
            if(filter == FilterType::LINEAR)
            {
                result = interpolate(coord, borderValue);
            }
            else
            {
                result = safeMemoryAccess(isaac_int_dim<T_textureDim>(coord), borderValue);
            }
            return result;
        }

        ISAAC_DEVICE_INLINE void set(const isaac_int_dim<T_textureDim>& coord, const T_Type& value)
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            bufferPtr[hash(offsetCoord)] = value;
        }

        ISAAC_DEVICE_INLINE T_Type get(const isaac_int_dim<T_textureDim>& coord) const
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            return bufferPtr[hash(offsetCoord)];
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<1>& coord) const
        {
            return coord.x;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<2>& coord) const
        {
            if(T_indexType == IndexType::MORTON)
                return (part1By1(coord.y) << 1) + part1By1(coord.x);
            else
                return coord.x + coord.y * sizeWithGuard.x;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<3>& coord) const
        {
            if(T_indexType == IndexType::MORTON)
                return (part1By2(coord.z) << 2) | (part1By2(coord.y) << 1) | part1By2(coord.x);
            else
                return coord.x + coord.y * sizeWithGuard.x + coord.z * sizeWithGuard.x * sizeWithGuard.y;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<T_textureDim> getSize() const
        {
            return size;
        }
        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<T_textureDim> getSizeWithGuard() const
        {
            return sizeWithGuard;
        }
        ISAAC_HOST_DEVICE_INLINE ISAAC_IDX_TYPE getGuardSize() const
        {
            return guardSize;
        }

        ISAAC_HOST_DEVICE_INLINE T_Type
        safeMemoryAccess(const isaac_int_dim<T_textureDim>& coord, const T_Type& borderValue = T_Type(0)) const
        {
            isaac_int_dim<T_textureDim> offsetCoord;
            if(border == BorderType::REPEAT)
            {
                // Modulo modification to also account for negative values
                for(int i = 0; i < T_textureDim; ++i)
                {
                    offsetCoord[i] = (sizeWithGuard[i] + ((coord[i] + isaac_int(guardSize)) % sizeWithGuard[i]))
                        % sizeWithGuard[i];
                }
            }
            else if(border == BorderType::VALUE)
            {
                offsetCoord = coord + isaac_int(guardSize);
                if(!isInLowerBounds(offsetCoord, isaac_int_dim<T_textureDim>(0))
                   || !isInUpperBounds(offsetCoord, isaac_int_dim<T_textureDim>(sizeWithGuard)))
                    return borderValue;
            }
            else
            {
                offsetCoord = glm::clamp(
                    coord + isaac_int(guardSize),
                    isaac_int_dim<T_textureDim>(0),
                    isaac_int_dim<T_textureDim>(sizeWithGuard) - 1);
            }
            return get(offsetCoord - isaac_int(guardSize));
        }

        ISAAC_HOST_DEVICE_INLINE T_Type
        interpolate(isaac_float_dim<1> coord, const T_Type& borderValue = T_Type(0)) const
        {
            coord -= isaac_float(0.5);
            T_Type data2[2];
            for(int x = 0; x < 2; x++)
            {
                data2[x] = safeMemoryAccess(isaac_int_dim<1>(glm::floor(coord)) + isaac_int_dim<1>(x), borderValue);
            }

            return linear(glm::fract(coord), data2);
        }

        ISAAC_HOST_DEVICE_INLINE T_Type
        interpolate(isaac_float_dim<2> coord, const T_Type& borderValue = T_Type(0)) const
        {
            coord -= isaac_float(0.5);
            T_Type data4[2][2];
            for(int y = 0; y < 2; y++)
            {
                for(int x = 0; x < 2; x++)
                {
                    data4[x][y] = safeMemoryAccess(isaac_int2(glm::floor(coord)) + isaac_int2(x, y), borderValue);
                }
            }

            return bilinear(glm::fract(coord), data4);
        }

        ISAAC_HOST_DEVICE_INLINE T_Type
        interpolate(isaac_float_dim<3> coord, const T_Type& borderValue = T_Type(0)) const
        {
            coord -= isaac_float(0.5);
            T_Type data8[2][2][2];
            if(border == BorderType::CLAMP)
            {
                coord = glm::clamp(
                    coord,
                    isaac_float3(-guardSize) + std::numeric_limits<isaac_float>::min(),
                    isaac_float3(sizeWithGuard - guardSize - ISAAC_IDX_TYPE(1))
                        - (std::numeric_limits<isaac_float>::epsilon()
                           * isaac_float3(sizeWithGuard - guardSize - ISAAC_IDX_TYPE(1))));

                for(int z = 0; z < 2; z++)
                {
                    for(int y = 0; y < 2; y++)
                    {
                        for(int x = 0; x < 2; x++)
                        {
                            data8[x][y][z] = (*this)[isaac_int3(glm::floor(coord)) + isaac_int3(x, y, z)];
                        }
                    }
                }
            }
            else
            {
                for(int z = 0; z < 2; z++)
                {
                    for(int y = 0; y < 2; y++)
                    {
                        for(int x = 0; x < 2; x++)
                        {
                            data8[x][y][z]
                                = safeMemoryAccess(isaac_int3(glm::floor(coord)) + isaac_int3(x, y, z), borderValue);
                        }
                    }
                }
            }


            return trilinear(glm::fract(coord), data8);
        }


    private:
        T_Type* bufferPtr;
        isaac_size_dim<T_textureDim> size;
        isaac_size_dim<T_textureDim> sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
        FilterType filter = FilterType::NEAREST;
        BorderType border = BorderType::VALUE;
    };
#ifdef ALPAKA_ACC_GPU_CUDA_ONLY_MODE
    template<>
    class Texture<isaac_float, 3, IndexType::SWEEP>
    {
    public:
        Texture() = default;

        /**
         * @brief Initialize texture
         *
         * @param bufferPtr Valid pointer to free memory
         * @param size Size of the texture in T_TextureDim dimensions
         * @param guardSize Size of the memory access guard, default = 0
         */
        Texture(
            isaac_float* bufferPtr,
            const isaac_size_dim<3>& size,
            ISAAC_IDX_TYPE guardSize = 0,
            FilterType filter = FilterType::NEAREST,
            BorderType border = BorderType::CLAMP)
            : bufferPtr(bufferPtr)
            , size(size)
            , sizeWithGuard(size + ISAAC_IDX_TYPE(2) * guardSize)
            , guardSize(guardSize)
            , filter(filter)
            , border(border)
        {
            const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            cudaMalloc3DArray(
                &cudaArray,
                &channelDesc,
                make_cudaExtent(sizeWithGuard.x, sizeWithGuard.y, sizeWithGuard.z),
                cudaArraySurfaceLoadStore);


            cudaResourceDesc rescDesc;
            memset(&rescDesc, 0, sizeof(rescDesc));
            rescDesc.resType = cudaResourceTypeArray;
            rescDesc.res.array.array = cudaArray;

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            for(int i = 0; i < 3; i++)
                texDesc.addressMode[i] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            for(int i = 0; i < 3; i++)
                texDesc.borderColor[i] = 0;
            texDesc.normalizedCoords = 0;

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaCreateTextureObject(&textureObj, &rescDesc, &texDesc, NULL));
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaCreateSurfaceObject(&surfaceObj, &rescDesc));
        }

        ISAAC_DEVICE_INLINE isaac_float
        sample(const isaac_float_dim<3>& coord, const isaac_float& borderValue = isaac_float(0)) const
        {
            isaac_float_dim<3> offsetCoord = coord + isaac_float(guardSize);
            float result = tex3D<float>(textureObj, offsetCoord.x, offsetCoord.y, offsetCoord.z);
            return result;
        }

        ISAAC_DEVICE_INLINE void set(const isaac_int_dim<3>& coord, const isaac_float& value)
        {
            isaac_uint_dim<3> offsetCoord = coord + isaac_int(guardSize);
            // for some reason cuda requires the x coordinate to be byte addressed
            surf3Dwrite(value, surfaceObj, offsetCoord.x * sizeof(float), offsetCoord.y, offsetCoord.z);
        }

        ISAAC_DEVICE_INLINE isaac_float get(const isaac_int_dim<3>& coord) const
        {
            isaac_uint_dim<3> offsetCoord = coord + isaac_int(guardSize);
            // for some reason cuda requires the x coordinate to be byte addressed
            float value = surf3Dread<float>(surfaceObj, offsetCoord.x * sizeof(float), offsetCoord.y, offsetCoord.z);
            return value;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<3> getSize() const
        {
            return size;
        }
        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<3> getSizeWithGuard() const
        {
            return sizeWithGuard;
        }
        ISAAC_HOST_DEVICE_INLINE ISAAC_IDX_TYPE getGuardSize() const
        {
            return guardSize;
        }

    private:
        isaac_float* bufferPtr;
        isaac_size_dim<3> size;
        isaac_size_dim<3> sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
        FilterType filter = FilterType::NEAREST;
        BorderType border = BorderType::VALUE;
        cudaArray_t cudaArray;
        cudaTextureObject_t textureObj;
        cudaSurfaceObject_t surfaceObj;
    };

    template<>
    class Texture<isaac_float4, 3, IndexType::SWEEP>
    {
    public:
        Texture() = default;

        /**
         * @brief Initialize texture
         *
         * @param bufferPtr Valid pointer to free memory
         * @param size Size of the texture in T_TextureDim dimensions
         * @param guardSize Size of the memory access guard, default = 0
         */
        Texture(
            isaac_float4* bufferPtr,
            const isaac_size_dim<3>& size,
            ISAAC_IDX_TYPE guardSize = 0,
            FilterType filter = FilterType::NEAREST,
            BorderType border = BorderType::CLAMP)
            : bufferPtr(bufferPtr)
            , size(size)
            , sizeWithGuard(size + ISAAC_IDX_TYPE(2) * guardSize)
            , guardSize(guardSize)
            , filter(filter)
            , border(border)
        {
            const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
            cudaMalloc3DArray(
                &cudaArray,
                &channelDesc,
                make_cudaExtent(sizeWithGuard.x, sizeWithGuard.y, sizeWithGuard.z),
                cudaArraySurfaceLoadStore);


            cudaResourceDesc rescDesc;
            memset(&rescDesc, 0, sizeof(rescDesc));
            rescDesc.resType = cudaResourceTypeArray;
            rescDesc.res.array.array = cudaArray;

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            for(int i = 0; i < 3; i++)
                texDesc.addressMode[i] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            for(int i = 0; i < 3; i++)
                texDesc.borderColor[i] = 0;
            texDesc.normalizedCoords = 0;

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaCreateTextureObject(&textureObj, &rescDesc, &texDesc, NULL));
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaCreateSurfaceObject(&surfaceObj, &rescDesc));
        }

        ISAAC_DEVICE_INLINE isaac_float4
        sample(const isaac_float_dim<3>& coord, const isaac_float4& borderValue = isaac_float4(0)) const
        {
            isaac_float_dim<3> offsetCoord = coord + isaac_float(guardSize);
            float4 result = tex3D<float4>(textureObj, offsetCoord.x, offsetCoord.y, offsetCoord.z);
            return isaac_float4(result.x, result.y, result.z, result.w);
        }

        ISAAC_DEVICE_INLINE void set(const isaac_int_dim<3>& coord, const isaac_float4& value)
        {
            isaac_uint_dim<3> offsetCoord = coord + isaac_int(guardSize);
            float4 cudaValue = {value.r, value.g, value.b, value.a};
            // for some reason cuda requires the x coordinate to be byte addressed
            surf3Dwrite(cudaValue, surfaceObj, offsetCoord.x * sizeof(float4), offsetCoord.y, offsetCoord.z);
        }

        ISAAC_DEVICE_INLINE isaac_float4 get(const isaac_int_dim<3>& coord) const
        {
            isaac_uint_dim<3> offsetCoord = coord + isaac_int(guardSize);
            // for some reason cuda requires the x coordinate to be byte addressed
            float4 value
                = surf3Dread<float4>(surfaceObj, offsetCoord.x * sizeof(float4), offsetCoord.y, offsetCoord.z);
            return isaac_float4(value.x, value.y, value.z, value.w);
        }

        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<3> getSize() const
        {
            return size;
        }
        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<3> getSizeWithGuard() const
        {
            return sizeWithGuard;
        }
        ISAAC_HOST_DEVICE_INLINE ISAAC_IDX_TYPE getGuardSize() const
        {
            return guardSize;
        }

    private:
        isaac_float4* bufferPtr;
        isaac_size_dim<3> size;
        isaac_size_dim<3> sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
        FilterType filter = FilterType::NEAREST;
        BorderType border = BorderType::VALUE;
        cudaArray_t cudaArray;
        cudaTextureObject_t textureObj;
        cudaSurfaceObject_t surfaceObj;
    };

#endif

    /**
     * @brief Allocator class for textures
     *
     * @tparam T_DevAcc Alpaka device description for the buffer allocation
     * @tparam T_Type Type of the buffer values
     * @tparam T_textureDim Dimension of the Texture
     */
    template<typename T_DevAcc, typename T_Type, int T_textureDim, IndexType T_indexType = IndexType::SWEEP>
    class TextureAllocator
    {
        using FraDim = alpaka::DimInt<1>;

    public:
        TextureAllocator(
            const T_DevAcc& devAcc,
            const isaac_size_dim<T_textureDim>& size,
            ISAAC_IDX_TYPE guardSize = 0,
            FilterType filter = FilterType::NEAREST,
            BorderType border = BorderType::CLAMP)
            : bufferExtent(0)
            , buffer(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent))
        {
            const isaac_size_dim<T_textureDim> sizeWithGuard = size + ISAAC_IDX_TYPE(2) * guardSize;

            if(T_indexType == IndexType::MORTON)
            {
                ISAAC_IDX_TYPE maxDim = sizeWithGuard[0];
                std::cout << sizeWithGuard[0] << ", ";
                for(int i = 1; i < T_textureDim; ++i)
                {
                    std::cout << sizeWithGuard[i] << ", ";
                    maxDim = glm::max(maxDim, sizeWithGuard[i]);
                }
                bufferExtent = glm::pow(maxDim, ISAAC_IDX_TYPE(T_textureDim));
                std::cout << std::endl << bufferExtent << std::endl;
            }
            else
            {
                bufferExtent = sizeWithGuard[0];
                for(int i = 1; i < T_textureDim; ++i)
                {
                    bufferExtent *= (sizeWithGuard[i]);
                }
            }

            buffer = alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent);

            texture = Texture<T_Type, T_textureDim, T_indexType>(
                alpaka::getPtrNative(buffer),
                size,
                guardSize,
                filter,
                border);

            // std::cout << "Finished texture init!" << std::endl;
        }

        template<typename T_Queue, typename T_ViewDst>
        void copyToBuffer(T_Queue& queue, T_ViewDst& viewDst) const
        {
            alpaka::memcpy(queue, viewDst, buffer, bufferExtent);
        }

        template<typename T_Queue, typename T_DstDev>
        void copyToTexture(T_Queue& queue, TextureAllocator<T_DstDev, T_Type, T_textureDim, T_indexType>& textureDst)
            const
        {
            assert(bufferExtent == textureDst.getBufferExtent());
            alpaka::memcpy(queue, textureDst.getTextureView(), buffer, bufferExtent);
        }

        template<typename T_Queue>
        void clearColor(T_Queue& queue)
        {
            alpaka::memset(queue, buffer, 0, bufferExtent);
        }

        Texture<T_Type, T_textureDim, T_indexType>& getTexture()
        {
            return texture;
        }

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE>& getTextureView()
        {
            return buffer;
        }

        ISAAC_IDX_TYPE getBufferExtent()
        {
            return bufferExtent;
        }

    private:
        Texture<T_Type, T_textureDim, T_indexType> texture;

        ISAAC_IDX_TYPE bufferExtent;

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE> buffer;
    };


    template<typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    using Texture2D = Texture<T_Type, 2, T_indexType>;

    template<typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    using Texture3D = Texture<T_Type, 3, T_indexType>;

    template<typename T_DevAcc, typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    using Texture2DAllocator = TextureAllocator<T_DevAcc, T_Type, 2, T_indexType>;

    template<typename T_DevAcc, typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    using Texture3DAllocator = TextureAllocator<T_DevAcc, T_Type, 3, T_indexType>;


    template<int T_n>
    struct PersistentArrayStruct
    {
        Texture3D<isaac_float> textures[ZeroCheck<T_n>::value];
    };

    struct GBuffer
    {
        isaac_size2 size;
        isaac_uint2 startOffset;
        Texture2D<isaac_byte4> color;
        Texture2D<isaac_float> depth;
        Texture2D<isaac_float3> normal;
        Texture2D<isaac_float> aoStrength;
    };

} // namespace isaac