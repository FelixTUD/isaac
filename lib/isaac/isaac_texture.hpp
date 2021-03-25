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

    /**
     * @brief Software texture implementation
     *
     * @tparam T_Type Type of the buffer values
     * @tparam T_textureDim Dimension of the Texture
     */
    template<typename T_Type, int T_textureDim>
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
        ISAAC_HOST_DEVICE_INLINE void init(
            T_Type* bufferPtr,
            const isaac_size_dim<T_textureDim>& size,
            ISAAC_IDX_TYPE guardSize = 0)
        {
            this->bufferPtr = bufferPtr;
            this->size = size;
            this->sizeWithGuard = size + ISAAC_IDX_TYPE(2) * guardSize;
            this->guardSize = guardSize;
        }

        ISAAC_HOST_DEVICE_INLINE T_Type operator[](const isaac_int_dim<T_textureDim>& coord) const
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            return bufferPtr[hash(offsetCoord)];
        }


        ISAAC_HOST_DEVICE_INLINE T_Type& operator[](const isaac_int_dim<T_textureDim>& coord)
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
            return coord.x + coord.y * sizeWithGuard.x;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<3>& coord) const
        {
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

        // TODO: this
        /* Not yet working because of partial template specialization
        template<typename T_Acc, typename T_AccDim, typename T_Stream, typename T_Kernel, typename... T_Args>
        ISAAC_HOST_INLINE void kernelOnEachElement(T_Stream& stream, T_Kernel& kernel, T_Args&&... args)
        {
            isaac_size2 gridSize
                = {ISAAC_IDX_TYPE((sizeWithGuard.x + 15) / 16), ISAAC_IDX_TYPE((sizeWithGuard.y + 15) / 16)};
            isaac_size2 blockSize = {ISAAC_IDX_TYPE(16), ISAAC_IDX_TYPE(16)};
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
            if(boost::mpl::not_<boost::is_same<T_Acc, alpaka::AccGpuCudaRt<T_AccDim, ISAAC_IDX_TYPE>>>::value)
#endif
            {
                gridSize.x = ISAAC_IDX_TYPE(sizeWithGuard.x);
                gridSize.y = ISAAC_IDX_TYPE(sizeWithGuard.y);
                blockSize.x = ISAAC_IDX_TYPE(1);
                blockSize.y = ISAAC_IDX_TYPE(1);
            }
            const alpaka::Vec<T_AccDim, ISAAC_IDX_TYPE> threads(
                ISAAC_IDX_TYPE(1),
                ISAAC_IDX_TYPE(1),
                ISAAC_IDX_TYPE(1));
            const alpaka::Vec<T_AccDim, ISAAC_IDX_TYPE> blocks(ISAAC_IDX_TYPE(1), blockSize.x, blockSize.y);
            const alpaka::Vec<T_AccDim, ISAAC_IDX_TYPE> grid(ISAAC_IDX_TYPE(1), gridSize.x, gridSize.y);
            auto const workdiv(alpaka::WorkDivMembers<T_AccDim, ISAAC_IDX_TYPE>(grid, blocks, threads));


            auto const instance(alpaka::createTaskKernel<T_Acc>(workdiv, kernel, this, args));
            alpaka::enqueue(stream, instance);
            alpaka::wait(stream);
        }
        */


    private:
        T_Type* bufferPtr = nullptr;
        isaac_size_dim<T_textureDim> size;
        isaac_size_dim<T_textureDim> sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
    };


    template<FilterType T_filter, BorderType T_border>
    class Sampler
    {
    public:
        template<typename T_Type, int T_textureDim>
        ISAAC_HOST_DEVICE_INLINE T_Type sample(
            const Texture<T_Type, T_textureDim>& texture,
            const isaac_float_dim<T_textureDim>& coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            T_Type result;
            if(T_filter == FilterType::LINEAR)
            {
                result = interpolate(texture, coord, borderValue);
            }
            else
            {
                result = safeMemoryAccess(texture, isaac_int_dim<T_textureDim>(coord), borderValue);
            }
            return result;
        }

        template<typename T_Type, int T_textureDim>
        ISAAC_HOST_DEVICE_INLINE T_Type safeMemoryAccess(
            const Texture<T_Type, T_textureDim>& texture,
            const isaac_int_dim<T_textureDim>& coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            const isaac_size_dim<T_textureDim> sizeWithGuard = texture.getSizeWithGuard();
            const ISAAC_IDX_TYPE guardSize = texture.getGuardSize();

            isaac_int_dim<T_textureDim> offsetCoord;
            if(T_border == BorderType::REPEAT)
            {
                // Modulo modification to also account for negative values
                for(int i = 0; i < T_textureDim; ++i)
                {
                    offsetCoord[i] = (sizeWithGuard[i] + ((coord[i] + isaac_int(guardSize)) % sizeWithGuard[i]))
                        % sizeWithGuard[i];
                }
            }
            else if(T_border == BorderType::VALUE)
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
            return texture[offsetCoord - isaac_int(guardSize)];
        }

        template<typename T_Type>
        ISAAC_HOST_DEVICE_INLINE T_Type interpolate(
            const Texture<T_Type, 1>& texture,
            isaac_float_dim<1> coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            T_Type data2[2];
            for(int x = 0; x < 2; x++)
            {
                data2[x] = safeMemoryAccess(texture, isaac_float_dim<1>(coord) + isaac_float_dim<1>(x), borderValue);
            }

            return linear(glm::fract(coord), data2);
        }

        template<typename T_Type>
        ISAAC_HOST_DEVICE_INLINE T_Type interpolate(
            const Texture<T_Type, 2>& texture,
            isaac_float_dim<2> coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            T_Type data4[2][2];
            for(int x = 0; x < 2; x++)
            {
                for(int y = 0; y < 2; y++)
                {
                    data4[x][y] = safeMemoryAccess(texture, isaac_int2(coord) + isaac_int2(x, y), borderValue);
                }
            }

            return bilinear(glm::fract(coord), data4);
        }

        template<typename T_Type>
        ISAAC_HOST_DEVICE_INLINE T_Type interpolate(
            const Texture<T_Type, 3>& texture,
            isaac_float_dim<3> coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            T_Type data8[2][2][2];
            for(int x = 0; x < 2; x++)
            {
                for(int y = 0; y < 2; y++)
                {
                    for(int z = 0; z < 2; z++)
                    {
                        data8[x][y][z]
                            = safeMemoryAccess(texture, isaac_int3(coord) + isaac_int3(x, y, z), borderValue);
                    }
                }
            }

            return trilinear(glm::fract(coord), data8);
        }

        ISAAC_HOST_DEVICE_INLINE isaac_byte4 interpolate(
            const Texture<isaac_byte4, 3>& texture,
            isaac_float_dim<3> coord,
            const isaac_byte4& borderValue = isaac_byte4(0)) const
        {
            isaac_float4 data8[2][2][2];
            for(int x = 0; x < 2; x++)
            {
                for(int y = 0; y < 2; y++)
                {
                    for(int z = 0; z < 2; z++)
                    {
                        data8[x][y][z]
                            = isaac_float4(
                                  safeMemoryAccess(texture, isaac_int3(coord) + isaac_int3(x, y, z), borderValue))
                            / isaac_float(255);
                    }
                }
            }

            return isaac_byte4(trilinear(glm::fract(coord), data8) * isaac_float(255));
        }
    };


    /**
     * @brief Allocator class for textures
     *
     * @tparam T_DevAcc Alpaka device description for the buffer allocation
     * @tparam T_Type Type of the buffer values
     * @tparam T_textureDim Dimension of the Texture
     */
    template<typename T_DevAcc, typename T_Type, int T_textureDim>
    class TextureAllocator
    {
        using FraDim = alpaka::DimInt<1>;

    public:
        TextureAllocator(
            const T_DevAcc& devAcc,
            const isaac_size_dim<T_textureDim>& size,
            ISAAC_IDX_TYPE guardSize = 0)
            : bufferExtent(0)
            , buffer(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent))
        {
            const isaac_size_dim<T_textureDim> sizeWithGuard = size + ISAAC_IDX_TYPE(2) * guardSize;

            bufferExtent = sizeWithGuard[0];
            for(int i = 1; i < T_textureDim; ++i)
            {
                bufferExtent *= (sizeWithGuard[i]);
            }

            buffer = alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent);

            texture.init(alpaka::getPtrNative(buffer), size, guardSize);
        }

        template<typename T_Queue, typename T_ViewDst>
        void copyToBuffer(T_Queue& queue, T_ViewDst& viewDst) const
        {
            alpaka::memcpy(queue, viewDst, buffer, bufferExtent);
        }

        template<typename T_Queue, typename T_DstDev>
        void copyToTexture(T_Queue& queue, TextureAllocator<T_DstDev, T_Type, T_textureDim>& textureDst) const
        {
            assert(bufferExtent == textureDst.getBufferExtent());
            alpaka::memcpy(queue, textureDst.getTextureView(), buffer, bufferExtent);
        }

        template<typename T_Queue>
        void clearColor(T_Queue& queue)
        {
            alpaka::memset(queue, buffer, 0, bufferExtent);
        }

        Texture<T_Type, T_textureDim>& getTexture()
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
        Texture<T_Type, T_textureDim> texture;

        ISAAC_IDX_TYPE bufferExtent;

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE> buffer;
    };


    template<typename T_Type>
    using Tex2D = Texture<T_Type, 2>;

    template<typename T_Type>
    using Tex3D = Texture<T_Type, 3>;

    template<typename T_DevAcc, typename T_Type>
    using Tex2DAllocator = TextureAllocator<T_DevAcc, T_Type, 2>;

    template<typename T_DevAcc, typename T_Type>
    using Tex3DAllocator = TextureAllocator<T_DevAcc, T_Type, 3>;


    template<int T_n>
    struct PersistentArrayStruct
    {
        Tex3D<isaac_float> textures[T_n];
    };

    struct GBuffer
    {
        isaac_size2 size;
        isaac_uint2 startOffset;
        Tex2D<isaac_byte4> color;
        Tex2D<isaac_float> depth;
        Tex2D<isaac_float3> normal;
        Tex2D<isaac_float> aoStrength;
    };

} // namespace isaac