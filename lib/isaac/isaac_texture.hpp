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

        template<FilterType T_filter = FilterType::NEAREST, BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type
        sample(const isaac_float_dim<T_textureDim>& coord, const T_Type& borderValue = T_Type(0)) const
        {
            T_Type result;
            if(T_filter == FilterType::LINEAR)
            {
                result = interpolate<T_border>(coord, borderValue);
            }
            else
            {
                result = safeMemoryAccess<T_border>(isaac_int_dim<T_textureDim>(coord), borderValue);
            }
            return result;
        }

        ISAAC_HOST_DEVICE_INLINE void set(const isaac_int_dim<T_textureDim>& coord, const T_Type& value)
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            bufferPtr[hash(offsetCoord)] = value;
        }

        ISAAC_HOST_DEVICE_INLINE T_Type get(const isaac_int_dim<T_textureDim>& coord) const
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            return bufferPtr[hash(offsetCoord)];
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

        template<BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type
        safeMemoryAccess(const isaac_int_dim<T_textureDim>& coord, const T_Type& borderValue = T_Type(0)) const
        {
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
            return get(offsetCoord - isaac_int(guardSize));
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


    private:
        T_Type* bufferPtr = nullptr;
        isaac_size_dim<T_textureDim> size;
        isaac_size_dim<T_textureDim> sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;


        template<BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type
        interpolate(isaac_float_dim<1> coord, const T_Type& borderValue = T_Type(0)) const
        {
            T_Type data2[2];
            for(int x = 0; x < 2; x++)
            {
                data2[x] = safeMemoryAccess<T_border>(isaac_float_dim<1>(coord) + isaac_float_dim<1>(x), borderValue);
            }

            return linear(glm::fract(coord), data2);
        }

        template<BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type
        interpolate(isaac_float_dim<2> coord, const T_Type& borderValue = T_Type(0)) const
        {
            T_Type data4[2][2];
            for(int x = 0; x < 2; x++)
            {
                for(int y = 0; y < 2; y++)
                {
                    data4[x][y] = safeMemoryAccess<T_border>(isaac_int2(coord) + isaac_int2(x, y), borderValue);
                }
            }

            return bilinear(glm::fract(coord), data4);
        }

        template<BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type
        interpolate(isaac_float_dim<3> coord, const T_Type& borderValue = T_Type(0)) const
        {
            T_Type data8[2][2][2];
            for(int x = 0; x < 2; x++)
            {
                for(int y = 0; y < 2; y++)
                {
                    for(int z = 0; z < 2; z++)
                    {
                        data8[x][y][z]
                            = safeMemoryAccess<T_border>(isaac_int3(coord) + isaac_int3(x, y, z), borderValue);
                    }
                }
            }

            return trilinear(glm::fract(coord), data8);
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

        Texture<T_Type, T_textureDim> getTexture() const
        {
            return texture;
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