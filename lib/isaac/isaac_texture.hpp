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

#include <alpaka/alpaka.hpp>


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


    template<typename T_Type, int T_textureDim, ISAAC_IDX_TYPE T_guardSize = 0>
    class Texture
    {
    public:
        Texture() = default;

        ISAAC_HOST_DEVICE_INLINE void init(T_Type* bufferPtr, const isaac_size_dim<T_textureDim>& size)
        {
            this->bufferPtr = bufferPtr;
            this->size = size;
            this->sizeWithGuard = size + ISAAC_IDX_TYPE(2) * T_guardSize;
        }

        // access between 0-1 in each dimension + guard
        template<FilterType T_filter = FilterType::NEAREST, BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type
        sample(const isaac_float_dim<T_textureDim>& normalizedCoord, const T_Type& borderValue = T_Type(0))
        {
            T_Type result;
            isaac_float_dim<T_textureDim> coord = isaac_float_dim<T_textureDim>(normalizedCoord * size);
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


        ISAAC_HOST_DEVICE_INLINE T_Type operator[](const isaac_int_dim<T_textureDim>& coord) const
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int(T_guardSize);
            return bufferPtr[hash(offsetCoord)];
        }


        ISAAC_HOST_DEVICE_INLINE T_Type& operator[](const isaac_int_dim<T_textureDim>& coord)
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int(T_guardSize);
            return bufferPtr[hash(offsetCoord)];
        }

        template<BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type
        safeMemoryAccess(const isaac_int_dim<T_textureDim>& coord, const T_Type& borderValue = T_Type(0))
        {
            isaac_int_dim<T_textureDim> offsetCoord;
            if(T_border == BorderType::REPEAT)
            {
                for(int i = 0; i < T_textureDim; ++i)
                    offsetCoord[i] = (coord[i] + isaac_int(T_guardSize)) % sizeWithGuard[i];
            }
            else if(T_border == BorderType::VALUE)
            {
                offsetCoord = coord + isaac_int(T_guardSize);
                if(!isInLowerBounds(coord, isaac_int_dim<T_textureDim>(0))
                   || !isInUpperBounds(coord, isaac_int_dim<T_textureDim>(sizeWithGuard)))
                    return borderValue;
            }
            else
            {
                offsetCoord = glm::clamp(
                    coord + isaac_int(T_guardSize),
                    isaac_int_dim<T_textureDim>(0),
                    isaac_int_dim<T_textureDim>(sizeWithGuard));
            }
            return (*this)[offsetCoord - isaac_int(T_guardSize)];
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<1>& coord)
        {
            return coord.x;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<2>& coord)
        {
            return coord.x + coord.y * sizeWithGuard.x;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<3>& coord)
        {
            return coord.x + coord.y * sizeWithGuard.x + coord.z * sizeWithGuard.y * sizeWithGuard.z;
        }


    private:
        T_Type* bufferPtr;
        isaac_size_dim<T_textureDim> size;
        isaac_size_dim<T_textureDim> sizeWithGuard;


        template<BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type interpolate(isaac_float_dim<1> coord, const T_Type& borderValue = T_Type(0))
        {
            T_Type data2[2];
            for(int x = 0; x < 2; x++)
            {
                data2[x] = safeMemoryAccess<T_border>(isaac_float_dim<1>(coord) + isaac_float_dim<1>(x), borderValue);
            }

            return linear(glm::fract(coord), data2);
        }

        template<BorderType T_border = BorderType::CLAMP>
        ISAAC_HOST_DEVICE_INLINE T_Type interpolate(isaac_float_dim<2> coord, const T_Type& borderValue = T_Type(0))
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
        ISAAC_HOST_DEVICE_INLINE T_Type interpolate(isaac_float_dim<3> coord, const T_Type& borderValue = T_Type(0))
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

    template<typename T_DevAcc, typename T_Type, int T_textureDim, ISAAC_IDX_TYPE T_guardSize = 0>
    class TextureWrapper
    {
        using FraDim = alpaka::DimInt<1>;

    public:
        TextureWrapper(const T_DevAcc& devAcc, const isaac_size_dim<T_textureDim>& size)
            : bufferExtent(1000)
            , buffer(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent))

        {
            const isaac_size_dim<T_textureDim> sizeWithGuard = size + ISAAC_IDX_TYPE(2) * T_guardSize;

            bufferExtent = sizeWithGuard[0];
            for(int i = 1; i < T_textureDim; ++i)
            {
                bufferExtent *= (sizeWithGuard[i]);
            }

            buffer = alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent);

            texture.init(alpaka::getPtrNative(buffer), size);
        }

        template<typename T_Queue, typename T_ViewDst>
        void copyToBuffer(T_Queue& queue, T_ViewDst& viewDst)
        {
            alpaka::memcpy(queue, viewDst, buffer, bufferExtent);
        }

        Texture<T_Type, T_textureDim, T_guardSize> getTexture()
        {
            return texture;
        }

    private:
        Texture<T_Type, T_textureDim, T_guardSize> texture;

        ISAAC_IDX_TYPE bufferExtent;

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE> buffer;
    };

    template<typename T_Type, ISAAC_IDX_TYPE T_guardSize = 0>
    using Tex2D = Texture<T_Type, 2, T_guardSize>;

    template<typename T_Type, ISAAC_IDX_TYPE T_guardSize = 0>
    using Tex3D = Texture<T_Type, 3, T_guardSize>;

    template<typename T_DevAcc, typename T_Type, ISAAC_IDX_TYPE T_guardSize = 0>
    using Tex2DWrapper = TextureWrapper<T_DevAcc, T_Type, 2, T_guardSize>;

    template<typename T_DevAcc, typename T_Type, ISAAC_IDX_TYPE T_guardSize = 0>
    using Tex3DWrapper = TextureWrapper<T_DevAcc, T_Type, 3, T_guardSize>;

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