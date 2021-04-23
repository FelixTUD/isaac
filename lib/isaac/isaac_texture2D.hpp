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


    template<typename T_Type>
    class Texture2D
    {
    public:
        Texture2D() = default;


        Texture2D(
            T_Type* bufferPtr,
            const isaac_size2& size,
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


        ISAAC_DEVICE_INLINE T_Type sample(const isaac_float2& coord) const
        {
            T_Type result;
            if(filter == FilterType::LINEAR)
            {
                result = interpolate(coord);
            }
            else
            {
                result = safeMemoryAccess(isaac_int2(coord));
            }
            return result;
        }

        ISAAC_DEVICE_INLINE void set(const isaac_int2& coord, const T_Type& value)
        {
            isaac_uint2 offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            bufferPtr[hash(offsetCoord)] = value;
        }

        ISAAC_DEVICE_INLINE T_Type get(const isaac_int2& coord) const
        {
            isaac_uint2 offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            return bufferPtr[hash(offsetCoord)];
        }


        ISAAC_DEVICE_INLINE isaac_uint hash(const isaac_uint2& coord) const
        {
            return coord.x + coord.y * sizeWithGuard.x;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_size2 getSize() const
        {
            return size;
        }
        ISAAC_HOST_DEVICE_INLINE isaac_size2 getSizeWithGuard() const
        {
            return sizeWithGuard;
        }
        ISAAC_HOST_DEVICE_INLINE ISAAC_IDX_TYPE getGuardSize() const
        {
            return guardSize;
        }

        ISAAC_DEVICE_INLINE T_Type safeMemoryAccess(const isaac_int2& coord) const
        {
            isaac_int2 offsetCoord;
            if(border == BorderType::REPEAT)
            {
                // Modulo modification to also account for negative values
                for(int i = 0; i < 2; ++i)
                {
                    offsetCoord[i] = (sizeWithGuard[i] + ((coord[i] + isaac_int(guardSize)) % sizeWithGuard[i]))
                        % sizeWithGuard[i];
                }
            }
            else if(border == BorderType::VALUE)
            {
                offsetCoord = coord + isaac_int(guardSize);
                if(!isInLowerBounds(offsetCoord, isaac_int2(0))
                   || !isInUpperBounds(offsetCoord, isaac_int2(sizeWithGuard)))
                    return T_Type(0);
            }
            else
            {
                offsetCoord = glm::clamp(coord + isaac_int(guardSize), isaac_int2(0), isaac_int2(sizeWithGuard) - 1);
            }
            return get(offsetCoord - isaac_int(guardSize));
        }


        ISAAC_DEVICE_INLINE T_Type interpolate(isaac_float_dim<2> coord) const
        {
            coord -= isaac_float(0.5);
            T_Type data4[2][2];
            for(int y = 0; y < 2; y++)
            {
                for(int x = 0; x < 2; x++)
                {
                    data4[x][y] = safeMemoryAccess(isaac_int2(glm::floor(coord)) + isaac_int2(x, y));
                }
            }

            return bilinear(glm::fract(coord), data4);
        }

    private:
        T_Type* bufferPtr;
        isaac_size2 size;
        isaac_size2 sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
        FilterType filter = FilterType::NEAREST;
        BorderType border = BorderType::VALUE;
    };


    template<typename T_DevAcc, typename T_Type>
    class Texture2DAllocator
    {
        using FraDim = alpaka::DimInt<1>;

    public:
        Texture2DAllocator(
            const T_DevAcc& devAcc,
            const isaac_size2& size,
            ISAAC_IDX_TYPE guardSize = 0,
            FilterType filter = FilterType::NEAREST,
            BorderType border = BorderType::CLAMP)
            : bufferExtent(0)
            , buffer(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent))
        {
            const isaac_size2 sizeWithGuard = size + ISAAC_IDX_TYPE(2) * guardSize;

            bufferExtent = sizeWithGuard.x * sizeWithGuard.y;

            buffer = alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent);

            texture = Texture2D<T_Type>(alpaka::getPtrNative(buffer), size, guardSize, filter, border);
        }

        template<typename T_Queue, typename T_ViewDst>
        void copyToBuffer(T_Queue& queue, T_ViewDst& viewDst) const
        {
            alpaka::memcpy(queue, viewDst, buffer, bufferExtent);
        }

        template<typename T_Queue>
        void clearColor(T_Queue& queue)
        {
            alpaka::memset(queue, buffer, 0, bufferExtent);
        }

        Texture2D<T_Type>& getTexture()
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
        Texture2D<T_Type> texture;

        ISAAC_IDX_TYPE bufferExtent;

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE> buffer;
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