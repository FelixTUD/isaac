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

#include "isaac_texture2D.hpp"

namespace isaac
{
    template<
        typename T_Type,
        FilterType T_filter = FilterType::NEAREST,
        BorderType T_border = BorderType::CLAMP,
        IndexType T_indexType = IndexType::SWEEP>
    class Texture3D
    {
    public:
        Texture3D() = default;


        Texture3D(T_Type* bufferPtr, const isaac_size3& size, ISAAC_IDX_TYPE guardSize = 0)
            : bufferPtr(bufferPtr)
            , size(size)
            , sizeWithGuard(size + ISAAC_IDX_TYPE(2) * guardSize)
            , guardSize(guardSize)
        {
        }


        ISAAC_DEVICE_INLINE T_Type sample(const isaac_float3& coord) const
        {
            T_Type result;
            if(T_filter == FilterType::LINEAR)
            {
                result = interpolate(coord);
            }
            else
            {
                result = safeMemoryAccess(isaac_int3(coord));
            }
            return result;
        }

        ISAAC_DEVICE_INLINE void set(const isaac_int3& coord, const T_Type& value)
        {
            isaac_uint3 offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            bufferPtr[hash(offsetCoord)] = value;
        }

        ISAAC_DEVICE_INLINE T_Type get(const isaac_int3& coord) const
        {
            isaac_uint3 offsetCoord = coord + isaac_int(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            return bufferPtr[hash(offsetCoord)];
        }

        ISAAC_DEVICE_INLINE isaac_uint hash(const isaac_uint3& coord) const
        {
            if(T_indexType == IndexType::MORTON)
                return (part1By2(coord.z) << 2) | (part1By2(coord.y) << 1) | part1By2(coord.x);
            else
                return coord.x + coord.y * sizeWithGuard.x + coord.z * sizeWithGuard.x * sizeWithGuard.y;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_size3 getSize() const
        {
            return size;
        }
        ISAAC_HOST_DEVICE_INLINE isaac_size3 getSizeWithGuard() const
        {
            return sizeWithGuard;
        }
        ISAAC_HOST_DEVICE_INLINE ISAAC_IDX_TYPE getGuardSize() const
        {
            return guardSize;
        }

        ISAAC_DEVICE_INLINE T_Type safeMemoryAccess(const isaac_int3& coord) const
        {
            isaac_int3 offsetCoord;
            if(T_border == BorderType::REPEAT)
            {
                // Modulo modification to also account for negative values
                for(int i = 0; i < 3; ++i)
                {
                    offsetCoord[i] = (sizeWithGuard[i] + ((coord[i] + isaac_int(guardSize)) % sizeWithGuard[i]))
                        % sizeWithGuard[i];
                }
            }
            else if(T_border == BorderType::VALUE)
            {
                offsetCoord = coord + isaac_int(guardSize);
                if(!isInLowerBounds(offsetCoord, isaac_int3(0))
                   || !isInUpperBounds(offsetCoord, isaac_int3(sizeWithGuard)))
                    return T_Type(0);
            }
            else
            {
                offsetCoord = glm::clamp(coord + isaac_int(guardSize), isaac_int3(0), isaac_int3(sizeWithGuard) - 1);
            }
            return get(offsetCoord - isaac_int(guardSize));
        }

        ISAAC_DEVICE_INLINE T_Type interpolate(isaac_float3 coord) const
        {
            coord -= isaac_float(0.5);
            T_Type data8[2][2][2];
            if(T_border == BorderType::CLAMP)
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
                            data8[x][y][z] = get(isaac_int3(glm::floor(coord)) + isaac_int3(x, y, z));
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
                            data8[x][y][z] = safeMemoryAccess(isaac_int3(glm::floor(coord)) + isaac_int3(x, y, z));
                        }
                    }
                }
            }


            return trilinear(glm::fract(coord), data8);
        }


    private:
        T_Type* bufferPtr;
        isaac_size3 size;
        isaac_size3 sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
    };


    template<typename T_DevAcc, typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    class Texture3DAllocator
    {
        using FraDim = alpaka::DimInt<1>;

    public:
        Texture3DAllocator(const T_DevAcc& devAcc, const isaac_size3& size, ISAAC_IDX_TYPE guardSize = 0)
            : bufferExtent(0)
            , size(size)
            , guardSize(guardSize)
            , sizeWithGuard(size + ISAAC_IDX_TYPE(2) * guardSize)
            , buffer(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent))
        {
            if(T_indexType == IndexType::MORTON)
            {
                ISAAC_IDX_TYPE maxDim = glm::max(sizeWithGuard.x, glm::max(sizeWithGuard.y, sizeWithGuard.z));
                bufferExtent = glm::pow(maxDim, ISAAC_IDX_TYPE(3));
            }
            else
            {
                bufferExtent = sizeWithGuard.x * sizeWithGuard.y * sizeWithGuard.z;
            }

            buffer = alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent);
        }

        template<typename T_Queue>
        void clearColor(T_Queue& queue)
        {
            alpaka::memset(queue, buffer, 0, bufferExtent);
        }

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE>& getTextureView()
        {
            return buffer;
        }


        ISAAC_IDX_TYPE getBufferExtent()
        {
            return bufferExtent;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_size3 getSize() const
        {
            return size;
        }
        ISAAC_HOST_DEVICE_INLINE isaac_size3 getSizeWithGuard() const
        {
            return sizeWithGuard;
        }
        ISAAC_HOST_DEVICE_INLINE ISAAC_IDX_TYPE getGuardSize() const
        {
            return guardSize;
        }
        T_Type* getBufferPtr()
        {
            return alpaka::getPtrNative(buffer);
        }

    private:
        ISAAC_IDX_TYPE bufferExtent;
        isaac_size3 size;
        isaac_size3 sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE> buffer;
    };
} // namespace isaac