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

        Texture3D(cudaArray_t cudaArray, const isaac_size3& size, ISAAC_IDX_TYPE guardSize = 0)
            : size(size)
            , sizeWithGuard(size + ISAAC_IDX_TYPE(2) * guardSize)
            , guardSize(guardSize)
        {
            cudaResourceDesc rescDesc;
            memset(&rescDesc, 0, sizeof(rescDesc));
            rescDesc.resType = cudaResourceTypeArray;
            rescDesc.res.array.array = cudaArray;

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            for(int i = 0; i < 3; i++)
            {
                switch(T_border)
                {
                case BorderType::REPEAT:
                    texDesc.addressMode[i] = cudaAddressModeBorder;
                    break;
                case BorderType::CLAMP:
                    texDesc.addressMode[i] = cudaAddressModeClamp;
                    break;
                case BorderType::VALUE:
                    texDesc.addressMode[i] = cudaAddressModeBorder;
                    break;
                }
            }
            if(T_filter == FilterType::LINEAR)
            {
                texDesc.filterMode = cudaFilterModeLinear;
            }
            texDesc.readMode = cudaReadModeElementType;
            for(int i = 0; i < 3; i++)
                texDesc.borderColor[i] = 0;
            texDesc.normalizedCoords = 0;

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaCreateTextureObject(&textureObj, &rescDesc, &texDesc, NULL));
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaCreateSurfaceObject(&surfaceObj, &rescDesc));
        }

        ISAAC_DEVICE_INLINE T_Type sample(const isaac_float3& coord) const
        {
            isaac_float3 offsetCoord = coord + isaac_float(guardSize);
            return convertCudaType(
                tex3D<CudaType<T_Type>::type>(textureObj, offsetCoord.x, offsetCoord.y, offsetCoord.z));
        }

        ISAAC_DEVICE_INLINE void set(const isaac_int3& coord, const T_Type& value)
        {
            isaac_uint3 offsetCoord = coord + isaac_int(guardSize);
            // for some reason cuda requires the x coordinate to be byte addressed
            surf3Dwrite(
                convertCudaType(value),
                surfaceObj,
                offsetCoord.x * sizeof(CudaType<T_Type>::type),
                offsetCoord.y,
                offsetCoord.z);
        }

        ISAAC_DEVICE_INLINE T_Type get(const isaac_int3& coord) const
        {
            isaac_uint3 offsetCoord = coord + isaac_int(guardSize);
            return convertCudaType(surf3Dread<CudaType<T_Type>::type>(
                surfaceObj,
                offsetCoord.x * sizeof(CudaType<T_Type>::type),
                offsetCoord.y,
                offsetCoord.z));
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

    private:
        isaac_size3 size;
        isaac_size3 sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
        cudaTextureObject_t textureObj;
        cudaSurfaceObject_t surfaceObj;
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
        {
            bufferExtent = sizeWithGuard.x * sizeWithGuard.y * sizeWithGuard.z;


            const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<typename CudaType<T_Type>::type>();
            cudaMalloc3DArray(
                &cudaArray,
                &channelDesc,
                make_cudaExtent(sizeWithGuard.x, sizeWithGuard.y, sizeWithGuard.z),
                cudaArraySurfaceLoadStore);
        }


        template<typename T_Queue>
        void clearColor(T_Queue& queue)
        {
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
        cudaArray_t getBufferPtr()
        {
            return cudaArray;
        }

    private:
        cudaArray_t cudaArray;
        ISAAC_IDX_TYPE bufferExtent;
        isaac_size3 size;
        isaac_size3 sizeWithGuard;
        ISAAC_IDX_TYPE guardSize;
    };
} // namespace isaac