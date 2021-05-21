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


#include "isaac/isaac_iso_kernel.hpp"
#include "isaac/isaac_min_max_kernel.hpp"
#include "isaac/isaac_particle_kernel.hpp"
#include "isaac/isaac_ssao_kernel.hpp"
#include "isaac/isaac_volume_kernel.hpp"

namespace isaac
{
    template<typename T_Acc, typename T_Queue, typename T_KernelFnObj, typename... T_Args>
    inline void executeKernelOnVolume(
        isaac_size3 volumeSize,
        T_Queue& queue,
        const T_KernelFnObj& kernelFnObj,
        T_Args&&... args)
    {
        using Dim = alpaka::DimInt<3>;
        isaac_size3 blockSize(glm::min(volumeSize.x, ISAAC_IDX_TYPE(512)), 1, 1);
        isaac_size3 gridSize = (volumeSize + blockSize - ISAAC_IDX_TYPE(1)) / blockSize;

#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
        if(boost::mpl::not_<boost::is_same<T_Acc, alpaka::AccGpuCudaRt<Dim, ISAAC_IDX_TYPE>>>::value)
#endif
        {
            blockSize = isaac_size3(1);
            gridSize = volumeSize;
        }
        const alpaka::Vec<Dim, ISAAC_IDX_TYPE> threadElements(ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1));
        // Coordinates need to be swapped for alpaka
        const alpaka::Vec<Dim, ISAAC_IDX_TYPE> blocks(blockSize.z, blockSize.y, blockSize.x);
        const alpaka::Vec<Dim, ISAAC_IDX_TYPE> grid(gridSize.z, gridSize.y, gridSize.x);
        auto const workdiv = alpaka::WorkDivMembers<Dim, ISAAC_IDX_TYPE>(grid, blocks, threadElements);
        auto const instance = alpaka::createTaskKernel<T_Acc>(workdiv, kernelFnObj, args...);
        alpaka::enqueue(queue, instance);
    }

    template<typename T_Acc, typename T_Queue, typename T_AccDev, typename T_Type, IndexType T_indexType>
    void syncOwnGuardTextures(
        T_Queue& queue,
        SyncedTexture3DAllocator<T_AccDev, T_Type, T_indexType>& texture,
        Neighbours<isaac_int>& neighbourIds)
    {
        const SyncToOwnGuard kernel;
        for(int z = 0; z < 3; z++)
        {
            for(int y = 0; y < 3; y++)
            {
                for(int x = 0; x < 3; x++)
                {
                    if(x != 1 || y != 1 || z != 1)
                    {
                        isaac_int3 side(x, y, z);
                        isaac_int3 signedSide = side - ISAAC_IDX_TYPE(1);
                        if(neighbourIds.get(signedSide) != -1)
                        {
                            Texture<T_Type, 3>& dstTexture = texture.getOwnGuardTexture(signedSide);
                            executeKernelOnVolume<T_Acc>(
                                dstTexture.getSize(),
                                queue,
                                kernel,
                                signedSide,
                                texture.getTexture(),
                                dstTexture);
                        }
                    }
                }
            }
        }
        alpaka::wait(queue);
    }

    template<typename T_Acc, typename T_Queue, typename T_AccDev, typename T_Type, IndexType T_indexType>
    void syncNeighbourGuardTextures(
        T_Queue& queue,
        SyncedTexture3DAllocator<T_AccDev, T_Type, T_indexType>& texture,
        Neighbours<isaac_int>& neighbourIds)
    {
        const SyncFromNeighbourGuard kernel;
        for(int z = 0; z < 3; z++)
        {
            for(int y = 0; y < 3; y++)
            {
                for(int x = 0; x < 3; x++)
                {
                    if(x != 1 || y != 1 || z != 1)
                    {
                        isaac_int3 side(x, y, z);
                        isaac_int3 signedSide = side - ISAAC_IDX_TYPE(1);
                        if(neighbourIds.get(signedSide) != -1)
                        {
                            Texture<T_Type, 3>& guardTexture = texture.getNeighbourGuardTexture(signedSide);
                            executeKernelOnVolume<T_Acc>(
                                guardTexture.getSize(),
                                queue,
                                kernel,
                                signedSide,
                                guardTexture,
                                texture.getTexture());
                        }
                    }
                }
            }
        }
        alpaka::wait(queue);
    }
} // namespace isaac