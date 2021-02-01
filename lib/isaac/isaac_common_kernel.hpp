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

#include "isaac_macros.hpp"
#include "isaac_fusion_extension.hpp"
#include "isaac_functor_chain.hpp"

#include <limits>

namespace isaac
{
    //inverse mvp matrix
    ISAAC_CONSTANT isaac_mat4 InverseMVPMatrix;

    //modelview matrix
    ISAAC_CONSTANT isaac_mat4 ModelViewMatrix;

    //projection matrix
    ISAAC_CONSTANT isaac_mat4 ProjectionMatrix;

    //simulation size properties
    ISAAC_CONSTANT SimulationSizeStruct SimulationSize;

    struct Ray
    {
        isaac_float3 dir;
        isaac_float3 start;
        isaac_float3 end;
        isaac_float startDepth;
        isaac_float endDepth;
        bool isClipped;
        isaac_float3 clippingNormal;
    };

    ISAAC_DEVICE_INLINE Ray pixelToRay( 
        const isaac_float2 pixel, 
        const isaac_float2 framebuffer_size 
        )
    {
        //relative pixel position in framebuffer [-1.0 ... 1.0]
        //get normalized pixel position in framebuffer
        isaac_float2 viewportPos = isaac_float2( pixel ) / isaac_float2( framebuffer_size ) * isaac_float( 2 ) - isaac_float( 1 );
        
        //ray start position
        isaac_float4 startPos;
        startPos.x = viewportPos.x;
        startPos.y = viewportPos.y;
        startPos.z = -1.0f;
        startPos.w = 1.0f;

        //ray end position
        isaac_float4 endPos;
        endPos.x = viewportPos.x;
        endPos.y = viewportPos.y;
        endPos.z = 1.0f;
        endPos.w = 1.0f;

        //apply inverse modelview transform to ray start/end and get ray start/end as worldspace
        startPos = InverseMVPMatrix * startPos;
        endPos = InverseMVPMatrix * endPos;

        Ray ray;
        //apply the w-clip
        ray.start = startPos / startPos.w;
        ray.end = endPos / endPos.w;

        isaac_float max_size = SimulationSize.maxGlobalSizeScaled * isaac_float( 0.5 );

        //scale to globale grid size
        ray.start = ray.start * max_size;
        ray.end = ray.end * max_size;

        //get step vector
        ray.dir = glm::normalize( ray.end - ray.start );
        return ray;
    }

    template <int T_N, typename T_Type1, typename T_Type2>
    ISAAC_DEVICE_INLINE bool isInLowerBounds( 
        const isaac_vec_dim<T_N, T_Type1>& vec, 
        const isaac_vec_dim<T_N, T_Type2>& lBounds )
    {
        for( int i = 0; i < T_N; ++i)
        {
            if( vec[i] < lBounds[i] )
                return false;
        }
        return true;
    }

    template <int T_N, typename T_Type1, typename T_Type2>
    ISAAC_DEVICE_INLINE bool isInUpperBounds( 
        const isaac_vec_dim<T_N, T_Type1>& vec, 
        const isaac_vec_dim<T_N, T_Type2>& uBounds )
    {
        for( int i = 0; i < T_N; ++i)
        {
            if( vec[i] >= uBounds[i] )
                return false;
        }
        return true;
    }

    template<
        typename T_Filter
    >
    struct CheckNoSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_Result
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const T_NR & nr,
            const T_Source & source,
            T_Result & result
        ) const
        {
            result |= boost::mpl::at_c<
                T_Filter,
                T_NR::value
            >::type::value;
        }
    };



    template<
        typename T_Source
    >
    struct UpdateBufferKernel
    {
        template<
            typename T_Acc
        >
        ALPAKA_FN_ACC void operator()(
            T_Acc const & acc,
            const T_Source source,
            void * const pointer,
            const isaac_int3 localSize
        ) const
        {
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_int3 dest = {
                isaac_int( alpThreadIdx[1] ),
                isaac_int( alpThreadIdx[2] ),
                0
            };
            isaac_int3 coord = dest;
            coord.x -= ISAAC_GUARD_SIZE;
            coord.y -= ISAAC_GUARD_SIZE;
            if( !isInUpperBounds( dest, localSize + isaac_int3( 2 * ISAAC_GUARD_SIZE) ) )
                return;
            isaac_float_dim <T_Source::feature_dim> * ptr =
                ( isaac_float_dim < T_Source::feature_dim > * )( pointer );
            if( T_Source::has_guard )
            {
                coord.z = -ISAAC_GUARD_SIZE;
                for( ; dest.z < localSize.z + 2 * ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                            * ( localSize.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                    coord.z++;
                }
            }
            else
            {
                coord.x = glm::clamp( coord.x, 0, localSize.x - 1 );
                coord.y = glm::clamp( coord.y, 0, localSize.y - 1 );
                coord.z = 0;
                for( ; dest.z < ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                            * ( localSize.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                }
                for( ; dest.z < localSize.z + ISAAC_GUARD_SIZE - 1; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                            * ( localSize.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                    coord.z++;
                }
                for( ; dest.z < localSize.z + 2 * ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                            * ( localSize.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                }
            }
        }
    };
}