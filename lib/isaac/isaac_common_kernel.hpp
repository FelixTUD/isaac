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
    ISAAC_CONSTANT isaac_mat4 isaac_inverse_d;

    //modelview matrix
    ISAAC_CONSTANT isaac_mat4 isaac_modelview_d;

    //projection matrix
    ISAAC_CONSTANT isaac_mat4 isaac_projection_d;

    //simulation size properties
    ISAAC_CONSTANT isaac_size_struct isaac_size_d;

    template <int N, typename Type1, typename Type2>
    ISAAC_DEVICE_INLINE bool isInLowerBounds(const isaac_vec_dim<N, Type1>& vec, const isaac_vec_dim<N, Type2>& lBounds)
    {
        for( int i = 0; i < N; ++i)
        {
            if( vec[i] < lBounds[i] )
                return false;
        }
        return true;
    }

    template <int N, typename Type1, typename Type2>
    ISAAC_DEVICE_INLINE bool isInUpperBounds(const isaac_vec_dim<N, Type1>& vec, const isaac_vec_dim<N, Type2>& uBounds)
    {
        for( int i = 0; i < N; ++i)
        {
            if( vec[i] >= uBounds[i] )
                return false;
        }
        return true;
    }

    template<
        typename TFilter
    >
    struct check_no_source_iterator
    {
        template<
            typename NR,
            typename TSource,
            typename TResult
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const NR & nr,
            const TSource & source,
            TResult & result
        ) const
        {
            result |= boost::mpl::at_c<
                TFilter,
                NR::value
            >::type::value;
        }
    };



    template<
        typename TSource
    >
    struct updateBufferKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
            const TSource source,
            void * const pointer,
            const isaac_int3 local_size
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
            if( !isInUpperBounds( dest, local_size + isaac_int3( 2 * ISAAC_GUARD_SIZE) ) )
                return;
            isaac_float_dim <TSource::feature_dim> * ptr =
                ( isaac_float_dim < TSource::feature_dim > * )( pointer );
            if( TSource::has_guard )
            {
                coord.z = -ISAAC_GUARD_SIZE;
                for( ; dest.z < local_size.z + 2 * ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                    coord.z++;
                }
            }
            else
            {
                coord.x = glm::clamp( coord.x, 0, local_size.x - 1 );
                coord.y = glm::clamp( coord.y, 0, local_size.y - 1 );
                coord.z = 0;
                for( ; dest.z < ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                }
                for( ; dest.z < local_size.z + ISAAC_GUARD_SIZE - 1; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                    coord.z++;
                }
                for( ; dest.z < local_size.z + 2 * ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                }
            }
        }
    };
}