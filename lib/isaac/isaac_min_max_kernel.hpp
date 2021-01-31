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

#include "isaac_common_kernel.hpp"

namespace isaac
{
    template<
        typename TSource
    >
    struct minMaxKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
            const TSource source,
            const int nr,
            minmax_struct * const result,
            const isaac_size3 local_size,
            void const * const pointer
        ) const
        {
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_int3 coord = {
                isaac_int( alpThreadIdx[1] ),
                isaac_int( alpThreadIdx[2] ),
                0
            };

            if( !isInUpperBounds(coord, local_size) )
                return;
            isaac_float min = FLT_MAX;
            isaac_float max = -FLT_MAX;
            for( ; coord.z < local_size.z; coord.z++ )
            {
                isaac_float_dim <TSource::feature_dim> data;
                if( TSource::persistent )
                {
                    data = source[coord];
                }
                else
                {
                    isaac_float_dim <TSource::feature_dim> * ptr = (
                        isaac_float_dim < TSource::feature_dim > *
                    )( pointer );
                    data = ptr[coord.x + ISAAC_GUARD_SIZE
                               + ( coord.y + ISAAC_GUARD_SIZE )
                                 * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                               + ( coord.z + ISAAC_GUARD_SIZE ) * (
                                   ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                                   * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                               )];
                };
                isaac_float value = applyFunctorChain<TSource::feature_dim>(&data, nr);
                min = glm::min( min, value );
                max = glm::max( max, value );
            }
            result[coord.x + coord.y * local_size.x].min = min;
            result[coord.x + coord.y * local_size.x].max = max;
        }

    };



    template<
        typename TParticleSource
    >
    struct minMaxParticleKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
            const TParticleSource particle_source,
            const int nr,
            minmax_struct * const result,
            const isaac_size3 local_size
        ) const
        {
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_uint3 coord = {
                isaac_uint( alpThreadIdx[1] ),
                isaac_uint( alpThreadIdx[2] ),
                0
            };
            if( !isInUpperBounds(coord, local_size) )
                return;
            isaac_float min = FLT_MAX;
            isaac_float max = -FLT_MAX;
            for( ; coord.z < local_size.z; coord.z++ )
            {
                auto particle_iterator = particle_source.getIterator( coord );
                for( int i = 0; i < particle_iterator.size; i++ )
                {
                    isaac_float_dim <TParticleSource::feature_dim> data;

                    data = particle_iterator.getAttribute( );

                    isaac_float value = applyFunctorChain<TParticleSource::feature_dim>(&data, nr);
                    min = glm::min( min, value );
                    max = glm::max( max, value );
                }

            }
            result[coord.x + coord.y * local_size.x].min = min;
            result[coord.x + coord.y * local_size.x].max = max;
        }

    };
}