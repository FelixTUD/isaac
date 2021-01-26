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
    /**
     * @brief Checks for collision with source particles
     * 
     * Returns color, normal, position of particle
     * 
     * @tparam Ttransfer_size 
     * @tparam TOffset 
     * @tparam TFilter 
     */
    template<
        ISAAC_IDX_TYPE Ttransfer_size,
        int TOffset,
        typename TFilter
    >
    struct merge_particle_iterator
    {
        template<
            typename NR,
            typename TSource,
            typename TTransferArray,
            typename TSourceWeight,
            typename TParticleScale
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const NR & nr,
            const TSource & source,                   //particle source
            const isaac_float3 & start,               //ray start position in local volume
            const isaac_float3 & dir,
            const isaac_uint3 & cell_pos,             //cell to test in local volume
            const TTransferArray & transferArray,     //transfer function
            const TSourceWeight & sourceWeight,       //weight of this particle source for radius
            const TParticleScale & particle_scale,    //scale of volume to prevent stretched particles
            const isaac_float3 & clipping_normal,     //normal of the intersecting clipping plane
            const bool & is_clipped,
            isaac_float4 & out_color,                 //resulting particle color
            isaac_float3 & out_normal,                //resulting particle normal
            isaac_float3 & out_position,              //resulting particle hit position
            bool & out_particle_hit,                  //true or false if particle has been hit or not
            isaac_float & out_depth                       //resulting particle depth
        ) const
        {
            const int sourceNumber = NR::value + TOffset;
            if( mpl::at_c<
                TFilter,
                NR::value
            >::type::value )
            {
                auto particle_iterator = source.getIterator( cell_pos );

                // iterate over all particles in current cell
                for( int i = 0; i < particle_iterator.size; ++i )
                {
                    // ray sphere intersection
                    isaac_float3 particle_pos =
                        ( particle_iterator.getPosition( ) + isaac_float3( cell_pos ) )
                        * particle_scale;
                    isaac_float3 L = particle_pos - start;
                    isaac_float radius = particle_iterator.getRadius( )
                                         * sourceWeight.value[NR::value + TOffset];
                    isaac_float radius2 = radius * radius;
                    isaac_float tca = glm::dot( L, dir );
                    isaac_float d2 = glm::dot( L, L ) - tca * tca;
                    if( d2 <= radius2 )
                    {
                        isaac_float thc = sqrt( radius2 - d2 );
                        isaac_float t0 = tca - thc;
                        isaac_float t1 = tca + thc;

                        // if the ray hits the sphere
                        if( t1 >= 0 && t0 < out_depth )
                        {
                            isaac_float_dim <TSource::feature_dim>
                                data = particle_iterator.getAttribute( );

                            isaac_float result = isaac_float( 0 );

                            // apply functorchain
                           result = applyFunctorChain<TSource::feature_dim>(&data, sourceNumber);

                            // apply transferfunction
                            ISAAC_IDX_TYPE lookup_value = ISAAC_IDX_TYPE(
                                glm::round( result * isaac_float( Ttransfer_size ) )
                            );
                            lookup_value = glm::clamp( lookup_value, ISAAC_IDX_TYPE( 0 ), Ttransfer_size - 1 );
                            isaac_float4 value = transferArray.pointer[sourceNumber][lookup_value];

                            // check if the alpha value is greater or equal than 0.5
                            if( value.w >= 0.5f )
                            {
                                out_color = value;
                                out_depth = t0;
                                out_particle_hit = 1;
                                out_position = particle_pos;
                                out_normal = start + t0 * dir - particle_pos;
                                if( t0 < 0 && is_clipped )
                                {
                                    #if ISAAC_AO_BUG_FIX == 1
                                        out_depth = 0;
                                    #endif
                                        out_normal = -clipping_normal;
                                }
                            }
                        }
                    }
                    particle_iterator.next( );
                }
            }
        }
    };


    template<
        typename TParticleList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        ISAAC_IDX_TYPE Ttransfer_size,
        int TSourceOffset
    >
    struct ParticleRenderKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
            uint32_t * const pixels,          //ptr to output pixels
            isaac_float3 * const gDepth,      //depth buffer
            isaac_float3 * const gNormal,     //normal buffer
            isaac_size2 framebuffer_size,     //size of framebuffer
            isaac_uint2 framebuffer_start,    //framebuffer offset
            TParticleList particle_sources,   //source simulation particles
            isaac_float4 background_color,    //color of render background
            TTransferArray transferArray,     //mapping to simulation memory
            TSourceWeight sourceWeight,       //weights of sources for blending
            TPointerArray pointerArray,
            isaac_float3 scale,               //isaac set scaling
            clipping_struct input_clipping,   //clipping planes
            ao_struct ambientOcclusion        //ambient occlusion params
        ) const
        {
            //get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_uint2 pixel = isaac_uint2( alpThreadIdx[2], alpThreadIdx[1] );
            //apply framebuffer offset to pixel
            //stop if pixel position is out of bounds
            pixel = pixel + framebuffer_start;
            if( pixel.x >= framebuffer_size.x || pixel.y >= framebuffer_size.y )
                return;

            //gNormalBuffer default value
            isaac_float3 default_normal = {0.0, 0.0, 0.0};
            isaac_float3 default_depth = {0.0, 0.0, 1.0};

            //set background color
            isaac_float4 color = background_color;
            bool at_least_one = true;
            isaac_for_each_with_mpl_params(
                particle_sources,
                check_no_source_iterator< TFilter >( ),
                at_least_one
            );
            if( !at_least_one )
            {
                ISAAC_SET_COLOR ( 
                    pixels[pixel.x + pixel.y * framebuffer_size.x], 
                    color 
                )
                gNormal[pixel.x + pixel.y * framebuffer_size.x] = default_normal;
                gDepth[pixel.x + pixel.y * framebuffer_size.x] = default_depth;
                return;
            }

            //relative pixel position in framebuffer [-1.0 ... 1.0]
            isaac_float2 pixel_f = isaac_float2( pixel ) / isaac_float2( framebuffer_size ) * isaac_float( 2 ) - isaac_float( 1 );

            //ray start position
            isaac_float4 start_p;
            start_p.x = pixel_f.x;
            start_p.y = pixel_f.y;
            start_p.z = -1.0f;
            start_p.w = 1.0f;

            //ray end position
            isaac_float4 end_p;
            end_p.x = pixel_f.x;
            end_p.y = pixel_f.y;
            end_p.z = 1.0f;
            end_p.w = 1.0f;

            //apply inverse modelview transform to ray start/end and get ray start/end as worldspace
            isaac_float4 start_w = isaac_inverse_d * start_p;
            isaac_float4 end_w = isaac_inverse_d * end_p;
            isaac_float3 start = start_w / start_w.w;
            isaac_float3 end = end_w / end_w.w;

            isaac_float max_size = isaac_size_d.max_global_size_scaled * isaac_float( 0.5 );

            //scale to globale grid size
            start = start * max_size;
            end = end * max_size;


            //clipping planes with transformed positions
            clipping_struct clipping;
            //set values for clipping planes
            //scale position to global size
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                clipping.elem[i].position = input_clipping.elem[i].position * isaac_float3( isaac_size_d.global_size ) * scale * isaac_float( 0.5 );
                clipping.elem[i].normal = input_clipping.elem[i].normal;
            }

            //move to local (scaled) grid
            //get offset of subvolume in global volume
            isaac_float3 position_offset = isaac_float3( isaac_int3( isaac_size_d.global_size_scaled ) / 2 - isaac_int3( isaac_size_d.position_scaled ) );

            //apply subvolume offset to start and end
            start = start + position_offset;
            end = end + position_offset;

            //apply subvolume offset to position checked clipping plane
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                clipping.elem[i].position =
                    clipping.elem[i].position + position_offset;
            }

            //get step vector
            isaac_float3 ray_dir = glm::normalize( end - start );

            //clip ray on volume bounding box
            isaac_float3 bb_intersection_min = -start / ray_dir;
            isaac_float3 bb_intersection_max = ( isaac_float3( isaac_size_d.local_size ) * scale - start ) / ray_dir;

            //bb_intersection_min shall have the smaller values
            ISAAC_SWITCH_IF_SMALLER ( bb_intersection_max.x, bb_intersection_min.x )
            ISAAC_SWITCH_IF_SMALLER ( bb_intersection_max.y, bb_intersection_min.y )
            ISAAC_SWITCH_IF_SMALLER ( bb_intersection_max.z, bb_intersection_min.z )

            isaac_float first_f = glm::max( bb_intersection_min.x, glm::max( bb_intersection_min.y, bb_intersection_min.z ) );
            isaac_float last_f = glm::min( bb_intersection_max.x, glm::min( bb_intersection_max.y, bb_intersection_max.z ) );

            bool is_clipped = false;
            isaac_float3 clipping_normal;
            //Iterate over clipping planes and adjust ray
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                isaac_float d = glm::dot( ray_dir, clipping.elem[i].normal);

                isaac_float intersection_depth = ( glm::dot( clipping.elem[i].position, clipping.elem[i].normal )
                                                    - glm::dot( start, clipping.elem[i].normal ) ) / d;
                if( d > 0 )
                {
                    if( last_f < intersection_depth )
                    {
                        ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebuffer_size.x], color )
                        return;
                    }
                    if( first_f <= intersection_depth )
                    {
                        clipping_normal = clipping.elem[i].normal;
                        is_clipped = true;
                        first_f = intersection_depth;
                    }
                }
                else
                {
                    if( first_f > intersection_depth )
                    {
                        ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebuffer_size.x], color )
                        return;
                    }
                    if( last_f > intersection_depth )
                    {
                        last_f = intersection_depth;
                    }
                }
            }
            first_f = glm::max( first_f, 0.0f );

            if( first_f > last_f )
            {
                ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebuffer_size.x], color )

                //this function aborts drawing and therfore wont set any normal or depth values
                //defaults will be applied for clean images
                gNormal[pixel.x + pixel.y * framebuffer_size.x] = default_normal;
                gDepth[pixel.x + pixel.y * framebuffer_size.x] = default_depth;

                return;
            }


            isaac_float4 particle_color = background_color;
            isaac_float depth = std::numeric_limits<isaac_float>::max( );
            bool particle_hit = false;
            // light direction is camera direction
            isaac_float3 light_dir = -ray_dir;

            // get the signs of the direction for the raymarch
            isaac_int3 dir_sign = glm::sign( ray_dir );

            // calculate current position in scaled object space
            isaac_float3 current_pos = start + ray_dir * first_f;

            // calculate current local cell coordinates
            isaac_uint3 current_cell = isaac_uint3( glm::clamp( 
                                    isaac_int3( current_pos / scale ), 
                                    isaac_int3( 0 ), 
                                    isaac_int3( isaac_size_d.local_particle_size - ISAAC_IDX_TYPE( 1 ) ) 
                                ) );

            isaac_float ray_length = last_f - first_f;
            isaac_float tested_length = 0;


            // calculate next intersection with each dimension
            isaac_float3 t = ( ( isaac_float3( current_cell ) + isaac_float3( glm::max( dir_sign, 0 ) ) ) 
                    * scale - current_pos ) / ray_dir;

            // calculate delta length to next intersection in the same dimension
            isaac_float3 delta_t = scale / ray_dir * isaac_float3( dir_sign );

            isaac_float3 particle_hitposition(0);

            // check for 0 to stop infinite looping
            if( ray_dir.x == 0 )
            {
                t.x = std::numeric_limits<isaac_float>::max( );
            }
            if( ray_dir.y == 0 )
            {
                t.y = std::numeric_limits<isaac_float>::max( );
            }
            if( ray_dir.z == 0 )
            {
                t.z = std::numeric_limits<isaac_float>::max( );
            }


            //normal at particle hit position
            isaac_float3 particle_normal = default_normal;

            // iterate over all cells on the ray path
            // check if the ray leaves the local volume, has a particle hit or exceeds the max ray distance
            while( isInUpperBounds(current_cell, isaac_size_d.local_particle_size)
                && particle_hit == false
                && tested_length <= ray_length )
            {

                // calculate particle intersections for each particle source
                isaac_for_each_with_mpl_params(
                    particle_sources,
                    merge_particle_iterator<
                        Ttransfer_size,
                        TSourceOffset,
                        TFilter
                    >( ),
                    current_pos,
                    ray_dir,
                    current_cell,
                    transferArray,
                    sourceWeight,
                    scale,
                    clipping_normal,
                    is_clipped,
                    particle_color,
                    particle_normal,
                    particle_hitposition,
                    particle_hit,
                    depth
                );


                // adds the delta t value to the smallest dimension t and increment the cell index in the dimension
                if( t.x < t.y && t.x < t.z )
                {
                    current_cell.x += dir_sign.x;
                    tested_length = t.x;
                    t.x += delta_t.x;
                }
                else if( t.y < t.x && t.y < t.z )
                {
                    current_cell.y += dir_sign.y;
                    tested_length = t.y;
                    t.y += delta_t.y;
                }
                else
                {
                    current_cell.z += dir_sign.z;
                    tested_length = t.z;
                    t.z += delta_t.z;
                }

            }
            // if there was a hit set maximum volume raycast distance to particle hit distance and set particle color
            if( particle_hit )
            {

                // calculate lighting properties for the last hit particle
                particle_normal = glm::normalize( particle_normal );

                isaac_float light_factor = glm::dot( particle_normal, light_dir );

                isaac_float3 half_vector = glm::normalize( -ray_dir + light_dir );

                isaac_float specular = glm::dot( particle_normal, half_vector );

                specular = pow( specular, 10 );
                specular *= 0.5f;
                light_factor = light_factor * 0.5f + 0.5f;


                particle_color = glm::min( particle_color * light_factor + specular, isaac_float( 1 ) );
                particle_color.a = 1.0f;
            }


            ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebuffer_size.x], particle_color )
            //save the particle normal in the normal g buffer
            gNormal[pixel.x + pixel.y * framebuffer_size.x] = particle_normal;
            
            //save the cell depth in our g buffer (depth)
            isaac_float3 depth_value = {
                0.0f,
                1.0f,
                depth
            };
            gDepth[pixel.x + pixel.y * framebuffer_size.x] = depth_value;
        }
    };



    template<
        typename TParticleList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        ISAAC_IDX_TYPE TTransfer_size,
        typename TAccDim,
        typename TAcc,
        typename TStream,
        typename TFunctionChain,
        int TSourceOffset,
        int N
    >
    struct ParticleRenderKernelCaller
    {
        inline static void call(
            TStream stream,
            uint32_t * framebuffer,
            isaac_float3 * depthBuffer,
            isaac_float3 * normalBuffer,
            const isaac_size2 & framebuffer_size,
            const isaac_uint2 & framebuffer_start,
            const TParticleList & particle_sources,
            const isaac_float4 & background_color,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            const TPointerArray & pointerArray,
            IceTInt const * const readback_viewport,
            const isaac_float3 & scale,
            const clipping_struct & clipping,
            const ao_struct & ambientOcclusion
        )
        {
            if( sourceWeight.value[TSourceOffset + mpl::size< TParticleList >::type::value - N] == isaac_float( 0 ) )
            {
                ParticleRenderKernelCaller<
                    TParticleList,
                    TTransferArray,
                    TSourceWeight,
                    TPointerArray,
                    typename mpl::push_back<
                        TFilter,
                        mpl::false_
                    >::type,
                    TTransfer_size,
                    TAccDim,
                    TAcc,
                    TStream,
                    TFunctionChain,
                    TSourceOffset,
                    N - 1
                >::call(
                    stream,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    particle_sources,
                    background_color,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    readback_viewport,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
            else
            {
                ParticleRenderKernelCaller<
                    TParticleList,
                    TTransferArray,
                    TSourceWeight,
                    TPointerArray,
                    typename mpl::push_back<
                        TFilter,
                        mpl::true_
                    >::type,
                    TTransfer_size,
                    TAccDim,
                    TAcc,
                    TStream,
                    TFunctionChain,
                    TSourceOffset,
                    N - 1
                >::call(
                    stream,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    particle_sources,
                    background_color,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    readback_viewport,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
        }
    };

    template<
        typename TParticleList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        ISAAC_IDX_TYPE TTransfer_size,
        typename TAccDim,
        typename TAcc,
        typename TStream,
        typename TFunctionChain,
        int TSourceOffset
    >
    struct ParticleRenderKernelCaller<
        TParticleList,
        TTransferArray,
        TSourceWeight,
        TPointerArray,
        TFilter,
        TTransfer_size,
        TAccDim,
        TAcc,
        TStream,
        TFunctionChain,
        TSourceOffset,
        0 //<-- spezialisation
    >
    {
        inline static void call(
            TStream stream,
            uint32_t *  framebuffer,
            isaac_float3 * depthBuffer,
            isaac_float3 * normalBuffer,
            const isaac_size2 & framebuffer_size,
            const isaac_uint2 & framebuffer_start,
            const TParticleList & particle_sources,
            const isaac_float4 & background_color,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            const TPointerArray & pointerArray,
            IceTInt const * const readback_viewport,
            const isaac_float3 & scale,
            const clipping_struct & clipping,
            const ao_struct & ambientOcclusion
        )
        {
            isaac_size2 block_size = {
                ISAAC_IDX_TYPE( 8 ),
                ISAAC_IDX_TYPE( 16 )
            };
            isaac_size2 grid_size = {
                ISAAC_IDX_TYPE( ( readback_viewport[2] + block_size.x - 1 ) / block_size.x ),
                ISAAC_IDX_TYPE( ( readback_viewport[3] + block_size.y - 1 ) / block_size.y )
            };
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
            if ( mpl::not_<boost::is_same<TAcc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
#endif
            {
                grid_size.x = ISAAC_IDX_TYPE( readback_viewport[2] );
                grid_size.y = ISAAC_IDX_TYPE( readback_viewport[3] );
                block_size.x = ISAAC_IDX_TYPE( 1 );
                block_size.y = ISAAC_IDX_TYPE( 1 );
            }
            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> threads(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> blocks(
                ISAAC_IDX_TYPE( 1 ),
                block_size.y,
                block_size.x
            );
            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> grid(
                ISAAC_IDX_TYPE( 1 ),
                grid_size.y,
                grid_size.x
            );
            auto const workdiv(
                alpaka::WorkDivMembers<
                    TAccDim,
                    ISAAC_IDX_TYPE
                >(
                    grid,
                    blocks,
                    threads
                )
            );
            ParticleRenderKernel
            <
                TParticleList,
                TTransferArray,
                TSourceWeight,
                TPointerArray,
                TFilter,
                TTransfer_size,
                TSourceOffset
            >
            kernel;
            auto const instance
            (
                alpaka::createTaskKernel<TAcc>
                (
                    workdiv,
                    kernel,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    particle_sources,
                    background_color,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    scale,
                    clipping,
                    ambientOcclusion
                )
            );
            alpaka::enqueue(stream, instance);
        }
    };
}