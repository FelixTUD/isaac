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
        isaac_int TInterpolation,
        typename NR,
        typename TSource,
        typename TPointerArray
    >
    ISAAC_HOST_DEVICE_INLINE isaac_float
    get_value(
        const TSource & source,
        const isaac_float3 & pos,
        const TPointerArray & pointerArray,
        const isaac_size3 & local_size,
        const isaac_float3 & scale
    )
    {
        isaac_float_dim <TSource::feature_dim> data;
        isaac_float_dim <TSource::feature_dim> * ptr = (
        isaac_float_dim < TSource::feature_dim > *
        )( pointerArray.pointer[NR::value] );
        if( TInterpolation == 0 )
        {
            isaac_int3 coord = pos;
            if( TSource::persistent )
            {
                data = source[coord];
            }
            else
            {
                data = ptr[coord.x + ISAAC_GUARD_SIZE + ( coord.y + ISAAC_GUARD_SIZE ) 
                            * ( local_size.x + 2 * ISAAC_GUARD_SIZE ) + ( coord.z + ISAAC_GUARD_SIZE ) 
                            * ( ( local_size.x + 2 * ISAAC_GUARD_SIZE ) 
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE ) )];
            }
        }
        else
        {
            isaac_int3 coord;
            isaac_float_dim <TSource::feature_dim> data8[2][2][2];
            for( int x = 0; x < 2; x++ )
            {
                for( int y = 0; y < 2; y++ )
                {
                    for( int z = 0; z < 2; z++ )
                    {
                        coord.x = isaac_int( x ? ceil( pos.x ) : floor( pos.x ) );
                        coord.y = isaac_int( y ? ceil( pos.y ) : floor( pos.y ) );
                        coord.z = isaac_int( z ? ceil( pos.z ) : floor( pos.z ) );
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            if( isaac_uint( coord.x ) >= local_size.x )
                            {
                                coord.x = isaac_int(
                                    x ? floor( pos.x ) : ceil( pos.x )
                                );
                            }
                            if( isaac_uint( coord.y ) >= local_size.y )
                            {
                                coord.y = isaac_int(
                                    y ? floor( pos.y ) : ceil( pos.y )
                                );
                            }
                            if( isaac_uint( coord.z ) >= local_size.z )
                            {
                                coord.z = isaac_int(
                                    z ? floor( pos.z ) : ceil( pos.z )
                                );
                            }
                            
                        }
                        if( TSource::persistent )
                        {
                            data8[x][y][z] = source[coord];
                        }
                        else
                        {
                            data8[x][y][z] = ptr[coord.x + ISAAC_GUARD_SIZE + ( coord.y + ISAAC_GUARD_SIZE ) 
                                                    * ( local_size.x + 2 * ISAAC_GUARD_SIZE ) + ( coord.z + ISAAC_GUARD_SIZE ) 
                                                    * ( ( local_size.x + 2 * ISAAC_GUARD_SIZE ) 
                                                    * ( local_size.y + 2 * ISAAC_GUARD_SIZE ) )];
                        }
                    }
                }
            }
            isaac_float_dim< 3 > pos_in_cube = pos - glm::floor( pos );
            
            isaac_float_dim <TSource::feature_dim> data4[2][2];
            for( int x = 0; x < 2; x++ )
            {
                for( int y = 0; y < 2; y++ )
                {
                    data4[x][y] = data8[x][y][0] * (
                        isaac_float( 1 ) - pos_in_cube.z
                    ) + data8[x][y][1] * (
                        pos_in_cube.z
                    );
                }
            }
            isaac_float_dim <TSource::feature_dim> data2[2];
            for( int x = 0; x < 2; x++ )
            {
                data2[x] = data4[x][0] * (
                    isaac_float( 1 ) - pos_in_cube.y
                ) + data4[x][1] * (
                    pos_in_cube.y
                );
            }
            data = data2[0] * (
                isaac_float( 1 ) - pos_in_cube.x
            ) + data2[1] * (
                pos_in_cube.x
            );
        }
        isaac_float result = isaac_float( 0 );


        result = applyFunctorChain<TSource::feature_dim>(&data, NR::value);

        return result;
    }

    /**
     * @brief Clamps coordinates to min/max
     * 
     * @tparam TInterpolation 
     * @param coord 
     * @param local_size 
     * @return ISAAC_HOST_DEVICE_INLINE check_coord clamped coordiantes
     */
    template<
        bool TInterpolation
    >
    ISAAC_HOST_DEVICE_INLINE void
    check_coord(
        isaac_float3 & coord,
        const isaac_size3 &  local_size
    )
    {
        constexpr ISAAC_IDX_TYPE extra_border = static_cast<ISAAC_IDX_TYPE>(TInterpolation);

        coord = glm::clamp(coord, isaac_float3(0), isaac_float3( local_size - extra_border ) - std::numeric_limits<isaac_float>::min( ) );
    }

    /**
     * @brief Clamps coordinates to min/max +- Guard margin
     * 
     * @tparam TInterpolation 
     * @param coord 
     * @param local_size 
     * @return ISAAC_HOST_DEVICE_INLINE check_coord_with_guard clamped coordinate
     */
    template<
        bool TInterpolation
    >
    ISAAC_HOST_DEVICE_INLINE void
    check_coord_with_guard(
        isaac_float3 & coord,
        const isaac_size3 & local_size
    )
    {
        constexpr ISAAC_IDX_TYPE extra_border = static_cast<ISAAC_IDX_TYPE>(TInterpolation);

        coord = glm::clamp(coord, isaac_float3( -ISAAC_GUARD_SIZE ), 
                            isaac_float3( local_size + ISAAC_IDX_TYPE( ISAAC_GUARD_SIZE ) - extra_border )
                             - std::numeric_limits<isaac_float>::min( ) );
    }

    template<
        ISAAC_IDX_TYPE Ttransfer_size,
        typename TFilter,
        isaac_int TInterpolation,
        isaac_int TIsoSurface
    >
    struct merge_source_iterator
    {
        template<
            typename NR,
            typename TSource,
            typename TTransferArray,
            typename TSourceWeight,
            typename TPointerArray,
            typename TFeedback
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const NR & nr,
            const TSource & source,
            isaac_float4 & color,
            const isaac_float3 & pos,
            const isaac_size3 & local_size,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            const TPointerArray & pointerArray,
            TFeedback & feedback,
            const isaac_float3 & step,
            const isaac_float & stepLength,
            const isaac_float3 & scale,
            const bool & first,
            const isaac_float3 & start_normal
        ) const
        {
            if( boost::mpl::at_c<
                TFilter,
                NR::value
            >::type::value )
            {
                isaac_float result = get_value<
                    TInterpolation,
                    NR
                >(
                    source,
                    pos,
                    pointerArray,
                    local_size,
                    scale
                );
                ISAAC_IDX_TYPE lookup_value = ISAAC_IDX_TYPE(
                    glm::round( result * isaac_float( Ttransfer_size ) )
                );
                lookup_value = glm::clamp( lookup_value, ISAAC_IDX_TYPE( 0 ), Ttransfer_size - 1 );
                isaac_float4 value = transferArray.pointer[NR::value][lookup_value];
                if( TIsoSurface )
                {
                    if( value.w >= isaac_float( 0.5 ) )
                    {
                        isaac_float3 left = {
                            -1,
                            0,
                            0
                        };
                        left = left + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                left,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                left,
                                local_size
                            );
                        }
                        isaac_float3 right = {
                            1,
                            0,
                            0
                        };
                        right = right + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                right,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                right,
                                local_size
                            );
                        }
                        isaac_float d1;
                        if( TInterpolation )
                        {
                            d1 = right.x - left.x;
                        }
                        else
                        {
                            d1 = isaac_int( right.x ) - isaac_int( left.x );
                        }

                        isaac_float3 up = {
                            0,
                            -1,
                            0
                        };
                        up = up + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                up,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                up,
                                local_size
                            );
                        }
                        isaac_float3 down = {
                            0,
                            1,
                            0
                        };
                        down = down + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                down,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                down,
                                local_size
                            );
                        }
                        isaac_float d2;
                        if( TInterpolation )
                        {
                            d2 = down.y - up.y;
                        }
                        else
                        {
                            d2 = isaac_int( down.y ) - isaac_int( up.y );
                        }

                        isaac_float3 front = {
                            0,
                            0,
                            -1
                        };
                        front = front + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                front,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                front,
                                local_size
                            );
                        }
                        isaac_float3 back = {
                            0,
                            0,
                            1
                        };
                        back = back + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                back,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                back,
                                local_size
                            );
                        }
                        isaac_float d3;
                        if( TInterpolation )
                        {
                            d3 = back.z - front.z;
                        }
                        else
                        {
                            d3 = isaac_int( back.z ) - isaac_int( front.z );
                        }

                        isaac_float3 gradient = {
                            (
                                get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    right,
                                    pointerArray,
                                    local_size,
                                    scale
                                ) - get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    left,
                                    pointerArray,
                                    local_size,
                                    scale
                                )
                            ) / d1,
                            (
                                get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    down,
                                    pointerArray,
                                    local_size,
                                    scale
                                ) - get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    up,
                                    pointerArray,
                                    local_size,
                                    scale
                                )
                            ) / d2,
                            (
                                get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    back,
                                    pointerArray,
                                    local_size,
                                    scale
                                ) - get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    front,
                                    pointerArray,
                                    local_size,
                                    scale
                                )
                            ) / d3
                        };
                        if( first )
                        {
                            gradient = start_normal;
                        }
                        //gradient *= scale;
                        isaac_float l = glm::length( gradient );
                        if( l == isaac_float( 0 ) )
                        {
                            color = value;
                        }
                        else
                        {
                            gradient = gradient / l;
                            //gradient.z = -gradient.z;
                            isaac_float3 light = glm::normalize( step );
                            isaac_float ac = fabs( glm::dot( gradient, light ) );
#if ISAAC_SPECULAR == 1
                            color = value * ac + ac * ac * ac * ac;
#else
                            color = value * ac;
#endif
                            //color = glm::vec4( gradient, 1.0f );
                        }
                        color.w = isaac_float( 1 );
                        feedback = 1;
                    }
                }
                else
                {
                    value.w *= sourceWeight.value[NR::value];
                    color.x = color.x + value.x * value.w;
                    color.y = color.y + value.y * value.w;
                    color.z = color.z + value.z * value.w;
                    color.w = color.w + value.w;
                }
            }
        }
    };

    template<
        typename TSourceList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        ISAAC_IDX_TYPE Ttransfer_size,
        isaac_int TInterpolation,
        isaac_int TIsoSurface
    >
    struct IsoRenderKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
            uint32_t * const pixels,                //ptr to output pixels
            isaac_float3 * const gDepth,            //depth buffer
            isaac_float3 * const gNormal,           //normal buffer
            const isaac_size2 framebuffer_size,     //size of framebuffer
            const isaac_uint2 framebuffer_start,    //framebuffer offset
            const TSourceList sources,              //source of volumes
            isaac_float step,                       //ray step length
            const isaac_float4 background_color,    //color of render background
            const TTransferArray transferArray,     //mapping to simulation memory
            const TSourceWeight sourceWeight,       //weights of sources for blending
            const TPointerArray pointerArray,
            const isaac_float3 scale,               //isaac set scaling
            const clipping_struct input_clipping,   //clipping planes
            const ao_struct ambientOcclusion        //ambient occlusion params
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
                sources,
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


            Ray ray = pixelToRay( isaac_float2( pixel ), isaac_float2( framebuffer_size ) );


            //clipping planes with transformed positions
            clipping_struct clipping;
            //set values for clipping planes
            //scale position to global size
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                clipping.elem[i].position = input_clipping.elem[i].position * isaac_float3( isaac_size_d.global_size_scaled ) * isaac_float( 0.5 );
                clipping.elem[i].normal = input_clipping.elem[i].normal;
            }

            //move to local (scaled) grid
            //get offset of subvolume in global volume
            isaac_float3 position_offset = isaac_float3( isaac_int3( isaac_size_d.global_size_scaled ) / 2 - isaac_int3( isaac_size_d.position_scaled ) );

            //apply subvolume offset to start and end
            ray.start = ray.start + position_offset;
            ray.end = ray.end + position_offset;

            //apply subvolume offset to position checked clipping plane
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                clipping.elem[i].position =
                    clipping.elem[i].position + position_offset;
            }

            //clip ray on volume bounding box
            isaac_float3 bb_intersection_min = -ray.start / ray.dir;
            isaac_float3 bb_intersection_max = ( isaac_float3( isaac_size_d.local_size_scaled ) - ray.start ) / ray.dir;

            //bb_intersection_min shall have the smaller values
            ISAAC_SWITCH_IF_SMALLER ( bb_intersection_max.x, bb_intersection_min.x )
            ISAAC_SWITCH_IF_SMALLER ( bb_intersection_max.y, bb_intersection_min.y )
            ISAAC_SWITCH_IF_SMALLER ( bb_intersection_max.z, bb_intersection_min.z )

            isaac_float start_distance = glm::max( bb_intersection_min.x, glm::max( bb_intersection_min.y, bb_intersection_min.z ) );
            isaac_float end_distance = glm::min( bb_intersection_max.x, glm::min( bb_intersection_max.y, bb_intersection_max.z ) );

            bool is_clipped = false;
            isaac_float3 clipping_normal;
            //Iterate over clipping planes and adjust ray
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                isaac_float d = glm::dot( ray.dir, clipping.elem[i].normal);

                isaac_float intersection_depth = ( glm::dot( clipping.elem[i].position, clipping.elem[i].normal )
                                                    - glm::dot( ray.start, clipping.elem[i].normal ) ) / d;
                if( d > 0 )
                {
                    if( end_distance < intersection_depth )
                    {
                        ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebuffer_size.x], color )
                        return;
                    }
                    if( start_distance <= intersection_depth )
                    {
                        clipping_normal = clipping.elem[i].normal;
                        is_clipped = true;
                        start_distance = intersection_depth;
                    }
                }
                else
                {
                    if( start_distance > intersection_depth )
                    {
                        ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebuffer_size.x], color )
                        return;
                    }
                    if( end_distance > intersection_depth )
                    {
                        end_distance = intersection_depth;
                    }
                }
            }
            start_distance = glm::max( start_distance, isaac_float( 0 ) );

            //return if the ray doesn't hit the volume
            if( start_distance > end_distance )
            {
                ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebuffer_size.x], color )

                //this function aborts drawing and therfore wont set any normal or depth values
                //defaults will be applied for clean images
                gNormal[pixel.x + pixel.y * framebuffer_size.x] = default_normal;
                gDepth[pixel.x + pixel.y * framebuffer_size.x] = default_depth;

                return;
            }


            //Starting the main loop
            isaac_float min_size = ISAAC_MIN(
                int(
                    isaac_size_d.global_size.x
                ),
                ISAAC_MIN(
                    int(
                        isaac_size_d.global_size.y
                    ),
                    int(
                        isaac_size_d.global_size.z
                    )
                )
            );
            isaac_float factor = step / min_size * 2.0f;
            isaac_float4 value = isaac_float4(0);
            isaac_int result = 0;
            isaac_float oma;
            isaac_float4 color_add;
            isaac_int start_steps = glm::ceil( start_distance / step );
            isaac_int end_steps = glm::floor( end_distance / step );
            isaac_float3 step_vec = ray.dir * step;
            //unscale all data for correct memory access
            isaac_float3 start_unscaled = ray.start / scale;
            step_vec /= scale;

            //move start_steps and end_steps to valid positions in the volume
            isaac_float3 pos = start_unscaled + step_vec * isaac_float( start_steps );
            while( ( !isInLowerBounds( pos, isaac_float3(0) )
                    || !isInUpperBounds( pos, isaac_size_d.local_size ) )
                    && start_steps <= end_steps)
            {
                start_steps++;
                pos = start_unscaled + step_vec * isaac_float( start_steps );
            }
            pos = start_unscaled + step_vec * isaac_float( end_steps );
            while( ( !isInLowerBounds( pos, isaac_float3(0) )
                    || !isInUpperBounds( pos, isaac_size_d.local_size ) )
                    && start_steps <= end_steps)
            {
                end_steps--;
                pos = start_unscaled + step_vec * isaac_float( end_steps );
            }
            isaac_float depth = std::numeric_limits<isaac_float>::max();
            //iterate over the volume
            for( isaac_int i = start_steps; i <= end_steps; i++ )
            {
                pos = start_unscaled + step_vec * isaac_float( i );
                result = 0;
                bool first = is_clipped && i == start_steps;
                isaac_for_each_with_mpl_params(
                    sources,
                    merge_source_iterator<
                        Ttransfer_size,
                        TFilter,
                        TInterpolation,
                        TIsoSurface
                    >( ),
                    value,
                    pos,
                    isaac_size_d.local_size,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    result,
                    step_vec,
                    step,
                    scale,
                    first,
                    clipping_normal
                );
                if( TIsoSurface )
                {
                    if( result )
                    {
                        depth = i * step;
                        color = value;
                        break;
                    }
                }
                else
                {
                    oma = isaac_float( 1 ) - color.w;
                    value *= factor;
                    color_add = oma * value;
                    color += color_add;
                    if( color.w > isaac_float( 0.99 ) )
                    {
                        break;
                    }
                }
            }
            //indicates how strong particle ao should be when gas is overlapping
            //isaac_float ao_blend = 0.0f;
            //if (!isInLowerBounds(start_unscaled + step_vec * isaac_float(start_steps), isaac_float3(0))
            //    || !isInUpperBounds(start_unscaled + step_vec * isaac_float(end_steps), isaac_float3( isaac_size_d.local_size )))
            //    color = isaac_float4(1, 1, 1, 1);
#if ISAAC_SHOWBORDER == 1
            if ( color.w <= isaac_float ( 0.99 ) ) {
                oma = isaac_float ( 1 ) - color.w;
                color_add.x = 0;
                color_add.y = 0;
                color_add.z = 0;
                color_add.w = oma * factor * isaac_float ( 10 );
                color += color_add;
            }
#endif


            ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebuffer_size.x], color )
            
            //save the particle normal in the normal g buffer
            //gNormal[pixel.x + pixel.y * framebuffer_size.x] = particle_normal;
            
            //save the cell depth in our g buffer (depth)
            //march_length takes the old particle_color w component 
            //the w component stores the particle depth and will be replaced later by new alpha values and 
            //is therefore stored in march_length
            //LINE 2044
            if( TIsoSurface )
            {
                isaac_float3 depth_value = {
                    0.0f,
                    1.0f,
                    depth
                };               
                gDepth[pixel.x + pixel.y * framebuffer_size.x] = depth_value;
            }
        }
    };


    template<
        typename TSourceList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        ISAAC_IDX_TYPE TTransfer_size,
        typename TAccDim,
        typename TAcc,
        typename TStream,
        typename TFunctionChain,
        int N
    >
    struct IsoRenderKernelCaller
    {
        inline static void call(
            TStream stream,
            uint32_t * framebuffer,
            isaac_float3 * depthBuffer,
            isaac_float3 * normalBuffer,
            const isaac_size2 & framebuffer_size,
            const isaac_uint2 & framebuffer_start,
            const TSourceList & sources,
            const isaac_float & step,
            const isaac_float4 & background_color,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            const TPointerArray & pointerArray,
            IceTInt const * const readback_viewport,
            const isaac_int interpolation,
            const isaac_int iso_surface,
            const isaac_float3 & scale,
            const clipping_struct & clipping,
            const ao_struct & ambientOcclusion
        )
        {
            if( sourceWeight.value[boost::mpl::size< TSourceList >::type::value
                                   - N] == isaac_float( 0 ) )
            {
                IsoRenderKernelCaller<
                    TSourceList,
                    TTransferArray,
                    TSourceWeight,
                    TPointerArray,
                    typename boost::mpl::push_back<
                        TFilter,
                        boost::mpl::false_
                    >::type,
                    TTransfer_size,
                    TAccDim,
                    TAcc,
                    TStream,
                    TFunctionChain,
                    N - 1
                >::call(
                    stream,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    sources,
                    step,
                    background_color,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    readback_viewport,
                    interpolation,
                    iso_surface,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
            else
            {
                IsoRenderKernelCaller<
                    TSourceList,
                    TTransferArray,
                    TSourceWeight,
                    TPointerArray,
                    typename boost::mpl::push_back<
                        TFilter,
                        boost::mpl::true_
                    >::type,
                    TTransfer_size,
                    TAccDim,
                    TAcc,
                    TStream,
                    TFunctionChain,
                    N - 1
                >::call(
                    stream,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    sources,
                    step,
                    background_color,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    readback_viewport,
                    interpolation,
                    iso_surface,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
        }
    };

    template<
        typename TSourceList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        ISAAC_IDX_TYPE TTransfer_size,
        typename TAccDim,
        typename TAcc,
        typename TStream,
        typename TFunctionChain
    >
    struct IsoRenderKernelCaller<
        TSourceList,
        TTransferArray,
        TSourceWeight,
        TPointerArray,
        TFilter,
        TTransfer_size,
        TAccDim,
        TAcc,
        TStream,
        TFunctionChain,
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
            const TSourceList & sources,
            const isaac_float & step,
            const isaac_float4 & background_color,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            const TPointerArray & pointerArray,
            IceTInt const * const readback_viewport,
            const isaac_int interpolation,
            const isaac_int iso_surface,
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
            if ( boost::mpl::not_<boost::is_same<TAcc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
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
#define ISAAC_KERNEL_START \
            { \
                IsoRenderKernel \
                < \
                    TSourceList, \
                    TTransferArray, \
                    TSourceWeight, \
                    TPointerArray, \
                    TFilter, \
                    TTransfer_size,
#define ISAAC_KERNEL_END \
                > \
                kernel; \
                auto const instance \
                ( \
                    alpaka::createTaskKernel<TAcc> \
                    ( \
                        workdiv, \
                        kernel, \
                        framebuffer, \
                        depthBuffer, \
                        normalBuffer, \
                        framebuffer_size, \
                        framebuffer_start, \
                        sources, \
                        step, \
                        background_color, \
                        transferArray, \
                        sourceWeight, \
                        pointerArray, \
                        scale, \
                        clipping, \
                        ambientOcclusion \
                    ) \
                ); \
                alpaka::enqueue(stream, instance); \
            }
            if( interpolation )
            {
                if( iso_surface )
                ISAAC_KERNEL_START 1,
                        1 ISAAC_KERNEL_END
                else
                ISAAC_KERNEL_START 1,
                        0 ISAAC_KERNEL_END
            }
            else
            {
                if( iso_surface )
                ISAAC_KERNEL_START 0,
                        1 ISAAC_KERNEL_END
                else
                ISAAC_KERNEL_START 0,
                        0 ISAAC_KERNEL_END
            }
#undef ISAAC_KERNEL_START
#undef ISAAC_KERNEL_END
        }
    };
}