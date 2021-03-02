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


#include "isaac_particle_kernel.hpp"
#include "isaac_ssao_kernel.hpp"
#include "isaac_min_max_kernel.hpp"
#include "isaac_iso_kernel.hpp"
#include "isaac_volume_kernel.hpp"
#include "isaac_texture.hpp"

#include <float.h>


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wsign-compare"

namespace isaac
{
#if 0
    template<
        isaac_int T_interpolation,
        typename T_NR,
        typename T_Source,
        typename T_PointerArray
    >
    ISAAC_HOST_DEVICE_INLINE isaac_float
    get_value(
        const T_Source & source,
        const isaac_float3 & pos,
        const T_PointerArray & pointerArray,
        const isaac_size3 & localSize,
        const isaac_float3 & scale
    )
    {
        isaac_float_dim <T_Source::featureDim> data;
        isaac_float_dim <T_Source::featureDim> * ptr = (
        isaac_float_dim < T_Source::featureDim > *
        )( pointerArray.pointer[T_NR::value] );
        if( T_interpolation == 0 )
        {
            isaac_int3 coord = pos;
            if( T_Source::persistent )
            {
                data = source[coord];
            }
            else
            {
                data = ptr[coord.x + ISAAC_GUARD_SIZE + ( coord.y + ISAAC_GUARD_SIZE ) 
                            * ( localSize.x + 2 * ISAAC_GUARD_SIZE ) + ( coord.z + ISAAC_GUARD_SIZE ) 
                            * ( ( localSize.x + 2 * ISAAC_GUARD_SIZE ) 
                            * ( localSize.y + 2 * ISAAC_GUARD_SIZE ) )];
            }
        }
        else
        {
            isaac_int3 coord;
            isaac_float_dim <T_Source::featureDim> data8[2][2][2];
            for( int x = 0; x < 2; x++ )
            {
                for( int y = 0; y < 2; y++ )
                {
                    for( int z = 0; z < 2; z++ )
                    {
                        coord.x = isaac_int( x ? ceil( pos.x ) : floor( pos.x ) );
                        coord.y = isaac_int( y ? ceil( pos.y ) : floor( pos.y ) );
                        coord.z = isaac_int( z ? ceil( pos.z ) : floor( pos.z ) );
                        if( !T_Source::hasGuard && T_Source::persistent )
                        {
                            if( isaac_uint( coord.x ) >= localSize.x )
                            {
                                coord.x = isaac_int(
                                    x ? floor( pos.x ) : ceil( pos.x )
                                );
                            }
                            if( isaac_uint( coord.y ) >= localSize.y )
                            {
                                coord.y = isaac_int(
                                    y ? floor( pos.y ) : ceil( pos.y )
                                );
                            }
                            if( isaac_uint( coord.z ) >= localSize.z )
                            {
                                coord.z = isaac_int(
                                    z ? floor( pos.z ) : ceil( pos.z )
                                );
                            }
                            
                        }
                        if( T_Source::persistent )
                        {
                            data8[x][y][z] = source[coord];
                        }
                        else
                        {
                            data8[x][y][z] = ptr[coord.x + ISAAC_GUARD_SIZE + ( coord.y + ISAAC_GUARD_SIZE ) 
                                                    * ( localSize.x + 2 * ISAAC_GUARD_SIZE ) + ( coord.z + ISAAC_GUARD_SIZE ) 
                                                    * ( ( localSize.x + 2 * ISAAC_GUARD_SIZE ) 
                                                    * ( localSize.y + 2 * ISAAC_GUARD_SIZE ) )];
                        }
                    }
                }
            }
            isaac_float_dim< 3 > pos_in_cube = pos - glm::floor( pos );
            
            isaac_float_dim <T_Source::featureDim> data4[2][2];
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
            isaac_float_dim <T_Source::featureDim> data2[2];
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


        result = applyFunctorChain(data, T_NR::value);

        return result;
    }

    /**
     * @brief Clamps coordinates to min/max
     * 
     * @tparam T_interpolation 
     * @tparam T_LocalSize 
     * @param coord 
     * @param localSize 
     * @return ISAAC_HOST_DEVICE_INLINE check_coord clamped coordiantes
     */
    template<
        bool T_interpolation
    >
    ISAAC_HOST_DEVICE_INLINE void
    check_coord(
        isaac_float3 & coord,
        const isaac_size3 &  localSize
    )
    {
        constexpr ISAAC_IDX_TYPE extra_border = static_cast<ISAAC_IDX_TYPE>(T_interpolation);

        coord = glm::clamp(coord, isaac_float3(0), isaac_float3( localSize - extra_border ) - std::numeric_limits<isaac_float>::min( ) );
    }

    /**
     * @brief Clamps coordinates to min/max +- Guard margin
     * 
     * @tparam T_interpolation 
     * @tparam T_LocalSize 
     * @param coord 
     * @param localSize 
     * @return ISAAC_HOST_DEVICE_INLINE check_coord_with_guard clamped coordinate
     */
    template<
        bool T_interpolation
    >
    ISAAC_HOST_DEVICE_INLINE void
    check_coord_with_guard(
        isaac_float3 & coord,
        const isaac_size3 & localSize
    )
    {
        constexpr ISAAC_IDX_TYPE extra_border = static_cast<ISAAC_IDX_TYPE>(T_interpolation);

        coord = glm::clamp(coord, isaac_float3( -ISAAC_GUARD_SIZE ), 
                            isaac_float3( localSize + ISAAC_IDX_TYPE( ISAAC_GUARD_SIZE ) - extra_border )
                             - std::numeric_limits<isaac_float>::min( ) );
    }

    template<
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Filter,
        isaac_int T_interpolation,
        isaac_int T_isoSurface
    >
    struct merge_source_iterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_SourceWeight,
            typename T_PointerArray,
            typename T_Feedback
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const T_NR & nr,
            const T_Source & source,
            isaac_float4 & color,
            const isaac_float3 & pos,
            const isaac_size3 & localSize,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            const T_PointerArray & pointerArray,
            T_Feedback & feedback,
            const isaac_float3 & step,
            const isaac_float & stepLength,
            const isaac_float3 & scale,
            const bool & first,
            const isaac_float3 & start_normal
        ) const
        {
            if( boost::mpl::at_c<
                T_Filter,
                T_NR::value
            >::type::value )
            {
                isaac_float result = get_value<
                    T_interpolation,
                    T_NR
                >(
                    source,
                    pos,
                    pointerArray,
                    localSize,
                    scale
                );
                ISAAC_IDX_TYPE lookup_value = ISAAC_IDX_TYPE(
                    glm::round( result * isaac_float( T_transferSize ) )
                );
                lookup_value = glm::clamp( lookup_value, ISAAC_IDX_TYPE( 0 ), T_transferSize - 1 );
                isaac_float4 value = transferArray.pointer[T_NR::value][lookup_value];
                if( T_isoSurface )
                {
                    if( value.w >= isaac_float( 0.5 ) )
                    {
                        isaac_float3 left = {
                            -1,
                            0,
                            0
                        };
                        left = left + pos;
                        if( !T_Source::hasGuard && T_Source::persistent )
                        {
                            check_coord< T_interpolation >(
                                left,
                                localSize
                            );
                        }
                        else
                        {
                            check_coord_with_guard< T_interpolation >(
                                left,
                                localSize
                            );
                        }
                        isaac_float3 right = {
                            1,
                            0,
                            0
                        };
                        right = right + pos;
                        if( !T_Source::hasGuard && T_Source::persistent )
                        {
                            check_coord< T_interpolation >(
                                right,
                                localSize
                            );
                        }
                        else
                        {
                            check_coord_with_guard< T_interpolation >(
                                right,
                                localSize
                            );
                        }
                        isaac_float d1;
                        if( T_interpolation )
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
                        if( !T_Source::hasGuard && T_Source::persistent )
                        {
                            check_coord< T_interpolation >(
                                up,
                                localSize
                            );
                        }
                        else
                        {
                            check_coord_with_guard< T_interpolation >(
                                up,
                                localSize
                            );
                        }
                        isaac_float3 down = {
                            0,
                            1,
                            0
                        };
                        down = down + pos;
                        if( !T_Source::hasGuard && T_Source::persistent )
                        {
                            check_coord< T_interpolation >(
                                down,
                                localSize
                            );
                        }
                        else
                        {
                            check_coord_with_guard< T_interpolation >(
                                down,
                                localSize
                            );
                        }
                        isaac_float d2;
                        if( T_interpolation )
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
                        if( !T_Source::hasGuard && T_Source::persistent )
                        {
                            check_coord< T_interpolation >(
                                front,
                                localSize
                            );
                        }
                        else
                        {
                            check_coord_with_guard< T_interpolation >(
                                front,
                                localSize
                            );
                        }
                        isaac_float3 back = {
                            0,
                            0,
                            1
                        };
                        back = back + pos;
                        if( !T_Source::hasGuard && T_Source::persistent )
                        {
                            check_coord< T_interpolation >(
                                back,
                                localSize
                            );
                        }
                        else
                        {
                            check_coord_with_guard< T_interpolation >(
                                back,
                                localSize
                            );
                        }
                        isaac_float d3;
                        if( T_interpolation )
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
                                    T_interpolation,
                                    T_NR
                                >(
                                    source,
                                    right,
                                    pointerArray,
                                    localSize,
                                    scale
                                ) - get_value<
                                    T_interpolation,
                                    T_NR
                                >(
                                    source,
                                    left,
                                    pointerArray,
                                    localSize,
                                    scale
                                )
                            ) / d1,
                            (
                                get_value<
                                    T_interpolation,
                                    T_NR
                                >(
                                    source,
                                    down,
                                    pointerArray,
                                    localSize,
                                    scale
                                ) - get_value<
                                    T_interpolation,
                                    T_NR
                                >(
                                    source,
                                    up,
                                    pointerArray,
                                    localSize,
                                    scale
                                )
                            ) / d2,
                            (
                                get_value<
                                    T_interpolation,
                                    T_NR
                                >(
                                    source,
                                    back,
                                    pointerArray,
                                    localSize,
                                    scale
                                ) - get_value<
                                    T_interpolation,
                                    T_NR
                                >(
                                    source,
                                    front,
                                    pointerArray,
                                    localSize,
                                    scale
                                )
                            ) / d3
                        };
                        if( first )
                        {
                            gradient = start_normal;
                        }
                        isaac_float l = glm::length( gradient );
                        if( l == isaac_float( 0 ) )
                        {
                            color = value;
                        }
                        else
                        {
                            gradient = gradient / l;
                            isaac_float3 light = step / stepLength;
                            isaac_float ac = fabs( glm::dot( gradient, light ) );
#if ISAAC_SPECULAR == 1
                            color = value * ac + ac * ac * ac * ac;
#else
                            color = value * ac;
#endif
                        }
                        color.w = isaac_float( 1 );
                        feedback = 1;
                    }
                }
                else
                {
                    value.w *= sourceWeight.value[T_NR::value];
                    color.x = color.x + value.x * value.w;
                    color.y = color.y + value.y * value.w;
                    color.z = color.z + value.z * value.w;
                    color.w = color.w + value.w;
                }
            }
        }
    };




    template<
        typename T_ParticleList,
        typename T_SourceList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        isaac_int T_interpolation,
        isaac_int T_isoSurface
    >
    struct isaacRenderKernel
    {
        template<
            typename T_Acc
        >
        ALPAKA_FN_ACC void operator()(
            T_Acc const & acc,
            uint32_t * const pixels,                //ptr to output pixels
            isaac_float3 * const gDepth,            //depth buffer
            isaac_float3 * const gNormal,           //normal buffer
            const isaac_size2 framebuffer_size,     //size of framebuffer
            const isaac_uint2 framebuffer_start,    //framebuffer offset
            const T_ParticleList particleSources,   //source simulation particles
            const T_SourceList sources,              //source of volumes
            isaac_float step,                       //ray step length
            const isaac_float4 backgroundColor,    //color of render background
            const T_TransferArray transferArray,     //mapping to simulation memory
            const T_SourceWeight sourceWeight,       //weights of sources for blending
            const T_PointerArray pointerArray,
            const isaac_float3 scale,                     //isaac set scaling
            const ClippingStruct input_clipping,   //clipping planes
            const AOParams ambientOcclusion        //ambient occlusion params
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
            isaac_float3 default_depth = {1.0, 1.0, 1.0};

            //set background color
            isaac_float4 color = backgroundColor;
            bool at_least_one = true;
            forEachWithMplParams(
                sources,
                CheckNoSourceIterator< T_Filter >( ),
                at_least_one
            );
            if( !at_least_one )
            {
                setColor ( 
                    pixels[pixel.x + pixel.y * framebuffer_size.x], 
                    color 
                );
                gNormal[pixel.x + pixel.y * framebuffer_size.x] = default_normal;
                gDepth[pixel.x + pixel.y * framebuffer_size.x] = default_depth;
                return;
            }


            bool global_front = false;

            //relative pixel position in framebuffer [0.0 ... 1.0]
            //get normalized pixel position in framebuffer
            isaac_float2 pixel_f = isaac_float2( pixel ) / isaac_float2( framebuffer_size ) * isaac_float( 2 ) - isaac_float( 1 );
            
            //ray start position
            isaac_float4 start_p;
            start_p.x = pixel_f.x * ISAAC_Z_NEAR;
            start_p.y = pixel_f.y * ISAAC_Z_NEAR;
            start_p.z = -1.0f * ISAAC_Z_NEAR;
            start_p.w = 1.0f * ISAAC_Z_NEAR;

            //ray end position
            isaac_float4 end_p;
            end_p.x = pixel_f.x * ISAAC_Z_FAR;
            end_p.y = pixel_f.y * ISAAC_Z_FAR;
            end_p.z = 1.0f * ISAAC_Z_FAR;
            end_p.w = 1.0f * ISAAC_Z_FAR;

            //apply inverse modelview transform to ray start/end and get ray start/end as worldspace
            isaac_float3 start = InverseMVPMatrix * start_p;
            isaac_float3 end = InverseMVPMatrix * end_p;

            isaac_float maxSize = SimulationSize.maxGlobalSizeScaled / 2.0f;

            //scale to globale grid size
            start = start * maxSize;
            end = end * maxSize;


            //clipping planes with transformed positions
            ClippingStruct clipping;
            //set values for clipping planes
            //scale position to global size
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                clipping.elem[i].position =
                    input_clipping.elem[i].position * maxSize;
                clipping.elem[i].normal = input_clipping.elem[i].normal;
            }

            //move to local (scaled) grid
            //get offset of subvolume in global volume
            isaac_float3 move_f = isaac_float3( isaac_int3( SimulationSize.globalSizeScaled ) / 2 - isaac_int3( SimulationSize.positionScaled ) );

            //apply subvolume offset to start and end
            start = start + move_f;
            end = end + move_f;

            //apply subvolume offset to position checked clipping plane
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                clipping.elem[i].position =
                    clipping.elem[i].position + move_f;
            }

            //get ray length
            isaac_float3 vec = end - start;
            isaac_float l_scaled = glm::length(vec);

            //apply isaac scaling to start, end and position tested by clipping plane
            start = start / scale;
            end = end / scale;

            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                clipping.elem[i].position = clipping.elem[i].position / scale;
            }

            //get ray length (scaled by isaac scaling)
            vec = end - start;
            isaac_float l = glm::length(vec);

            //get step vector
            isaac_float3 step_vec = vec / l * step;

            //start index for ray
            isaac_float3 count_start = -start / step_vec;

            //get subvolume size as float
            isaac_float3 local_size_f = isaac_float3( SimulationSize.localSize );

            //end index for ray
            isaac_float3 count_end = ( local_size_f - start ) / step_vec;

            //count_start shall have the smaller values
            ISAAC_SWITCH_IF_SMALLER ( count_end.x,
                count_start.x )
            ISAAC_SWITCH_IF_SMALLER ( count_end.y,
                count_start.y )
            ISAAC_SWITCH_IF_SMALLER ( count_end.z,
                count_start.z )

            //calc intersection of all three super planes and save in [count_start.x ; count_end.x]
            isaac_float max_start = ISAAC_MAX(
                ISAAC_MAX(
                    count_start.x,
                    count_start.y
                ),
                count_start.z
            );

            isaac_float3 start_normal;
            if( ceil( count_start.x ) == ceil( max_start ) )
            {
                if( step_vec.x > 0.0f )
                {
                    if( SimulationSize.position.x == 0 )
                    {
                        global_front = true;
                        start_normal = {
                            1.0f,
                            0,
                            0
                        };
                    }
                }
                else
                {
                    if( SimulationSize.position.x == SimulationSize.globalSize.x - SimulationSize.localSize.x )
                    {
                        global_front = true;
                        start_normal = {
                            -1.0f,
                            0,
                            0
                        };
                    }
                }
            }
            if( ceil( count_start.y ) == ceil( max_start ) )
            {
                if( step_vec.y > 0.0f )
                {
                    if( SimulationSize.position.y == 0 )
                    {
                        global_front = true;
                        start_normal = {
                            0,
                            1.0f,
                            0
                        };
                    }
                }
                else
                {
                    if( SimulationSize.position.y == SimulationSize.globalSize.y - SimulationSize.localSize.y )
                    {
                        global_front = true;
                        start_normal = {
                            0,
                            -1.0f,
                            0
                        };
                    }
                }
            }
            if( ceil( count_start.z ) == ceil( max_start ) )
            {
                if( step_vec.z > 0.0f )
                {
                    if( SimulationSize.position.z == 0 )
                    {
                        global_front = true;
                        start_normal = {
                            0,
                            0,
                            1.0f
                        };
                    }
                }
                else
                {
                    if( SimulationSize.position.z == SimulationSize.globalSize.z - SimulationSize.localSize.z )
                    {
                        global_front = true;
                        start_normal = {
                            0,
                            0,
                            -1.0f
                        };
                    }
                }
            }
            count_start.x = max_start;
            count_end.x = ISAAC_MIN(
                ISAAC_MIN(
                    count_end.x,
                    count_end.y
                ),
                count_end.z
            );
            if( count_start.x > count_end.x )
            {
                //TODO understand the superplanes stuff...
                setColor ( pixels[pixel.x + pixel.y * framebuffer_size.x], color );

                //this function aborts drawing and therfore wont set any normal or depth values
                //defaults will be applied for clean images
                gNormal[pixel.x + pixel.y * framebuffer_size.x] = default_normal;
                gDepth[pixel.x + pixel.y * framebuffer_size.x] = default_depth;

                return;
            }

            //set start and end index of ray
            isaac_int first = isaac_int( ceil( count_start.x ) );
            isaac_int last = isaac_int( floor( count_end.x ) );

            isaac_float first_f = count_start.x;
            isaac_float last_f = count_end.x;

            //Moving last and first until their points are valid
            isaac_float3 pos = start + step_vec * isaac_float( last );
            isaac_int3 coord = isaac_int3( glm::floor( pos ) );
            while( (
                ISAAC_FOR_EACH_DIM_TWICE ( 3,
                    coord,
                    >= SimulationSize.localSize,
                    || )
                ISAAC_FOR_EACH_DIM ( 3,
                    coord,
                    < 0 || ) 0 )
                && first <= last )
            {
                last--;
                pos = start + step_vec * isaac_float( last );
                coord = isaac_int3( glm::floor( pos ) );
            }
            pos = start + step_vec * isaac_float( first );
            coord = isaac_int3( glm::floor( pos ) );
            while( (
                ISAAC_FOR_EACH_DIM_TWICE ( 3,
                    coord,
                    >= SimulationSize.localSize,
                    || )
                ISAAC_FOR_EACH_DIM ( 3,
                    coord,
                    < 0 || ) 0 )
                && first <= last )
            {
                first++;
                pos = start + step_vec * isaac_float( first );
                coord = isaac_int3( glm::floor( pos ) );
            }
            first = ISAAC_MAX(
                first,
                0
            );
            first_f = ISAAC_MAX(
                first_f,
                0.0f
            );

            bool is_clipped = false;
            isaac_float3 clipping_normal;
            //Extra clipping
            for( isaac_int i = 0; i < input_clipping.count; i++ )
            {
                isaac_float d = glm::dot( step_vec, clipping.elem[i].normal);

                isaac_float intersection_step = ( glm::dot( clipping.elem[i].position, clipping.elem[i].normal )
                                                    - glm::dot( start, clipping.elem[i].normal ) ) / d;
                if( d > 0 )
                {
                    if( last_f < intersection_step )
                    {
                        setColor ( pixels[pixel.x + pixel.y * framebuffer_size.x], color );
                        return;
                    }
                    if( first_f <= intersection_step )
                    {
                        first = ceil( intersection_step );
                        first_f = intersection_step;
                        clipping_normal = clipping.elem[i].normal;
                        is_clipped = true;
                        global_front = true;
                        start_normal = clipping.elem[i].normal;
                    }
                }
                else
                {
                    if( first_f > intersection_step )
                    {
                        setColor ( pixels[pixel.x + pixel.y * framebuffer_size.x], color );
                        return;
                    }
                    if( last_f > intersection_step )
                    {
                        last = floor( intersection_step );
                        last_f = intersection_step;
                    }
                }
            }


            // set distance check in alpha channel on scaled max distance
            isaac_float4 particle_color;
            isaac_float depth = ( last_f - first_f ) * step * l_scaled / l;
            isaac_float3 local_start = ( start + step_vec * first_f ) * scale;
            bool particle_hit = false;
            isaac_float3 normalized_dir = glm::normalize( step_vec * scale / step );
            // light direction is camera direction
            isaac_float3 light_dir = -normalized_dir;

            /* RAYMARCH */

            // get the signs of the direction for the raymarch
            isaac_int3 dir_sign = glm::sign( normalized_dir );

            //TODO: alternative for constant 0.001f
            // calculate current position in scaled object space
            isaac_float3 current_pos = ( start + step_vec * ISAAC_MAX( first_f, 0.0f ) ) * scale;

            isaac_float3 particle_scale = isaac_float3( SimulationSize.localSizeScaled ) / isaac_float3( SimulationSize.localParticleSize );
            // calculate current local cell coordinates
            isaac_uint3 current_cell = isaac_uint3( glm::clamp( 
                                    isaac_int3( current_pos / particle_scale ), 
                                    isaac_int3( 0 ), 
                                    isaac_int3( SimulationSize.localParticleSize - ISAAC_IDX_TYPE( 1 ) ) 
                                ) );

            isaac_float ray_length = ( last_f - first_f ) * step * l_scaled / l;
            isaac_float march_length = 0;


            // calculate next intersection with each dimension
            isaac_float3 t = ( ( isaac_float3( current_cell ) + isaac_float3( glm::max( dir_sign, 0 ) ) ) 
                    * particle_scale - current_pos ) / normalized_dir;

            // calculate delta length to next intersection in the same dimension
            isaac_float3 delta_t = particle_scale / normalized_dir * isaac_float3( dir_sign );

            isaac_float3 particle_hitposition(0);

            // check for 0 to stop infinite looping
            if( normalized_dir.x == 0 )
            {
                t.x = std::numeric_limits<isaac_float>::max( );
            }
            if( normalized_dir.y == 0 )
            {
                t.y = std::numeric_limits<isaac_float>::max( );
            }
            if( normalized_dir.z == 0 )
            {
                t.z = std::numeric_limits<isaac_float>::max( );
            }


            //normal at particle hit position
            isaac_float3 particle_normal;
            // check if the ray leaves the local volume, has a particle hit or exceeds the max ray distance
            while( current_cell.x < SimulationSize.localParticleSize.x 
                && current_cell.y < SimulationSize.localParticleSize.y 
                && current_cell.z < SimulationSize.localParticleSize.z 
                && particle_hit == false
                && march_length <= ray_length )
            {

                // calculate particle intersections for each particle source
                forEachWithMplParams(
                    particleSources,
                    merge_particle_iterator<
                        T_transferSize,
                        boost::mpl::size< T_SourceList >::type::value,
                        T_Filter
                    >( ),
                    local_start,
                    normalized_dir,
                    light_dir,
                    current_cell,
                    transferArray,
                    sourceWeight,
                    particle_scale,
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
                    march_length = t.x;
                    t.x += delta_t.x;
                }
                else if( t.y < t.x && t.y < t.z )
                {
                    current_cell.y += dir_sign.y;
                    march_length = t.y;
                    t.y += delta_t.y;
                }
                else
                {
                    current_cell.z += dir_sign.z;
                    march_length = t.z;
                    t.z += delta_t.z;
                }

            }
            // if there was a hit set maximum volume raycast distance to particle hit distance and set particle color
            if( particle_hit )
            {
                last = ISAAC_MIN(
                    last,
                    int(
                        ceil(
                            first_f + depth
                                            / ( step * l_scaled / l )
                        )
                    )
                );

                // calculate lighting properties for the last hit particle
                particle_normal = glm::normalize( particle_normal );

                isaac_float light_factor = glm::dot( particle_normal, light_dir );

                isaac_float3 half_vector = glm::normalize( -normalized_dir + light_dir );

                isaac_float specular = glm::dot( particle_normal, half_vector );

                specular = pow( specular, 10 );
                specular *= 0.5f;
                light_factor = light_factor * 0.5f + 0.5f;


                particle_color = glm::min( particle_color * light_factor + specular, isaac_float( 1 ) );
            }


            //Starting the main loop
            isaac_float min_size = ISAAC_MIN(
                int(
                    SimulationSize.globalSize.x
                ),
                ISAAC_MIN(
                    int(
                        SimulationSize.globalSize.y
                    ),
                    int(
                        SimulationSize.globalSize.z
                    )
                )
            );
            isaac_float factor = step / min_size * 2.0f;
            isaac_float4 value = isaac_float4(0);
            isaac_int result = 0;
            isaac_float oma;
            isaac_float4 color_add;
            for( isaac_int i = first; i <= last; i++ )
            {
                pos = start + step_vec * isaac_float( i );
                value = isaac_float4( 0 );
                result = 0;
                bool firstRound = ( global_front && i == first );
                forEachWithMplParams(
                    sources,
                    merge_source_iterator<
                        T_transferSize,
                        T_Filter,
                        T_interpolation,
                        T_isoSurface
                    >( ),
                    value,
                    pos,
                    SimulationSize.localSize,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    result,
                    step_vec,
                    step,
                    scale,
                    firstRound,
                    start_normal
                );
                if( T_isoSurface )
                {
                    if( result )
                    {
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
            isaac_float ao_blend = 0.0f;
            if( particle_hit && !result )
            {
                ao_blend = (1 - color.w);

                particle_color.w = 1;
                
                color = color + particle_color * ( 1 - color.w );
                
            }

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


            setColor ( pixels[pixel.x + pixel.y * framebuffer_size.x], color );
            //save the particle normal in the normal g buffer
            gNormal[pixel.x + pixel.y * framebuffer_size.x] = particle_normal;
            
            //save the cell depth in our g buffer (depth)
            //march_length takes the old particle_color w component 
            //the w component stores the particle depth and will be replaced later by new alpha values and 
            //is therefore stored in march_length
            //LINE 2044
            isaac_float3 depth_value = {
                0.0f,
                ao_blend,
                depth
            };                

            gDepth[pixel.x + pixel.y * framebuffer_size.x] = depth_value;
        }
    };

    template<
        typename T_ParticleList,
        typename T_SourceList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename TAccDim,
        typename T_Acc,
        typename T_Stream,
        typename T_FunctionChain,
        int N
    >
    struct IsaacRenderKernelCaller
    {
        inline static void call(
            T_Stream stream,
            uint32_t * framebuffer,
            isaac_float3 * depthBuffer,
            isaac_float3 * normalBuffer,
            const isaac_size2 & framebuffer_size,
            const isaac_uint2 & framebuffer_start,
            const T_ParticleList & particleSources,
            const T_SourceList & sources,
            const isaac_float & step,
            const isaac_float4 & backgroundColor,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            const T_PointerArray & pointerArray,
            IceTInt const * const readback_viewport,
            const isaac_int interpolation,
            const isaac_int isoSurface,
            const isaac_float3 & scale,
            const ClippingStruct & clipping,
            const AOParams & ambientOcclusion
        )
        {
            if( sourceWeight.value[boost::mpl::size< T_SourceList >::type::value
                                   + boost::mpl::size< T_ParticleList >::type::value
                                   - N] == isaac_float( 0 ) )
            {
                IsaacRenderKernelCaller<
                    T_ParticleList,
                    T_SourceList,
                    T_TransferArray,
                    T_SourceWeight,
                    T_PointerArray,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::false_
                    >::type,
                    T_transferSize,
                    TAccDim,
                    T_Acc,
                    T_Stream,
                    T_FunctionChain,
                    N - 1
                >::call(
                    stream,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    particleSources,
                    sources,
                    step,
                    backgroundColor,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    readback_viewport,
                    interpolation,
                    isoSurface,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
            else
            {
                IsaacRenderKernelCaller<
                    T_ParticleList,
                    T_SourceList,
                    T_TransferArray,
                    T_SourceWeight,
                    T_PointerArray,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::true_
                    >::type,
                    T_transferSize,
                    TAccDim,
                    T_Acc,
                    T_Stream,
                    T_FunctionChain,
                    N - 1
                >::call(
                    stream,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    particleSources,
                    sources,
                    step,
                    backgroundColor,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    readback_viewport,
                    interpolation,
                    isoSurface,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
        }
    };

    template<
        typename T_ParticleList,
        typename T_SourceList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename TAccDim,
        typename T_Acc,
        typename T_Stream,
        typename T_FunctionChain
    >
    struct IsaacRenderKernelCaller<
        T_ParticleList,
        T_SourceList,
        T_TransferArray,
        T_SourceWeight,
        T_PointerArray,
        T_Filter,
        T_transferSize,
        TAccDim,
        T_Acc,
        T_Stream,
        T_FunctionChain,
        0 //<-- spezialisation
    >
    {
        inline static void call(
            T_Stream stream,
            uint32_t *  framebuffer,
            isaac_float3 * depthBuffer,
            isaac_float3 * normalBuffer,
            const isaac_size2 & framebuffer_size,
            const isaac_uint2 & framebuffer_start,
            const T_ParticleList & particleSources,
            const T_SourceList & sources,
            const isaac_float & step,
            const isaac_float4 & backgroundColor,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            const T_PointerArray & pointerArray,
            IceTInt const * const readback_viewport,
            const isaac_int interpolation,
            const isaac_int isoSurface,
            const isaac_float3 & scale,
            const ClippingStruct & clipping,
            const AOParams & ambientOcclusion
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
            if ( boost::mpl::not_<boost::is_same<T_Acc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
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
                isaacRenderKernel \
                < \
                    T_ParticleList, \
                    T_SourceList, \
                    T_TransferArray, \
                    T_SourceWeight, \
                    T_PointerArray, \
                    T_Filter, \
                    T_transferSize,
#define ISAAC_KERNEL_END \
                > \
                kernel; \
                auto const instance \
                ( \
                    alpaka::createTaskKernel<T_Acc> \
                    ( \
                        workdiv, \
                        kernel, \
                        framebuffer, \
                        depthBuffer, \
                        normalBuffer, \
                        framebuffer_size, \
                        framebuffer_start, \
                        particleSources, \
                        sources, \
                        step, \
                        backgroundColor, \
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
                if( isoSurface )
                ISAAC_KERNEL_START 1,
                        1 ISAAC_KERNEL_END
                else
                ISAAC_KERNEL_START 1,
                        0 ISAAC_KERNEL_END
            }
            else
            {
                if( isoSurface )
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
#endif

} //namespace isaac;

#pragma GCC diagnostic pop
