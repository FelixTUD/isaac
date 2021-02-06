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
        isaac_int T_interpolation,
        typename T_NR,
        typename T_Source,
        typename T_PointerArray
    >
    ISAAC_HOST_DEVICE_INLINE isaac_float
    getValue(
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
            isaac_float3 posInCube = pos - glm::floor( pos );
            
            isaac_float_dim <T_Source::featureDim> data4[2][2];
            for( int x = 0; x < 2; x++ )
            {
                for( int y = 0; y < 2; y++ )
                {
                    data4[x][y] = data8[x][y][0] * (
                        isaac_float( 1 ) - posInCube.z
                    ) + data8[x][y][1] * (
                        posInCube.z
                    );
                }
            }
            isaac_float_dim <T_Source::featureDim> data2[2];
            for( int x = 0; x < 2; x++ )
            {
                data2[x] = data4[x][0] * (
                    isaac_float( 1 ) - posInCube.y
                ) + data4[x][1] * (
                    posInCube.y
                );
            }
            data = data2[0] * (
                isaac_float( 1 ) - posInCube.x
            ) + data2[1] * (
                posInCube.x
            );
        }
        isaac_float result = isaac_float( 0 );


        result = applyFunctorChain<T_Source::featureDim>(&data, T_NR::value);

        return result;
    }

    /**
     * @brief Clamps coordinates to min/max
     * 
     * @tparam T_interpolation 
     * @param coord 
     * @param localSize 
     * @return ISAAC_HOST_DEVICE_INLINE check_coord clamped coordiantes
     */
    template<
        typename T_Source
    >
    ISAAC_HOST_DEVICE_INLINE void
    checkCoord(
        isaac_float3 & coord,
        const isaac_size3 &  localSize
    )
    {
        if( T_Source::hasGuard || !T_Source::persistent )
        {
            coord = glm::clamp(coord, isaac_float3( -ISAAC_GUARD_SIZE ), 
                    isaac_float3( localSize + ISAAC_IDX_TYPE( ISAAC_GUARD_SIZE - 1 ) )
                        - std::numeric_limits<isaac_float>::min( ) );
        }
        else
        {
            coord = glm::clamp(coord, isaac_float3(0), isaac_float3( localSize - ISAAC_IDX_TYPE( 1 ) ) );
        }
    }

    template<
        isaac_int T_interpolation,
        isaac_int T_index,
        typename T_NR,
        typename T_Source,
        typename T_PointerArray
    >
    ISAAC_HOST_DEVICE_INLINE isaac_float
    getCompGradient(
        const T_Source & source,
        const isaac_float3 & pos,
        const T_PointerArray & pointerArray,
        const isaac_size3 &  localSize,
        const isaac_float3 & scale
    )
    {
        isaac_float3 front = { 0, 0, 0 };
        front[T_index] = -1;
        front = front + pos;
        checkCoord< 
            T_Source
        >(
            front,
            localSize
        );

        isaac_float3 back = { 0, 0, 0 };
        back[T_index] = 1;
        back = back + pos;
        checkCoord< 
            T_Source
        >(
            back,
            localSize
        );

        isaac_float d;
        if( T_interpolation )
        {
            d = back[T_index] - front[T_index];
        }
        else
        {
            d = isaac_int( back[T_index] ) - isaac_int( front[T_index] );
        }

        return (
            getValue<
                T_interpolation,
                T_NR
            >(
                source,
                back,
                pointerArray,
                localSize,
                scale
            ) - getValue<
                T_interpolation,
                T_NR
            >(
                source,
                front,
                pointerArray,
                localSize,
                scale
            )
        ) / d;
    }

    template<
        isaac_int T_interpolation,
        typename T_NR,
        typename T_Source,
        typename T_PointerArray
    >
    ISAAC_HOST_DEVICE_INLINE isaac_float3
    getGradient(
        const T_Source & source,
        const isaac_float3 & pos,
        const T_PointerArray & pointerArray,
        const isaac_size3 &  localSize,
        const isaac_float3 & scale
    )
    {

        isaac_float3 gradient = {
            getCompGradient<
                T_interpolation,
                0,
                T_NR
            >(
                source,
                pos,
                pointerArray,
                localSize,
                scale
            ),
            getCompGradient<
                T_interpolation,
                1,
                T_NR
            >(
                source,
                pos,
                pointerArray,
                localSize,
                scale
            ),
            getCompGradient<
                T_interpolation,
                2,
                T_NR
            >(
                source,
                pos,
                pointerArray,
                localSize,
                scale
            )
        };
        return gradient;
    }

    template<
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Filter,
        isaac_int T_interpolation,
        isaac_int T_isoSurface
    >
    struct MergeVolumeSourceIterator
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
            const isaac_float3 & pos,
            const isaac_size3 & localSize,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            const T_PointerArray & pointerArray,
            const isaac_float3 & stepSize,
            const isaac_float & stepLength,
            const isaac_float3 & scale,
            const bool & first,
            const isaac_float3 & clippingNormal,
            T_Feedback & feedback,
            isaac_float4 & color,
            isaac_float3 & normal
        ) const
        {
            if( boost::mpl::at_c<
                T_Filter,
                T_NR::value
            >::type::value )
            {
                isaac_float result = getValue<
                    T_interpolation,
                    T_NR
                >(
                    source,
                    pos,
                    pointerArray,
                    localSize,
                    scale
                );
                ISAAC_IDX_TYPE lookupValue = ISAAC_IDX_TYPE(
                    glm::round( result * isaac_float( T_transferSize ) )
                );
                lookupValue = glm::clamp( lookupValue, ISAAC_IDX_TYPE( 0 ), T_transferSize - 1 );
                isaac_float4 value = transferArray.pointer[T_NR::value][lookupValue];
                if( T_isoSurface )
                {
                    if( value.w >= isaac_float( 0.5 ) )
                    {
                        isaac_float3 gradient = getGradient<
                            T_interpolation,
                            T_NR
                        >(
                            source,
                            pos,
                            pointerArray,
                            localSize,
                            scale
                        );

                        if( first )
                        {
                            gradient = clippingNormal;
                        }
                        //gradient *= scale;
                        normal = glm::normalize(-gradient);
                        color = value;
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
        typename T_SourceList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        isaac_int T_interpolation,
        isaac_int T_isoSurface
    >
    struct VolumeRenderKernel
    {
        template<
            typename T_Acc
        >
        ALPAKA_FN_ACC void operator()(
            T_Acc const & acc,
            GBuffer gBuffer,
            const T_SourceList sources,              //source of volumes
            isaac_float stepSize,                       //ray stepSize length
            const T_TransferArray transferArray,     //mapping to simulation memory
            const T_SourceWeight sourceWeight,       //weights of sources for blending
            const T_PointerArray pointerArray,
            const isaac_float3 scale,               //isaac set scaling
            const ClippingStruct inputClipping     //clipping planes
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
            pixel = pixel + gBuffer.startOffset;
            if( !isInUpperBounds( pixel, gBuffer.size ) )
                return;

            //set background color
            bool atLeastOne = true;
            forEachWithMplParams(
                sources,
                CheckNoSourceIterator< T_Filter >( ),
                atLeastOne
            );
            if( !atLeastOne )
                return;

            Ray ray = pixelToRay( isaac_float2( pixel ), isaac_float2( gBuffer.size ) );

            if( !clipRay(ray, inputClipping ) )
                return;

            ray.endDepth = glm::min(ray.endDepth, gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x]);

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
            isaac_float factor = stepSize / min_size * 2.0f;
            isaac_float4 value = isaac_float4(0);
            isaac_int result = 0;
            isaac_float oma;
            isaac_float4 colorAdd;
            isaac_int startSteps = glm::ceil( ray.startDepth / stepSize );
            isaac_int endSteps = glm::floor( ray.endDepth / stepSize );
            isaac_float3 stepVec =  stepSize * ray.dir / scale;
            //unscale all data for correct memory access
            isaac_float3 startUnscaled = ray.start / scale;

            //move startSteps and endSteps to valid positions in the volume
            isaac_float3 pos = startUnscaled + stepVec * isaac_float( startSteps );
            while( ( !isInLowerBounds( pos, isaac_float3(0) )
                    || !isInUpperBounds( pos, SimulationSize.localSize ) )
                    && startSteps <= endSteps)
            {
                startSteps++;
                pos = startUnscaled + stepVec * isaac_float( startSteps );
            }
            pos = startUnscaled + stepVec * isaac_float( endSteps );
            while( ( !isInLowerBounds( pos, isaac_float3(0) )
                    || !isInUpperBounds( pos, SimulationSize.localSize ) )
                    && startSteps <= endSteps)
            {
                endSteps--;
                pos = startUnscaled + stepVec * isaac_float( endSteps );
            }
            isaac_float depth = 0;
            isaac_float4 color = isaac_float4( 0 );
            isaac_float3 normal;
            //iterate over the volume
            for( isaac_int i = startSteps; i <= endSteps; i++ )
            {
                pos = startUnscaled + stepVec * isaac_float( i );
                result = 0;
                bool first = ray.isClipped && i == startSteps;
                forEachWithMplParams(
                    sources,
                    MergeVolumeSourceIterator<
                        T_transferSize,
                        T_Filter,
                        T_interpolation,
                        T_isoSurface
                    >( ),
                    pos,
                    SimulationSize.localSize,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    stepVec,
                    stepSize,
                    scale,
                    first,
                    ray.clippingNormal,
                    result,
                    value,
                    normal
                );
                if( T_isoSurface )
                {
                    if( result )
                    {
                        depth = i * stepSize;
                        color = value;
                        normal = normal;
                        break;
                    }
                }
                else
                {
                    oma = isaac_float( 1 ) - color.w;
                    value *= factor;
                    colorAdd = oma * value;
                    color += colorAdd;
                    if( color.w > isaac_float( 0.99 ) )
                    {
                        break;
                    }
                }
            }

            //indicates how strong particle ao should be when gas is overlapping
            //isaac_float ao_blend = 0.0f;
            //if (!isInLowerBounds(startUnscaled + stepVec * isaac_float(startSteps), isaac_float3(0))
            //    || !isInUpperBounds(startUnscaled + stepVec * isaac_float(endSteps), isaac_float3( SimulationSize.localSize )))
            //    color = isaac_float4(1, 1, 1, 1);
#if ISAAC_SHOWBORDER == 1
            if ( color.w <= isaac_float ( 0.99 ) ) {
                oma = isaac_float ( 1 ) - color.w;
                colorAdd.x = 0;
                colorAdd.y = 0;
                colorAdd.z = 0;
                colorAdd.w = oma * factor * isaac_float ( 10 );
                color += colorAdd;
            }
#endif


            
            //save the particle normal in the normal g buffer
            //gBuffer.normal[pixel.x + pixel.y * gBuffer.size.x] = particle_normal;
            
            //save the cell depth in our g buffer (depth)
            //march_length takes the old particle_color w component 
            //the w component stores the particle depth and will be replaced later by new alpha values and 
            //is therefore stored in march_length
            //LINE 2044
            //color = isaac_float4(endSteps / 1000.0f);
            //color.w = 1;
            //setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], color );
            //return;
            if( !T_isoSurface )
            {
                isaac_float4 solidColor = getColor( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x] );
                color = color + ( 1 - color.w ) * solidColor;
                setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], color );
            }
            else if( result )
            {   
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = depth;
                gBuffer.normal[pixel.x + pixel.y * gBuffer.size.x] = normal;
                setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], color );
            }
        }
    };


    template<
        typename T_SourceList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream,
        int T_n
    >
    struct VolumeRenderKernelCaller
    {
        inline static void call(
            T_Stream stream,
            const GBuffer & gBuffer,
            const T_SourceList & sources,
            const isaac_float & stepSize,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            const T_PointerArray & pointerArray,
            const T_WorkDiv & workdiv,
            const isaac_int interpolation,
            const isaac_int isoSurface,
            const isaac_float3 & scale,
            const ClippingStruct & clipping
        )
        {
            if( sourceWeight.value[boost::mpl::size< T_SourceList >::type::value
                                   - T_n] == isaac_float( 0 ) )
            {
                VolumeRenderKernelCaller<
                    T_SourceList,
                    T_TransferArray,
                    T_SourceWeight,
                    T_PointerArray,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::false_
                    >::type,
                    T_transferSize,
                    T_WorkDiv,
                    T_Acc,
                    T_Stream,
                    T_n - 1
                >::call(
                    stream,
                    gBuffer,
                    sources,
                    stepSize,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    workdiv,
                    interpolation,
                    isoSurface,
                    scale,
                    clipping
                );
            }
            else
            {
                VolumeRenderKernelCaller<
                    T_SourceList,
                    T_TransferArray,
                    T_SourceWeight,
                    T_PointerArray,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::true_
                    >::type,
                    T_transferSize,
                    T_WorkDiv,
                    T_Acc,
                    T_Stream,
                    T_n - 1
                >::call(
                    stream,
                    gBuffer,
                    sources,
                    stepSize,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    workdiv,
                    interpolation,
                    isoSurface,
                    scale,
                    clipping
                );
            }
        }
    };

    template<
        typename T_SourceList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream
    >
    struct VolumeRenderKernelCaller<
        T_SourceList,
        T_TransferArray,
        T_SourceWeight,
        T_PointerArray,
        T_Filter,
        T_transferSize,
        T_WorkDiv,
        T_Acc,
        T_Stream,
        0 //<-- spezialisation
    >
    {
        inline static void call(
            T_Stream stream,
            const GBuffer & gBuffer,
            const T_SourceList & sources,
            const isaac_float & stepSize,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            const T_PointerArray & pointerArray,
            const T_WorkDiv & workdiv,
            const isaac_int interpolation,
            const isaac_int isoSurface,
            const isaac_float3 & scale,
            const ClippingStruct & clipping
        )
        {

#define ISAAC_KERNEL_START \
            { \
                VolumeRenderKernel \
                < \
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
                        gBuffer, \
                        sources, \
                        stepSize, \
                        transferArray, \
                        sourceWeight, \
                        pointerArray, \
                        scale, \
                        clipping \
                    ) \
                ); \
                alpaka::enqueue(stream, instance); \
            }
            if( interpolation )
            {
                ISAAC_KERNEL_START 1,
                        0 ISAAC_KERNEL_END
            }
            else
            {
                ISAAC_KERNEL_START 0,
                        0 ISAAC_KERNEL_END
            }
#undef ISAAC_KERNEL_START
#undef ISAAC_KERNEL_END
        }
    };
}