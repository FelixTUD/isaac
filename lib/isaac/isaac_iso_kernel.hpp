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
#include "isaac_volume_kernel.hpp"


namespace isaac
{
#if 0
    template<
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Filter,
        isaac_int T_interpolation,
        isaac_int T_isoSurface
    >
    struct MergeIsoSourceIterator
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
            const isaac_float3 & stepSize,
            const isaac_float & stepLength,
            const isaac_float3 & scale,
            const bool & first,
            const isaac_float3 & clippingNormal
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
                        isaac_float l = glm::length( gradient );
                        if( l == isaac_float( 0 ) )
                        {
                            color = value;
                        }
                        else
                        {
                            gradient = gradient / l;
                            //gradient.z = -gradient.z;
                            isaac_float3 light = glm::normalize( stepSize );
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
    struct IsoRenderKernel
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
            //iterate over the volume
            for( isaac_int i = startSteps; i <= endSteps; i++ )
            {
                pos = startUnscaled + stepVec * isaac_float( i );
                result = 0;
                bool first = ray.isClipped && i == startSteps;
                forEachWithMplParams(
                    sources,
                    MergeIsoSourceIterator<
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
                    stepVec,
                    stepSize,
                    scale,
                    first,
                    ray.clippingNormal
                );
                if( T_isoSurface )
                {
                    if( result )
                    {
                        depth = i * stepSize;
                        color = value;
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
            //color = isaac_float4(ray.endDepth / 100.0f);
            //color.w = 1;
            //setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], color );
            //return;
            if( !T_isoSurface )
            {
                uint32_t colorValue = gBuffer.color[pixel.x + pixel.y * gBuffer.size.x];
                isaac_float4 solidColor = {
                    ((colorValue >>  0) & 0xff) / 255.0f,
                    ((colorValue >>  8) & 0xff) / 255.0f,
                    ((colorValue >> 16) & 0xff) / 255.0f,
                    ((colorValue >> 24) & 0xff) / 255.0f
                };
                color = color.w * color + ( 1 - color.w ) * solidColor;
                setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], color );
            }
            else if( result )
            {   
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = depth;
                setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], color );
            }
        }
    };
#endif

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
    struct IsoRenderKernelCaller
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
                IsoRenderKernelCaller<
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
                IsoRenderKernelCaller<
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
    struct IsoRenderKernelCaller<
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
                if( isoSurface )
                ISAAC_KERNEL_START 1,
                        1 ISAAC_KERNEL_END
            }
            else
            {
                if( isoSurface )
                ISAAC_KERNEL_START 0,
                        1 ISAAC_KERNEL_END
            }
#undef ISAAC_KERNEL_START
#undef ISAAC_KERNEL_END
        }
    };
}