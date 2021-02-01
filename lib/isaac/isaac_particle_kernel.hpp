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
     * @tparam T_transferSize 
     * @tparam T_offset 
     * @tparam T_Filter 
     */
    template<
        ISAAC_IDX_TYPE T_transferSize,
        int T_offset,
        typename T_Filter
    >
    struct MergeParticleSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_SourceWeight
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const T_NR & nr,
            const T_Source & source,                   //particle source
            const isaac_float3 & start,               //ray start position in local volume
            const isaac_float3 & dir,
            const isaac_uint3 & cellPos,             //cell to test in local volume
            const T_TransferArray & transferArray,     //transfer function
            const T_SourceWeight & sourceWeight,       //weight of this particle source for radius
            const isaac_float3 & particleScale,    //scale of volume to prevent stretched particles
            const isaac_float3 & clippingNormal,     //normal of the intersecting clipping plane
            const bool & isClipped,
            isaac_float4 & out_color,                 //resulting particle color
            isaac_float3 & out_normal,                //resulting particle normal
            isaac_float3 & out_position,              //resulting particle hit position
            bool & out_particleHit,                  //true or false if particle has been hit or not
            isaac_float & out_depth                       //resulting particle depth
        ) const
        {
            const int sourceNumber = T_NR::value + T_offset;
            if( boost::mpl::at_c<
                T_Filter,
                T_NR::value
            >::type::value )
            {
                auto particleIterator = source.getIterator( cellPos );

                // iterate over all particles in current cell
                for( int i = 0; i < particleIterator.size; ++i )
                {
                    // ray sphere intersection
                    isaac_float3 particlePos =
                        ( particleIterator.getPosition( ) + isaac_float3( cellPos ) )
                        * particleScale;
                    isaac_float3 L = particlePos - start;
                    isaac_float radius = particleIterator.getRadius( )
                                         * sourceWeight.value[T_NR::value + T_offset];
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
                            isaac_float_dim <T_Source::feature_dim>
                                data = particleIterator.getAttribute( );

                            isaac_float result = isaac_float( 0 );

                            // apply functorchain
                           result = applyFunctorChain<T_Source::feature_dim>(&data, sourceNumber);

                            // apply transferfunction
                            ISAAC_IDX_TYPE lookupValue = ISAAC_IDX_TYPE(
                                glm::round( result * isaac_float( T_transferSize ) )
                            );
                            lookupValue = glm::clamp( lookupValue, ISAAC_IDX_TYPE( 0 ), T_transferSize - 1 );
                            isaac_float4 value = transferArray.pointer[sourceNumber][lookupValue];

                            // check if the alpha value is greater or equal than 0.5
                            if( value.w >= 0.5f )
                            {
                                out_color = value;
                                out_depth = t0;
                                out_particleHit = 1;
                                out_position = particlePos;
                                out_normal = start + t0 * dir - particlePos;
                                if( t0 < 0 && isClipped )
                                {
                                    #if ISAAC_AO_BUG_FIX == 1
                                        out_depth = 0;
                                    #endif
                                        out_normal = -clippingNormal;
                                }
                            }
                        }
                    }
                    particleIterator.next( );
                }
            }
        }
    };


    template<
        typename T_ParticleList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        int T_sourceOffset
    >
    struct ParticleRenderKernel
    {
        template<
            typename T_Acc
        >
        ALPAKA_FN_ACC void operator()(
            T_Acc const & acc,
            uint32_t * const pixels,          //ptr to output pixels
            isaac_float3 * const gDepth,      //depth buffer
            isaac_float3 * const gNormal,     //normal buffer
            const isaac_size2 framebufferSize,     //size of framebuffer
            const isaac_uint2 framebufferStart,    //framebuffer offset
            const T_ParticleList particleSources,   //source simulation particles
            const isaac_float4 backgroundColor,    //color of render background
            const T_TransferArray transferArray,     //array of pointers to transfer functions
            const T_SourceWeight sourceWeight,       //weights of all sources 
            const isaac_float3 scale,               //isaac set scaling
            const ClippingStruct inputClipping,   //clipping planes
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
            pixel = pixel + framebufferStart;
            if( pixel.x >= framebufferSize.x || pixel.y >= framebufferSize.y )
                return;

            //gNormalBuffer default value
            isaac_float3 defaultNormal = {0.0, 0.0, 0.0};
            isaac_float3 defaultDepth = {0.0, 0.0, 1.0};

            //set background color
            isaac_float4 color = backgroundColor;
            bool atLeastOne = true;
            isaac_for_each_with_mpl_params(
                particleSources,
                CheckNoSourceIterator< T_Filter >( ),
                atLeastOne
            );
            if( !atLeastOne )
            {
                ISAAC_SET_COLOR ( 
                    pixels[pixel.x + pixel.y * framebufferSize.x], 
                    color 
                )
                gNormal[pixel.x + pixel.y * framebufferSize.x] = defaultNormal;
                gDepth[pixel.x + pixel.y * framebufferSize.x] = defaultDepth;
                return;
            }

            Ray ray = pixelToRay( isaac_float2( pixel ), isaac_float2( framebufferSize ) );

            if( !clipRay(ray, inputClipping ) )
            {
                ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebufferSize.x], color )

                //this function aborts drawing and therfore wont set any normal or depth values
                //defaults will be applied for clean images
                gNormal[pixel.x + pixel.y * framebufferSize.x] = defaultNormal;
                gDepth[pixel.x + pixel.y * framebufferSize.x] = defaultDepth;

                return;
            }


            isaac_float4 particleColor = backgroundColor;
            isaac_float depth = std::numeric_limits<isaac_float>::max( );
            bool particleHit = false;
            // light direction is camera direction
            isaac_float3 lightDir = -ray.dir;

            // get the signs of the direction for the raymarch
            isaac_int3 dirSign = glm::sign( ray.dir );

            // calculate current position in scaled object space
            isaac_float3 currentPos = ray.start + ray.dir * startDepth;

            // calculate current local cell coordinates
            isaac_uint3 currentCell = isaac_uint3( glm::clamp( 
                                    isaac_int3( currentPos / scale ), 
                                    isaac_int3( 0 ), 
                                    isaac_int3( SimulationSize.localParticleSize - ISAAC_IDX_TYPE( 1 ) ) 
                                ) );

            isaac_float rayLength = endDepth - startDepth;
            isaac_float testedLength = 0;


            // calculate next intersection with each dimension
            isaac_float3 t = ( ( isaac_float3( currentCell ) + isaac_float3( glm::max( dirSign, 0 ) ) ) 
                    * scale - currentPos ) / ray.dir;

            // calculate delta length to next intersection in the same dimension
            isaac_float3 deltaT = scale / ray.dir * isaac_float3( dirSign );

            isaac_float3 particleHitposition(0);

            // check for 0 to stop infinite looping
            if( ray.dir.x == 0 )
            {
                t.x = std::numeric_limits<isaac_float>::max( );
            }
            if( ray.dir.y == 0 )
            {
                t.y = std::numeric_limits<isaac_float>::max( );
            }
            if( ray.dir.z == 0 )
            {
                t.z = std::numeric_limits<isaac_float>::max( );
            }


            //normal at particle hit position
            isaac_float3 particleNormal = defaultNormal;

            // iterate over all cells on the ray path
            // check if the ray leaves the local volume, has a particle hit or exceeds the max ray distance
            while( isInUpperBounds(currentCell, SimulationSize.localParticleSize)
                && particleHit == false
                && testedLength <= rayLength )
            {

                // calculate particle intersections for each particle source
                isaac_for_each_with_mpl_params(
                    particleSources,
                    MergeParticleSourceIterator<
                        T_transferSize,
                        T_sourceOffset,
                        T_Filter
                    >( ),
                    currentPos,
                    ray.dir,
                    currentCell,
                    transferArray,
                    sourceWeight,
                    scale,
                    clippingNormal,
                    isClipped,
                    particleColor,
                    particleNormal,
                    particleHitposition,
                    particleHit,
                    depth
                );


                // adds the delta t value to the smallest dimension t and increment the cell index in the dimension
                if( t.x < t.y && t.x < t.z )
                {
                    currentCell.x += dirSign.x;
                    testedLength = t.x;
                    t.x += deltaT.x;
                }
                else if( t.y < t.x && t.y < t.z )
                {
                    currentCell.y += dirSign.y;
                    testedLength = t.y;
                    t.y += deltaT.y;
                }
                else
                {
                    currentCell.z += dirSign.z;
                    testedLength = t.z;
                    t.z += deltaT.z;
                }

            }
            // if there was a hit set maximum volume raycast distance to particle hit distance and set particle color
            if( particleHit )
            {

                // calculate lighting properties for the last hit particle
                particleNormal = glm::normalize( particleNormal );

                isaac_float lightFactor = glm::dot( particleNormal, lightDir );

                isaac_float3 halfVector = glm::normalize( -ray.dir + lightDir );

                isaac_float specular = glm::dot( particleNormal, halfVector );

                specular = pow( specular, 10 );
                specular *= 0.5f;
                lightFactor = lightFactor * 0.5f + 0.5f;


                particleColor = glm::min( particleColor * lightFactor + specular, isaac_float( 1 ) );
                particleColor.a = 1.0f;
            }


            ISAAC_SET_COLOR ( pixels[pixel.x + pixel.y * framebufferSize.x], particleColor )
            //save the particle normal in the normal g buffer
            gNormal[pixel.x + pixel.y * framebufferSize.x] = particleNormal;
            
            //save the cell depth in our g buffer (depth)
            isaac_float3 depthValue = {
                0.0f,
                1.0f,
                depth + startDepth
            };
            gDepth[pixel.x + pixel.y * framebufferSize.x] = depthValue;
        }
    };



    template<
        typename T_ParticleList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_AccDim,
        typename T_Acc,
        typename T_Stream,
        typename T_FunctionChain,
        int T_sourceOffset,
        int T_n
    >
    struct ParticleRenderKernelCaller
    {
        inline static void call(
            T_Stream stream,
            uint32_t * framebuffer,
            isaac_float3 * depthBuffer,
            isaac_float3 * normalBuffer,
            const isaac_size2 & framebufferSize,
            const isaac_uint2 & framebufferStart,
            const T_ParticleList & particleSources,
            const isaac_float4 & backgroundColor,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            IceTInt const * const readback_viewport,
            const isaac_float3 & scale,
            const ClippingStruct & clipping,
            const AOParams & ambientOcclusion
        )
        {
            if( sourceWeight.value[T_sourceOffset + boost::mpl::size< T_ParticleList >::type::value - T_n] == isaac_float( 0 ) )
            {
                ParticleRenderKernelCaller<
                    T_ParticleList,
                    T_TransferArray,
                    T_SourceWeight,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::false_
                    >::type,
                    T_transferSize,
                    T_AccDim,
                    T_Acc,
                    T_Stream,
                    T_FunctionChain,
                    T_sourceOffset,
                    T_n - 1
                >::call(
                    stream,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebufferSize,
                    framebufferStart,
                    particleSources,
                    backgroundColor,
                    transferArray,
                    sourceWeight,
                    readback_viewport,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
            else
            {
                ParticleRenderKernelCaller<
                    T_ParticleList,
                    T_TransferArray,
                    T_SourceWeight,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::true_
                    >::type,
                    T_transferSize,
                    T_AccDim,
                    T_Acc,
                    T_Stream,
                    T_FunctionChain,
                    T_sourceOffset,
                    T_n - 1
                >::call(
                    stream,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebufferSize,
                    framebufferStart,
                    particleSources,
                    backgroundColor,
                    transferArray,
                    sourceWeight,
                    readback_viewport,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
        }
    };

    template<
        typename T_ParticleList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_AccDim,
        typename T_Acc,
        typename T_Stream,
        typename T_FunctionChain,
        int T_sourceOffset
    >
    struct ParticleRenderKernelCaller<
        T_ParticleList,
        T_TransferArray,
        T_SourceWeight,
        T_Filter,
        T_transferSize,
        T_AccDim,
        T_Acc,
        T_Stream,
        T_FunctionChain,
        T_sourceOffset,
        0 //<-- spezialisation
    >
    {
        inline static void call(
            T_Stream stream,
            uint32_t *  framebuffer,
            isaac_float3 * depthBuffer,
            isaac_float3 * normalBuffer,
            const isaac_size2 & framebufferSize,
            const isaac_uint2 & framebufferStart,
            const T_ParticleList & particleSources,
            const isaac_float4 & backgroundColor,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            IceTInt const * const readback_viewport,
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
            if ( boost::mpl::not_<boost::is_same<T_Acc, alpaka::AccGpuCudaRt<T_AccDim, ISAAC_IDX_TYPE> > >::value )
#endif
            {
                grid_size.x = ISAAC_IDX_TYPE( readback_viewport[2] );
                grid_size.y = ISAAC_IDX_TYPE( readback_viewport[3] );
                block_size.x = ISAAC_IDX_TYPE( 1 );
                block_size.y = ISAAC_IDX_TYPE( 1 );
            }
            const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> threads(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> blocks(
                ISAAC_IDX_TYPE( 1 ),
                block_size.y,
                block_size.x
            );
            const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> grid(
                ISAAC_IDX_TYPE( 1 ),
                grid_size.y,
                grid_size.x
            );
            auto const workdiv(
                alpaka::WorkDivMembers<
                    T_AccDim,
                    ISAAC_IDX_TYPE
                >(
                    grid,
                    blocks,
                    threads
                )
            );
            ParticleRenderKernel
            <
                T_ParticleList,
                T_TransferArray,
                T_SourceWeight,
                T_Filter,
                T_transferSize,
                T_sourceOffset
            >
            kernel;
            auto const instance
            (
                alpaka::createTaskKernel<T_Acc>
                (
                    workdiv,
                    kernel,
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebufferSize,
                    framebufferStart,
                    particleSources,
                    backgroundColor,
                    transferArray,
                    sourceWeight,
                    scale,
                    clipping,
                    ambientOcclusion
                )
            );
            alpaka::enqueue(stream, instance);
        }
    };
}