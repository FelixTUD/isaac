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
    /* 
     * SSAO
     * Kernels for ssao calculation
     */

    //filter kernel
    ISAAC_CONSTANT isaac_float3 ssao_kernel_d[64];

    //vector rotation noise kernel
    ISAAC_CONSTANT isaac_float3 ssao_noise_d[16];

    /**
     * @brief Calculate SSAO factor
     * 
     * Requires AO Buffer     (dim 1)
     *          Depth Buffer  (dim 1)
     *          Normal Buffer (dim 3)
     * 
     */
    struct isaacSSAOKernel {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator() (
            TAcc__ const &acc,
            isaac_float * const gAOBuffer,       //ao buffer
            isaac_float3 * const gDepth,         //depth buffer (will be used as y=blending of particles and volume, z=depth of pixels)
            isaac_float3 * const gNormal,        //normal buffer
            const isaac_size2 framebuffer_size,  //size of framebuffer
            const isaac_uint2 framebuffer_start, //framebuffer offset
            ao_struct ao_properties              //properties for ambient occlusion
            ) const
        {

            isaac_uint2 pixel;
            //get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads> ( acc );
            pixel.x = isaac_uint ( alpThreadIdx[2] );
            pixel.y = isaac_uint ( alpThreadIdx[1] );

            pixel = pixel + framebuffer_start;    

            


            /*
            * TODO
            * 
            * Old standart ssao by crytech 
            * 
            * First implemntation failed and the source code is below
            * Possible errors could be mv or proj matrix
            */

            //search radius for depth testing
            isaac_int radius = 10;

            /*
            //isaac_float3 origin = gDepth[pixel.x + pixel.y * framebuffer_size.x];

            

            //get the normal value from the gbuffer
            isaac_float3 normal = gNormal[pixel.x + pixel.y * framebuffer_size.x];

            //normalize the normal
            isaac_float len = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
            if(len == 0) {
                gAOBuffer[pixel.x + pixel.y * framebuffer_size.x] = 0.0f;
                return;
            }

            normal = normal / len;
            
            

            isaac_float3 rvec = {0.7f, 0.1f, 0.3f};
            isaac_float3 tangent = rvec - normal * (rvec.x * normal.x + rvec.y * normal.y + rvec.z * normal.z);
            len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y + tangent.z * tangent.z);
            tangent = tangent / len;
            isaac_float3 bitangent = {
                normal.y * tangent.z - normal.z * tangent.y,
                normal.z * tangent.x - normal.x * tangent.z,
                normal.x * tangent.y - normal.y * tangent.y
            };

            isaac_float tbn[9];
            tbn[0] = tangent.x;
            tbn[1] = tangent.y;
            tbn[2] = tangent.z;

            tbn[3] = bitangent.x;
            tbn[4] = bitangent.y;
            tbn[5] = bitangent.z;

            tbn[6] = normal.x;
            tbn[7] = normal.y;
            tbn[8] = normal.z;

            isaac_float occlusion = 0.0f;
            for(int i = 0; i < 1; i++) {
                //sample = tbn * sample_kernel
                isaac_float3 sample = {
                    tbn[0] * ssao_kernel_d[i].x + tbn[3] * ssao_kernel_d[i].y + tbn[6] * ssao_kernel_d[i].z,
                    tbn[1] * ssao_kernel_d[i].x + tbn[4] * ssao_kernel_d[i].y + tbn[7] * ssao_kernel_d[i].z,
                    tbn[2] * ssao_kernel_d[i].x + tbn[5] * ssao_kernel_d[i].y + tbn[8] * ssao_kernel_d[i].z,
                };

                sample = sample * radius + origin;

                isaac_float4 offset = {
                    sample.x,
                    sample.y,
                    sample.z,
                    1.0
                };

                //offset = projection * offset
                offset = isaac_float4({
                    isaac_projection_d[0] * offset.x + isaac_projection_d[4] * offset.y + isaac_projection_d[8 ] * offset.z + isaac_projection_d[12] * offset.w,
                    isaac_projection_d[1] * offset.x + isaac_projection_d[5] * offset.y + isaac_projection_d[9 ] * offset.z + isaac_projection_d[13] * offset.w,
                    isaac_projection_d[2] * offset.x + isaac_projection_d[6] * offset.y + isaac_projection_d[10] * offset.z + isaac_projection_d[14] * offset.w,
                    isaac_projection_d[3] * offset.x + isaac_projection_d[7] * offset.y + isaac_projection_d[11] * offset.z + isaac_projection_d[15] * offset.w
                });

                isaac_float2 offset2d = isaac_float2({offset.x / offset.w, offset.y / offset.w});
                offset2d.x = MAX(MIN(offset2d.x * 0.5 + 0.5, 1.0f), 0.0f);
                offset2d.y = MAX(MIN(offset2d.y * 0.5 + 0.5, 1.0f), 0.0f);

                isaac_uint2 offsetFramePos = {
                    isaac_uint(framebuffer_size.x * offset2d.x) + framebuffer_start.x,
                    isaac_uint(framebuffer_size.y * offset2d.y) + framebuffer_start.y,
                };
                //printf("%f %f -- %u %u\n", offset2d.x, offset2d.y, offsetFramePos.x, offsetFramePos.y);
                isaac_float sampleDepth = gDepth[offsetFramePos.x + offsetFramePos.y * framebuffer_size.x].z; 
                occlusion += (sampleDepth - sample.z ? 1.0f : 0.0f);
            }*/
            

            /* 
            * 1. compare all neighbour (+-2 pixel) depth values with the current one and increase the counter if the neighbour is
            *    closer to the camera
            * 
            * 2. get average value by dividing the counter by the cell count (7x7=49)       * 
            *
            */
            //closer to the camera
            isaac_float occlusion = 0.0f;
            isaac_float ref_depth = gDepth[pixel.x + pixel.y * framebuffer_size.x].z;
            for(int i = -3; i <= 3; i++) {
                for(int j = -3; j <= 3; j++) {
                    //avoid out of bounds by simple min max
                    isaac_int x = glm::clamp(pixel.x + i * radius, framebuffer_start.x, framebuffer_start.x + framebuffer_size.x);
                    isaac_int y = glm::clamp(pixel.y + j * radius, framebuffer_start.y, framebuffer_start.y + framebuffer_size.y);

                    //get the neighbour depth value
                    isaac_float depth_sample = gDepth[x + y * framebuffer_size.x].z;

                    // only increase the counter if the neighbour depth is closer to the camera
                    // use <= because we will discard pixels with a depth/ao value 0.0 (for background pixels and image merging), 
                    // but planes will have pixels with depth/ao with 0 because of neighbor pixels
                    if(depth_sample <= ref_depth) {
                        occlusion += 1.0f;
                    }
                }
            }
            isaac_float depth = (occlusion / 49.0f);

            //save the depth value in our ao buffer
            gAOBuffer[pixel.x + pixel.y * framebuffer_size.x] = depth;
        }
    };

    /**
     * @brief Filter SSAO artifacts and return the color with depth simulation
     * 
     * Requires Color Buffer      (dim 4)
     * Requires AO Values Buffer  (dim 1)
     * 
     */
    struct isaacSSAOFilterKernel {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator() (
            TAcc__ const &acc,
            uint32_t * const gColor,             //ptr to output pixels
            isaac_float * const gAOBuffer,       //ambient occlusion values from ssao kernel
            isaac_float3 * const gDepthBuffer,   //depth and blending values
            const isaac_size2 framebuffer_size,  //size of framebuffer
            const isaac_uint2 framebuffer_start, //framebuffer offset
            ao_struct ao_properties              //properties for ambient occlusion
            ) const
        {

            isaac_uint2 pixel;
            //get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads> ( acc );
            pixel.x = isaac_uint ( alpThreadIdx[2] );
            pixel.y = isaac_uint ( alpThreadIdx[1] );

            //get real pixel coordinate by offset
            pixel = pixel + framebuffer_start;

            /* TODO
            * Normally the depth values are smoothed
            * in this case the smooting filter is not applied for simplicity
            * 
            * If the real ssao algorithm is implemented, a real filter will be necessary
            */
            isaac_float depth = gAOBuffer[pixel.x + pixel.y * framebuffer_size.x];
            
            //convert uint32 back to 4x 1 Byte color values
            uint32_t color = gColor[pixel.x + pixel.y * framebuffer_size.x];
            isaac_float4 color_values = {
                ((color >>  0) & 0xff) / 255.0f,
                ((color >>  8) & 0xff) / 255.0f,
                ((color >> 16) & 0xff) / 255.0f,
                ((color >> 24) & 0xff) / 255.0f
            };        

            //read the weight from the global ao settings and merge them with the color value
            isaac_float weight = ao_properties.weight;
            isaac_float ao_factor = ((1.0f - weight) + weight * (1.0f - depth));
            isaac_float particle_blend = gDepthBuffer[pixel.x + pixel.y * framebuffer_size.x].y;
            
            isaac_float4 final_color = { 
                particle_blend * ao_factor * color_values.x + (1.0f - particle_blend) * color_values.x,
                particle_blend * ao_factor * color_values.y + (1.0f - particle_blend) * color_values.y,
                particle_blend * ao_factor * color_values.z + (1.0f - particle_blend) * color_values.z,
                1.0f  * color_values.w
            };
        
            //if the depth value is 0 the ssao kernel found a background value and the color
            //merging is therefore removed
            if(depth == 0.0f) { 
                final_color = { 0, 0, 0, 1.0 };
            }

            //finally replace the old color value with the new ssao filtered color value
            ISAAC_SET_COLOR(gColor[pixel.x + pixel.y * framebuffer_size.x], final_color);
            
        }
    };
}