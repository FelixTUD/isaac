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

#include <alpaka/alpaka.hpp>


#include "isaac_types.hpp"
#include "isaac_common_kernel.hpp"



namespace isaac
{
    enum class FilterType{
        NEAREST,
        LINEAR
    };

    enum class BorderType{
        CLAMP,
        REPEAT,
        VALUE
    };


    template<
        typename T_VecType,
        int T_vecDim,
        int T_textureDim,
        ISAAC_IDX_TYPE T_guardSize = 0
    >
    class Texture
    {
        public:
            ISAAC_HOST_DEVICE_INLINE
            void init(isaac_vec_dim<T_vecDim, T_VecType>* bufferPtr, const isaac_size_dim<T_textureDim> & size)
            {
                this->bufferPtr = bufferPtr;
                this->size = size;
                this->sizeWithGuard = size + ISAAC_IDX_TYPE( 2 ) * T_guardSize;
            }

            ISAAC_HOST_DEVICE_INLINE
            void setValue( const isaac_float_dim<T_textureDim>& coord, const isaac_vec_dim<T_vecDim, T_VecType>& value )
            {
                isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int( T_guardSize );
                assert( isInUpperBounds( coord, isaac_int_dim<T_textureDim>( sizeWithGuard ) ) );
                bufferPtr[hash( offsetCoord )] = value;
            }

            // access between 0-1 in each dimension + guard
            template<FilterType T_filter = FilterType::NEAREST, BorderType T_border = BorderType::CLAMP>
            ISAAC_HOST_DEVICE_INLINE 
            isaac_vec_dim<T_vecDim, T_VecType> sample( 
                const isaac_float_dim<T_textureDim>& normalizedCoord, 
                const isaac_vec_dim<T_vecDim, T_VecType> & borderValue = isaac_vec_dim<T_vecDim, T_VecType>( 0 )
            )
            {
                isaac_vec_dim<T_vecDim, T_VecType> result;
                isaac_float_dim<T_textureDim> coord = isaac_float_dim<T_textureDim>( normalizedCoord * size );
                if( T_filter == FilterType::LINEAR )
                {
                    result = interpolate<T_border>( coord, borderValue );
                }
                else
                {
                    result = directMemoryAccess<T_border>( isaac_int_dim<T_textureDim>( coord ), borderValue );
                }
                return result;
            }

            template<BorderType T_border = BorderType::CLAMP>
            ISAAC_HOST_DEVICE_INLINE
            isaac_vec_dim<T_vecDim, T_VecType> directMemoryAccess( 
                const isaac_int_dim<T_textureDim>& coord,
                const isaac_vec_dim<T_vecDim, T_VecType> & borderValue = isaac_vec_dim<T_vecDim, T_VecType>( 0 )
            )
            {
                isaac_uint_dim<T_textureDim> offsetCoord;
                if( T_border == BorderType::REPEAT )
                {
                    offsetCoord = glm::mod( coord + isaac_int( T_guardSize ), isaac_int_dim<T_textureDim>( sizeWithGuard ) );
                }
                else if( T_border == BorderType::VALUE )
                {
                    offsetCoord = coord + isaac_int( T_guardSize );
                    if( !isInLowerBounds( coord, isaac_int( 0 ) ) 
                        || !isInUpperBounds(coord, isaac_int_dim<T_textureDim>( sizeWithGuard ) ) )
                            return borderValue;
                }
                else
                {
                    offsetCoord = glm::clamp( coord + isaac_int( T_guardSize ), isaac_int(0), isaac_int_dim<T_textureDim>( sizeWithGuard ) );
                }
                return bufferPtr[hash( offsetCoord )];
            }

            ISAAC_HOST_DEVICE_INLINE 
            isaac_uint hash( const isaac_uint_dim<1>& coord )
            {
                return coord.x;
            }

            ISAAC_HOST_DEVICE_INLINE 
            isaac_uint hash( const isaac_uint_dim<2>& coord )
            {
                return coord.x + coord.y * sizeWithGuard.x;
            }

            ISAAC_HOST_DEVICE_INLINE 
            isaac_uint hash( const isaac_uint_dim<3>& coord )
            {
                return coord.x + coord.y * sizeWithGuard.x 
                    + coord.z * sizeWithGuard.y * sizeWithGuard.z;
            }


        private:
            isaac_vec_dim<T_vecDim, T_VecType>* bufferPtr;
            isaac_size_dim<T_textureDim> size;
            isaac_size_dim<T_textureDim> sizeWithGuard;


            template<BorderType T_border = BorderType::CLAMP>
            ISAAC_HOST_DEVICE_INLINE 
            isaac_vec_dim<T_vecDim, T_VecType> interpolate ( 
                isaac_float_dim<1> coord, 
                const isaac_vec_dim<T_vecDim, T_VecType> & borderValue = isaac_vec_dim<T_vecDim, T_VecType>( 0 )
            )
            {
                isaac_vec_dim<T_vecDim, T_VecType> data2[2];
                for( int x = 0; x < 2; x++ )
                {
                    data2[x] = directMemoryAccess<T_border>( isaac_float_dim<1>( coord ) + isaac_float_dim<1>( x ), borderValue );
                }

                return linear( glm::fract( coord ), data2 );
            }

            template<BorderType T_border = BorderType::CLAMP>
            ISAAC_HOST_DEVICE_INLINE 
            isaac_vec_dim<T_vecDim, T_VecType> interpolate ( 
                isaac_float_dim<2> coord, 
                const isaac_vec_dim<T_vecDim, T_VecType> & borderValue = isaac_vec_dim<T_vecDim, T_VecType>( 0 ) 
            )
            {
                isaac_vec_dim<T_vecDim, T_VecType> data4[2][2];
                for( int x = 0; x < 2; x++ )
                {
                    for( int y = 0; y < 2; y++ )
                    {
                        data4[x][y] = directMemoryAccess<T_border>( isaac_int2( coord ) + isaac_int2( x, y ), borderValue );
                    }
                }

                return bilinear( glm::fract( coord ), data4 );
            }

            template<BorderType T_border = BorderType::CLAMP>
            ISAAC_HOST_DEVICE_INLINE 
            isaac_vec_dim<T_vecDim, T_VecType> interpolate ( 
                isaac_float_dim<3> coord, 
                const isaac_vec_dim<T_vecDim, T_VecType> & borderValue = isaac_vec_dim<T_vecDim, T_VecType>( 0 ) 
            )
            {
                isaac_vec_dim<T_vecDim, T_VecType> data8[2][2][2];
                for( int x = 0; x < 2; x++ )
                {
                    for( int y = 0; y < 2; y++ )
                    {
                        for( int z = 0; z < 2; z++ )
                        {
                            data8[x][y][z] = directMemoryAccess<T_border>( isaac_int3( coord ) + isaac_int3( x, y, z ), borderValue );
                        }
                    }
                }

                return trilinear( glm::fract( coord ), data8 );
            }
    };

    template<
        typename T_DevAcc,
        typename T_VecType,
        int T_vecDim,
        int T_textureDim,
        ISAAC_IDX_TYPE T_guardSize = 0
    >
    class TextureWrapper
    {
        using FraDim = alpaka::DimInt< 1 >;

        static const int featureDim = T_vecDim;

        public:
            TextureWrapper() = default;

            void init( const T_DevAcc& devAcc, const isaac_size_dim<T_textureDim>& size )
            {
                const isaac_size_dim<T_textureDim> sizeWithGuard = size + ISAAC_IDX_TYPE( 2 ) * T_guardSize;

                ISAAC_IDX_TYPE totalSize = sizeWithGuard[0];
                for( int i = 1; i < T_textureDim; ++i )
                {
                    totalSize *= ( sizeWithGuard[i] );
                }

                alpaka::Vec<
                    FraDim,
                    ISAAC_IDX_TYPE
                > bufferExtent( totalSize );

                buffer =
                    alpaka::allocBuf<
                        isaac_vec_dim<T_vecDim, T_VecType>,
                        ISAAC_IDX_TYPE
                    >(
                        devAcc,
                        bufferExtent
                    );

                texture.init( alpaka::getPtrNative( buffer ), size );
            }

        private:
            Texture<
                T_VecType,
                T_vecDim,
                T_textureDim,
                T_guardSize
            > texture;

            alpaka::Buf<
                T_DevAcc, 
                isaac_vec_dim<T_vecDim, T_VecType>, 
                FraDim, 
                ISAAC_IDX_TYPE
            > buffer;
    };

    template<
        typename T_VecType,
        int T_vecDim,
        ISAAC_IDX_TYPE T_guardSize = 0
    >
    using Tex2D = Texture<T_VecType, T_vecDim, 2>;

    template<
        typename T_VecType,
        int T_vecDim,
        ISAAC_IDX_TYPE T_guardSize = 0
    >
    using Tex3D = Texture<T_VecType, T_vecDim, 3>;

    template<
        typename T_DevAcc,
        typename T_VecType,
        int T_vecDim,
        ISAAC_IDX_TYPE T_guardSize = 0
    >
    using Tex2DWrapper = TextureWrapper<T_DevAcc, T_VecType, T_vecDim, 2>;

    template<
        typename T_DevAcc,
        typename T_VecType,
        int T_vecDim,
        ISAAC_IDX_TYPE T_guardSize = 0
    >
    using Tex3DWrapper = TextureWrapper<T_DevAcc, T_VecType, T_vecDim, 3>;

} //namespace isaac;