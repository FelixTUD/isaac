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
#include "isaac_functors.hpp"

#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/mpl/size.hpp>

namespace isaac
{
    typedef isaac_float( * isaac_functor_chain_pointer_4 )(
        isaac_float_dim< 4 >,
        isaac_int
    );

    typedef isaac_float( * isaac_functor_chain_pointer_3 )(
        isaac_float_dim< 3 >,
        isaac_int
    );

    typedef isaac_float( * isaac_functor_chain_pointer_2 )(
        isaac_float_dim< 2 >,
        isaac_int
    );

    typedef isaac_float( * isaac_functor_chain_pointer_1 )(
        isaac_float_dim< 1 >,
        isaac_int
    );

    typedef isaac_float( * isaac_functor_chain_pointer_N )(
        void *,
        isaac_int
    );

    ISAAC_CONSTANT isaac_float4 isaac_parameter_d[ ISAAC_MAX_SOURCES*ISAAC_MAX_FUNCTORS ];

    ISAAC_CONSTANT isaac_functor_chain_pointer_N isaac_function_chain_d[ ISAAC_MAX_SOURCES ];

    template<
        int N
    >
    struct dest_array_struct
    {
        isaac_int nr[N];
    };

    template<int TFeatureDim>
    ISAAC_DEVICE_INLINE isaac_float applyFunctorChain(isaac_float_dim <TFeatureDim>* data, const int nr)
    {
        if( TFeatureDim == 1 )
        {
            return reinterpret_cast<isaac_functor_chain_pointer_1> ( isaac_function_chain_d[nr] )(
                    *( reinterpret_cast< isaac_float_dim< 1 > * > ( data ) ),
                    nr
                );
        }
        if( TFeatureDim == 2 )
        {
            return reinterpret_cast<isaac_functor_chain_pointer_2> ( isaac_function_chain_d[nr] )(
                    *( reinterpret_cast< isaac_float_dim< 2 > * > ( data ) ),
                    nr
                );
        }
        if( TFeatureDim == 3 )
        {
            return reinterpret_cast<isaac_functor_chain_pointer_3> ( isaac_function_chain_d[nr] )(
                    *( reinterpret_cast< isaac_float_dim< 3 > * > ( data ) ),
                    nr
                );
        }
        if( TFeatureDim == 4 )
        {
            return reinterpret_cast<isaac_functor_chain_pointer_4> ( isaac_function_chain_d[nr] )(
                    *( reinterpret_cast< isaac_float_dim< 4 > * > ( data ) ),
                    nr
                );
        }
        return 0;
    }

    template<
        typename TFunctorVector,
        int TFeatureDim,
        int NR
    >
    struct FillFunctorChainPointerKernelStruct
    {
        ISAAC_DEVICE static isaac_functor_chain_pointer_N
        call( isaac_int const * const bytecode )
        {
#define ISAAC_SUB_CALL( Z, I, U ) \
            if (bytecode[ISAAC_MAX_FUNCTORS-NR] == I) \
                return FillFunctorChainPointerKernelStruct \
                < \
                    typename boost::mpl::push_back< TFunctorVector, typename boost::mpl::at_c<IsaacFunctorPool,I>::type >::type, \
                    TFeatureDim, \
                    NR - 1 \
                > ::call( bytecode );
            BOOST_PP_REPEAT(
                ISAAC_FUNCTOR_COUNT,
                ISAAC_SUB_CALL,
                ~
            )
#undef ISAAC_SUB_CALL
            return NULL; //Should never be reached anyway
        }
    };


    template<
        typename TFunctorVector,
        int TFeatureDim
    >
    ISAAC_DEVICE isaac_float generateFunctorChain(
        isaac_float_dim <TFeatureDim> const value,
        isaac_int const src_id
    )
    {
#define  ISAAC_LEFT_DEF( Z, I, U ) boost::mpl::at_c< TFunctorVector, ISAAC_MAX_FUNCTORS - I - 1 >::type::call(
#define ISAAC_RIGHT_DEF( Z, I, U ) , isaac_parameter_d[ src_id * ISAAC_MAX_FUNCTORS + I ] )
#define  ISAAC_LEFT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_LEFT_DEF, ~)
#define ISAAC_RIGHT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_RIGHT_DEF, ~)
        // expands to: funcN( ... func1( func0( data, p[0] ), p[1] ) ... p[N] );
        return ISAAC_LEFT
        value
        ISAAC_RIGHT.x;
#undef ISAAC_LEFT_DEF
#undef ISAAC_LEFT
#undef ISAAC_RIGHT_DEF
#undef ISAAC_RIGHT
    }


    template<
        typename TFunctorVector,
        int TFeatureDim
    >
    struct FillFunctorChainPointerKernelStruct<
        TFunctorVector,
        TFeatureDim,
        0 //<- Specialization
    >
    {
        ISAAC_DEVICE static isaac_functor_chain_pointer_N
        call( isaac_int const * const bytecode )
        {
            return reinterpret_cast<isaac_functor_chain_pointer_N> ( generateFunctorChain<
                TFunctorVector,
                TFeatureDim
            > );
        }
    };



    struct fillFunctorChainPointerKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
            isaac_functor_chain_pointer_N * const functor_chain_d
        ) const
        {
            isaac_int bytecode[ISAAC_MAX_FUNCTORS];
            for( int i = 0; i < ISAAC_MAX_FUNCTORS; i++ )
            {
                bytecode[i] = 0;
            }
            for( int i = 0; i < ISAAC_FUNCTOR_COMPLEX;
            i++ )
            {
                functor_chain_d[i * 4 + 0] =
                    FillFunctorChainPointerKernelStruct<
                        boost::mpl::vector< >,
                        1,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functor_chain_d[i * 4 + 1] =
                    FillFunctorChainPointerKernelStruct<
                        boost::mpl::vector< >,
                        2,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functor_chain_d[i * 4 + 2] =
                    FillFunctorChainPointerKernelStruct<
                        boost::mpl::vector< >,
                        3,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functor_chain_d[i * 4 + 3] =
                    FillFunctorChainPointerKernelStruct<
                        boost::mpl::vector< >,
                        4,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                for( int j = ISAAC_MAX_FUNCTORS - 1; j >= 0; j-- )
                {
                    if( bytecode[j] < ISAAC_FUNCTOR_COUNT - 1 )
                    {
                        bytecode[j]++;
                        break;
                    }
                    else
                    {
                        bytecode[j] = 0;
                    }
                }
            }
        }
    };

    template<
        int count,
        typename TDest
    >
    struct updateFunctorChainPointerKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
            isaac_functor_chain_pointer_N * const functor_chain_choose_d,
            isaac_functor_chain_pointer_N const * const functor_chain_d,
            TDest dest
        ) const
        {
            for( int i = 0; i < count; i++ )
            {
                functor_chain_choose_d[i] = functor_chain_d[dest.nr[i]];
            }
        }
    };
}