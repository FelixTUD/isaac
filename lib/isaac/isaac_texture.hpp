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

#include "isaac_texture2D.hpp"
#ifdef ISAAC_USE_CUDA_TEXTURES
#    include "isaac_texture3D_cuda.hpp"
#else
#    include "isaac_texture3D.hpp"
#endif

namespace isaac
{
    template<FilterType T_filterType = FilterType::NEAREST, BorderType T_borderType = BorderType::CLAMP>
    struct Texture3DCreator
    {
        template<typename T_DevAcc, typename T_Type, IndexType T_indexType>
        inline static Texture3D<T_Type, T_filterType, T_borderType, T_indexType> create(
            Texture3DAllocator<T_DevAcc, T_Type, T_indexType>& allocator)
        {
            return Texture3D<T_Type, T_filterType, T_borderType, T_indexType>(
                allocator.getBufferPtr(),
                allocator.getSize(),
                allocator.getGuardSize());
        }
    };

    template<int T_n>
    struct PersistentArrayStruct
    {
        Texture3D<isaac_float, FilterType::LINEAR> textures[ZeroCheck<T_n>::value];
    };

    template<int T_n>
    struct LicArrayStruct
    {
        Texture3D<isaac_float, FilterType::LINEAR, BorderType::VALUE> textures[T_n];
    };

} // namespace isaac