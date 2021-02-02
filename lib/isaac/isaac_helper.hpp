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

namespace isaac
{

void mergeJSON(json_t* result, json_t* candidate)
{
    const char *cKey;
    const char *rKey;
    json_t *cValue;
    json_t *rValue;
    //metadata merge, old values stay, arrays are merged
    json_t* mCandidate = json_object_get(candidate, "metadata");
    json_t* mResult = json_object_get(result, "metadata");
    void *temp,*temp2;
    if (mCandidate && mResult)
    {
        json_object_foreach_safe( mCandidate, temp, cKey, cValue )
        {
            bool found_array = false;
            json_object_foreach_safe( mResult, temp2, rKey, rValue )
            {
                if (strcmp(rKey,cKey) == 0)
                {
                    if (json_is_array(rValue) && json_is_array(cValue))
                    {
                        json_array_extend(rValue, cValue);
                        found_array = true;
                    }
                    break;
                }
            }
            if (!found_array)
                json_object_set( mResult, cKey, cValue );
        }
    }
    //general merge, new values stay
    json_object_foreach_safe( candidate, temp, cKey, cValue )
    {
        bool foundMeta = false;
        json_object_foreach_safe( result, temp2, rKey, rValue )
        {
            if (strcmp(rKey,cKey) == 0 && strcmp(rKey,"metadata") == 0)
            {
                foundMeta = true;
                break;
            }
        }
        if (!foundMeta)
            json_object_set( result, cKey, cValue );
    }
}

isaac_float4 getHSVA(isaac_float h, isaac_float s, isaac_float v, isaac_float a)
{
    isaac_int hi = isaac_int(floor(h / (M_PI/3)));
    isaac_float f = h / (M_PI/3) - isaac_float(hi);
    isaac_float p = v*(isaac_float(1)-s);
    isaac_float q = v*(isaac_float(1)-s*f);
    isaac_float t = v*(isaac_float(1)-s*(isaac_float(1)-f));
    isaac_float4 result = {0,0,0,a};
    switch (hi)
    {
        case 0: case 6:
            result.x = v;
            result.y = t;
            result.z = p;
            break;
        case 1:
            result.x = q;
            result.y = v;
            result.z = p;
            break;
        case 2:
            result.x = p;
            result.y = v;
            result.z = t;
            break;
        case 3:
            result.x = p;
            result.y = q;
            result.z = v;
            break;
        case 4:
            result.x = t;
            result.y = p;
            result.z = v;
            break;
        case 5:
            result.x = v;
            result.y = p;
            result.z = q;
            break;
    }
    return result;
}

void setFrustum(isaac_double * const projection, const isaac_float left,const isaac_float right,const isaac_float bottom,const isaac_float top,const isaac_float znear,const isaac_float zfar )
{
    isaac_float  znear2 = znear * isaac_float(2);
    isaac_float  width = right - left;
    isaac_float  height = top - bottom;
    isaac_float  zRange = znear - zfar;
    projection[ 0] = znear2 / width;
    projection[ 1] = isaac_float( 0);
    projection[ 2] = isaac_float( 0);
    projection[ 3] = isaac_float( 0);
    projection[ 4] = isaac_float( 0);
    projection[ 5] = znear2 / height;
    projection[ 6] = isaac_float( 0);
    projection[ 7] = isaac_float( 0);
    projection[ 8] = ( right + left ) / width;
    projection[ 9] = ( top + bottom ) / height;
    projection[10] = ( zfar + znear) / zRange;
    projection[11] = isaac_float(-1);
    projection[12] = isaac_float( 0);
    projection[13] = isaac_float( 0);
    projection[14] = ( -znear2 * zfar ) / -zRange;
    projection[15] = isaac_float( 0);
}

void setPerspective(isaac_double * const projection, const isaac_float fovyInDegrees,const isaac_float aspectRatio,const isaac_float znear,const isaac_float zfar )
{
    isaac_float ymax = znear * tan( fovyInDegrees * M_PI / isaac_float(360) );
    isaac_float xmax = ymax * aspectRatio;
    setFrustum(projection, -xmax, xmax, -ymax, ymax, znear, zfar );
}

void spSetPerspectiveStereoscopic( isaac_double * const projection, const isaac_float fovyInDegrees,const isaac_float aspectRatio,const isaac_float znear,const isaac_float zfar,const isaac_float z0,const isaac_float distance )
{
    isaac_float t_z0 = znear * tan( fovyInDegrees * M_PI / isaac_float(360) );
    isaac_float xmin = -t_z0 + distance/2.0f*znear/z0;
    isaac_float xmax =  t_z0 + distance/2.0f*znear/z0;
    isaac_float ymin = -t_z0 / aspectRatio;
    isaac_float ymax =  t_z0 / aspectRatio;
    setFrustum(projection, xmin, xmax, ymin, ymax, znear, zfar );
    projection[12] += distance;
}


void setOrthographic(isaac_dmat4& projection, const isaac_double right,const isaac_double top,const isaac_double znear,const isaac_double zfar )
{
    projection = isaac_dmat4( 1 );
    projection[0][0] = 1.0 / right;
    projection[1][1] = 1.0 / top;
    projection[2][2] = -2.0 / (zfar-znear);
    projection[3][2] = - (zfar+znear) / (zfar-znear);
}



#if ISAAC_VALGRIND_TWEAKS == 1
    static void *extra_malloc(size_t size)
    {
        /* Get 4 bytes more than requested */
        void *ptr = malloc(size + 4);
        return ptr;
    }

    static void extra_free(void *ptr)
    {
        free(ptr);
    }
#endif


} //namespace isaac;
