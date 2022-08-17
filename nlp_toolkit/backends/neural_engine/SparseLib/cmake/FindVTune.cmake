##  Copyright (c) 2022 Intel Corporation
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

if(SPARSE_LIB_USE_VTUNE)
  if( UNIX )
    if( CMAKE_VTUNE_HOME )
      set( VTUNE_HOME ${CMAKE_VTUNE_HOME} )
    elseif( DEFINED ENV{CMAKE_VTUNE_HOME} )
      set( VTUNE_HOME $ENV{CMAKE_VTUNE_HOME} )
    else()
      set( VTUNE_HOME /opt/intel/oneapi/vtune/latest )
    endif()

    message( STATUS "Refer to ${VTUNE_HOME} for vtune" )
    set( arch "64" )

    find_path( VTUNE_INCLUDE ittnotify.h PATHS ${VTUNE_HOME}/include )
    find_library( VTUNE_LIBRARY libittnotify.a PATHS ${VTUNE_HOME}/lib${arch}/ )

    if( NOT VTUNE_INCLUDE MATCHES NOTFOUND )
      if( NOT VTUNE_LIBRARY MATCHES NOTFOUND )
        set( VTUNE_FOUND TRUE )
        message( STATUS "ITT was found here ${VTUNE_HOME}" )

        get_filename_component( VTUNE_LIBRARY_PATH ${VTUNE_LIBRARY} PATH )

        set(EXTERNAL_INCLUDE_PAYH "${VTUNE_HOME}/include ${EXTERNAL_INCLUDE_PAYH}")
        set(EXTERNAL_LIBRARY_PAYH "${VTUNE_HOME}/lib${arch}/libittnotify.a ${EXTERNAL_LIBRARY_PAYH}")

        set(CMAKE_C_FLAGS "-DSPARSE_LIB_USE_VTUNE ${CMAKE_C_FLAGS}" )
        set(CMAKE_CXX_FLAGS "-DSPARSE_LIB_USE_VTUNE ${CMAKE_CXX_FLAGS}" )
        set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}" ) # better to provide debug symbol to vtune

        set( ITT_CFLAGS "-I${VTUNE_INCLUDE} -DITT_SUPPORT" )
        set( ITT_LIBRARY_DIRS "${VTUNE_LIBRARY_PATH}" )

        set( ITT_LIBS "" )
        list( APPEND ITT_LIBS
          ittnotify
        )
      endif()
    endif()
  else()
    message( STATUS "vtune is supported only for linux!" )
  endif()

endif()
