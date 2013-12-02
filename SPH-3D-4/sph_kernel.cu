#ifndef __SPHKERNEL_CU__
#define __SPHKERNEL_CU__

#include "sph_header.h"
#include "sph_system.h"
#include "sph_param.h"

//parameter container on dev memory. 
//SPHSyetm->m_host_param is on host memory
__constant__ SysParam dev_param;

#endif