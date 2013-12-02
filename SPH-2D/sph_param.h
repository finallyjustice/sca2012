#ifndef __SPHPARAM_H__
#define __SPHPARAM_H__

#include "sph_header.h"

class SysParam
{
public:
	uint num_particle;
	uint max_particle;

	float world_width;
	float world_depth;

	float large_mass;
	float small_mass;

	float large_kernel;
	float small_kernel;

	float large_kernel2;
	float small_kernel2;

	float large_kernel6;
	float small_kernel6;

	float large_radius;
	float small_radius;

	float large_poly6;
	float small_poly6;

	float large_spiky;
	float small_spiky;

	float large_visco;
	float small_visco;

	float large_grad_poly6;
	float small_grad_poly6;

	float large_lapc_poly6;
	float small_lapc_poly6;

	float2 gravity;
	float wall_damping;
	float rest_density;
	float gas_constant;
	float viscosity;
	float time_step;
	float surface_normal;

	float cell_size;
	uint row_cell;
	uint col_cell;
	uint total_cell;

	float co_surface;
	float co_energy;

	float thresh_split;
	float thresh_merge;

	bool force_waiting; 
	float2 extern_force;
	float2 extern_click;
	uint2 extern_pos;
	float extern_len; 
	float extern_coe;
};

#endif
