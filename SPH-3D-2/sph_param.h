#ifndef __SPHPARAM_H__
#define __SPHPARAM_H__

#include "sph_object.h"
#include "sph_header.h"

/* This class is used to pass the parameters from host memory to device memory(constant mem) */
class SysParam
{
public:
	uint num_particle;
	uint max_particle;

	float world_width;
	float world_height;
	float world_length;

	float large_radius;
	float small_radius;

	float large_kernel;
	float small_kernel;

	float large_mass;
	float small_mass;

	float cell_size;
	uint row_cell; 
	uint col_cell;
	uint len_cell;
	uint total_cell;

	float3 gravity;
	float wall_damping;
	float rest_density;
	float gas_constant;
	float time_step;
	float viscosity;
	float surface_tension;

	float co_surface;
	float co_energy;

	float split_criteria;
	float merge_criteria;

	float large_poly6;
	float large_spiky;
	float large_visco;

	float large_grad_poly6;
	float large_lapc_poly6;

	float small_poly6;
	float small_spiky;
	float small_visco;

	float small_grad_poly6;
	float small_lapc_poly6;

	float large_kernel_2;
	float large_kernel_6;

	float small_kernel_2;
	float small_kernel_6;

	float large_poly6_radius;
	float small_poly6_radius;

	uint dens_mul;
	float den_size;
	uint row_dens;
	uint col_dens;
	uint len_dens;
	uint tot_dens;

	Sphere sphere1;
	Sphere sphere2;
	Sphere sphere3;
	Sphere sphere4;

	float split_energy;
	float merge_energy;

	bool force_waiting;
	bool use_cylinder;
};

#endif
