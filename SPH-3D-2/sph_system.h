#ifndef __SMOKESYSTEM_H__
#define __SMOKESYSTEM_H__

#include "sph_header.h"
#include "sph_param.h"
#include "sph_object.h"

class Particle
{
public:
	float3 pos;
	float3 vel;
	float3 ev;
	float3 acc;

	float dens;
	float press;

	float energy;
	float surface;
	float tension;

	uint level;
};

class SPHSystem
{
public:
	Particle *m_host_mem;
	Particle *m_dev_mem;

	uint m_num_particle;
	uint m_max_particle;

	float m_world_width;
	float m_world_height;
	float m_world_length;

	float m_large_radius;
	float m_small_radius;

	float m_large_kernel;
	float m_small_kernel;

	float m_large_mass;
	float m_small_mass;

	float m_cell_size;
	uint m_row_cell;
	uint m_col_cell;
	uint m_len_cell;
	uint m_total_cell;

	uint *m_dev_hash;	
	uint *m_dev_index;
	uint *m_cell_start;
	uint *m_cell_end;

	uint *m_dev_status;
	float3 *m_trans_vel;

	uint *m_dev_split;
	uint *m_dev_split_index;
	uint *m_dev_merge;
	 
	float3 m_gravity;
	float m_wall_damping;
	float m_rest_density;
	float m_gas_constant;
	float m_time_step;
	float m_viscosity;
	float m_surface_tension;

	float m_co_surface;
	float m_co_energy;

	float m_split_criteria;
	float m_merge_criteria;

	float m_large_poly6;
	float m_large_spiky;
	float m_large_visco;

	float m_large_grad_poly6;
	float m_large_lapc_poly6;

	float m_small_poly6;
	float m_small_spiky;
	float m_small_visco;

	float m_small_grad_poly6;
	float m_small_lapc_poly6;

	float m_large_kernel_2;
	float m_large_kernel_6;

	float m_small_kernel_2;
	float m_small_kernel_6;

	float m_large_poly6_radius;
	float m_small_poly6_radius;

	uint m_dens_mul;
	float m_den_size;
	uint m_row_dens;
	uint m_col_dens;
	uint m_len_dens;
	uint m_tot_dens;
	float *m_host_dens;
	float *m_dev_dens;

	float3 *m_host_dens_pos;
	float3 *m_dev_dens_pos;
	float3 *m_host_dens_normal;
	float3 *m_dev_dens_normal;

	uint m_num_lines;
	float3 *m_host_line0;
	float3 *m_host_line1;

	uint m_num_triangle;
	float3 *m_host_triangle0;
	float3 *m_host_triangle1;
	float3 *m_host_triangle2;
	float3 *m_host_triangle_normal0;
	float3 *m_host_triangle_normal1;
	float3 *m_host_triangle_normal2;

	Sphere m_sphere1;
	Sphere m_sphere2;
	Sphere m_sphere3;
	Sphere m_sphere4;

	SysParam *m_host_param;

	uint m_sys_running;

	uint m_trans_once;
	uint m_use_split;
	uint m_use_merge;

	uint m_disp_mode;

public:
	SPHSystem();
	~SPHSystem();
	void add_box_particle();
	void add_new_particle();
	void animation();
	void add_particle(float3 pos, float3 vel, uint level);

	void marching_cube();
	void marching_cube_cell(uint count_x, uint count_y, uint count_z);
};

#endif
