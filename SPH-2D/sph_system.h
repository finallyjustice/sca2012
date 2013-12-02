#ifndef __SMOKESYSTEM_H__
#define __SMOKESYSTEM_H__

#include "sph_header.h"
#include "sph_param.h"

class Particle
{
public:
	float2 pos;
	float2 vel;
	float2 ev;
	float2 acc;

	float dens;
	float pres;

	float energy;
	uint level;

	float surface_normal;
};

class SPHSystem
{
public:
	uint m_num_particle;
	uint m_max_particle;

	float m_world_width;
	float m_world_depth;

	float m_large_mass;
	float m_small_mass;

	float m_large_kernel;
	float m_small_kernel;

	float m_large_kernel2;
	float m_small_kernel2;

	float m_large_kernel6;
	float m_small_kernel6;

	float m_large_radius;
	float m_small_radius;

	float m_large_poly6;
	float m_small_poly6;

	float m_large_spiky;
	float m_small_spiky;

	float m_large_visco;
	float m_small_visco;

	float m_large_grad_poly6;
	float m_small_grad_poly6;

	float m_large_lapc_poly6;
	float m_small_lapc_poly6;

	float2 m_gravity;
	float m_wall_damping;
	float m_rest_density;
	float m_gas_constant;
	float m_viscosity;
	float m_time_step;
	float m_surface_normal;
	uint m_sys_running;

	float m_cell_size;
	uint m_row_cell;
	uint m_col_cell;
	uint m_total_cell;

	float m_co_surface;
	float m_co_energy;

	float m_thresh_split;
	float m_thresh_merge;

	Particle *m_host_mem;
	Particle *m_dev_mem;

	uint *m_dev_hash;	
	uint *m_dev_index;
	uint *m_cell_start;
	uint *m_cell_end;

	float2 *m_trans_vel;
	uint *m_dev_status;
	uint *m_dev_split;
	uint *m_dev_merge;
	uint *m_dev_split_index;

	uint m_enable_split;
	uint m_enable_merge;

	SysParam *m_host_param;

public:
	SPHSystem();
	void add_init_particle();
	void add_particle(float2 pos, float2 vel, uint level);
	void animate();
	void add_extern_force(float2 pos, float2 force);
};

#endif
