#include "sph_system.h"
#include "sph_system.cuh"
#include <cutil_math.h>

SPHSystem::SPHSystem()
{
	m_num_particle=0;
	m_max_particle=100000;

	m_world_width=32.0f;
	m_world_depth=32.0f;

	m_large_mass=1.0f;
	m_small_mass=0.5f;

	m_large_kernel=1.0f;
	m_small_kernel=0.9f;

	m_large_kernel2=m_large_kernel*m_large_kernel;
	m_small_kernel2=m_small_kernel*m_small_kernel;

	m_large_kernel6=pow(m_large_kernel, 6);
	m_small_kernel6=pow(m_small_kernel, 6);

	m_large_radius=1.0f;
	m_small_radius=0.8f;

	m_large_poly6=315.0f/(64.0f * PI * pow(m_large_kernel, 9));
	m_small_poly6=315.0f/(64.0f * PI * pow(m_small_kernel, 9));;

	m_large_spiky=-45.0f/(PI * pow(m_large_kernel, 6));
	m_small_spiky=-45.0f/(PI * pow(m_small_kernel, 6));

	m_large_visco=45.0f/(PI * pow(m_large_kernel, 6));
	m_small_visco=45.0f/(PI * pow(m_small_kernel, 6));

	m_large_grad_poly6=-945/(32 * PI * pow(m_large_kernel, 9));
	m_small_grad_poly6=-945/(32 * PI * pow(m_small_kernel, 9));

	m_large_lapc_poly6=-945/(8 * PI * pow(m_large_kernel, 9));
	m_small_lapc_poly6=-945/(8 * PI * pow(m_small_kernel, 9));

	m_gravity=make_float2(0.0f, -4.5f);
	m_wall_damping=-0.6f;
	m_rest_density=2.0f;
	m_gas_constant=1.0f;
	m_viscosity=2.0f;
	m_time_step=0.1f;
	m_surface_normal=0.5f;
	m_sys_running=0;

	m_cell_size=m_large_kernel;
	m_row_cell=(uint)ceil(m_world_width/m_cell_size);;
	m_col_cell=(uint)ceil(m_world_depth/m_cell_size);;
	m_total_cell=m_row_cell*m_col_cell;

	m_world_width=m_row_cell*m_cell_size;
	m_world_depth=m_col_cell*m_cell_size;

	m_co_surface=0.8f;
	m_co_energy=1.0f;

	m_thresh_split=0.7f;
	m_thresh_merge=0.04f;

	m_host_mem=(Particle *)malloc(sizeof(Particle)*m_max_particle);
	alloc_array((void**)&(m_dev_mem), sizeof(Particle)*m_max_particle);

	alloc_array((void**)&m_dev_hash, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_dev_index, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_cell_start, sizeof(uint)*m_total_cell);
	alloc_array((void**)&m_cell_end, sizeof(uint)*m_total_cell);

	alloc_array((void**)&m_trans_vel, sizeof(float2)*m_max_particle);
	alloc_array((void**)&m_dev_status, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_dev_split, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_dev_merge, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_dev_split_index, sizeof(uint)*m_max_particle);

	m_enable_split=0;
	m_enable_merge=0;

	m_host_param=new SysParam();

	m_host_param->num_particle=m_num_particle;
	m_host_param->max_particle=m_max_particle;

	m_host_param->world_width=m_world_width;
	m_host_param->world_depth=m_world_depth;

	m_host_param->large_mass=m_large_mass;
	m_host_param->small_mass=m_small_mass;

	m_host_param->large_kernel=m_large_kernel;
	m_host_param->small_kernel=m_small_kernel;

	m_host_param->large_kernel2=m_large_kernel2;
	m_host_param->small_kernel2=m_small_kernel2;

	m_host_param->large_kernel6=m_large_kernel6;
	m_host_param->small_kernel6=m_small_kernel6;

	m_host_param->large_radius=m_large_radius;
	m_host_param->small_radius=m_small_radius;

	m_host_param->large_poly6=m_large_poly6;
	m_host_param->small_poly6=m_small_poly6;

	m_host_param->large_spiky=m_large_spiky;
	m_host_param->small_spiky=m_small_spiky;

	m_host_param->large_visco=m_large_visco;
	m_host_param->small_visco=m_small_visco;

	m_host_param->large_grad_poly6=m_large_grad_poly6;
	m_host_param->small_grad_poly6=m_small_grad_poly6;

	m_host_param->large_lapc_poly6=m_large_lapc_poly6;
	m_host_param->small_lapc_poly6=m_small_lapc_poly6;

	m_host_param->gravity=m_gravity;
	m_host_param->wall_damping=m_wall_damping;
	m_host_param->rest_density=m_rest_density;
	m_host_param->gas_constant=m_gas_constant;
	m_host_param->viscosity=m_viscosity;
	m_host_param->time_step=m_time_step;
	m_host_param->surface_normal=m_surface_normal;

	m_host_param->cell_size=m_cell_size;
	m_host_param->row_cell=m_row_cell;
	m_host_param->col_cell=m_col_cell;
	m_host_param->total_cell=m_total_cell;

	m_host_param->co_surface=m_co_surface;
	m_host_param->co_energy=m_co_energy;

	m_host_param->thresh_split=m_thresh_split;
	m_host_param->thresh_merge=m_thresh_merge;

	m_host_param->force_waiting=false;
	m_host_param->extern_force=make_float2(0.0f, 0.0f);
	m_host_param->extern_click=make_float2(0.0f, 0.0f);
	m_host_param->extern_pos=make_uint2(0, 0);
	m_host_param->extern_len=2.0f;
	m_host_param->extern_coe=1.0f;
}

void SPHSystem::add_init_particle()
{
	float2 pos;
	float2 vel=make_float2(0.0f, 0.0f);

	for(pos.x=m_large_radius/2.0f; pos.x<m_world_width/2; pos.x+=m_large_kernel*0.5f)
	{
		for(pos.y=m_large_radius/2.0f; m_world_depth-pos.y>=m_large_radius/2.0f; pos.y+=m_large_kernel*0.5f)
		{
			add_particle(pos, vel, 1);
		}
	}

	copy_array(m_dev_mem, m_host_mem, sizeof(Particle)*m_num_particle, CUDA_HOST_TO_DEV);
}

void SPHSystem::add_particle(float2 pos, float2 vel, uint level)
{
	Particle *p=&(m_host_mem[m_num_particle]);

	p->pos=pos;
	p->vel=vel;
	p->ev=make_float2(0.0f);
	p->acc=make_float2(0.0f);
	p->dens=0.0f;
	p->pres=0.0f;
	p->level=level;

	m_num_particle++;
}

void SPHSystem::add_extern_force(float2 pos, float2 force)
{
	uint cell_x=(uint)floor(pos.x * m_row_cell);
	uint cell_y=(uint)floor(pos.y * m_col_cell);

	m_host_param->extern_click=make_float2(pos.x*m_world_width, pos.y*m_world_depth);
	m_host_param->extern_pos=make_uint2(cell_x, cell_y);
	m_host_param->extern_force=make_float2(force.x, force.y);
	m_host_param->force_waiting=true;
}

void SPHSystem::animate()
{
	if(m_sys_running == 0)
	{
		return;
	}

	m_host_param->num_particle=m_num_particle;
	set_parameters(m_host_param);

	 calc_hash(m_dev_hash,
			m_dev_index,
			m_dev_mem,
			m_num_particle);

	sort_particles(m_dev_hash, m_dev_index, m_num_particle);

	find_start_end(m_cell_start,
					m_cell_end,
					m_dev_hash,
					m_dev_index,
					m_num_particle,
					m_total_cell);

	compute(m_dev_mem,
			m_dev_hash,
			m_dev_index,
			m_cell_start,
			m_cell_end,
			m_num_particle,
			m_total_cell);

	integrate_velocity(m_dev_mem, m_num_particle);

	compute_energy(m_dev_mem,
				m_dev_hash,
				m_dev_index,
				m_cell_start,
				m_cell_end,
				m_num_particle,
				m_total_cell,
				m_trans_vel);

	if(m_enable_split == 1)
	{
		m_num_particle=split_particle(m_dev_mem,
									m_dev_hash,
									m_dev_split_index,
									m_cell_start,
									m_cell_end,
									m_num_particle,
									m_total_cell,
									m_dev_split,
									m_dev_status);
	}

	if(m_enable_merge == 1)
	{
		merge(m_dev_mem,
			m_dev_hash,
			m_dev_index,
			m_cell_start,
			m_cell_end,
			m_num_particle,
			m_total_cell,
			m_dev_merge,
			m_dev_status);

		m_num_particle=rearrange(m_dev_mem, m_dev_status, m_num_particle);
	}

	copy_array(m_host_mem, m_dev_mem, sizeof(Particle)*m_num_particle, CUDA_DEV_TO_HOST);

	m_host_param->force_waiting=false;
}