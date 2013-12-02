#ifndef __SPHSYSTEM_CU__
#define __SPHSYSTEM_CU__

#include "sph_header.h"
#include "sph_param.h"

#define CUDA_HOST_TO_DEV	1
#define CUDA_DEV_TO_HOST	2
#define CUDA_DEV_TO_DEV		3

void set_parameters(SysParam *host_param);
void alloc_array(void **dev_ptr, size_t size);
void free_array(void *dev_ptr);
void copy_array(void *ptr_a, void *ptr_b, size_t size, int type);
void compute_grid_size(uint num_particle, uint block_size, uint &num_blocks, uint &num_threads);

void calc_hash(uint*  dev_hash,
              uint*  dev_index,
              Particle *dev_mem,
              uint    num_particle);

void sort_particles(uint *dev_hash, uint *dev_index, uint num_particle);

void find_start_end(uint *cell_start,
					uint *cell_end,
					uint *dev_hash,
					uint *dev_index,
					uint num_particle,
					uint num_cell);

void integrate_velocity(Particle *dev_mem, uint num_particle);

void compute(Particle *dev_mem,
				uint *dev_hash,
				uint *dev_index,
				uint *cell_start,
				uint *cell_end,
				uint num_particle,
				uint total_cell);

void compute_energy(Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					float3 *trans_vel);

uint adaptive_particle(Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					uint *dev_split,
					uint *dev_status);

void merge(Particle *dev_mem,
			uint *dev_hash,
			uint *dev_index,
			uint *cell_start,
			uint *cell_end,
			uint num_particle,
			uint total_cell,
			uint *cell_merge,
			uint *dev_status);

uint rearrange(Particle *dev_mem, uint *dev_status, uint num_particle);

void integrate_dens(Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					uint row_dens,
					uint col_dens,
					uint len_dens,
					uint tot_dens,
					float den_size,
					float *dev_dens,
					float3 *dev_dens_pos);

void integrate_normal(float *dev_dens,
					float3 *dev_dens_pos,
					float3 *dev_dens_normal,
					uint tot_dens);

#endif