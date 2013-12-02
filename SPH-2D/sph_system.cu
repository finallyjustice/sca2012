#include "sph_header.h"
#include "sph_param.h"
#include "sph_math.h"
#include "sph_kernel.cu"
#include <cutil_math.h>

#define EXP 2.718281f
#define EXT2 1e-12f
#define EXT 1e-6f

__device__ 
uint2 calc_grid_pos(float x, float y)
{
    uint2 cell_pos;

    cell_pos.x=(uint)floor(x / dev_param.cell_size);
    cell_pos.y=(uint)floor(y / dev_param.cell_size);

    return cell_pos;
}

__device__ 
uint calc_grid_hash(uint2 cell_pos)
{  
    return cell_pos.y*dev_param.row_cell+cell_pos.x;
}

void set_parameters(SysParam *host_param)
{
    cudaMemcpyToSymbol((char *)&dev_param, host_param, sizeof(SysParam));
}

void alloc_array(void **dev_ptr, size_t size)
{
    cudaMalloc(dev_ptr, size);
}

void free_array(void *dev_ptr)
{
    cudaFree(dev_ptr);
}

void copy_array(void *ptr_a, void *ptr_b, size_t size, int type)
{
	if(type == 1)
	{
		cudaMemcpy(ptr_a, ptr_b, size, cudaMemcpyHostToDevice);
		return;
	}

	if(type == 2)
	{
		cudaMemcpy(ptr_a, ptr_b, size, cudaMemcpyDeviceToHost);
		return;
	}

	if(type == 3)
	{	
		cudaMemcpy(ptr_a, ptr_b, size, cudaMemcpyDeviceToDevice);
		return;
	}
	
	return;
}

void compute_grid_size(uint num_particle, uint block_size, uint &num_blocks, uint &num_threads)
{
    num_threads=min(block_size, num_particle);
    num_blocks=iDivUp(num_particle, num_threads);
}

__global__
void calcHashD(uint *dev_hash,		
               uint *dev_index,		
               Particle *dev_mem,
               uint num_particle)
{
    uint index=blockIdx.x*blockDim.x+threadIdx.x;

    if(index >= num_particle) 
	{
		return;
	}

    uint2 grid_pos=calc_grid_pos(dev_mem[index].pos.x, dev_mem[index].pos.y);
    uint hash=calc_grid_hash(grid_pos);

    dev_hash[index]=hash;
    dev_index[index]=index;
}

void calc_hash(uint*  dev_hash,
              uint*  dev_index,
              Particle *dev_mem,
              uint    num_particle)
{
	if(num_particle == 0)
	{
		return;
	}

    uint num_threads;
	uint num_blocks;

    compute_grid_size(num_particle, 256, num_blocks, num_threads);

	calcHashD<<< num_blocks, num_threads >>>(dev_hash,
                                           dev_index,
                                           dev_mem,
                                           num_particle);
}

void sort_particles(uint *dev_hash, uint *dev_index, uint num_particle)
{
	if(num_particle == 0)
	{
		return;
	}

    thrust::sort_by_key(thrust::device_ptr<uint>(dev_hash),
                        thrust::device_ptr<uint>(dev_hash + num_particle),
                        thrust::device_ptr<uint>(dev_index));
}

__global__
void find_start_end_kernel(uint *cell_start,
					uint *cell_end,	
					uint *dev_hash,	
					uint *dev_index,
					uint num_particle)
{
	extern __shared__ uint shared_hash[];    
    uint index=blockIdx.x*blockDim.x+threadIdx.x;
	
    uint hash;

    if(index < num_particle) 
	{
        hash=dev_hash[index];
	    shared_hash[threadIdx.x+1]=hash;

	    if(index > 0 && threadIdx.x == 0)
	    {
		    shared_hash[0]=dev_hash[index-1];
	    }
	}

	__syncthreads();
	
	if(index < num_particle) 
	{
		if(index == 0 || hash != shared_hash[threadIdx.x])
	    {
		    cell_start[hash]=index;

            if(index > 0)
			{
                cell_end[shared_hash[threadIdx.x]]=index;
			}
	    }

        if (index == num_particle-1)
        {
            cell_end[hash]=index+1;
        }
	}
}

void find_start_end(uint *cell_start,
					uint *cell_end,
					uint *dev_hash,
					uint *dev_index,
					uint num_particle,
					uint num_cell)
{
	if(num_particle == 0)
	{
		return;
	}

    uint num_thread;
	uint num_block;
    compute_grid_size(num_particle, 256, num_block, num_thread);

    cudaMemset(cell_start, 0xffffffff, num_cell*sizeof(int));
	cudaMemset(cell_end, 0x0, num_cell*sizeof(int));

    uint smemSize=sizeof(int)*(num_thread+1);

    find_start_end_kernel<<< num_block, num_thread, smemSize>>>(
													cell_start,
													cell_end,
													dev_hash,
													dev_index,
													num_particle);
}

__global__
void integrate_velocity_kernel(Particle* dev_mem, uint num_particle)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= num_particle)
	{
		return;
	}

	Particle *p=&(dev_mem[index]);
	float radius;

	if(p->level == 1)
	{
		radius=dev_param.large_radius;
	}

	if(p->level == 2)
	{
		radius=dev_param.small_radius;
	}
	
	p->acc=p->acc+dev_param.gravity*dev_param.time_step;
	p->acc=p->acc/p->dens;

	p->vel=p->vel+p->acc*dev_param.time_step;	
	p->ev=(p->ev+p->vel)/2;

	p->pos=p->pos+p->vel*dev_param.time_step;

	if(p->pos.x >= dev_param.world_width-radius/2)
	{
		p->vel.x=p->vel.x*dev_param.wall_damping;
		p->pos.x=dev_param.world_width-radius/2;
	}

	if(p->pos.x < radius/2)
	{
		p->vel.x=p->vel.x*dev_param.wall_damping;
		p->pos.x=radius/2;
	}

	if(p->pos.y >= dev_param.world_depth-radius/2)
	{
		p->vel.y=p->vel.y*dev_param.wall_damping;
		p->pos.y=dev_param.world_depth-radius/2;
	}

	if(p->pos.y < radius/2)
	{
		p->vel.y=p->vel.y*dev_param.wall_damping;
		p->pos.y=radius/2;
	}
}

void integrate_velocity(Particle *dev_mem, uint num_particle)
{
	if(num_particle == 0)
	{
		return;
	}

	uint num_thread;
	uint num_block;
    compute_grid_size(num_particle, 256, num_block, num_thread);

	integrate_velocity_kernel<<< num_block, num_thread >>>(dev_mem, num_particle);
}

__device__
float compute_cell_density(uint index,
					uint2 neighbor,
					Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell)
{
	uint grid_hash=calc_grid_hash(neighbor);
	uint start_index=cell_start[grid_hash];

	float total_cell_density=0.0f;

	float2 rel_pos;
	float r2;

	Particle *p=&(dev_mem[index]);
	Particle *np;
	uint neighbor_index;

	float i_kernel2;
	float i_kernel6;
	float i_poly6;
	float i_mass;

	float j_mass;

	if(p->level == 1)
	{
		i_kernel2=dev_param.large_kernel2;
		i_kernel6=dev_param.large_kernel6;
		i_poly6=dev_param.large_poly6;
		i_mass=dev_param.large_mass;
	}
	if(p->level == 2)
	{
		i_kernel2=dev_param.small_kernel2;
		i_kernel6=dev_param.small_kernel6;
		i_poly6=dev_param.small_poly6;
		i_mass=dev_param.small_mass;
	}

	if(start_index != 0xffffffff)
	{        
        uint end_index=cell_end[grid_hash];

        for(uint count_index=start_index; count_index<end_index; count_index++) 
		{
			neighbor_index=dev_index[count_index];
			np=&(dev_mem[neighbor_index]);

            if(neighbor_index != index)
			{
				rel_pos=np->pos-p->pos;
				r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y;

				if(r2 < EXT2)
				{
					continue;
				}

				if(r2 < i_kernel2)
				{
					if(np->level == 1)
					{
						j_mass=dev_param.large_mass;
					}
					if(np->level == 2)
					{
						j_mass=dev_param.small_mass;
					}

					total_cell_density=total_cell_density + j_mass * i_poly6 * pow(i_kernel2-r2, 3);
				}
            }
			if(neighbor_index == index)
			{
				total_cell_density=total_cell_density + i_mass * i_poly6 * i_kernel6;
			}
        }
	}

	return total_cell_density;
}

__global__
void compute_density_kernel(Particle *dev_mem,
							uint *dev_hash,
							uint *dev_index,
							uint *cell_start,
							uint *cell_end,
							uint num_particle,
							uint total_cell)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= num_particle)
	{
		return;
	}

	uint2 cell_pos=calc_grid_pos(dev_mem[index].pos.x, dev_mem[index].pos.y);
	
	uint count;
	uint2 neighbor_pos[9];
	uint neighbor_status[9];

	for(count=0; count<9; count++)
	{
		neighbor_status[count]=1;
	}

	neighbor_pos[0].x=cell_pos.x;
	neighbor_pos[0].y=cell_pos.y;

	neighbor_pos[1].x=cell_pos.x-1;
	neighbor_pos[1].y=cell_pos.y-1;

	neighbor_pos[2].x=cell_pos.x-1;
	neighbor_pos[2].y=cell_pos.y;

	neighbor_pos[3].x=cell_pos.x;
	neighbor_pos[3].y=cell_pos.y-1;

	neighbor_pos[4].x=cell_pos.x+1;
	neighbor_pos[4].y=cell_pos.y+1;

	neighbor_pos[5].x=cell_pos.x+1;
	neighbor_pos[5].y=cell_pos.y;

	neighbor_pos[6].x=cell_pos.x;
	neighbor_pos[6].y=cell_pos.y+1;

	neighbor_pos[7].x=cell_pos.x+1;
	neighbor_pos[7].y=cell_pos.y-1;

	neighbor_pos[8].x=cell_pos.x-1;
	neighbor_pos[8].y=cell_pos.y+1;

	if(cell_pos.x == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[2]=0;
		neighbor_status[8]=0;
	}

	if(cell_pos.x == dev_param.row_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[5]=0;
		neighbor_status[7]=0;
	}

	if(cell_pos.y == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[3]=0;
		neighbor_status[7]=0;
	}

	if(cell_pos.y == dev_param.col_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[6]=0;
		neighbor_status[8]=0;
	}

	float total_density=0.0f;

	for(count=0; count<9; count++)
	{
		if(neighbor_status[count] == 0)
		{
			continue;
		}

		total_density=total_density+compute_cell_density(index,
													neighbor_pos[count],
													dev_mem,
													dev_hash,
													dev_index,
													cell_start,
													cell_end,
													num_particle,
													total_cell);
	}
	
	dev_mem[index].dens=total_density;
	dev_mem[index].pres=(pow(dev_mem[index].dens / dev_param.rest_density, 3) - 1) * dev_param.gas_constant;
}

__device__
float2 compute_cell_force(uint index,
					uint2 neighbor,
					Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					float3 *surface_color)
{
	uint grid_hash=calc_grid_hash(neighbor);
	uint start_index=cell_start[grid_hash];
	
	float2 total_cell_force=make_float2(0.0f);

	float i_kernel;
	float i_mass;
	float i_kernel_2;
	float i_kernel_6;
	float i_poly6_value;
	float i_spiky_value;
	float i_visco_value;
	float i_grad_poly6;
	float i_lapc_poly6;

	float j_kernel;
	float j_mass;
	float j_kernel_2;
	float j_kernel_6;
	float j_spiky_value;
	float j_visco_value;

	float i_press_kernel;
	float i_visco_kernel;

	float j_press_kernel;
	float j_visco_kernel;

	float i_kernel_r;
	float j_kernel_r;

	float iV;
	float jV;

	uint neighbor_index;

	Particle *p=&(dev_mem[index]);
	Particle *np;

	float2 rel_pos;
	float r2;
	float r;
	float temp_force;
	float2 rel_vel;

	if(p->level == 1)
	{
		i_kernel=dev_param.large_kernel;
		i_mass=dev_param.large_mass;
		i_kernel_2=dev_param.large_kernel2;
		i_kernel_6=dev_param.large_kernel6;
		i_poly6_value=dev_param.large_poly6;
		i_spiky_value=dev_param.large_spiky;
		i_visco_value=dev_param.large_visco;
		i_grad_poly6=dev_param.large_grad_poly6;
		i_lapc_poly6=dev_param.large_lapc_poly6;
	}

	if(p->level == 2)
	{
		i_kernel=dev_param.small_kernel;
		i_mass=dev_param.small_mass;
		i_kernel_2=dev_param.small_kernel2;
		i_kernel_6=dev_param.small_kernel6;
		i_poly6_value=dev_param.small_poly6;
		i_spiky_value=dev_param.small_spiky;
		i_visco_value=dev_param.small_visco;
		i_grad_poly6=dev_param.small_grad_poly6;
		i_lapc_poly6=dev_param.small_lapc_poly6;
	}

	if(start_index != 0xffffffff)
	{        
		uint end_index=cell_end[grid_hash];

        for(uint count_index=start_index; count_index<end_index; count_index++) 
		{
			neighbor_index=dev_index[count_index];

			np=&(dev_mem[neighbor_index]);
			
            if(neighbor_index != index)
			{
				rel_pos=p->pos-np->pos;
				r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y;

				if(np->level == 1)
				{
					j_kernel=dev_param.large_kernel;
					j_mass=dev_param.large_mass;
					j_kernel_2=dev_param.large_kernel2;
					j_kernel_6=dev_param.large_kernel6;
					j_spiky_value=dev_param.large_spiky;
					j_visco_value=dev_param.large_visco;
				}

				if(np->level == 2)
				{
					j_kernel=dev_param.small_kernel;
					j_mass=dev_param.small_mass;
					j_kernel_2=dev_param.small_kernel2;
					j_kernel_6=dev_param.small_kernel6;
					j_spiky_value=dev_param.small_spiky;
					j_visco_value=dev_param.small_visco;
				}

				float max_kernel_2=i_kernel_2>j_kernel_2?i_kernel_2:j_kernel_2;

				if(r2 < max_kernel_2)
				{
					if(r2 < 0.0001f)
					{
						continue;
					}
					else
					{
						r=sqrt(r2);
					}

					iV=i_mass/p->dens;
					jV=j_mass/np->dens;

					i_kernel_r=i_kernel-r;
					j_kernel_r=j_kernel-r;

					if(i_kernel_r > 0)
					{
						i_press_kernel=i_spiky_value * i_kernel_r * i_kernel_r / i_kernel_6;
						i_visco_kernel=i_visco_value/i_kernel_6*(i_kernel_r);

						float temp=(-1) * i_grad_poly6 * jV * pow(i_kernel_2-r2, 2);
						surface_color->x += temp * rel_pos.x;
						surface_color->y += temp * rel_pos.y;

						//surface_color->z += i_lapc_poly6 * jV * (i_kernel_2-r2) * (3*i_kernel_2-7*r2);
						surface_color->z += i_lapc_poly6 * jV * (i_kernel_2-r2) * (r2-3/4*(i_kernel_2-r2));
					}
					else
					{
						i_press_kernel=0.0f;
						i_visco_kernel=0.0f;
					}

					if(j_kernel_r > 0)
					{
						j_press_kernel=j_spiky_value * j_kernel_r * j_kernel_r / j_kernel_6;
						j_visco_kernel=j_visco_value/j_kernel_6*(j_kernel_r);
					}
					else
					{
						j_press_kernel=0.0f;
						j_visco_kernel=0.0f;
					}

					temp_force=i_mass*j_mass * (p->pres/(p->dens*p->dens)+np->pres/(np->dens*np->dens)) * (i_press_kernel+j_press_kernel)/2;
					total_cell_force=total_cell_force-rel_pos*temp_force/r;

					rel_vel=np->ev-p->ev;

					temp_force=(iV*jV) * dev_param.viscosity * (i_visco_kernel+j_visco_kernel)/2;
					total_cell_force=total_cell_force + rel_vel*temp_force; 
				}
            }
			if(neighbor_index == index)
			{
				//surface_color->z += i_lapc_poly6 * i_mass / p->dens * i_kernel_2 * (3*i_kernel_2);
				surface_color->z += i_lapc_poly6 * j_mass / np->dens * (i_kernel_2) * (0-3/4*(i_kernel_2));
			}
        }
	}

	return total_cell_force;
}

__global__
void compute_force_kernel(Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= num_particle)
	{
		return;
	}

	uint2 cell_pos=calc_grid_pos(dev_mem[index].pos.x, dev_mem[index].pos.y);

	uint count;
	uint2 neighbor_pos[9];
	uint neighbor_status[9];

	for(count=0; count<9; count++)
	{
		neighbor_status[count]=1;
	}

	neighbor_pos[0].x=cell_pos.x;
	neighbor_pos[0].y=cell_pos.y;

	neighbor_pos[1].x=cell_pos.x-1;
	neighbor_pos[1].y=cell_pos.y-1;

	neighbor_pos[2].x=cell_pos.x-1;
	neighbor_pos[2].y=cell_pos.y;

	neighbor_pos[3].x=cell_pos.x;
	neighbor_pos[3].y=cell_pos.y-1;

	neighbor_pos[4].x=cell_pos.x+1;
	neighbor_pos[4].y=cell_pos.y+1;

	neighbor_pos[5].x=cell_pos.x+1;
	neighbor_pos[5].y=cell_pos.y;

	neighbor_pos[6].x=cell_pos.x;
	neighbor_pos[6].y=cell_pos.y+1;

	neighbor_pos[7].x=cell_pos.x+1;
	neighbor_pos[7].y=cell_pos.y-1;

	neighbor_pos[8].x=cell_pos.x-1;
	neighbor_pos[8].y=cell_pos.y+1;

	if(cell_pos.x == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[2]=0;
		neighbor_status[8]=0;
	}

	if(cell_pos.x == dev_param.row_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[5]=0;
		neighbor_status[7]=0;
	}

	if(cell_pos.y == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[3]=0;
		neighbor_status[7]=0;
	}

	if(cell_pos.y == dev_param.col_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[6]=0;
		neighbor_status[8]=0;
	}

	float2 total_force=make_float2(0.0f);
	float3 surface_color=make_float3(0.0f);;
	
	for(count=0; count<9; count++)
	{
		if(neighbor_status[count] == 0)
		{
			continue;
		}

		total_force=total_force+compute_cell_force(index,
												neighbor_pos[count],
												dev_mem,
												dev_hash,
												dev_index,
												cell_start,
												cell_end,
												num_particle,
												total_cell,
												&surface_color);
	}

	dev_mem[index].acc=total_force;
	
	//////
	float2 grad;
	float lapc;

	grad.x=surface_color.x;
	grad.y=surface_color.y;
	lapc=surface_color.z;

	float2 force;
	float normal;
	normal=sqrt(grad.x*grad.x+grad.y*grad.y);

	dev_mem[index].surface_normal=normal;

	if(normal > dev_param.surface_normal)
	{
		force=0.04f * lapc * grad / normal;
	}
	else
	{
		force=make_float2(0.0f, 0.0f);
	}
	//////

	dev_mem[index].acc=total_force+force;

	//add external force
	if(dev_param.force_waiting == true)
	{
		float x_len=dev_mem[index].pos.x-dev_param.extern_click.x;
		float y_len=dev_mem[index].pos.y-dev_param.extern_click.y;
		if((x_len*x_len+y_len*y_len) <= (dev_param.extern_len*dev_param.extern_len))
		{
			dev_mem[index].acc+=dev_param.extern_force*dev_param.extern_coe;
		}
	}
}

void compute(Particle *dev_mem,
				uint *dev_hash,
				uint *dev_index,
				uint *cell_start,
				uint *cell_end,
				uint num_particle,
				uint total_cell)
{
	if(num_particle == 0)
	{
		return;
	}

	uint num_thread;
	uint num_block;

    compute_grid_size(num_particle, 128, num_block, num_thread);

	compute_density_kernel<<< num_block, num_thread >>>(dev_mem,
													dev_hash,
													dev_index,
													cell_start,
													cell_end,
													num_particle,
													total_cell);

	compute_force_kernel<<< num_block, num_thread >>>(dev_mem,
												dev_hash,
												dev_index,
												cell_start,
												cell_end,
												num_particle,
												total_cell);
}

__device__ float wavelet(float input)
{
	float sigma=2.0;

	return 2.0f/(pow(PI, 0.25f)*pow(3.0f*sigma, 0.5f)) * ((input*input)/(sigma*sigma)-1) * pow(EXP, -(input*input)/(2*sigma*sigma));
}

__device__
float2 compute_cell_energy(uint index,
						uint2 neighbor,
						Particle *dev_mem,
						uint *dev_hash,
						uint *dev_index,
						uint *cell_start,
						uint *cell_end,
						uint num_particle,
						uint total_cell,
						float2 *trans_vel)
{
	uint grid_hash=calc_grid_hash(neighbor);
	uint start_index=cell_start[grid_hash];

	if(start_index == 0xffffffff)
	{
		return make_float2(0.0f);
	}
	
	float2 cell_trans=make_float2(0.0f, 0.0f);
	float2 transform=make_float2(0.0f, 0.0f);

	uint neighbor_index;

	Particle *p=&(dev_mem[index]);
	Particle *np;

	float2 rel_pos;
	float r2;

	float kernel;
	float kernel2;

	if(p->level == 1)
	{
		kernel=dev_param.large_kernel;
		kernel2=dev_param.large_kernel2;
	}

	if(p->level == 2)
	{
		kernel=dev_param.small_kernel;
		kernel2=dev_param.small_kernel2;
	}

	if(start_index != 0xffffffff)
	{        
		uint end_index=cell_end[grid_hash];

        for(uint count_index=start_index; count_index<end_index; count_index++) 
		{
			neighbor_index=dev_index[count_index];

			np=&(dev_mem[neighbor_index]);
			
            if(neighbor_index != index)
			{
				rel_pos=p->pos-np->pos;
				r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y;

				if(r2 < kernel2)
				{
					if(r2 < EXT2)
					{
						r2=EXT2;
					}

					transform.x=wavelet((p->pos.x-np->pos.x)/kernel);
					transform.y=wavelet((p->pos.y-np->pos.y)/kernel);

					cell_trans=cell_trans+transform;

					trans_vel[index].x=trans_vel[index].x+(np->vel.x*transform.x/pow(kernel, 0.5f));
					trans_vel[index].y=trans_vel[index].y+(np->vel.y*transform.y/pow(kernel, 0.5f));

				}
            }
			if(neighbor_index == index)
			{
				
			}
        }
	}

	return cell_trans;
}

__global__
void compute_energy_kernel(Particle *dev_mem,
							uint *dev_hash,
							uint *dev_index,
							uint *cell_start,
							uint *cell_end,
							uint num_particle,
							uint total_cell,
							float2 *trans_vel)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= num_particle)
	{
		return;
	}

	uint2 cell_pos=calc_grid_pos(dev_mem[index].pos.x, dev_mem[index].pos.y);

	uint count;
	uint2 neighbor_pos[9];
	uint neighbor_status[9];

	for(count=0; count<9; count++)
	{
		neighbor_status[count]=1;
	}

	neighbor_pos[0].x=cell_pos.x;
	neighbor_pos[0].y=cell_pos.y;

	neighbor_pos[1].x=cell_pos.x-1;
	neighbor_pos[1].y=cell_pos.y-1;

	neighbor_pos[2].x=cell_pos.x-1;
	neighbor_pos[2].y=cell_pos.y;

	neighbor_pos[3].x=cell_pos.x;
	neighbor_pos[3].y=cell_pos.y-1;

	neighbor_pos[4].x=cell_pos.x+1;
	neighbor_pos[4].y=cell_pos.y+1;

	neighbor_pos[5].x=cell_pos.x+1;
	neighbor_pos[5].y=cell_pos.y;

	neighbor_pos[6].x=cell_pos.x;
	neighbor_pos[6].y=cell_pos.y+1;

	neighbor_pos[7].x=cell_pos.x+1;
	neighbor_pos[7].y=cell_pos.y-1;

	neighbor_pos[8].x=cell_pos.x-1;
	neighbor_pos[8].y=cell_pos.y+1;

	if(cell_pos.x == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[2]=0;
		neighbor_status[8]=0;
	}

	if(cell_pos.x == dev_param.row_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[5]=0;
		neighbor_status[7]=0;
	}

	if(cell_pos.y == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[3]=0;
		neighbor_status[7]=0;
	}

	if(cell_pos.y == dev_param.col_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[6]=0;
		neighbor_status[8]=0;
	}

	float2 total_trans=make_float2(0.0f, 0.0f);
	trans_vel[index]=make_float2(0.0f, 0.0f);
	
	for(count=0; count<9; count++)
	{
		if(neighbor_status[count] == 0)
		{
			continue;
		}

		total_trans=total_trans+compute_cell_energy(index,
													neighbor_pos[count],
													dev_mem,
													dev_hash,
													dev_index,
													cell_start,
													cell_end,
													num_particle,
													total_cell,
													trans_vel);
	}

	float mass;

	if(dev_mem[index].level == 1)
	{
		mass=dev_param.large_mass;
	}

	if(dev_mem[index].level == 2)
	{
		mass=dev_param.small_mass;
	}

	trans_vel[index]=trans_vel[index]/total_trans;
	dev_mem[index].energy=0.5f*(trans_vel[index].x*trans_vel[index].x+trans_vel[index].y*trans_vel[index].y);
}

void compute_energy(Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					float2 *trans_vel)
{
	uint num_thread;
	uint num_block;

    compute_grid_size(num_particle, 256, num_block, num_thread);

	compute_energy_kernel<<<num_block, num_thread>>>(dev_mem,
												dev_hash,
												dev_index,
												cell_start,
												cell_end,
												num_particle,
												total_cell,
												trans_vel);
}

__global__
void decide_split_kernel(Particle *dev_mem,
						uint num_particle,
						uint *dev_split,
						uint *dev_index)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= num_particle)
	{
		return;
	}

	if(dev_mem[index].energy*dev_param.co_energy+dev_mem[index].surface_normal*dev_param.co_surface > dev_param.thresh_split && dev_mem[index].level == 1)
	{
		dev_split[index]=0;
	}
	else
	{
		dev_split[index]=1;
	}

	dev_index[index]=index;
}

__global__ 
void split_particle_kernel(Particle *dev_mem,
							uint *dev_split,
							uint *dev_index,
							uint num_particle,
							uint num_split,
							uint *dev_status)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= num_split)
	{
		return;
	}

	uint new1st=dev_index[index];
	uint new2nd=num_particle+index;
		
	//2
	dev_mem[new2nd].level=2;

	dev_mem[new2nd].pos.x=dev_mem[new1st].pos.x-dev_param.small_kernel/2;
	dev_mem[new2nd].pos.y=dev_mem[new1st].pos.y;

	dev_mem[new2nd].vel=dev_mem[new1st].vel;
	dev_mem[new2nd].ev=dev_mem[new1st].ev;
	dev_mem[new2nd].acc=dev_mem[new1st].acc;
	dev_mem[new2nd].dens=dev_mem[new1st].dens;
	dev_mem[new2nd].pres=dev_mem[new1st].pres;
	dev_mem[new2nd].energy=dev_mem[new1st].energy;
	dev_mem[new2nd].surface_normal=dev_mem[new1st].surface_normal;

	dev_status[new2nd]=0;

	//1
	dev_mem[new1st].level=2;
	dev_mem[new1st].pos.x=dev_mem[new1st].pos.x+dev_param.small_kernel/2;

	dev_status[new1st]=0;
}

uint split_particle(Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					uint *dev_split,
					uint *dev_status)
{
	if(num_particle == 0)
	{
		return num_particle;
	}

	uint num_thread;
	uint num_block;

    compute_grid_size(num_particle, 256, num_block, num_thread);

	decide_split_kernel<<< num_block, num_thread >>>(dev_mem, num_particle, dev_split, dev_index);

	uint num_split=num_particle-thrust::reduce(thrust::device_ptr<uint>(dev_split), thrust::device_ptr<uint>(dev_split + num_particle), (uint) 0, thrust::plus<uint>());

	thrust::sort_by_key(thrust::device_ptr<uint>(dev_split),
                        thrust::device_ptr<uint>(dev_split + num_particle),
                        thrust::device_ptr<uint>(dev_index));

	if(num_split != 0)
	{
		compute_grid_size(num_split, 256, num_block, num_thread);

		split_particle_kernel<<< num_block, num_thread >>>(dev_mem,
															dev_split,
															dev_index,
															num_particle,
															num_split,
															dev_status);
		num_particle=num_particle+num_split;
	}

	return num_particle;
}

__global__
void merge_kernel(Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					uint *cell_merge,
					uint *dev_status)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= total_cell)
	{
		return;
	}

	uint start_index=cell_start[index];
	cell_merge[index]=0;

	if(start_index != 0xffffffff)
	{        
        uint end_index=cell_end[index];

        for(uint count_index=start_index; count_index<end_index; count_index++) 
		{
			uint p_index=dev_index[count_index];
			dev_status[p_index]=0;

			if(dev_mem[p_index].level == 2 && dev_mem[index].energy*dev_param.co_energy+dev_mem[index].surface_normal*dev_param.co_surface < dev_param.thresh_merge)
			{
				cell_merge[index]++;
			}
        }

		if(cell_merge[index] < 2)
		{
			return;
		}

		float num_merge=cell_merge[index] / 2;
		
		uint current=start_index;
		uint p1st;
		uint p2nd;
		uint p_index;

		while(num_merge > 0)
		{
			while(current < end_index)
			{
				p_index=dev_index[current];
				if(dev_mem[p_index].level == 2 && dev_mem[index].energy*dev_param.co_energy+dev_mem[index].surface_normal*dev_param.co_surface < dev_param.thresh_merge)
				{
					p1st=p_index;
					current++;

					break;
				}

				current++;
			}

			while(current < end_index)
			{
				p_index=dev_index[current];
				if(dev_mem[p_index].level == 2 && dev_mem[index].energy*dev_param.co_energy+dev_mem[index].surface_normal*dev_param.co_surface < dev_param.thresh_merge)
				{
					p2nd=p_index;
					current++;

					break;
				}

				current++;
			}

			dev_mem[p1st].level=1;

			dev_mem[p1st].pos=(dev_mem[p1st].pos+dev_mem[p2nd].pos)/2;
			dev_mem[p1st].vel=(dev_mem[p1st].vel+dev_mem[p2nd].vel)/2;
			dev_mem[p1st].ev=(dev_mem[p1st].ev+dev_mem[p2nd].ev)/2;
			dev_mem[p1st].acc=(dev_mem[p1st].acc+dev_mem[p2nd].acc)/2;
			dev_mem[p1st].energy=(dev_mem[p1st].energy+dev_mem[p2nd].energy)/2;

			dev_status[p1st]=0;
			dev_status[p2nd]=1;

			num_merge--;
		}

	}
}

void merge(Particle *dev_mem,
			uint *dev_hash,
			uint *dev_index,
			uint *cell_start,
			uint *cell_end,
			uint num_particle,
			uint total_cell,
			uint *cell_merge,
			uint *dev_status)
{
	uint num_thread;
	uint num_block;

    compute_grid_size(total_cell, 256, num_block, num_thread);

	merge_kernel<<<num_block, num_thread>>>(dev_mem,
											dev_hash,
											dev_index,
											cell_start,
											cell_end,
											num_particle,
											total_cell,
											cell_merge,
											dev_status);
}

uint rearrange(Particle *dev_mem, uint *dev_status, uint num_particle)
{	
	if(num_particle == 0)
	{
		return num_particle;
	}

	uint temp_num_particle=num_particle;

	uint num_threads;
	uint num_blocks;

    compute_grid_size(temp_num_particle, 256, num_blocks, num_threads);
	
	thrust::sort_by_key(thrust::device_ptr<uint>(dev_status),
                        thrust::device_ptr<uint>(dev_status + temp_num_particle),
                        thrust::device_ptr<Particle>(dev_mem));
	
	temp_num_particle=thrust::reduce(thrust::device_ptr<uint>(dev_status), thrust::device_ptr<uint>(dev_status + temp_num_particle), (uint) 0, thrust::plus<uint>());
	num_particle=num_particle-temp_num_particle;

	return num_particle;
}