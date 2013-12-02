#include "sph_header.h"
#include "sph_param.h"
#include "sph_math.h"
#include "sph_kernel.cu"
#include <cutil_math.h>

#define EXP 2.718281f
#define EXT2 1e-12f
#define EXT 1e-6f

__device__ float wavelet(float input)
{
	float sigma=2.0;

	return 2.0f/(pow(PI, 0.25f)*pow(3.0f*sigma, 0.5f)) * ((input*input)/(sigma*sigma)-1) * pow(EXP, -(input*input)/(2*sigma*sigma));
}

__device__ uint3 calc_grid_pos(float x, float y ,float z)
{
    uint3 cell_pos;

    cell_pos.x=(uint)floor(x / dev_param.cell_size);
    cell_pos.y=(uint)floor(y / dev_param.cell_size);
	cell_pos.z=(uint)floor(z / dev_param.cell_size);

    return cell_pos;
}

__device__ uint calc_grid_hash(uint3 cell_pos)
{  
	return cell_pos.z*dev_param.row_cell*dev_param.col_cell + cell_pos.y*dev_param.row_cell + cell_pos.x;
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

	uint3 grid_pos=calc_grid_pos(dev_mem[index].pos.x, dev_mem[index].pos.y, dev_mem[index].pos.z);
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

	//external force
	if(dev_param.force_waiting_left == true)
	{
		if(p->pos.x < dev_param.world_width/3 && p->pos.y > dev_param.world_height/10)
		{
			p->acc.x=p->acc.x+20.0f;
		}
	}
	if(dev_param.force_waiting_right == true)
	{
		if(p->pos.x > dev_param.world_width/3*2 && p->pos.y > dev_param.world_height/10)
		{
			p->acc.x=p->acc.x-20.0f;
		}
	}
	
	p->vel=p->vel+p->acc*dev_param.time_step/p->dens;
	p->vel=p->vel+dev_param.gravity*dev_param.time_step/p->dens;	

	p->pos=p->pos+p->vel*dev_param.time_step;

	if(dev_param.use_cylinder == true)
	{
		//handle collision with the sphere 1
		float3 dist_vec=dev_mem[index].pos-dev_param.sphere1.pos;
		float distance=sqrt(dist_vec.x*dist_vec.x+dist_vec.y*dist_vec.y+dist_vec.z*dist_vec.z);
		if(distance < dev_param.sphere1.radius+radius)
		{
			float3 poxyz=dev_mem[index].pos-dev_param.sphere1.pos;
			float3 dxyz=poxyz/distance*(dev_param.sphere1.radius+radius);
			float3 normal=dxyz;
			normal=normalize(normal);
			dev_mem[index].pos=dev_param.sphere1.pos+dxyz;
			dev_mem[index].vel=dev_mem[index].vel-dev_param.sphere1.damping*(dot(dev_mem[index].vel, normal))*normal;
		}

		//handle collision with the sphere 2
		dist_vec=dev_mem[index].pos-dev_param.sphere2.pos;
		distance=sqrt(dist_vec.x*dist_vec.x+dist_vec.y*dist_vec.y+dist_vec.z*dist_vec.z);
		if(distance < dev_param.sphere2.radius+radius)
		{
			float3 poxyz=dev_mem[index].pos-dev_param.sphere2.pos;
			float3 dxyz=poxyz/distance*(dev_param.sphere2.radius+radius);
			float3 normal=dxyz;
			normal=normalize(normal);
			dev_mem[index].pos=dev_param.sphere2.pos+dxyz;
			dev_mem[index].vel=dev_mem[index].vel-dev_param.sphere2.damping*(dot(dev_mem[index].vel, normal))*normal;
		}

		//handle collision with the sphere 3
		dist_vec=dev_mem[index].pos-dev_param.sphere3.pos;
		distance=sqrt(dist_vec.x*dist_vec.x+dist_vec.y*dist_vec.y+dist_vec.z*dist_vec.z);
		if(distance < dev_param.sphere3.radius+radius)
		{
			float3 poxyz=dev_mem[index].pos-dev_param.sphere3.pos;
			float3 dxyz=poxyz/distance*(dev_param.sphere3.radius+radius);
			float3 normal=dxyz;
			normal=normalize(normal);
			dev_mem[index].pos=dev_param.sphere3.pos+dxyz;
			dev_mem[index].vel=dev_mem[index].vel-dev_param.sphere3.damping*(dot(dev_mem[index].vel, normal))*normal;
		}
	}


	if(p->pos.x >= dev_param.world_width-radius)
	{
		p->vel.x=p->vel.x*dev_param.wall_damping;
		p->pos.x=dev_param.world_width-radius;
	}

	if(p->pos.x < radius)
	{
		p->vel.x=p->vel.x*dev_param.wall_damping;
		p->pos.x=radius;
	}

	if(p->pos.y >= dev_param.world_height-radius)
	{
		p->vel.y=p->vel.y*dev_param.wall_damping;
		p->pos.y=dev_param.world_height-radius;
	}

	if(p->pos.y < radius)
	{
		p->vel.y=p->vel.y*dev_param.wall_damping;
		p->pos.y=radius;
	}

	if(p->pos.z >= dev_param.world_length-radius)
	{
		p->vel.z=p->vel.z*dev_param.wall_damping;
		p->pos.z=dev_param.world_length-radius;
	}

	if(p->pos.z < radius)
	{
		p->vel.z=p->vel.z*dev_param.wall_damping;
		p->pos.z=radius;
	}

	p->ev=(p->ev+p->vel)/2;
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
					uint3 neighbor,
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

	float i_mass;
	float i_kernel_2;
	float i_kernel_6;
	float i_poly6_value;

	float j_mass;

	float3 rel_pos;
	float r2;

	Particle *p=&(dev_mem[index]);
	Particle *np;
	uint neighbor_index;

	if(p->level == 1)
	{
		i_mass=dev_param.large_mass;
		i_kernel_2=dev_param.large_kernel_2;
		i_kernel_6=dev_param.large_kernel_6;
		i_poly6_value=dev_param.large_poly6;
	}

	if(p->level == 2)
	{
		i_mass=dev_param.small_mass;
		i_kernel_2=dev_param.small_kernel_2;
		i_kernel_6=dev_param.small_kernel_6;
		i_poly6_value=dev_param.small_poly6;
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
				r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y+rel_pos.z*rel_pos.z;

				if(r2 < 0.0001f)
				{
					continue;
				}

				if(r2 < i_kernel_2)
				{
					if(np->level == 1)
					{
						j_mass=dev_param.large_mass;
					}

					if(np->level == 2)
					{
						j_mass=dev_param.small_mass;
					}

					total_cell_density=total_cell_density + j_mass * i_poly6_value * pow(i_kernel_2-r2, 3);
				}
            }
			if(neighbor_index == index)
			{
				total_cell_density=total_cell_density + i_mass * i_poly6_value * i_kernel_6;
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

	uint3 grid_pos=calc_grid_pos(dev_mem[index].pos.x, dev_mem[index].pos.y, dev_mem[index].pos.z);
	
	uint count;
	uint3 neighbor_pos[27];
	uint neighbor_status[27];

	for(count=0; count<27; count++)
	{
		neighbor_status[count]=1;
	}

	neighbor_pos[0].x=grid_pos.x;
	neighbor_pos[0].y=grid_pos.y;
	neighbor_pos[0].z=grid_pos.z;

	neighbor_pos[1].x=grid_pos.x-1;
	neighbor_pos[1].y=grid_pos.y-1;
	neighbor_pos[1].z=grid_pos.z;

	neighbor_pos[2].x=grid_pos.x-1;
	neighbor_pos[2].y=grid_pos.y;
	neighbor_pos[2].z=grid_pos.z;

	neighbor_pos[3].x=grid_pos.x;
	neighbor_pos[3].y=grid_pos.y-1;
	neighbor_pos[3].z=grid_pos.z;

	neighbor_pos[4].x=grid_pos.x+1;
	neighbor_pos[4].y=grid_pos.y+1;
	neighbor_pos[4].z=grid_pos.z;

	neighbor_pos[5].x=grid_pos.x+1;
	neighbor_pos[5].y=grid_pos.y;
	neighbor_pos[5].z=grid_pos.z;

	neighbor_pos[6].x=grid_pos.x;
	neighbor_pos[6].y=grid_pos.y+1;
	neighbor_pos[6].z=grid_pos.z;

	neighbor_pos[7].x=grid_pos.x+1;
	neighbor_pos[7].y=grid_pos.y-1;
	neighbor_pos[7].z=grid_pos.z;

	neighbor_pos[8].x=grid_pos.x-1;
	neighbor_pos[8].y=grid_pos.y+1;
	neighbor_pos[8].z=grid_pos.z;

	neighbor_pos[9].x=grid_pos.x;
	neighbor_pos[9].y=grid_pos.y;
	neighbor_pos[9].z=grid_pos.z-1;

	neighbor_pos[10].x=grid_pos.x-1;
	neighbor_pos[10].y=grid_pos.y-1;
	neighbor_pos[10].z=grid_pos.z-1;

	neighbor_pos[11].x=grid_pos.x-1;
	neighbor_pos[11].y=grid_pos.y;
	neighbor_pos[11].z=grid_pos.z-1;

	neighbor_pos[12].x=grid_pos.x;
	neighbor_pos[12].y=grid_pos.y-1;
	neighbor_pos[12].z=grid_pos.z-1;

	neighbor_pos[13].x=grid_pos.x+1;
	neighbor_pos[13].y=grid_pos.y+1;
	neighbor_pos[13].z=grid_pos.z-1;

	neighbor_pos[14].x=grid_pos.x+1;
	neighbor_pos[14].y=grid_pos.y;
	neighbor_pos[14].z=grid_pos.z-1;

	neighbor_pos[15].x=grid_pos.x;
	neighbor_pos[15].y=grid_pos.y+1;
	neighbor_pos[15].z=grid_pos.z-1;

	neighbor_pos[16].x=grid_pos.x+1;
	neighbor_pos[16].y=grid_pos.y-1;
	neighbor_pos[16].z=grid_pos.z-1;

	neighbor_pos[17].x=grid_pos.x-1;
	neighbor_pos[17].y=grid_pos.y+1;
	neighbor_pos[17].z=grid_pos.z-1;

	neighbor_pos[18].x=grid_pos.x;
	neighbor_pos[18].y=grid_pos.y;
	neighbor_pos[18].z=grid_pos.z+1;

	neighbor_pos[19].x=grid_pos.x-1;
	neighbor_pos[19].y=grid_pos.y-1;
	neighbor_pos[19].z=grid_pos.z+1;

	neighbor_pos[20].x=grid_pos.x-1;
	neighbor_pos[20].y=grid_pos.y;
	neighbor_pos[20].z=grid_pos.z+1;

	neighbor_pos[21].x=grid_pos.x;
	neighbor_pos[21].y=grid_pos.y-1;
	neighbor_pos[21].z=grid_pos.z+1;

	neighbor_pos[22].x=grid_pos.x+1;
	neighbor_pos[22].y=grid_pos.y+1;
	neighbor_pos[22].z=grid_pos.z+1;

	neighbor_pos[23].x=grid_pos.x+1;
	neighbor_pos[23].y=grid_pos.y;
	neighbor_pos[23].z=grid_pos.z+1;

	neighbor_pos[24].x=grid_pos.x;
	neighbor_pos[24].y=grid_pos.y+1;
	neighbor_pos[24].z=grid_pos.z+1;

	neighbor_pos[25].x=grid_pos.x+1;
	neighbor_pos[25].y=grid_pos.y-1;
	neighbor_pos[25].z=grid_pos.z+1;

	neighbor_pos[26].x=grid_pos.x-1;
	neighbor_pos[26].y=grid_pos.y+1;
	neighbor_pos[26].z=grid_pos.z+1;

	if(grid_pos.x == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[2]=0;
		neighbor_status[8]=0;

		neighbor_status[10]=0;
		neighbor_status[11]=0;
		neighbor_status[17]=0;

		neighbor_status[19]=0;
		neighbor_status[20]=0;
		neighbor_status[26]=0;
	}

	if(grid_pos.x == dev_param.row_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[5]=0;
		neighbor_status[7]=0;

		neighbor_status[13]=0;
		neighbor_status[14]=0;
		neighbor_status[16]=0;

		neighbor_status[22]=0;
		neighbor_status[23]=0;
		neighbor_status[25]=0;
	}

	if(grid_pos.y == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[3]=0;
		neighbor_status[7]=0;

		neighbor_status[10]=0;
		neighbor_status[12]=0;
		neighbor_status[16]=0;

		neighbor_status[19]=0;
		neighbor_status[21]=0;
		neighbor_status[25]=0;
	}

	if(grid_pos.y == dev_param.col_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[6]=0;
		neighbor_status[8]=0;

		neighbor_status[13]=0;
		neighbor_status[15]=0;
		neighbor_status[17]=0;

		neighbor_status[22]=0;
		neighbor_status[24]=0;
		neighbor_status[26]=0;
	}

	if(grid_pos.z == 0)
	{
		neighbor_status[9]=0;
		neighbor_status[10]=0;
		neighbor_status[11]=0;

		neighbor_status[12]=0;
		neighbor_status[13]=0;
		neighbor_status[14]=0;

		neighbor_status[15]=0;
		neighbor_status[16]=0;
		neighbor_status[17]=0;
	}

	if(grid_pos.z == dev_param.len_cell-1)
	{
		neighbor_status[18]=0;
		neighbor_status[19]=0;
		neighbor_status[20]=0;

		neighbor_status[21]=0;
		neighbor_status[22]=0;
		neighbor_status[23]=0;

		neighbor_status[24]=0;
		neighbor_status[25]=0;
		neighbor_status[26]=0;
	}

	float total_density=0;

	for(count=0; count<27; count++)
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
	dev_mem[index].press=(pow(dev_mem[index].dens / dev_param.rest_density, 3) - 1) * dev_param.gas_constant;
	//dev_mem[index].press=(dev_mem[index].dens / dev_param.rest_density - 1) * dev_param.gas_constant;
}

__device__
float3 compute_cell_force(uint index,
					uint3 neighbor,
					Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					float4 *surface)
{
	uint grid_hash=calc_grid_hash(neighbor);
	uint start_index=cell_start[grid_hash];
	
	float3 total_cell_force=make_float3(0.0f);

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

	float3 rel_pos;
	float r2;
	float r;
	float temp_force;
	float3 rel_vel;

	if(p->level == 1)
	{
		i_kernel=dev_param.large_kernel;
		i_mass=dev_param.large_mass;
		i_kernel_2=dev_param.large_kernel_2;
		i_kernel_6=dev_param.large_kernel_6;
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
		i_kernel_2=dev_param.small_kernel_2;
		i_kernel_6=dev_param.small_kernel_6;
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
				r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y+rel_pos.z*rel_pos.z;

				if(np->level == 1)
				{
					j_kernel=dev_param.large_kernel;
					j_mass=dev_param.large_mass;
					j_kernel_2=dev_param.large_kernel_2;
					j_kernel_6=dev_param.large_kernel_6;
					j_spiky_value=dev_param.large_spiky;
					j_visco_value=dev_param.large_visco;
				}

				if(np->level == 2)
				{
					j_kernel=dev_param.small_kernel;
					j_mass=dev_param.small_mass;
					j_kernel_2=dev_param.small_kernel_2;
					j_kernel_6=dev_param.small_kernel_6;
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

						//surface tension
						float temp=(-1) * i_grad_poly6 * j_mass / np->dens * pow(i_kernel_2-r2, 2);
						
						surface->x += temp * rel_pos.x;
						surface->y += temp * rel_pos.y;
						surface->z += temp * rel_pos.z;

						surface->w += i_lapc_poly6 * j_mass / np->dens * (i_kernel_2-r2) * (r2-3/4*(i_kernel_2-r2));
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

					temp_force=i_mass*j_mass * (p->press/(p->dens*p->dens)+np->press/(np->dens*np->dens)) * (i_press_kernel+j_press_kernel)/2;
					total_cell_force=total_cell_force-rel_pos*temp_force/r;

					rel_vel=np->ev-p->ev;

					temp_force=(iV*jV) * dev_param.viscosity * (i_visco_kernel+j_visco_kernel)/2;
					total_cell_force=total_cell_force + rel_vel*temp_force; 
				}
            }
			if(neighbor_index == index)
			{
				
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

	uint3 grid_pos=calc_grid_pos(dev_mem[index].pos.x, dev_mem[index].pos.y, dev_mem[index].pos.z);

	uint count;
	uint3 neighbor_pos[27];
	uint neighbor_status[27];

	for(count=0; count<27; count++)
	{
		neighbor_status[count]=1;
	}

	neighbor_pos[0].x=grid_pos.x;
	neighbor_pos[0].y=grid_pos.y;
	neighbor_pos[0].z=grid_pos.z;

	neighbor_pos[1].x=grid_pos.x-1;
	neighbor_pos[1].y=grid_pos.y-1;
	neighbor_pos[1].z=grid_pos.z;

	neighbor_pos[2].x=grid_pos.x-1;
	neighbor_pos[2].y=grid_pos.y;
	neighbor_pos[2].z=grid_pos.z;

	neighbor_pos[3].x=grid_pos.x;
	neighbor_pos[3].y=grid_pos.y-1;
	neighbor_pos[3].z=grid_pos.z;

	neighbor_pos[4].x=grid_pos.x+1;
	neighbor_pos[4].y=grid_pos.y+1;
	neighbor_pos[4].z=grid_pos.z;

	neighbor_pos[5].x=grid_pos.x+1;
	neighbor_pos[5].y=grid_pos.y;
	neighbor_pos[5].z=grid_pos.z;

	neighbor_pos[6].x=grid_pos.x;
	neighbor_pos[6].y=grid_pos.y+1;
	neighbor_pos[6].z=grid_pos.z;

	neighbor_pos[7].x=grid_pos.x+1;
	neighbor_pos[7].y=grid_pos.y-1;
	neighbor_pos[7].z=grid_pos.z;

	neighbor_pos[8].x=grid_pos.x-1;
	neighbor_pos[8].y=grid_pos.y+1;
	neighbor_pos[8].z=grid_pos.z;

	neighbor_pos[9].x=grid_pos.x;
	neighbor_pos[9].y=grid_pos.y;
	neighbor_pos[9].z=grid_pos.z-1;

	neighbor_pos[10].x=grid_pos.x-1;
	neighbor_pos[10].y=grid_pos.y-1;
	neighbor_pos[10].z=grid_pos.z-1;

	neighbor_pos[11].x=grid_pos.x-1;
	neighbor_pos[11].y=grid_pos.y;
	neighbor_pos[11].z=grid_pos.z-1;

	neighbor_pos[12].x=grid_pos.x;
	neighbor_pos[12].y=grid_pos.y-1;
	neighbor_pos[12].z=grid_pos.z-1;

	neighbor_pos[13].x=grid_pos.x+1;
	neighbor_pos[13].y=grid_pos.y+1;
	neighbor_pos[13].z=grid_pos.z-1;

	neighbor_pos[14].x=grid_pos.x+1;
	neighbor_pos[14].y=grid_pos.y;
	neighbor_pos[14].z=grid_pos.z-1;

	neighbor_pos[15].x=grid_pos.x;
	neighbor_pos[15].y=grid_pos.y+1;
	neighbor_pos[15].z=grid_pos.z-1;

	neighbor_pos[16].x=grid_pos.x+1;
	neighbor_pos[16].y=grid_pos.y-1;
	neighbor_pos[16].z=grid_pos.z-1;

	neighbor_pos[17].x=grid_pos.x-1;
	neighbor_pos[17].y=grid_pos.y+1;
	neighbor_pos[17].z=grid_pos.z-1;

	neighbor_pos[18].x=grid_pos.x;
	neighbor_pos[18].y=grid_pos.y;
	neighbor_pos[18].z=grid_pos.z+1;

	neighbor_pos[19].x=grid_pos.x-1;
	neighbor_pos[19].y=grid_pos.y-1;
	neighbor_pos[19].z=grid_pos.z+1;

	neighbor_pos[20].x=grid_pos.x-1;
	neighbor_pos[20].y=grid_pos.y;
	neighbor_pos[20].z=grid_pos.z+1;

	neighbor_pos[21].x=grid_pos.x;
	neighbor_pos[21].y=grid_pos.y-1;
	neighbor_pos[21].z=grid_pos.z+1;

	neighbor_pos[22].x=grid_pos.x+1;
	neighbor_pos[22].y=grid_pos.y+1;
	neighbor_pos[22].z=grid_pos.z+1;

	neighbor_pos[23].x=grid_pos.x+1;
	neighbor_pos[23].y=grid_pos.y;
	neighbor_pos[23].z=grid_pos.z+1;

	neighbor_pos[24].x=grid_pos.x;
	neighbor_pos[24].y=grid_pos.y+1;
	neighbor_pos[24].z=grid_pos.z+1;

	neighbor_pos[25].x=grid_pos.x+1;
	neighbor_pos[25].y=grid_pos.y-1;
	neighbor_pos[25].z=grid_pos.z+1;

	neighbor_pos[26].x=grid_pos.x-1;
	neighbor_pos[26].y=grid_pos.y+1;
	neighbor_pos[26].z=grid_pos.z+1;

	if(grid_pos.x == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[2]=0;
		neighbor_status[8]=0;

		neighbor_status[10]=0;
		neighbor_status[11]=0;
		neighbor_status[17]=0;

		neighbor_status[19]=0;
		neighbor_status[20]=0;
		neighbor_status[26]=0;
	}

	if(grid_pos.x == dev_param.row_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[5]=0;
		neighbor_status[7]=0;

		neighbor_status[13]=0;
		neighbor_status[14]=0;
		neighbor_status[16]=0;

		neighbor_status[22]=0;
		neighbor_status[23]=0;
		neighbor_status[25]=0;
	}

	if(grid_pos.y == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[3]=0;
		neighbor_status[7]=0;

		neighbor_status[10]=0;
		neighbor_status[12]=0;
		neighbor_status[16]=0;

		neighbor_status[19]=0;
		neighbor_status[21]=0;
		neighbor_status[25]=0;
	}

	if(grid_pos.y == dev_param.col_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[6]=0;
		neighbor_status[8]=0;

		neighbor_status[13]=0;
		neighbor_status[15]=0;
		neighbor_status[17]=0;

		neighbor_status[22]=0;
		neighbor_status[24]=0;
		neighbor_status[26]=0;
	}

	if(grid_pos.z == 0)
	{
		neighbor_status[9]=0;
		neighbor_status[10]=0;
		neighbor_status[11]=0;

		neighbor_status[12]=0;
		neighbor_status[13]=0;
		neighbor_status[14]=0;

		neighbor_status[15]=0;
		neighbor_status[16]=0;
		neighbor_status[17]=0;
	}

	if(grid_pos.z == dev_param.len_cell-1)
	{
		neighbor_status[18]=0;
		neighbor_status[19]=0;
		neighbor_status[20]=0;

		neighbor_status[21]=0;
		neighbor_status[22]=0;
		neighbor_status[23]=0;

		neighbor_status[24]=0;
		neighbor_status[25]=0;
		neighbor_status[26]=0;
	}

	float3 total_force=make_float3(0.0f, 0.0f, 0.0f);
	float4 surface=make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for(count=0; count<27; count++)
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
												&surface);
	}

	float3 grad;
	float lapc;

	grad.x=surface.x;
	grad.y=surface.y;
	grad.z=surface.z;
	lapc=surface.w;

	float3 force;
	float normal;
	normal=sqrt(grad.x*grad.x+grad.y*grad.y+grad.z*grad.z);

	dev_mem[index].surface=normal;

	if(normal > dev_param.surface_tension)
	{
		force=0.02f * lapc * grad / normal;
	}
	else
	{
		force=make_float3(0.0f, 0.0f, 0.0f);
	}

	dev_mem[index].acc=total_force+force;
}

void compute(Particle *dev_mem,
				uint *dev_hash,
				uint *dev_index,
				uint *cell_start,
				uint *cell_end,
				uint num_particle,
				uint total_cell)
{
	uint num_thread;
	uint num_block;

    compute_grid_size(num_particle, 256, num_block, num_thread);

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

__device__
float3 compute_cell_energy(uint index,
						uint3 neighbor,
						Particle *dev_mem,
						uint *dev_hash,
						uint *dev_index,
						uint *cell_start,
						uint *cell_end,
						uint num_particle,
						uint total_cell,
						float3 *trans_vel)
{
	uint grid_hash=calc_grid_hash(neighbor);
	uint start_index=cell_start[grid_hash];
	
	float3 cell_trans=make_float3(0.0f, 0.0f, 0.0f);
	float3 transform=make_float3(0.0f, 0.0f, 0.0f);

	uint neighbor_index;

	Particle *p=&(dev_mem[index]);
	Particle *np;

	float3 rel_pos;
	float r2;
	float r;

	float kernel;
	float kernel2;

	if(p->level == 1)
	{
		kernel=dev_param.large_kernel;
		kernel2=dev_param.large_kernel_2;
	}

	if(p->level == 2)
	{
		kernel=dev_param.large_kernel;
		kernel2=dev_param.large_kernel_2;
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
				r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y+rel_pos.z*rel_pos.z;

				if(r2 < kernel2)
				{
					if(r2 < 0.0001f)
					{
						continue;
					}
					else
					{
						r=sqrt(r2);
					}

					transform.x=wavelet((p->pos.x-np->pos.x)/kernel);
					transform.y=wavelet((p->pos.y-np->pos.y)/kernel);
					transform.z=wavelet((p->pos.z-np->pos.z)/kernel);

					cell_trans=cell_trans+transform;

					trans_vel[index].x=trans_vel[index].x+(np->vel.x*transform.x/pow(kernel, 0.5f));
					trans_vel[index].y=trans_vel[index].y+(np->vel.y*transform.y/pow(kernel, 0.5f));
					trans_vel[index].z=trans_vel[index].z+(np->vel.z*transform.z/pow(kernel, 0.5f));

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
							float3 *trans_vel)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= num_particle)
	{
		return;
	}

	uint3 grid_pos=calc_grid_pos(dev_mem[index].pos.x, dev_mem[index].pos.y, dev_mem[index].pos.z);

	uint count;
	uint3 neighbor_pos[27];
	uint neighbor_status[27];

	for(count=0; count<27; count++)
	{
		neighbor_status[count]=1;
	}

	neighbor_pos[0].x=grid_pos.x;
	neighbor_pos[0].y=grid_pos.y;
	neighbor_pos[0].z=grid_pos.z;

	neighbor_pos[1].x=grid_pos.x-1;
	neighbor_pos[1].y=grid_pos.y-1;
	neighbor_pos[1].z=grid_pos.z;

	neighbor_pos[2].x=grid_pos.x-1;
	neighbor_pos[2].y=grid_pos.y;
	neighbor_pos[2].z=grid_pos.z;

	neighbor_pos[3].x=grid_pos.x;
	neighbor_pos[3].y=grid_pos.y-1;
	neighbor_pos[3].z=grid_pos.z;

	neighbor_pos[4].x=grid_pos.x+1;
	neighbor_pos[4].y=grid_pos.y+1;
	neighbor_pos[4].z=grid_pos.z;

	neighbor_pos[5].x=grid_pos.x+1;
	neighbor_pos[5].y=grid_pos.y;
	neighbor_pos[5].z=grid_pos.z;

	neighbor_pos[6].x=grid_pos.x;
	neighbor_pos[6].y=grid_pos.y+1;
	neighbor_pos[6].z=grid_pos.z;

	neighbor_pos[7].x=grid_pos.x+1;
	neighbor_pos[7].y=grid_pos.y-1;
	neighbor_pos[7].z=grid_pos.z;

	neighbor_pos[8].x=grid_pos.x-1;
	neighbor_pos[8].y=grid_pos.y+1;
	neighbor_pos[8].z=grid_pos.z;

	neighbor_pos[9].x=grid_pos.x;
	neighbor_pos[9].y=grid_pos.y;
	neighbor_pos[9].z=grid_pos.z-1;

	neighbor_pos[10].x=grid_pos.x-1;
	neighbor_pos[10].y=grid_pos.y-1;
	neighbor_pos[10].z=grid_pos.z-1;

	neighbor_pos[11].x=grid_pos.x-1;
	neighbor_pos[11].y=grid_pos.y;
	neighbor_pos[11].z=grid_pos.z-1;

	neighbor_pos[12].x=grid_pos.x;
	neighbor_pos[12].y=grid_pos.y-1;
	neighbor_pos[12].z=grid_pos.z-1;

	neighbor_pos[13].x=grid_pos.x+1;
	neighbor_pos[13].y=grid_pos.y+1;
	neighbor_pos[13].z=grid_pos.z-1;

	neighbor_pos[14].x=grid_pos.x+1;
	neighbor_pos[14].y=grid_pos.y;
	neighbor_pos[14].z=grid_pos.z-1;

	neighbor_pos[15].x=grid_pos.x;
	neighbor_pos[15].y=grid_pos.y+1;
	neighbor_pos[15].z=grid_pos.z-1;

	neighbor_pos[16].x=grid_pos.x+1;
	neighbor_pos[16].y=grid_pos.y-1;
	neighbor_pos[16].z=grid_pos.z-1;

	neighbor_pos[17].x=grid_pos.x-1;
	neighbor_pos[17].y=grid_pos.y+1;
	neighbor_pos[17].z=grid_pos.z-1;

	neighbor_pos[18].x=grid_pos.x;
	neighbor_pos[18].y=grid_pos.y;
	neighbor_pos[18].z=grid_pos.z+1;

	neighbor_pos[19].x=grid_pos.x-1;
	neighbor_pos[19].y=grid_pos.y-1;
	neighbor_pos[19].z=grid_pos.z+1;

	neighbor_pos[20].x=grid_pos.x-1;
	neighbor_pos[20].y=grid_pos.y;
	neighbor_pos[20].z=grid_pos.z+1;

	neighbor_pos[21].x=grid_pos.x;
	neighbor_pos[21].y=grid_pos.y-1;
	neighbor_pos[21].z=grid_pos.z+1;

	neighbor_pos[22].x=grid_pos.x+1;
	neighbor_pos[22].y=grid_pos.y+1;
	neighbor_pos[22].z=grid_pos.z+1;

	neighbor_pos[23].x=grid_pos.x+1;
	neighbor_pos[23].y=grid_pos.y;
	neighbor_pos[23].z=grid_pos.z+1;

	neighbor_pos[24].x=grid_pos.x;
	neighbor_pos[24].y=grid_pos.y+1;
	neighbor_pos[24].z=grid_pos.z+1;

	neighbor_pos[25].x=grid_pos.x+1;
	neighbor_pos[25].y=grid_pos.y-1;
	neighbor_pos[25].z=grid_pos.z+1;

	neighbor_pos[26].x=grid_pos.x-1;
	neighbor_pos[26].y=grid_pos.y+1;
	neighbor_pos[26].z=grid_pos.z+1;

	if(grid_pos.x == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[2]=0;
		neighbor_status[8]=0;

		neighbor_status[10]=0;
		neighbor_status[11]=0;
		neighbor_status[17]=0;

		neighbor_status[19]=0;
		neighbor_status[20]=0;
		neighbor_status[26]=0;
	}

	if(grid_pos.x == dev_param.row_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[5]=0;
		neighbor_status[7]=0;

		neighbor_status[13]=0;
		neighbor_status[14]=0;
		neighbor_status[16]=0;

		neighbor_status[22]=0;
		neighbor_status[23]=0;
		neighbor_status[25]=0;
	}

	if(grid_pos.y == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[3]=0;
		neighbor_status[7]=0;

		neighbor_status[10]=0;
		neighbor_status[12]=0;
		neighbor_status[16]=0;

		neighbor_status[19]=0;
		neighbor_status[21]=0;
		neighbor_status[25]=0;
	}

	if(grid_pos.y == dev_param.col_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[6]=0;
		neighbor_status[8]=0;

		neighbor_status[13]=0;
		neighbor_status[15]=0;
		neighbor_status[17]=0;

		neighbor_status[22]=0;
		neighbor_status[24]=0;
		neighbor_status[26]=0;
	}

	if(grid_pos.z == 0)
	{
		neighbor_status[9]=0;
		neighbor_status[10]=0;
		neighbor_status[11]=0;

		neighbor_status[12]=0;
		neighbor_status[13]=0;
		neighbor_status[14]=0;

		neighbor_status[15]=0;
		neighbor_status[16]=0;
		neighbor_status[17]=0;
	}

	if(grid_pos.z == dev_param.len_cell-1)
	{
		neighbor_status[18]=0;
		neighbor_status[19]=0;
		neighbor_status[20]=0;

		neighbor_status[21]=0;
		neighbor_status[22]=0;
		neighbor_status[23]=0;

		neighbor_status[24]=0;
		neighbor_status[25]=0;
		neighbor_status[26]=0;
	}

	float3 total_force=make_float3(0.0f, 0.0f, 0.0f);
	float3 total_trans=make_float3(0.0f, 0.0f, 0.0f);
	trans_vel[index]=make_float3(0.0f, 0.0f, 0.0f);

	for(count=0; count<27; count++)
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
	dev_mem[index].energy=0.5f*mass*(trans_vel[index].x*trans_vel[index].x+trans_vel[index].y*trans_vel[index].y);
}

void compute_energy(Particle *dev_mem,
					uint *dev_hash,
					uint *dev_index,
					uint *cell_start,
					uint *cell_end,
					uint num_particle,
					uint total_cell,
					float3 *trans_vel)
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
void decide_adaptive_kernel(Particle *dev_mem,
						uint num_particle,
						uint *dev_split,
						uint *dev_index)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= num_particle)
	{
		return;
	}

	float co_surface=dev_param.co_surface;
	float co_energy=dev_param.co_energy;

	if(dev_mem[index].energy*co_energy+dev_mem[index].surface*co_surface > dev_param.split_criteria && dev_mem[index].level == 1)
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
void adaptive_particle_kernel(Particle *dev_mem,
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
	//uint new2nd=num_particle+index;
	uint new2nd=num_particle+index*3+0;
	uint new3rd=num_particle+index*3+1;
	uint new4th=num_particle+index*3+2;
		
	//2
	dev_mem[new2nd].level=2;

	dev_mem[new2nd].pos.x=dev_mem[new1st].pos.x-dev_param.small_kernel/2;
	dev_mem[new2nd].pos.y=dev_mem[new1st].pos.y;
	dev_mem[new2nd].pos.z=dev_mem[new1st].pos.z;

	dev_mem[new2nd].vel=dev_mem[new1st].vel;
	dev_mem[new2nd].ev=dev_mem[new1st].ev;
	dev_mem[new2nd].acc=dev_mem[new1st].acc;

	dev_mem[new2nd].energy=dev_mem[new1st].energy/4;

	dev_status[new2nd]=0;

	//3
	dev_mem[new3rd].level=2;

	dev_mem[new3rd].pos.x=dev_mem[new1st].pos.x;
	dev_mem[new3rd].pos.y=dev_mem[new1st].pos.y;
	dev_mem[new3rd].pos.z=dev_mem[new1st].pos.z-dev_param.small_kernel/2;

	dev_mem[new3rd].vel=dev_mem[new1st].vel;
	dev_mem[new3rd].ev=dev_mem[new1st].ev;
	dev_mem[new3rd].acc=dev_mem[new1st].acc;

	dev_mem[new3rd].energy=dev_mem[new1st].energy/4;

	dev_status[new3rd]=0;

	//4
	dev_mem[new4th].level=2;

	dev_mem[new4th].pos.x=dev_mem[new1st].pos.x;
	dev_mem[new4th].pos.y=dev_mem[new1st].pos.y;
	dev_mem[new4th].pos.z=dev_mem[new1st].pos.z+dev_param.small_kernel/2;

	dev_mem[new4th].vel=dev_mem[new1st].vel;
	dev_mem[new4th].ev=dev_mem[new1st].ev;
	dev_mem[new4th].acc=dev_mem[new1st].acc;

	dev_mem[new4th].energy=dev_mem[new1st].energy/4;

	dev_status[new4th]=0;

	//1
	dev_mem[new1st].level=2;
	dev_mem[new1st].pos.x=dev_mem[new1st].pos.x+dev_param.small_kernel/2;
	dev_mem[new1st].pos.y=dev_mem[new1st].pos.y;
	dev_mem[new1st].energy=dev_mem[new1st].energy/4;

	dev_status[new1st]=0;
}

uint adaptive_particle(Particle *dev_mem,
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

	decide_adaptive_kernel<<< num_block, num_thread >>>(dev_mem, num_particle, dev_split, dev_index);

	uint num_split=num_particle-thrust::reduce(thrust::device_ptr<uint>(dev_split), thrust::device_ptr<uint>(dev_split + num_particle), (uint) 0, thrust::plus<uint>());

	thrust::sort_by_key(thrust::device_ptr<uint>(dev_split),
                        thrust::device_ptr<uint>(dev_split + num_particle),
                        thrust::device_ptr<uint>(dev_index));

	if(num_split != 0)
	{
		compute_grid_size(num_split, 256, num_block, num_thread);

		adaptive_particle_kernel<<< num_block, num_thread >>>(dev_mem,
															dev_split,
															dev_index,
															num_particle,
															num_split,
															dev_status);
		num_particle=num_particle+num_split*3;
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

	float co_surface=dev_param.co_surface;
	float co_energy=dev_param.co_energy;

	if(start_index != 0xffffffff)
	{        
        uint end_index=cell_end[index];

        for(uint count_index=start_index; count_index<end_index; count_index++) 
		{
			uint p_index=dev_index[count_index];
			dev_status[p_index]=0;

			if(dev_mem[p_index].level != 1 && dev_mem[p_index].energy*co_energy+dev_mem[p_index].surface*co_surface < dev_param.merge_criteria)
			{
				cell_merge[index]++;
			}
        }

		if(cell_merge[index] < 4)
		{
			return;
		}

		float num_merge=cell_merge[index] / 4;
		
		uint current=start_index;
		uint p1st;
		uint p2nd;
		uint p3rd;
		uint p4th;
		uint p_index;

		while(num_merge > 0)
		{
			//1
			while(current < end_index)
			{
				p_index=dev_index[current];
				if(dev_mem[p_index].level != 1 && dev_mem[p_index].energy*co_energy+dev_mem[p_index].surface*co_surface < dev_param.merge_criteria)
				{
					p1st=p_index;
					current++;

					break;
				}

				current++;
			}

			//2
			while(current < end_index)
			{
				p_index=dev_index[current];
				if(dev_mem[p_index].level != 1 && dev_mem[p_index].energy*co_energy+dev_mem[p_index].surface*co_surface < dev_param.merge_criteria)
				{
					p2nd=p_index;
					current++;

					break;
				}

				current++;
			}

			//3
			while(current < end_index)
			{
				p_index=dev_index[current];
				if(dev_mem[p_index].level != 1 && dev_mem[p_index].energy*co_energy+dev_mem[p_index].surface*co_surface < dev_param.merge_criteria)
				{
					p3rd=p_index;
					current++;

					break;
				}

				current++;
			}

			//4
			while(current < end_index)
			{
				p_index=dev_index[current];
				if(dev_mem[p_index].level != 1 && dev_mem[p_index].energy*co_energy+dev_mem[p_index].surface*co_surface < dev_param.merge_criteria)
				{
					p4th=p_index;
					current++;

					break;
				}

				current++;
			}

			dev_mem[p1st].level=1;

			/*dev_mem[p1st].pos=(dev_mem[p1st].pos+dev_mem[p2nd].pos)/2;
			dev_mem[p1st].vel=(dev_mem[p1st].vel+dev_mem[p2nd].vel)/2;
			dev_mem[p1st].ev=(dev_mem[p1st].ev+dev_mem[p2nd].ev)/2;
			dev_mem[p1st].acc=(dev_mem[p1st].acc+dev_mem[p2nd].acc)/2;*/

			//////

			float V1;
			float V2;
			float V3;
			float V4;

			if(dev_mem[p1st].level == 1)
			{
				V1=dev_param.large_mass/dev_mem[p1st].dens;
			}

			if(dev_mem[p1st].level == 2)
			{
				V1=dev_param.small_mass/dev_mem[p1st].dens;
			}

			if(dev_mem[p2nd].level == 1)
			{
				V2=dev_param.large_mass/dev_mem[p2nd].dens;
			}

			if(dev_mem[p2nd].level == 2)
			{
				V2=dev_param.small_mass/dev_mem[p2nd].dens;
			}

			if(dev_mem[p3rd].level == 1)
			{
				V3=dev_param.large_mass/dev_mem[p3rd].dens;
			}

			if(dev_mem[p3rd].level == 2)
			{
				V3=dev_param.small_mass/dev_mem[p3rd].dens;
			}

			if(dev_mem[p4th].level == 1)
			{
				V4=dev_param.large_mass/dev_mem[p4th].dens;
			}

			if(dev_mem[p4th].level == 2)
			{
				V4=dev_param.small_mass/dev_mem[p4th].dens;
			}

			dev_mem[p1st].pos=(dev_mem[p1st].pos*V1+dev_mem[p2nd].pos*V2+dev_mem[p3rd].pos*V3+dev_mem[p4th].pos*V4)/(V1+V2+V3+V4);
			dev_mem[p1st].vel=(dev_mem[p1st].vel*V1+dev_mem[p2nd].vel*V2+dev_mem[p3rd].vel*V3+dev_mem[p4th].vel*V4)/(V1+V2+V3+V4);
			dev_mem[p1st].ev=(dev_mem[p1st].ev*V1+dev_mem[p2nd].ev*V2+dev_mem[p3rd].ev*V3+dev_mem[p4th].ev*V4)/(V1+V2+V3+V4);
			dev_mem[p1st].acc=(dev_mem[p1st].acc*V1+dev_mem[p2nd].acc*V2+dev_mem[p3rd].acc*V3+dev_mem[p4th].acc*V4)/(V1+V2+V3+V4);

			//////

			dev_status[p1st]=0;

			dev_status[p2nd]=1;
			dev_status[p3rd]=1;
			dev_status[p4th]=1;

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

__device__
void integrate_cell_dens(float3 dens_pos,
						uint index,
						uint3 cell_pos,
						Particle *dev_mem,
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
						float *dev_dens)
{
	Particle *p;
	uint particle_index;

	float radius2;
	float mass;
	float poly6_value;

	uint cell_index=calc_grid_hash(cell_pos);

	uint start_index=cell_start[cell_index];

	if(start_index != 0xffffffff)
	{      
		uint end_index=cell_end[cell_index];

        for(uint count_index=start_index; count_index<end_index; count_index++) 
		{
			particle_index=dev_index[count_index];
			p=&(dev_mem[particle_index]);
			
			if(p->level == 1)
			{
				radius2=dev_param.large_radius*dev_param.large_radius;
				//radius2=dev_param.large_kernel_2;
				mass=dev_param.large_mass;
				poly6_value=dev_param.large_poly6_radius;
			}

			if(p->level == 2)
			{
				radius2=dev_param.small_radius*dev_param.small_radius;
				//radius2=dev_param.small_kernel_2;
				mass=dev_param.small_mass;
				poly6_value=dev_param.small_poly6_radius;
			}

			float3 rel_pos=p->pos-dens_pos;
			float r2=rel_pos.x*rel_pos.x+rel_pos.y*rel_pos.y+rel_pos.z*rel_pos.z;

			if(r2 < radius2)
			{
				dev_dens[index]+=mass*poly6_value*pow(radius2-r2, 3);
			}
		}
	}
}

__global__
void integrate_dens_kernel(Particle *dev_mem,
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
					float3 *dev_dens_pos)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= tot_dens)
	{
		return;
	}

	uint vox_x=index % (dev_param.row_dens*dev_param.col_dens) % dev_param.row_dens;
	uint vox_y=index % (dev_param.row_dens*dev_param.col_dens) / dev_param.row_dens;
	uint vox_z=index / (dev_param.row_dens*dev_param.col_dens);

	dev_dens[index]=0.0f;

	if(vox_x==0 || vox_y==0 || vox_z == 0 || vox_x==dev_param.row_dens || vox_y==dev_param.col_dens || vox_z==dev_param.len_dens)
	{
		return;
	}

	float3 dens_pos=dev_dens_pos[index];

	uint3 cell_pos;
	cell_pos.x=(uint)floor((float)vox_x / (float)(dev_param.row_dens) * dev_param.row_cell);
	cell_pos.y=(uint)floor((float)vox_y / (float)(dev_param.col_dens) * dev_param.col_cell);
	cell_pos.z=(uint)floor((float)vox_z / (float)(dev_param.len_dens) * dev_param.len_cell);

	uint count;
	uint3 neighbor_pos[27];
	uint neighbor_status[27];

	for(count=0; count<27; count++)
	{
		neighbor_status[count]=1;
	}

	neighbor_pos[0].x=cell_pos.x;
	neighbor_pos[0].y=cell_pos.y;
	neighbor_pos[0].z=cell_pos.z;

	neighbor_pos[1].x=cell_pos.x-1;
	neighbor_pos[1].y=cell_pos.y-1;
	neighbor_pos[1].z=cell_pos.z;

	neighbor_pos[2].x=cell_pos.x-1;
	neighbor_pos[2].y=cell_pos.y;
	neighbor_pos[2].z=cell_pos.z;

	neighbor_pos[3].x=cell_pos.x;
	neighbor_pos[3].y=cell_pos.y-1;
	neighbor_pos[3].z=cell_pos.z;

	neighbor_pos[4].x=cell_pos.x+1;
	neighbor_pos[4].y=cell_pos.y+1;
	neighbor_pos[4].z=cell_pos.z;

	neighbor_pos[5].x=cell_pos.x+1;
	neighbor_pos[5].y=cell_pos.y;
	neighbor_pos[5].z=cell_pos.z;

	neighbor_pos[6].x=cell_pos.x;
	neighbor_pos[6].y=cell_pos.y+1;
	neighbor_pos[6].z=cell_pos.z;

	neighbor_pos[7].x=cell_pos.x+1;
	neighbor_pos[7].y=cell_pos.y-1;
	neighbor_pos[7].z=cell_pos.z;

	neighbor_pos[8].x=cell_pos.x-1;
	neighbor_pos[8].y=cell_pos.y+1;
	neighbor_pos[8].z=cell_pos.z;

	neighbor_pos[9].x=cell_pos.x;
	neighbor_pos[9].y=cell_pos.y;
	neighbor_pos[9].z=cell_pos.z-1;

	neighbor_pos[10].x=cell_pos.x-1;
	neighbor_pos[10].y=cell_pos.y-1;
	neighbor_pos[10].z=cell_pos.z-1;

	neighbor_pos[11].x=cell_pos.x-1;
	neighbor_pos[11].y=cell_pos.y;
	neighbor_pos[11].z=cell_pos.z-1;

	neighbor_pos[12].x=cell_pos.x;
	neighbor_pos[12].y=cell_pos.y-1;
	neighbor_pos[12].z=cell_pos.z-1;

	neighbor_pos[13].x=cell_pos.x+1;
	neighbor_pos[13].y=cell_pos.y+1;
	neighbor_pos[13].z=cell_pos.z-1;

	neighbor_pos[14].x=cell_pos.x+1;
	neighbor_pos[14].y=cell_pos.y;
	neighbor_pos[14].z=cell_pos.z-1;

	neighbor_pos[15].x=cell_pos.x;
	neighbor_pos[15].y=cell_pos.y+1;
	neighbor_pos[15].z=cell_pos.z-1;

	neighbor_pos[16].x=cell_pos.x+1;
	neighbor_pos[16].y=cell_pos.y-1;
	neighbor_pos[16].z=cell_pos.z-1;

	neighbor_pos[17].x=cell_pos.x-1;
	neighbor_pos[17].y=cell_pos.y+1;
	neighbor_pos[17].z=cell_pos.z-1;

	neighbor_pos[18].x=cell_pos.x;
	neighbor_pos[18].y=cell_pos.y;
	neighbor_pos[18].z=cell_pos.z+1;

	neighbor_pos[19].x=cell_pos.x-1;
	neighbor_pos[19].y=cell_pos.y-1;
	neighbor_pos[19].z=cell_pos.z+1;

	neighbor_pos[20].x=cell_pos.x-1;
	neighbor_pos[20].y=cell_pos.y;
	neighbor_pos[20].z=cell_pos.z+1;

	neighbor_pos[21].x=cell_pos.x;
	neighbor_pos[21].y=cell_pos.y-1;
	neighbor_pos[21].z=cell_pos.z+1;

	neighbor_pos[22].x=cell_pos.x+1;
	neighbor_pos[22].y=cell_pos.y+1;
	neighbor_pos[22].z=cell_pos.z+1;

	neighbor_pos[23].x=cell_pos.x+1;
	neighbor_pos[23].y=cell_pos.y;
	neighbor_pos[23].z=cell_pos.z+1;

	neighbor_pos[24].x=cell_pos.x;
	neighbor_pos[24].y=cell_pos.y+1;
	neighbor_pos[24].z=cell_pos.z+1;

	neighbor_pos[25].x=cell_pos.x+1;
	neighbor_pos[25].y=cell_pos.y-1;
	neighbor_pos[25].z=cell_pos.z+1;

	neighbor_pos[26].x=cell_pos.x-1;
	neighbor_pos[26].y=cell_pos.y+1;
	neighbor_pos[26].z=cell_pos.z+1;

	if(cell_pos.x == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[2]=0;
		neighbor_status[8]=0;

		neighbor_status[10]=0;
		neighbor_status[11]=0;
		neighbor_status[17]=0;

		neighbor_status[19]=0;
		neighbor_status[20]=0;
		neighbor_status[26]=0;
	}

	if(cell_pos.x == dev_param.row_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[5]=0;
		neighbor_status[7]=0;

		neighbor_status[13]=0;
		neighbor_status[14]=0;
		neighbor_status[16]=0;

		neighbor_status[22]=0;
		neighbor_status[23]=0;
		neighbor_status[25]=0;
	}

	if(cell_pos.y == 0)
	{
		neighbor_status[1]=0;
		neighbor_status[3]=0;
		neighbor_status[7]=0;

		neighbor_status[10]=0;
		neighbor_status[12]=0;
		neighbor_status[16]=0;

		neighbor_status[19]=0;
		neighbor_status[21]=0;
		neighbor_status[25]=0;
	}

	if(cell_pos.y == dev_param.col_cell-1)
	{
		neighbor_status[4]=0;
		neighbor_status[6]=0;
		neighbor_status[8]=0;

		neighbor_status[13]=0;
		neighbor_status[15]=0;
		neighbor_status[17]=0;

		neighbor_status[22]=0;
		neighbor_status[24]=0;
		neighbor_status[26]=0;
	}

	if(cell_pos.z == 0)
	{
		neighbor_status[9]=0;
		neighbor_status[10]=0;
		neighbor_status[11]=0;

		neighbor_status[12]=0;
		neighbor_status[13]=0;
		neighbor_status[14]=0;

		neighbor_status[15]=0;
		neighbor_status[16]=0;
		neighbor_status[17]=0;
	}

	if(cell_pos.z == dev_param.len_cell-1)
	{
		neighbor_status[18]=0;
		neighbor_status[19]=0;
		neighbor_status[20]=0;

		neighbor_status[21]=0;
		neighbor_status[22]=0;
		neighbor_status[23]=0;

		neighbor_status[24]=0;
		neighbor_status[25]=0;
		neighbor_status[26]=0;
	}

	for(count=0; count<27; count++)
	{
		if(neighbor_status[count] == 0)
		{
			continue;
		}

		integrate_cell_dens(dens_pos,
							index,
							neighbor_pos[count],
							dev_mem,
							dev_hash,
							dev_index,
							cell_start,
							cell_end,
							num_particle,
							total_cell,
							row_dens,
							col_dens,
							len_dens,
							tot_dens,
							den_size,
							dev_dens);
	}

	dev_dens[index]=dev_dens[index];
	/*if(dev_dens[index] > 0.0f)
	{
		dev_dens[index]=1.0f;
	}*/
}

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
					float3 *dev_dens_pos)
{
	uint num_thread;
	uint num_block;

    compute_grid_size(tot_dens, 256, num_block, num_thread);

	integrate_dens_kernel<<< num_block, num_thread >>>(dev_mem,
													dev_hash,
													dev_index,
													cell_start,
													cell_end,
													num_particle,
													total_cell,
													row_dens,
													col_dens,
													len_dens,
													tot_dens,
													den_size,
													dev_dens,
													dev_dens_pos);
}

__global__
void integrate_normall(float *dev_dens,
					float3 *dev_dens_pos,
					float3 *dev_dens_normal,
					uint tot_dens)
{
	uint index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= tot_dens)
	{
		return;
	}

	uint vox_x=index % (dev_param.row_dens*dev_param.col_dens) % dev_param.row_dens;
	uint vox_y=index % (dev_param.row_dens*dev_param.col_dens) / dev_param.row_dens;
	uint vox_z=index / (dev_param.row_dens*dev_param.col_dens);

	if(vox_x == 0)
	{
		uint x_b=vox_z*dev_param.row_dens*dev_param.col_dens+vox_y*dev_param.row_dens+vox_x+1;

		float dens_a=0.0f;
		float dens_b=dev_dens[x_b];

		dev_dens_normal[index].x=(dens_a-dens_b)/dev_param.den_size;
	}
	else if(vox_x == dev_param.row_dens-1)
	{
		uint x_a=vox_z*dev_param.row_dens*dev_param.col_dens+vox_y*dev_param.row_dens+vox_x-1;

		float dens_a=dev_dens[x_a];
		float dens_b=0.0f;

		dev_dens_normal[index].x=(dens_a-dens_b)/dev_param.den_size;
	}
	else
	{
		uint x_a=vox_z*dev_param.row_dens*dev_param.col_dens+vox_y*dev_param.row_dens+vox_x-1;
		uint x_b=vox_z*dev_param.row_dens*dev_param.col_dens+vox_y*dev_param.row_dens+vox_x+1;

		float dens_a=dev_dens[x_a];
		float dens_b=dev_dens[x_b];

		dev_dens_normal[index].x=(dens_a-dens_b)/dev_param.den_size;
	}

	if(vox_y == 0)
	{
		uint x_b=vox_z*dev_param.row_dens*dev_param.col_dens+(vox_y+1)*dev_param.row_dens+vox_x;

		float dens_a=0.0f;
		float dens_b=dev_dens[x_b];;

		dev_dens_normal[index].y=(dens_a-dens_b)/dev_param.den_size;
	}
	else if(vox_y == dev_param.col_dens-1)
	{
		uint x_a=vox_z*dev_param.row_dens*dev_param.col_dens+(vox_y-1)*dev_param.row_dens+vox_x;

		float dens_a=dev_dens[x_a];;
		float dens_b=0.0f;

		dev_dens_normal[index].y=(dens_a-dens_b)/dev_param.den_size;
	}
	else
	{
		uint x_a=vox_z*dev_param.row_dens*dev_param.col_dens+(vox_y-1)*dev_param.row_dens+vox_x;
		uint x_b=vox_z*dev_param.row_dens*dev_param.col_dens+(vox_y+1)*dev_param.row_dens+vox_x;

		float dens_a=dev_dens[x_a];
		float dens_b=dev_dens[x_b];

		dev_dens_normal[index].y=(dens_a-dens_b)/dev_param.den_size;
	}

	if(vox_z == 0)
	{
		uint x_b=(vox_z+1)*dev_param.row_dens*dev_param.col_dens+vox_y*dev_param.row_dens+vox_x;

		float dens_a=0.0f;
		float dens_b=dev_dens[x_b];

		dev_dens_normal[index].z=(dens_a-dens_b)/dev_param.den_size;
	}
	else if(vox_z == dev_param.len_dens-1)
	{
		uint x_a=(vox_z-1)*dev_param.row_dens*dev_param.col_dens+vox_y*dev_param.row_dens+vox_x;

		float dens_a=dev_dens[x_a];
		float dens_b=0.0f;

		dev_dens_normal[index].z=(dens_a-dens_b)/dev_param.den_size;
	}
	else
	{
		uint x_a=(vox_z-1)*dev_param.row_dens*dev_param.col_dens+vox_y*dev_param.row_dens+vox_x;
		uint x_b=(vox_z+1)*dev_param.row_dens*dev_param.col_dens+vox_y*dev_param.row_dens+vox_x;

		float dens_a=dev_dens[x_a];
		float dens_b=dev_dens[x_b];

		dev_dens_normal[index].z=(dens_a-dens_b)/dev_param.den_size;
	}

	float normal=sqrt(dev_dens_normal[index].x*dev_dens_normal[index].x+dev_dens_normal[index].y*dev_dens_normal[index].y+dev_dens_normal[index].z*dev_dens_normal[index].z);

	if(normal == 0.0f)
	{
		dev_dens_normal[index]=make_float3(0.0f);
	}
	else
	{
		dev_dens_normal[index]=dev_dens_normal[index]/normal;
	}
}

void integrate_normal(float *dev_dens,
					float3 *dev_dens_pos,
					float3 *dev_dens_normal,
					uint tot_dens)
{
	uint num_thread;
	uint num_block;

    compute_grid_size(tot_dens, 256, num_block, num_thread);

	integrate_normall<<< num_block, num_thread >>>(dev_dens,
												dev_dens_pos,
												dev_dens_normal,
												tot_dens);
}