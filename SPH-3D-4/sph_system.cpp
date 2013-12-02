#include "sph_system.h"
#include "sph_system.cuh"
#include "sph_param.h"
#include <cutil_math.h>

SPHSystem::SPHSystem()
{
	m_num_particle=0;
	m_max_particle=1000000;

	m_world_width=48;
	m_world_height=24;
	m_world_length=24;

	m_large_radius=1.2f;
	m_small_radius=0.8f;

	m_large_kernel=1.2f;
	m_large_mass=1.0f;

	m_small_kernel=1.0f;
	m_small_mass=0.25f;

	m_cell_size=m_large_kernel;
	m_row_cell=(uint)ceil(m_world_width/m_cell_size);
	m_col_cell=(uint)ceil(m_world_height/m_cell_size);
	m_len_cell=(uint)ceil(m_world_length/m_cell_size);
	m_total_cell=m_row_cell*m_col_cell*m_len_cell;

	m_world_width=m_row_cell*m_cell_size;
	m_world_height=m_col_cell*m_cell_size;
	m_world_length=m_len_cell*m_cell_size;

	m_host_mem=(Particle *)malloc(sizeof(Particle)*m_max_particle);
	alloc_array((void**)&(m_dev_mem), sizeof(Particle)*m_max_particle);

	alloc_array((void**)&m_dev_hash, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_dev_index, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_cell_start, sizeof(uint)*m_total_cell);
	alloc_array((void**)&m_cell_end, sizeof(uint)*m_total_cell);

	alloc_array((void**)&m_dev_status, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_trans_vel, sizeof(float3)*m_max_particle);

	alloc_array((void**)&m_dev_split, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_dev_split_index, sizeof(uint)*m_max_particle);
	alloc_array((void**)&m_dev_merge, sizeof(uint)*m_max_particle);

	m_gravity=make_float3(0.0f, -4.0f, 0.0f);
	m_wall_damping=-0.6f;
	m_rest_density=0.5f;
	m_gas_constant=1.0f;
	m_time_step=0.1f;
	m_viscosity=1.0f;
	m_surface_tension=0.8f;

	m_co_surface=1.5f;
	m_co_energy=1.0f;

	m_split_criteria=1.0f;//0.8f;
	m_merge_criteria=0.6f;//0.4f;

	m_large_poly6=315.0f/(64.0f * PI * pow(m_large_kernel, 9));
	m_large_spiky=-45.0f/(PI * pow(m_large_kernel, 6));
	m_large_visco=45.0f/(PI * pow(m_large_kernel, 6));

	m_large_grad_poly6=-945.0f / (32.0f * PI * pow(m_large_kernel, 9));
	m_large_lapc_poly6=-945.0f / (8.0f * PI * pow(m_large_kernel, 9));

	m_small_poly6=315.0f/(64.0f * PI * pow(m_small_kernel, 9));
	m_small_spiky=-45.0f/(PI * pow(m_small_kernel, 6));
	m_small_visco=45.0f/(PI * pow(m_small_kernel, 6));

	m_small_grad_poly6=-945.0f / (32.0f * PI * pow(m_small_kernel, 9));
	m_small_lapc_poly6=-945.0f / (8.0f * PI * pow(m_small_kernel, 9));

	m_large_kernel_2=m_large_kernel*m_large_kernel;
	m_large_kernel_6=pow(m_large_kernel, 6);

	m_small_kernel_2=m_small_kernel*m_small_kernel;
	m_small_kernel_6=pow(m_small_kernel, 6);

	m_large_poly6_radius=315.0f/(64.0f * PI * pow(m_large_radius, 9));;
	m_small_poly6_radius=315.0f/(64.0f * PI * pow(m_small_radius, 9));;

	m_dens_mul=4;
	m_den_size=m_cell_size/m_dens_mul;

	m_row_dens=m_row_cell*m_dens_mul+1;
	m_col_dens=m_col_cell*m_dens_mul+1;
	m_len_dens=m_len_cell*m_dens_mul+1;
	m_tot_dens=m_row_dens*m_col_dens*m_len_dens;

	m_host_dens=(float *)malloc(sizeof(float)*m_tot_dens);
	alloc_array((void**)&(m_dev_dens), sizeof(float)*m_tot_dens);
	memset(m_host_dens, 0.0f, sizeof(float)*m_tot_dens);

	m_host_dens_pos=(float3 *)malloc(sizeof(float3)*m_tot_dens);
	alloc_array((void**)&(m_dev_dens_pos), sizeof(float3)*m_tot_dens);
	m_host_dens_normal=(float3 *)malloc(sizeof(float3)*m_tot_dens);
	alloc_array((void**)&(m_dev_dens_normal), sizeof(float3)*m_tot_dens);

	m_num_lines=0;
	m_host_line0=(float3 *)malloc(sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*2);
	m_host_line1=(float3 *)malloc(sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*2);

	m_num_triangle=0;
	m_host_triangle0=(float3 *)malloc(sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*3);
	m_host_triangle1=(float3 *)malloc(sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*3);
	m_host_triangle2=(float3 *)malloc(sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*3);

	m_host_triangle_normal0=(float3 *)malloc(sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*3);
	m_host_triangle_normal1=(float3 *)malloc(sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*3);
	m_host_triangle_normal2=(float3 *)malloc(sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*3);

	m_sphere1.pos=make_float3(11.0f, 3.0f, 4.0f);
	m_sphere1.radius=2.0f;
	m_sphere1.damping=1.0f;

	m_sphere2.pos=make_float3(11.0f, 3.0f, 12.0f);
	m_sphere2.radius=2.0f;
	m_sphere2.damping=1.0f;

	m_sphere3.pos=make_float3(18.0f, 3.0f, 8.0f);
	m_sphere3.radius=2.0f;
	m_sphere3.damping=1.0f;

	m_host_param=new SysParam();
	m_host_param->num_particle=m_num_particle;
	m_host_param->max_particle=m_max_particle;

	m_host_param->world_width=m_world_width;
	m_host_param->world_height=m_world_height;
	m_host_param->world_length=m_world_length;

	m_host_param->large_radius=m_large_radius;
	m_host_param->small_radius=m_small_radius;

	m_host_param->large_kernel=m_large_kernel;
	m_host_param->small_kernel=m_small_kernel;
	m_host_param->large_mass=m_large_mass;
	m_host_param->small_mass=m_small_mass;

	m_host_param->cell_size=m_cell_size;
	m_host_param->row_cell=m_row_cell;
	m_host_param->col_cell=m_col_cell;
	m_host_param->len_cell=m_len_cell;
	m_host_param->total_cell=m_total_cell;

	m_host_param->gravity=m_gravity;
	m_host_param->wall_damping=m_wall_damping;
	m_host_param->rest_density=m_rest_density;
	m_host_param->gas_constant=m_gas_constant;
	m_host_param->time_step=m_time_step;
	m_host_param->viscosity=m_viscosity;
	m_host_param->surface_tension=m_surface_tension;

	m_host_param->co_surface=m_co_surface;
	m_host_param->co_energy=m_co_energy;

	m_host_param->split_criteria=m_split_criteria;
	m_host_param->merge_criteria=m_merge_criteria;

	m_host_param->large_poly6=m_large_poly6;
	m_host_param->large_spiky=m_large_spiky;
	m_host_param->large_visco=m_large_visco;

	m_host_param->large_grad_poly6=m_large_grad_poly6;
	m_host_param->large_lapc_poly6=m_large_lapc_poly6;

	m_host_param->small_poly6=m_small_poly6;
	m_host_param->small_spiky=m_small_spiky;
	m_host_param->small_visco=m_small_visco;

	m_host_param->small_grad_poly6=m_small_grad_poly6;
	m_host_param->small_lapc_poly6=m_small_lapc_poly6;

	m_host_param->large_kernel_2=m_large_kernel_2;
	m_host_param->large_kernel_6=m_large_kernel_6;

	m_host_param->small_kernel_2=m_small_kernel_2;
	m_host_param->small_kernel_6=m_small_kernel_6;

	m_host_param->large_poly6_radius=m_large_poly6_radius;
	m_host_param->small_poly6_radius=m_small_poly6_radius;

	m_host_param->dens_mul=m_dens_mul;
	m_host_param->den_size=m_den_size;
	m_host_param->row_dens=m_row_dens;
	m_host_param->col_dens=m_col_dens;
	m_host_param->len_dens=m_len_dens;
	m_host_param->tot_dens=m_tot_dens;

	m_host_param->sphere1=m_sphere1;
	m_host_param->sphere2=m_sphere2;
	m_host_param->sphere3=m_sphere3;

	m_host_param->split_energy=0.8f;
	m_host_param->merge_energy=0.02f;

	m_host_param->force_waiting_left=false;
	m_host_param->force_waiting_right=false;
	m_host_param->use_cylinder=false;

	m_sys_running=0;

	m_trans_once=0;
	m_use_split=0;
	m_use_merge=0;

	m_disp_mode=1; //1-particle 2-dens

	//////

	for(uint count_x=0; count_x<m_row_dens; count_x++)
	{
		for(uint count_y=0; count_y<m_col_dens; count_y++)
		{
			for(uint count_z=0; count_z<m_len_dens; count_z++)
			{
				uint index=count_z*m_row_dens*m_col_dens+count_y*m_row_dens+count_x;
				m_host_dens_pos[index].x=(float)count_x*m_den_size;
				m_host_dens_pos[index].y=(float)count_y*m_den_size;
				m_host_dens_pos[index].z=(float)count_z*m_den_size;
			}
		}
	}

	copy_array(m_dev_dens_pos, m_host_dens_pos, sizeof(float3)*m_tot_dens, CUDA_HOST_TO_DEV);
}

SPHSystem::~SPHSystem()
{
	free(m_host_mem);
	cudaFree(m_dev_mem);
	cudaFree(m_dev_hash);
	cudaFree(m_dev_index);
	cudaFree(m_cell_start);
	cudaFree(m_cell_end);
	cudaFree(m_dev_status);
	cudaFree(m_trans_vel);
	cudaFree(m_dev_split);
	cudaFree(m_dev_split_index);
	cudaFree(m_dev_merge);
	free(m_host_param);
}

void SPHSystem::add_box_particle()
{
	/*float3 pos;
	float3 vel=make_float3(0.0f);
	float space=m_small_kernel;

	//large 0.8
	//small 0.6

	for(pos.x=space; pos.x<=m_world_width/3; pos.x+=space*0.6f)
	{
		for(pos.y=space; m_world_height-pos.y>=space; pos.y+=space*0.6f)
		{
			for(pos.z=space; m_world_length-pos.z>=space; pos.z+=space*0.6f)
			{
				add_particle(pos, vel, 2);
			}
		}
	}

	copy_array(m_dev_mem, m_host_mem, sizeof(Particle)*m_num_particle, CUDA_HOST_TO_DEV);*/

	float3 pos;
	float3 vel=make_float3(0.0f);
	float space=m_small_kernel;

	//large 0.8
	//small 0.6

	for(pos.x=space; m_world_width-pos.x>=space; pos.x+=space*0.5f)
	{
		for(pos.y=space; pos.y<m_world_height/4; pos.y+=space*0.5f)
		{
			for(pos.z=space; m_world_length-pos.z>=space; pos.z+=space*0.5f)
			{
				add_particle(pos, vel, 2);
			}
		}
	}

	copy_array(m_dev_mem, m_host_mem, sizeof(Particle)*m_num_particle, CUDA_HOST_TO_DEV);
}

void SPHSystem::add_new_particle()
{
	float r=10.0f;;
	float spacing=m_small_kernel*0.5;

	float3 vel=make_float3(0.0f, -10.0f, 0.0f);

    for(float z=-r; z<=r; z+=spacing) 
	{
        for(float y=-r; y<=r; y+=spacing) 
		{
            for(float x=-r; x<=r; x+=spacing) 
			{
                float dx = x;
                float dy = y;
                float dz = z;

                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_large_kernel*0.01f;

                if (l <= 5.0f) 
				{
					float3 pos;
					pos.x=24.0f + dx;
					pos.y=18.0f + dy;
					pos.z=12.0f + dz;
					add_particle(pos, vel, 2);

                    /*m_hPos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter; 
                    m_hPos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+3] = pos[3];*/
					
					m_num_particle++;
                }
            }
        }
    }

	copy_array(m_dev_mem, m_host_mem, sizeof(Particle)*m_num_particle, CUDA_HOST_TO_DEV);
}

void SPHSystem::add_particle(float3 pos, float3 vel, uint level)
{
	Particle *p=&(m_host_mem[m_num_particle]);

	p->pos=pos;
	p->vel=vel;
	p->ev=make_float3(0.0f);
	p->acc=make_float3(0.0f);
	p->dens=0.0f;
	p->press=0.0f;
	p->surface=0.0f;
	p->tension=0.0f;
	p->level=level;

	m_num_particle++;
}

void SPHSystem::animation()
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

	if(m_disp_mode == 2)
	{
		integrate_dens(m_dev_mem,
						m_dev_hash,
						m_dev_index,
						m_cell_start,
						m_cell_end,
						m_num_particle,
						m_total_cell,
						m_row_dens,
						m_col_dens,
						m_len_dens,
						m_tot_dens,
						m_den_size,
						m_dev_dens,
						m_dev_dens_pos);

		integrate_normal(m_dev_dens,
						m_dev_dens_pos,
						m_dev_dens_normal,
						m_tot_dens);

		copy_array(m_host_dens, m_dev_dens, sizeof(float)*m_tot_dens, CUDA_DEV_TO_HOST);
		copy_array(m_host_dens_normal, m_dev_dens_normal, sizeof(float3)*m_tot_dens, CUDA_DEV_TO_HOST);

		marching_cube();
	}

	compute_energy(m_dev_mem,
					m_dev_hash,
					m_dev_index,
					m_cell_start,
					m_cell_end,
					m_num_particle,
					m_total_cell,
					m_trans_vel);

	if(m_use_split == 1)
	{
		m_num_particle=adaptive_particle(m_dev_mem,
								m_dev_hash,
								m_dev_split_index,
								m_cell_start,
								m_cell_end,
								m_num_particle,
								m_total_cell,
								m_dev_split,
								m_dev_status);
	}

	if(m_use_merge == 1)
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

	m_host_param->force_waiting_left=false;
	m_host_param->force_waiting_right=false;
}

float fGetOffset(float fValue1, float fValue2, float fValueDesired)
{
        double fDelta = fValue2 - fValue1;

        if(fDelta == 0.0)
        {
                return 0.5;
        }
        return (fValueDesired - fValue1)/fDelta;
}

void SPHSystem::marching_cube_cell(uint count_x, uint count_y, uint count_z)
{
	extern int aiCubeEdgeFlags[256];
	extern int a2iTriangleConnectionTable[256][16];
	
	static const float a2fVertexOffset[8][3] =
	{
		{0.0, 0.0, 0.0},{1.0, 0.0, 0.0},{1.0, 1.0, 0.0},{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},{1.0, 0.0, 1.0},{1.0, 1.0, 1.0},{0.0, 1.0, 1.0}
	};

	static const int a2iEdgeConnection[12][2] = 
	{
		{0,1}, {1,2}, {2,3}, {3,0},
		{4,5}, {5,6}, {6,7}, {7,4},
		{0,4}, {1,5}, {2,6}, {3,7}
	};

	static const float a2fEdgeDirection[12][3] =
	{
		{1.0, 0.0, 0.0},{0.0, 1.0, 0.0},{-1.0, 0.0, 0.0},{0.0, -1.0, 0.0},
		{1.0, 0.0, 0.0},{0.0, 1.0, 0.0},{-1.0, 0.0, 0.0},{0.0, -1.0, 0.0},
		{0.0, 0.0, 1.0},{0.0, 0.0, 1.0},{ 0.0, 0.0, 1.0},{0.0,  0.0, 1.0}
	};
	
	float afCubeValue[8];
	float3 afCubePos[8];
	float3 afCubeNormal[8];
	float3 asEdgeVertex[12];
	float3 asEdgeVertexNormal[12];

	for(int iVertex = 0; iVertex < 8; iVertex++)
	{
		uint index=(count_z+(uint)(a2fVertexOffset[iVertex][2]))*m_row_dens*m_col_dens 
			+ (count_y+(uint)(a2fVertexOffset[iVertex][1]))*m_row_dens 
			+ (count_x+(uint)(a2fVertexOffset[iVertex][0]));
		
		afCubeValue[iVertex] = m_host_dens[index];
		afCubePos[iVertex] = m_host_dens_pos[index];
		afCubeNormal[iVertex] = m_host_dens_normal[index];
	}

	int iFlagIndex = 0;
	for(int iVertexTest = 0; iVertexTest < 8; iVertexTest++)
	{
		if(afCubeValue[iVertexTest] > ISOVALUE) 
		{
			iFlagIndex |= 1<<iVertexTest;
		}
	}

	int iEdgeFlags = aiCubeEdgeFlags[iFlagIndex];

	if(iEdgeFlags == 0) 
	{
		return;
	}

	for(int iEdge = 0; iEdge < 12; iEdge++)
	{
		if(iEdgeFlags & (1<<iEdge))
		{
			asEdgeVertex[iEdge]=(afCubePos[ a2iEdgeConnection[iEdge][0] ] + afCubePos[ a2iEdgeConnection[iEdge][1] ])/2;
			asEdgeVertexNormal[iEdge]=(afCubeNormal[ a2iEdgeConnection[iEdge][0] ] + afCubeNormal[ a2iEdgeConnection[iEdge][1] ])/2;

			//asEdgeVertex[iEdge]=(afCubePos[ a2iEdgeConnection[iEdge][0] ]*(afCubeValue[ a2iEdgeConnection[iEdge][0] ]-ISOVALUE)- afCubePos[ a2iEdgeConnection[iEdge][1] ]*(afCubeValue[ a2iEdgeConnection[iEdge][1] ]-ISOVALUE))/((afCubeValue[ a2iEdgeConnection[iEdge][0] ]-ISOVALUE)-(afCubeValue[ a2iEdgeConnection[iEdge][1] ]-ISOVALUE));
		}
	}

	for(int iTriangle = 0; iTriangle < 5; iTriangle++)
	{
		if(a2iTriangleConnectionTable[iFlagIndex][3*iTriangle] < 0)
			break;

		uint fVertex = a2iTriangleConnectionTable[iFlagIndex][3*iTriangle+0];
		uint sVertex = a2iTriangleConnectionTable[iFlagIndex][3*iTriangle+1];
		uint tVertex = a2iTriangleConnectionTable[iFlagIndex][3*iTriangle+2];

		m_host_triangle0[m_num_triangle]=asEdgeVertex[fVertex];
		m_host_triangle1[m_num_triangle]=asEdgeVertex[sVertex];
		m_host_triangle2[m_num_triangle]=asEdgeVertex[tVertex];
		m_host_triangle_normal0[m_num_triangle]=asEdgeVertexNormal[fVertex];
		m_host_triangle_normal1[m_num_triangle]=asEdgeVertexNormal[sVertex];
		m_host_triangle_normal2[m_num_triangle]=asEdgeVertexNormal[tVertex];
		m_num_triangle++;

		/*m_host_line0[m_num_lines]=asEdgeVertex[fVertex];
		m_host_line1[m_num_lines]=asEdgeVertex[sVertex];
		m_num_lines++;

		m_host_line0[m_num_lines]=asEdgeVertex[fVertex];
		m_host_line1[m_num_lines]=asEdgeVertex[tVertex];
		m_num_lines++;

		m_host_line0[m_num_lines]=asEdgeVertex[sVertex];
		m_host_line1[m_num_lines]=asEdgeVertex[tVertex];
		m_num_lines++;*/
	}
}

void SPHSystem::marching_cube()
{
	m_num_lines=0;
	memset(m_host_line0, 0, sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*2);
	memset(m_host_line1, 0, sizeof(float3)*m_row_dens*m_col_dens*m_len_dens*2);

	m_num_triangle=0;
	memset(m_host_triangle0, 0, sizeof(float3)*m_row_dens*m_col_dens*m_len_dens);
	memset(m_host_triangle1, 0, sizeof(float3)*m_row_dens*m_col_dens*m_len_dens);
	memset(m_host_triangle2, 0, sizeof(float3)*m_row_dens*m_col_dens*m_len_dens);

	uint count_x;
	uint count_y;
	uint count_z;

	for(count_x=0; count_x<m_row_dens-1; count_x++)
	{
		for(count_y=0; count_y<m_col_dens-1; count_y++)
		{
			for(count_z=0; count_z<m_len_dens-1; count_z++)
			{
				marching_cube_cell(count_x, count_y, count_z);
			}
		}
	}
}

int aiCubeEdgeFlags[256]=
{
        0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 
        0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 
        0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 
        0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 
        0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, 
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460, 
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230, 
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

int a2iTriangleConnectionTable[256][16] =  
{
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
        {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
        {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
        {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
        {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
        {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
        {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
        {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
        {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
        {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
        {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
        {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
        {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
        {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
        {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
        {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
        {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
        {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
        {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
        {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
        {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
        {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
        {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
        {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
        {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
        {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
        {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
        {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
        {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
        {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
        {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
        {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
        {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
        {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
        {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
        {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
        {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
        {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
        {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
        {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
        {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
        {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
        {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
        {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
        {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
        {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
        {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
        {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
        {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
        {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
        {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
        {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
        {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
        {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
        {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
        {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
        {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
        {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
        {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
        {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
        {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
        {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
        {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
        {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
        {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
        {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
        {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
        {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
        {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
        {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
        {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
        {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
        {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
        {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
        {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
        {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
        {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
        {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
        {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
        {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
        {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
        {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
        {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
        {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
        {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
        {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
        {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
        {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
        {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
        {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
        {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
        {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
        {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
        {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
        {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
        {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
        {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
        {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
        {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
        {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
        {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
        {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
        {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
        {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
        {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
        {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
        {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
        {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
        {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};