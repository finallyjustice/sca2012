#include "sph_timer.h"
#include "sph_system.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include "sph_header.h"
#include "bitmap.h"

float window_width=1000;
float window_height=600;

GLfloat xRot = 0.0f;
GLfloat yRot = 0.0f;
GLfloat xTrans = 0;
GLfloat yTrans = 0;
GLfloat zTrans = -32.0;

int ox;
int oy;
int buttonState;
GLfloat xRotLength = 0.0f;
GLfloat yRotLength = 0.0f;

float3 real_world_origin;
float3 real_world_side;
float3 sim_ratio;

float world_width;
float world_height;
float world_length;

SPHSystem *sph_system;

Timer *sph_timer;
char *window_title;

unsigned long *screenData;
int frameNum;

GLUquadricObj *cylinder;

int create_video=0;

GLuint dTex;

void screenShot(int n)
{
	Bitmap b;
	char filename[] = "Screen_Shots/000000.bmp";
	int i = 18;

	while (n > 0) {
		filename[i--] += n % 10;
		n /= 10;
	}

	glReadPixels(0, 0, window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, screenData);
	b.makeBMP(screenData, window_width, window_height);
	b.Save(filename);
}

void make_picture(int n)
{
	char filename[] = "Povray_Shots/000000.pov";
	char header[] = "#include \"colors.inc\" \
					#include \"textures.inc\" \
					#include \"finish.inc\" \
					background{ Black }\
					\
					#declare Water = pigment\
					{\
						color rgb <0.9, 0.9, 0.9> transmit 0.25\
					}\
					\
					camera {\
						location <1,5,-35>   \
						look_at <0,0,0>\
					}\
					\
					light_source { <10, 10, -45> color White }\
					\
					blob\
					{\
						threshold .1\n";
					
	int i = 18;

	while (n > 0) {
		filename[i--] += n % 10;
		n /= 10;
	}

	FILE *fp;
	float x;
	float y;
	float z;

	fp=fopen(filename, "w");

	fprintf(fp, header);

	for(uint count=0; count<sph_system->m_num_particle; count++)
	{
		x=sph_system->m_host_mem[count].pos.x*sim_ratio.x+real_world_origin.x;
		y=sph_system->m_host_mem[count].pos.y*sim_ratio.y+real_world_origin.y; 
		z=sph_system->m_host_mem[count].pos.z*sim_ratio.z+real_world_origin.z;
		fprintf(fp, "sphere { <%f,%f,%f>, 0.2, 0.2 pigment {Water} }\n", x, y, z);
	}

	fprintf(fp, "finish { ambient 0.0 diffuse 0.0 specular 0.4 roughness 0.003 reflection { 0.003, 1.0 fresnel on } } interior { ior 1.33 } }\n");

	fclose(fp);
}

void make_mesh(int n)
{
	char filename[] = "Povray_Shots/000000.pov";
	char header[] = "#include \"colors.inc\" \n \
					camera { \n \
					location <5,10,-35> \n \
					look_at <0,0,0> \n \
					} \n \
					light_source {<13, 13, -20> color White} \n \
					mesh {\n";
					
	int i = 18;

	while (n > 0) {
		filename[i--] += n % 10;
		n /= 10;
	}

	FILE *fp;
	float x;
	float y;
	float z;

	fp=fopen(filename, "w");

	fprintf(fp, header);

	for(uint count=0; count<sph_system->m_num_triangle; count++)
	{
		float t0x=sph_system->m_host_triangle0[count].x*sim_ratio.x+real_world_origin.x;
		float t0y=sph_system->m_host_triangle0[count].y*sim_ratio.y+real_world_origin.y;
		float t0z=sph_system->m_host_triangle0[count].z*sim_ratio.z+real_world_origin.z;

		float t1x=sph_system->m_host_triangle1[count].x*sim_ratio.x+real_world_origin.x;
		float t1y=sph_system->m_host_triangle1[count].y*sim_ratio.y+real_world_origin.y;
		float t1z=sph_system->m_host_triangle1[count].z*sim_ratio.z+real_world_origin.z;

		float t2x=sph_system->m_host_triangle2[count].x*sim_ratio.x+real_world_origin.x;
		float t2y=sph_system->m_host_triangle2[count].y*sim_ratio.y+real_world_origin.y;
		float t2z=sph_system->m_host_triangle2[count].z*sim_ratio.z+real_world_origin.z;

		float nt0x=sph_system->m_host_triangle_normal0[count].x;
		float nt0y=sph_system->m_host_triangle_normal0[count].y;
		float nt0z=sph_system->m_host_triangle_normal0[count].z;

		float nt1x=sph_system->m_host_triangle_normal1[count].x;
		float nt1y=sph_system->m_host_triangle_normal1[count].y;
		float nt1z=sph_system->m_host_triangle_normal1[count].z;

		float nt2x=sph_system->m_host_triangle_normal2[count].x;
		float nt2y=sph_system->m_host_triangle_normal2[count].y;
		float nt2z=sph_system->m_host_triangle_normal2[count].z;

		fprintf(fp, "smooth_triangle { <%f, %f, %f>, <%f, %f, %f>, <%f, %f, %f>, <%f, %f, %f>, <%f, %f, %f>, <%f, %f, %f> }  \n", t0x, t0y, t0z, nt0x, nt0y, nt0z, t1x, t1y, t1z, nt1x, nt1y, nt1z, t2x, t2y, t2z, nt2x, nt2y, nt2z);
	}

	fprintf(fp, "pigment { color red 0.8 green 0.8 blue 0.8 transmit 0.6} finish { ambient 0.2 diffuse 0.5 } } plane { <0,-1,0>, 10 pigment {checker color White, color Black}}\n");

	fclose(fp);
}
void make_particle(int n)
{
	char filename[] = "Povray_Shots/000000.pov";
	char header[] = "#include \"colors.inc\" \n \
					camera { \n \
					location <5,10,-35> \n \
					look_at <0,0,0> \n \
					} \n \
					light_source {<13, 13, -20> color White} \n";
					
	int i = 18;

	while (n > 0) {
		filename[i--] += n % 10;
		n /= 10;
	}

	FILE *fp;
	float x;
	float y;
	float z;

	fp=fopen(filename, "w");

	fprintf(fp, header);

	for(uint count=0; count<sph_system->m_num_particle; count++)
	{
		x=sph_system->m_host_mem[count].pos.x*sim_ratio.x+real_world_origin.x;
		y=sph_system->m_host_mem[count].pos.y*sim_ratio.y+real_world_origin.y; 
		z=sph_system->m_host_mem[count].pos.z*sim_ratio.z+real_world_origin.z;

		if(sph_system->m_host_mem[count].level == 1)
		{
			fprintf(fp, "sphere { <%f, %f, %f>, %f texture { pigment { color Red } } }\n", x, y, z, 0.3f);
		}

		if(sph_system->m_host_mem[count].level == 2)
		{
			fprintf(fp, "sphere { <%f, %f, %f>, %f texture { pigment { color Yellow } } }\n", x, y, z, 0.24f);
		}
	}

	fprintf(fp, "plane { <0,-1,0>, 10 pigment {checker color White, color Black}}\n");

	fclose(fp);
}

void draw_box(float ox, float oy, float oz, float width, float height, float length)
{
    glLineWidth(1.0f);

    glBegin(GL_LINES);   
        glColor3f(1.0f, 0.0f, 0.0f);
		
		//1
        glVertex3f(ox, oy, oz);
        glVertex3f(ox+width, oy, oz);

		//2
        glVertex3f(ox, oy, oz);
        glVertex3f(ox, oy+height, oz);

		//3
        glVertex3f(ox, oy, oz);
        glVertex3f(ox, oy, oz+length);

		//4
        glVertex3f(ox+width, oy, oz);
        glVertex3f(ox+width, oy+height, oz);

		//5
        glVertex3f(ox+width, oy+height, oz);
        glVertex3f(ox, oy+height, oz);

		//6
        glVertex3f(ox, oy+height, oz+length);
        glVertex3f(ox, oy, oz+length);

		//7
        glVertex3f(ox, oy+height, oz+length);
        glVertex3f(ox, oy+height, oz);

		//8
        glVertex3f(ox+width, oy, oz);
        glVertex3f(ox+width, oy, oz+length);

		//9
        glVertex3f(ox, oy, oz+length);
        glVertex3f(ox+width, oy, oz+length);

		//10
        glVertex3f(ox+width, oy+height, oz);
        glVertex3f(ox+width, oy+height, oz+length);

		//11
        glVertex3f(ox+width, oy+height, oz+length);
        glVertex3f(ox+width, oy, oz+length);

		//12
        glVertex3f(ox, oy+height, oz+length);
        glVertex3f(ox+width, oy+height, oz+length);
    glEnd();
}
void init_sph_system()
{
	real_world_side.x=30.0f;
	real_world_side.y=20.0f;
	real_world_side.z=20.0f;

	real_world_origin.x=0.0f-real_world_side.x/2;
	real_world_origin.y=0.0f-real_world_side.y/2;
	real_world_origin.z=0.0f-real_world_side.z/2;

	sph_system=new SPHSystem();

	printf("Initialize 3D SPH Particle System:\n");
	printf("World Size	:	%f, %f, %f\n", sph_system->m_world_width, sph_system->m_world_height, sph_system->m_world_length);
	printf("Cell Size	:	%f\n", sph_system->m_cell_size);
	printf("Large Kernel:	%f\n", sph_system->m_large_kernel);
	printf("Small Kernel:	%f\n", sph_system->m_small_kernel);
	printf("Side Cell	:	%u, %u, %u\n", sph_system->m_row_cell, sph_system->m_col_cell, sph_system->m_len_cell);
	printf("Total Cell	:	%u\n", sph_system->m_total_cell);

	printf("Large Poly6 :   %f\n", sph_system->m_large_poly6);
	printf("Large Spiky :   %f\n", sph_system->m_large_spiky);
	printf("Large Visco :   %f\n", sph_system->m_large_visco);

	printf("Small Poly6 :   %f\n", sph_system->m_small_poly6);
	printf("Small Spiky :   %f\n", sph_system->m_small_spiky);
	printf("Small Visco :   %f\n", sph_system->m_small_visco);

	printf("Large Kernel2:  %f\n", sph_system->m_large_kernel_2);
	printf("Small Kernel2:  %f\n", sph_system->m_small_kernel_2);

	printf("Row Dens Num : %u\n", sph_system->m_row_dens);
	printf("Col Dens Num : %u\n", sph_system->m_col_dens);
	printf("Len Dens Num : %u\n", sph_system->m_len_dens);
	printf("Dens Size    : %f\n", sph_system->m_den_size);
	printf("Total Dens   : %u\n", sph_system->m_tot_dens);

	sph_system->add_box_particle();
	printf("New particle: %u\n", sph_system->m_num_particle);

	sph_timer=new Timer();
	window_title=(char *)malloc(sizeof(char)*50);

	screenData = new unsigned long[(int)window_width * (int)window_height];
	frameNum = 0;
}

void init()
{
	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)window_width/window_height, 10.0f, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);

	cylinder=gluNewQuadric();
	gluQuadricDrawStyle(cylinder, GLU_FILL);
}

void init_ratio()
{
	sim_ratio.x=real_world_side.x/sph_system->m_world_width;
	sim_ratio.y=real_world_side.y/sph_system->m_world_height;
	sim_ratio.z=real_world_side.z/sph_system->m_world_length;
}

void render_particles()
{

	for(uint count=0; count<sph_system->m_num_particle; count++)
	{	
		if(sph_system->m_host_mem[count].level == 1)
		{
			glPointSize(3.0f);
			//glPointSize(3.0f);
			glColor4f(0.2f, 0.2f, 1.0f, 0.7f);

			if(sph_system->m_host_mem[count].surface > sph_system->m_surface_tension)
			{
				//continue;
				//glColor3f(1.0f, 0.0f, 0.0f);
			}
			else
			{	//continue;
				//glColor3f(0.2f, 0.2f, 1.0f);
			}
			
			/*if(sph_system->m_host_mem[count].energy > sph_system->m_host_param->split_energy)
			{
				glColor3f(1.0f, 1.0f, 0.2f);
			}

			if(sph_system->m_host_mem[count].energy < sph_system->m_host_param->merge_energy)
			{
				glColor3f(0.2f, 1.0f, 1.0f);
			}*/
		}

		if(sph_system->m_host_mem[count].level == 2)
		{
			glPointSize(2.4f);
			glColor4f(1.0f, 1.0f, 0.2f, 0.7f);

			if(sph_system->m_host_mem[count].surface > sph_system->m_surface_tension)
			{
				//continue;
				//glColor3f(1.0f, 0.0f, 0.0f);
			}
			else
			{	//continue;
				//glColor3f(1.0f, 1.0f, 0.2f);
			}

			/*if(sph_system->m_host_mem[count].energy > sph_system->m_host_param->split_energy)
			{
				glColor3f(1.0f, 1.0f, 0.2f);
			}

			if(sph_system->m_host_mem[count].energy < sph_system->m_host_param->merge_energy)
			{
				glColor3f(0.2f, 1.0f, 1.0f);
			}*/
		}

		glBegin(GL_POINTS);
			glVertex3f(sph_system->m_host_mem[count].pos.x*sim_ratio.x+real_world_origin.x, 
						sph_system->m_host_mem[count].pos.y*sim_ratio.y+real_world_origin.y, 
						sph_system->m_host_mem[count].pos.z*sim_ratio.z+real_world_origin.z);
		glEnd();
	}
}

void render_density()
{

	glPointSize(8.0f);
	glColor3f(0.0f, 0.0f, 1.0f);
	for(float count_x=0; count_x<sph_system->m_row_dens; count_x++)
	{
		for(float count_y=0; count_y<sph_system->m_col_dens; count_y++)
		{
			for(float count_z=0; count_z<sph_system->m_len_dens; count_z++)
			{
				uint index=count_z*sph_system->m_row_dens*sph_system->m_col_dens+count_y*sph_system->m_row_dens+count_x;
				glBegin(GL_POINTS);
				glVertex3f(sph_system->m_host_dens_pos[index].x*sim_ratio.x+real_world_origin.x, 
							sph_system->m_host_dens_pos[index].y*sim_ratio.y+real_world_origin.y, 
							sph_system->m_host_dens_pos[index].z*sim_ratio.z+real_world_origin.z);
				glEnd();
			}
		}
	}

	return;


	uint index;
	glPointSize(8.0f);
	for(float count_x=0; count_x<sph_system->m_row_dens; count_x++)
	{
		for(float count_y=0; count_y<sph_system->m_col_dens; count_y++)
		{
			for(float count_z=0; count_z<sph_system->m_len_dens; count_z++)
			{
				index=count_z*sph_system->m_row_dens*sph_system->m_col_dens+count_y*sph_system->m_row_dens+count_x;
				if(sph_system->m_host_dens[index] == 0.0f)
				{
					continue;
				}

				float x=count_x/sph_system->m_row_dens*sph_system->m_world_width;
				float y=count_y/sph_system->m_col_dens*sph_system->m_world_height;
				float z=count_z/sph_system->m_len_dens*sph_system->m_world_length;

				glColor4f(sph_system->m_host_dens[index]*0.2f, sph_system->m_host_dens[index]*0.5f, sph_system->m_host_dens[index]*0.7f, 0.2f);
				glBegin(GL_POINTS);
					glVertex3f(x*sim_ratio.x+real_world_origin.x, 
								y*sim_ratio.y+real_world_origin.y, 
								z*sim_ratio.z+real_world_origin.z);
				glEnd();
			}
		}
	}
}

void render_mesh()
{
	glColor4f(0.7f, 0.7f, 7.0f, 0.1f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	for(uint count=0; count<sph_system->m_num_triangle; count++)
	{
		glBegin(GL_TRIANGLES);
			//glColor4f(1.0f, 0.0f, 0.0f, 0.1f);
			glNormal3f(sph_system->m_host_triangle_normal0[count].x*sim_ratio.x+real_world_origin.x, 
					sph_system->m_host_triangle_normal0[count].y*sim_ratio.y+real_world_origin.y, 
					sph_system->m_host_triangle_normal0[count].z*sim_ratio.z+real_world_origin.z);
			glVertex3f(sph_system->m_host_triangle0[count].x*sim_ratio.x+real_world_origin.x, 
					sph_system->m_host_triangle0[count].y*sim_ratio.y+real_world_origin.y, 
					sph_system->m_host_triangle0[count].z*sim_ratio.z+real_world_origin.z);

			//glColor4f(0.0f, 1.0f, 0.0f, 0.1f);
			glNormal3f(sph_system->m_host_triangle_normal1[count].x*sim_ratio.x+real_world_origin.x, 
					sph_system->m_host_triangle_normal1[count].y*sim_ratio.y+real_world_origin.y, 
					sph_system->m_host_triangle_normal1[count].z*sim_ratio.z+real_world_origin.z);
			glVertex3f(sph_system->m_host_triangle1[count].x*sim_ratio.x+real_world_origin.x, 
					sph_system->m_host_triangle1[count].y*sim_ratio.y+real_world_origin.y, 
					sph_system->m_host_triangle1[count].z*sim_ratio.z+real_world_origin.z);

			//glColor4f(0.0f, 0.0f, 1.0f, 0.1f);
			glNormal3f(sph_system->m_host_triangle_normal2[count].x*sim_ratio.x+real_world_origin.x, 
					sph_system->m_host_triangle_normal2[count].y*sim_ratio.y+real_world_origin.y, 
					sph_system->m_host_triangle_normal2[count].z*sim_ratio.z+real_world_origin.z);
			glVertex3f(sph_system->m_host_triangle2[count].x*sim_ratio.x+real_world_origin.x, 
					sph_system->m_host_triangle2[count].y*sim_ratio.y+real_world_origin.y, 
					sph_system->m_host_triangle2[count].z*sim_ratio.z+real_world_origin.z);
		glEnd();
	}

	return;

	glColor3f(0.0f, 0.0f, 1.0f);
	for(uint count=0; count<sph_system->m_num_lines; count++)
	{
		glBegin(GL_LINES);
			glVertex3f(sph_system->m_host_line0[count].x*sim_ratio.x+real_world_origin.x, 
					sph_system->m_host_line0[count].y*sim_ratio.y+real_world_origin.y, 
					sph_system->m_host_line0[count].z*sim_ratio.z+real_world_origin.z);
			glVertex3f(sph_system->m_host_line1[count].x*sim_ratio.x+real_world_origin.x, 
					sph_system->m_host_line1[count].y*sim_ratio.y+real_world_origin.y, 
					sph_system->m_host_line1[count].z*sim_ratio.z+real_world_origin.z);
		glEnd();
	}
}

void draw_cylinders(float x0, float y0, float z0, float x1, float y1, float z1 , float r)  
{  
	glColor3f(0.64f, 0.16f, 0.16f);
    GLdouble  dir_x = x1 - x0;  
    GLdouble  dir_y = y1 - y0;  
    GLdouble  dir_z = z1 - z0;  
    GLdouble  bone_length = sqrt( dir_x*dir_x + dir_y*dir_y + dir_z*dir_z );  
    static GLUquadricObj *  quad_obj = NULL;  
    if ( quad_obj == NULL )  
        quad_obj = gluNewQuadric();  
    gluQuadricDrawStyle( quad_obj, GLU_FILL );  
    gluQuadricNormals( quad_obj, GLU_SMOOTH );  
    glPushMatrix();  
    // 平移到起始点  
    glTranslated( x0, y0, z0 );  
    // 计算长度  
    double  length;  
    length = sqrt( dir_x*dir_x + dir_y*dir_y + dir_z*dir_z );  
    if ( length < 0.0001 ) {   
        dir_x = 0.0; dir_y = 0.0; dir_z = 1.0;  length = 1.0;  
    }  
    dir_x /= length;  dir_y /= length;  dir_z /= length;  
    GLdouble  up_x, up_y, up_z;  
    up_x = 0.0;  
    up_y = 1.0;  
    up_z = 0.0;  
    double  side_x, side_y, side_z;  
    side_x = up_y * dir_z - up_z * dir_y;  
    side_y = up_z * dir_x - up_x * dir_z;  
    side_z = up_x * dir_y - up_y * dir_x;  
    length = sqrt( side_x*side_x + side_y*side_y + side_z*side_z );  
    if ( length < 0.0001 ) {  
        side_x = 1.0; side_y = 0.0; side_z = 0.0;  length = 1.0;  
    }  
    side_x /= length;  side_y /= length;  side_z /= length;  
    up_x = dir_y * side_z - dir_z * side_y;  
    up_y = dir_z * side_x - dir_x * side_z;  
    up_z = dir_x * side_y - dir_y * side_x;  
    // 计算变换矩阵  
    GLdouble  m[16] = { side_x, side_y, side_z, 0.0,  
        up_x,   up_y,   up_z,   0.0,  
        dir_x,  dir_y,  dir_z,  0.0,  
        0.0,    0.0,    0.0,    1.0 };  
    glMultMatrixd( m );  
    // 圆柱体参数  
    GLdouble radius= r;        // 半径  
    GLdouble slices = 150.0;      //  段数  
    GLdouble stack = 150.0;       // 递归次数  
    gluCylinder( quad_obj, radius, radius, bone_length, slices, stack );   
    glPopMatrix();  
}  

void draw_spheres()
{
	//sphere 1
	glPushMatrix();
	glTranslatef(sph_system->m_sphere1.pos.x*sim_ratio.x+real_world_origin.x, 
				sph_system->m_sphere1.pos.y*sim_ratio.y+real_world_origin.y, 
				sph_system->m_sphere1.pos.z*sim_ratio.z+real_world_origin.z);
	glColor3f(1.0f, 0.0f, 0.0f);
	glDisable(GL_BLEND);
	glutWireSphere(sph_system->m_sphere1.radius*sim_ratio.x, 40, 40);
	glPopMatrix();

	//sphere 2
	glPushMatrix();
	glTranslatef(sph_system->m_sphere2.pos.x*sim_ratio.x+real_world_origin.x, 
				sph_system->m_sphere2.pos.y*sim_ratio.y+real_world_origin.y, 
				sph_system->m_sphere2.pos.z*sim_ratio.z+real_world_origin.z);
	glColor3f(1.0f, 0.0f, 0.0f);
	glDisable(GL_BLEND);
	glutWireSphere(sph_system->m_sphere2.radius*sim_ratio.x, 40, 40);
	glPopMatrix();

	//sphere 3
	glPushMatrix();
	glTranslatef(sph_system->m_sphere3.pos.x*sim_ratio.x+real_world_origin.x, 
				sph_system->m_sphere3.pos.y*sim_ratio.y+real_world_origin.y, 
				sph_system->m_sphere3.pos.z*sim_ratio.z+real_world_origin.z);
	glColor3f(1.0f, 0.0f, 0.0f);
	glDisable(GL_BLEND);
	glutWireSphere(sph_system->m_sphere3.radius*sim_ratio.x, 40, 40);
	glPopMatrix();

	/*//sphere 4
	glPushMatrix();
	glTranslatef(sph_system->m_sphere4.pos.x*sim_ratio.x+real_world_origin.x, 
				sph_system->m_sphere4.pos.y*sim_ratio.y+real_world_origin.y, 
				sph_system->m_sphere4.pos.z*sim_ratio.z+real_world_origin.z);
	glColor3f(1.0f, 0.0f, 0.0f);
	glDisable(GL_BLEND);
	glutSolidSphere(sph_system->m_sphere4.radius*sim_ratio.x, 30, 30);
	glPopMatrix();*/
}

void display_func()
{
	sph_system->animation();

	/*GLfloat ambientLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glEnable(GL_LIGHTING);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT,ambientLight);
	glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);*/

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.2f, 0.2f, 0.2f, 0.1f);

	//glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	//glEnable(GL_ALPHA_TEST);
	//glAlphaFunc(GL_GREATER, 0);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_POLYGON_SMOOTH);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	////

	/*GLfloat afPropertiesAmbient [] = {0.50, 0.50, 0.50, 1.00}; 
	GLfloat afPropertiesDiffuse [] = {0.75, 0.75, 0.75, 1.00}; 
	GLfloat afPropertiesSpecular[] = {1.00, 1.00, 1.00, 1.00}; 

	GLfloat afAmbientWhite [] = {0.25, 0.25, 0.25, 1.00}; 
	GLfloat afAmbientRed   [] = {0.25, 0.00, 0.00, 1.00}; 
	GLfloat afAmbientGreen [] = {0.00, 0.25, 0.00, 1.00}; 
	GLfloat afAmbientBlue  [] = {0.00, 0.00, 0.25, 1.00}; 
	GLfloat afDiffuseWhite [] = {0.75, 0.75, 0.75, 1.00}; 
	GLfloat afDiffuseRed   [] = {0.75, 0.00, 0.00, 1.00}; 
	GLfloat afDiffuseGreen [] = {0.00, 0.75, 0.00, 1.00}; 
	GLfloat afDiffuseBlue  [] = {0.00, 0.00, 0.75, 1.00}; 
	GLfloat afSpecularWhite[] = {1.00, 1.00, 1.00, 1.00}; 
	GLfloat afSpecularRed  [] = {1.00, 0.25, 0.25, 1.00}; 
	GLfloat afSpecularGreen[] = {0.25, 1.00, 0.25, 1.00}; 
	GLfloat afSpecularBlue [] = {0.25, 0.25, 1.00, 1.00};

	GLfloat position [] = {40.0f, 40.0f, 40.0f};

	glLightfv( GL_LIGHT0, GL_AMBIENT,  afPropertiesAmbient); 
	glLightfv( GL_LIGHT0, GL_DIFFUSE,  afPropertiesDiffuse); 
	glLightfv( GL_LIGHT0, GL_SPECULAR, afPropertiesSpecular); 
	glLightfv( GL_LIGHT0, GL_POSITION, position);
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0); 

	glEnable( GL_LIGHT0 ); 

	glMaterialfv(GL_BACK,  GL_AMBIENT,   afAmbientGreen); 
	glMaterialfv(GL_BACK,  GL_DIFFUSE,   afDiffuseGreen); 
	glMaterialfv(GL_FRONT, GL_AMBIENT,   afAmbientBlue); 
	glMaterialfv(GL_FRONT, GL_DIFFUSE,   afDiffuseBlue); 
	glMaterialfv(GL_FRONT, GL_SPECULAR,  afSpecularWhite); 
	glMaterialf( GL_FRONT, GL_SHININESS, 25.0); */

	///
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPushMatrix();

	if(buttonState == 1)
	{
		xRot+=(xRotLength-xRot)*0.1f;
		yRot+=(yRotLength-yRot)*0.1f;
	}

	glTranslatef(xTrans, yTrans, zTrans);
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

	if(sph_system->m_disp_mode == 1)
	{
		render_particles();
	}

	if(sph_system->m_disp_mode == 2)
	{
		//render_density();
		render_mesh();
	}

	draw_box(real_world_origin.x, real_world_origin.y, real_world_origin.y, real_world_side.x, real_world_side.y, real_world_side.z);
	//draw_cylinders(0.0f, -10.0f, 5.0f, 0.0f, 10.0f, 5.0f, 1.5f);
	//draw_cylinders(0.0f, -10.0f, -5.0f, 0.0f, 10.0f, -5.0f, 1.5f);
	//draw_cylinders(10.0f, -10.0f, 0.0f, 10.0f, 10.0f, 0.0f, 1.5f);

	if(sph_system->m_host_param->use_cylinder == true)
	{
		draw_spheres();
	}

	glPopMatrix();

	if(create_video == 1)
	{
		if (frameNum % 2 == 0)
		{
			//screenShot(frameNum / 2);
			//make_picture(frameNum / 2);
			make_mesh(frameNum / 2);
			//make_particle(frameNum / 2);
		}

		frameNum++;
	}

    glutSwapBuffers();
	
	sph_timer->update();
	memset(window_title, 0, 50);
	sprintf(window_title, "SPH Smoke 2D. FPS: %f", sph_timer->get_fps());
	glutSetWindowTitle(window_title);
}

void idle_func()
{
	glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)width/height, 0.001, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
	if(key == 'w' || key == 'W')
	{
		zTrans += 1.0f;
	}

	if(key == 's' || key == 'S')
	{
		zTrans -= 1.0f;
	}

	if(key == 'd' || key == 'D')
	{
		xTrans -= 1.0f;
	}

	if(key == 'a' || key == 'A')
	{
		xTrans += 1.0f;
	}

	if(key == 'q' || key == 'Q')
	{
		yTrans -= 1.0f;
	}

	if(key == 'e' || key == 'E')
	{
		yTrans += 1.0f;
	}

	if(key == 'p' || key == 'P')
	{
		sph_system->add_new_particle();
	}

	if(key == 't' || key == 'T')
	{
		sph_system->m_trans_once=1;
	}

	if(key == 'f' || key == 'F')
	{
		glutFullScreen();
	}

	if(key == 27)
	{
		exit(0);
	}

	if(key == 32)
	{
		sph_system->m_sys_running=1-sph_system->m_sys_running;
	}

	if(key == 'y' || key == 'Y')
	{
		sph_system->m_host_param->force_waiting=true;
	}

	if(key == 'r' || key == 'R')
	{
		delete sph_system;
		init_sph_system();
		//init();
		//init_ratio();
		glutFullScreen();
	}
}

void specialkey_func(int key, int x, int y)
{
	if(key == GLUT_KEY_LEFT)
	{
		
	}

	if(key == GLUT_KEY_RIGHT)
	{
		
	}
}

void mouse_func(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
        buttonState = 1;
	}
    else if (state == GLUT_UP)
	{
        buttonState = 0;
	}

    ox = x; oy = y;

    glutPostRedisplay();
}

void motion_func(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

	if (buttonState == 1) 
	{
		xRotLength += dy / 5.0f;
		yRotLength += dx / 5.0f;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

void menu_key(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) 
    {
		case '1':
			sph_system->m_host_param->use_cylinder^=true;
			break;
		case '2':
			sph_system->m_use_split=1-sph_system->m_use_split;
			break;
		case '3': 
			sph_system->m_use_merge=1-sph_system->m_use_merge;
			break;
		case '4': 
			create_video=1-create_video;
			break;
		case '5':
			sph_system->m_disp_mode=2;
			break;
		case '6':
			sph_system->m_disp_mode=1;
			break;
		case '8':
			
			break;
	}
}

void main_menu(int i)
{
    menu_key((unsigned char) i, 0, 0);
}

void init_menu()
{
    glutCreateMenu(main_menu);
    glutAddMenuEntry("Use Cylinder", '1');
	glutAddMenuEntry("Use Spliting", '2');
	glutAddMenuEntry("Use Merging", '3');
	glutAddMenuEntry("Create Video", '4');
	glutAddMenuEntry("Display Dens", '5');
	glutAddMenuEntry("Display Particle", '6');
	glutAddMenuEntry("Display Mesh", '7');
	glutAddMenuEntry("Make Picture", '8');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("SPH Smoke 2D");

	init_sph_system();
	init();
	init_ratio();
	init_menu();

    glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutSpecialFunc(specialkey_func);
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutIdleFunc(idle_func);

    glutMainLoop();

    return 0;
}
