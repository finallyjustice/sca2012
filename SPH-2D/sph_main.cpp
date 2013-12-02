#include <GL\glut.h>
#include "sph_header.h"
#include "sph_timer.h"
#include "sph_system.h"
#include "bitmap.h"

Timer *sph_timer;
char *window_title;

SPHSystem *sph_system;

float window_width=600;
float window_height=600;

static int mouse_down[3];
static int x_from;
static int y_from;
static int x_to;
static int y_to;

unsigned long *screenData;
int frameNum=0;
int create_video=0;

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

void init_sph_system()
{
	sph_system=new SPHSystem();

	sph_system->add_init_particle();

	sph_timer=new Timer();
	window_title=(char *)malloc(sizeof(char)*50);

	screenData = new unsigned long[(int)window_width * (int)window_height];
	frameNum = 0;
}

void init()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

	gluOrtho2D(0.0, sph_system->m_world_width, 0.0, sph_system->m_world_depth);
}

void display_particle()
{
	glColor3f(0.0f, 0.0f, 1.0f);
	glPointSize(10.0f);
	Particle *p;
	for(uint count=0; count<sph_system->m_num_particle; count++)
	{
		p=&(sph_system->m_host_mem[count]);

		if(p->surface_normal > sph_system->m_surface_normal)
		{
			glColor3f(1.0f, 0.0f, 0.0f);
		}
		else
		{
			glColor3f(0.0f, 0.0f, 1.0f);
		}

		/*if(p->level == 1)
		{
			glColor3f(1.0f, 0.0f, 0.0f);
			glPointSize(10.0f);
		}

		if(p->level == 2)
		{
			glColor3f(1.0f, 1.0f, 0.0f);
			glPointSize(8.0f);
		}*/

		/*if(sph_system->m_co_energy*p->energy+sph_system->m_co_surface*p->surface_normal > sph_system->m_thresh_split)
		{
			glColor3f(1.0f, 0.0f, 0.0f);
		}
		else
		{
			glColor3f(0.0f, 0.0f, 1.0f);
		}*/

		glBegin(GL_POINTS);
			glVertex2f(p->pos.x, p->pos.y);
		glEnd();
	}
}

void get_extern_force()
{
	float2 extern_force;
	float2 pos;

	if(mouse_down[0])
    {
		pos.x=x_to/window_width;
		pos.y=(window_height-y_to)/window_height;

        if(pos.x>0 && pos.x<1 && pos.y>0 && pos.y< 1)
        {
			extern_force.x=x_to-x_from;
			extern_force.y=y_from-y_to;

			sph_system->add_extern_force(pos, extern_force);
           
			x_from=x_to;
            y_from=y_to;
        }
    }
}

void display_func()
{
	get_extern_force();
	sph_system->animate();

	glViewport(0, 0, window_width, window_height);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	
	glClearColor(0.7, 0.7, 0.7, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	display_particle();

	if(create_video == 1)
	{
		if (frameNum % 2 == 0)
		{
			screenShot(frameNum / 2);
		}

		frameNum++;
	}

	glutSwapBuffers();

	sph_timer->update();
	memset(window_title, 0, 50);
	sprintf(window_title, "SPH Smoke 2D. FPS: %0.2f", sph_timer->get_fps());
	glutSetWindowTitle(window_title);
}

void idle_func()
{
	glutPostRedisplay();
}

void reshape_func(int width, int height)
{	window_width=width;
	window_height=height;	glutReshapeWindow(window_width, window_height);
}

void process_keyboard(unsigned char key, int x, int y)
{
	if(key == ' ')
	{
		sph_system->m_sys_running=1-sph_system->m_sys_running;
	}
}

void special_keys(int key, int x, int y)
{
	if(key == GLUT_KEY_UP)
	{
		create_video=1;
	}

	if(key == GLUT_KEY_DOWN)
	{
		
	}

	if(key == GLUT_KEY_LEFT)
	{
		
	}

	if(key == GLUT_KEY_RIGHT)
	{
		
	}
           
	glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
    x_from=x;
    y_from=y;

    x_to=x;
    y_to=y;

    if(state==GLUT_DOWN)
    {
        mouse_down[button]=1;
    }
    else
    {
        mouse_down[button]=0;
    }
}

static void motion_func(int x, int y)
{
    x_to=x;
    y_to=y;
}

void menu_key(unsigned char key, int /*x*/, int /*y*/)
{
	switch(key)
	{
		case '1':
			sph_system->m_enable_split=1-sph_system->m_enable_split;
			break;	

		case '2':
			sph_system->m_enable_merge=1-sph_system->m_enable_merge;
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
    glutAddMenuEntry("Split particle", '1');
	glutAddMenuEntry("Merge particle", '2');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

int main(int argc, char **argv)
{
	init_sph_system();

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("SPH System 2D");

	init();
	init_menu();

    glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutIdleFunc(idle_func);
	glutKeyboardFunc(process_keyboard);
	glutSpecialFunc(special_keys);
	glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutMainLoop();

    return 0;
} 
