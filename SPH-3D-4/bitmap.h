#ifndef _BITMAP_H_
#define _BITMAP_H_

#include <stdio.h>
#include <string.h>

class Bitmap
{
private:
	unsigned long read_u_long_int(FILE *fp);
	int write_u_long_int(unsigned long a, FILE *fp);
	int write_u_short_int(unsigned short a, FILE *fp);
	unsigned long *LoadBMP32(const char *name, int *w, int *h);
	unsigned long *m_data;
	int m_w, m_h;

public:
	Bitmap();
	~Bitmap();
	int GetWidth();
	int GetHeight();
	unsigned long *GetBits();
	bool makeBMP(unsigned long *data, int w, int h);
	bool Load(const char *strFileName);
	bool Save(const char *strFileName);
	float* toFloat();
};

#endif