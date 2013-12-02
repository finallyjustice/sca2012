#include "bitmap.h"

Bitmap::Bitmap()
{
	m_data = NULL;
	m_w = m_h = 0;
}

Bitmap::~Bitmap()
{
	if(m_data != NULL)
	{
		delete []m_data;
		m_data = NULL;
	}
}

int Bitmap::GetWidth()
{
	return m_w;
}

int Bitmap::GetHeight()
{
	return m_h;
}

unsigned long *Bitmap::GetBits()
{
	return m_data;
}

bool Bitmap::makeBMP(unsigned long *data, int w, int h)
{
	int x, y;

	if (m_data != NULL) delete []m_data;
	m_data = new unsigned long[w * h];
	for (y = 0; y < h; y++)
		for (x = 0; x < w; x++)
			m_data[y * w + x] = data[y * w + x];
	m_w = w;
	m_h = h;
	return true;
}

unsigned long Bitmap::read_u_long_int(FILE *fp)
{
	int a = (unsigned)fgetc(fp);

	a += (unsigned)fgetc(fp) * 0x100;
	a += (unsigned)fgetc(fp) * 0x10000;
	a += (unsigned)fgetc(fp) * 0x1000000;

	return a;
}

int Bitmap::write_u_long_int(unsigned long a, FILE *fp)
{
	unsigned short int ah;
	unsigned short int al;

	ah = (unsigned short)(a / 65536);
	al = (unsigned short)(a % 65536);
    write_u_short_int(al, fp);
    write_u_short_int(ah, fp);

	return 4;
}

int Bitmap::write_u_short_int(unsigned short a, FILE *fp)
{
	unsigned char ch;
	unsigned char cl;

	ch = (unsigned char)(a / 256);
	cl = (unsigned char)(a % 256);
	fputc(cl, fp);
	fputc(ch, fp);

	return 2;
}

unsigned long *Bitmap::LoadBMP32(const char *name, int *w, int *h)
{
	FILE *fp;
	int x, y, overrun = 0;
	unsigned long *tex;
	char sig[10];
	unsigned char *temp;

	fopen_s(&fp, name, "rb");
	if (fp == NULL) return 0;

	fread(sig, 2, 1, fp);

	if(memcmp(sig, "BM", 2) != 0)
	{
		fclose(fp);
		return 0;
	}

	fseek(fp, 0x12, 0);

	*w = read_u_long_int(fp);
	*h = read_u_long_int(fp);

	overrun = (4 - ((*w * 3) & 3)) & 3;

	tex = new unsigned long[(*w) * (*h)];
	if (tex == NULL)
	{
		fclose(fp);
		return 0;
	}

	temp = new unsigned char[(*w + overrun) * (*h) * 3];
	if (temp == NULL)
	{
		fclose(fp);
		return 0;
	}

	fseek(fp, 54, 0);
	fread(temp, (*w + overrun) * (*h) * 3, 1, fp);
	fclose(fp);

	int p = 0;
	for (y = 0; y < *h; y++) {
		for (x = 0; x < *w; x++) {
			tex[y * (*w) + x] = (0xff << 24) + (temp[p] << 16) + (temp[p + 1] << 8) + temp[p + 2];
			p += 3;
		}
		p += overrun;
	}

	return tex;
}

bool Bitmap::Load(const char *strFileName)
{
	if (m_data != NULL) delete []m_data;
	m_data = LoadBMP32(strFileName, &m_w, &m_h);
	if (m_data == NULL) return false;

	return true;
}

bool Bitmap::Save(const char *strFileName)
{
	int i, x, y;
	unsigned long ul;
	unsigned short us;
	unsigned char uc;
	FILE *fp;
	
	fopen_s(&fp, strFileName, "wb");

	if (fp == NULL) return false;

	fputc('B', fp);
	fputc('M', fp);

	ul = 3 * m_w * m_h + 54; write_u_long_int(ul, fp);
	us = 0; write_u_short_int(us, fp);
	us = 0; write_u_short_int(us, fp);
	ul = 54; write_u_long_int(ul, fp);
	ul = 40; write_u_long_int(ul, fp);
	write_u_long_int(m_w, fp);
	write_u_long_int(m_h, fp);
	us = 1; write_u_short_int(us, fp); 
	us = 24; write_u_short_int(us, fp);
	for (i = 0; i <= 5; i++) {
		ul = 0;
		write_u_long_int(ul, fp);
	}

	int overrun = (4 - ((m_w * 3) & 3)) & 3;

	for (y = 0; y < m_h; y++) {
		for (x = 0; x < m_w; x++) {
			ul = m_data[y * m_w + x];
			uc = (unsigned char)((ul >> 16) & 0xff); fputc(uc, fp);
			uc = (unsigned char)((ul >> 8) & 0xff); fputc(uc, fp);
			uc = (unsigned char)(ul & 0xff); fputc(uc, fp);
		}
		for (x = 0; x < overrun; x++) fputc(0, fp);
	}

	fclose(fp);
	return true;
}

float* Bitmap::toFloat()
{
	float* tex = new float[m_w * m_h * 3];
	unsigned long ul;
	unsigned char uc;
	int x, y;

	for (y = 0; y < m_h; y++) {
		for (x = 0; x < m_w; x++) {
			ul = m_data[y * m_w + x];
			uc = (unsigned char)(ul & 0xff);
			tex[(y * m_w + x) * 3] = (float)uc / 256.0f;
			uc = (unsigned char)((ul >> 8) & 0xff);
			tex[(y * m_w + x) * 3 + 1] = (float)uc / 256.0f;
			uc = (unsigned char)((ul >> 16) & 0xff);
			tex[(y * m_w + x) * 3 + 2] = (float)uc / 256.0f;
		}
	}

	return tex;
}
