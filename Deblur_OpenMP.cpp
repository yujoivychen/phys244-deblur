#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
using namespace std;

//  FUNCTIONS FOR READING AND WRITING
double **readpicture(ifstream &infile, int &width, int &height);
char readchar(ifstream& infile, int &num);
int readint(ifstream& infile, int numbytes, int &num);
int char2int(char thechar);
int char2int(char *thechar, int numbytes);
void readrgb(ifstream& infile, double *rgb, int &num);
void writepicture(ofstream& outfile, double **f, int width, int height);
char int2char(unsigned long theint);
void int2char(char *thechar, int theint);
void writechar(ofstream &outfile, char thechar);
void writeint(ofstream &outfile, int theint, int numbytes);
void writergb(ofstream& outfile, double r, double g, double b);
//  FUNCTIONS FOR CREATING AND REMOVING MEMORY
double **matrix(int width, int height);
void free_matrix(double **f, int width, int height);
int **imatrix(int width, int height);
void free_matrix(int **f, int width, int height);

// heat function
void heat(double **newf, double **f, int N, int width, int height);

int main()
{
   int width;
   int height;
   double **f, **g, **A;
   double x, y, h, k;
   int i, j, N, Nblur;

// reading in picture
   ifstream infile("zebra_Original.bmp", ios::in | ios::binary);
   f = readpicture(infile,width,height);
   infile.close();


g = matrix(width, height);
A = matrix(width, height);
char notdone = 1;
long int step = 1;
double gradf, laplf, themax, eps, lambda;
double fx, fy, fxx, fyy, fxy, numerator;
double **tempf = matrix(width,height);
double **newf = matrix(width,height);



h = 0.1;
lambda = 300;
eps = h;
k = h*h*eps/(4.0 + (lambda*h*h*eps));
N = 100;
Nblur = 15;

// start timer
double wtime;
 wtime = omp_get_wtime ( );


#pragma omp parallel for collapse(2) shared(A, f, width, height) private(i, j)

   for(i=0; i<width; i++)
   {
      for(j=0; j<height; j++)
      {
         A[i][j] = f[i][j];
      }
   }



for(step=1; step < N+1; step++)
{  
   #pragma omp parallel for collapse(2) shared(tempf, f, width, height) private(i, j)

   for(i=0; i<width; i++)
   {
      for(j=0; j<height; j++)
      {
         tempf[i][j] = f[i][j];
      }
   }

   heat(newf, tempf, Nblur, width, height);

   #pragma omp parallel for collapse(2) shared(tempf, A, width, height) private(i, j)

   for(i = 0; i <width; i++)
   {
      for (j=0; j<height; j++)
      {
         tempf[i][j] = tempf[i][j] - A[i][j];
      }
   }

   heat(newf, tempf, Nblur, width, height);


   #pragma omp parallel for collapse(2) shared(tempf, g, f, width, height, h, lambda, eps, k) private(i, j, fx, fy, gradf, fxx, fyy, fxy, numerator, laplf)

   for(i=0; i<width; i++)
      for(j=0; j<height; j++)
      {
            fx = float((f[min(i+1, width -1)][j]) - (f[max(i-1,0)][j]))/float(2.0*h);
 

            fy = float((f[i][min(j+1,height-1)]) - (f[i][max(j-1, 0)]))/float(2.0*h);


            gradf = sqrt((fx*fx)+(fy*fy));


            fxx = float(f[min(i+1, width -1)][j] - (2*f[i][j]) + f[max(i-1,0)][j])/float(h*h);
            fyy = float(f[i][min(j+1,height-1)] - (2*f[i][j]) + f[i][max(j-1, 0)])/float(h*h);
            fxy = float(f[min(i+1, width -1)][min(j+1,height-1)] - f[max(i-1,0)][min(j+1,height-1)] - f[min(i+1, width -1)][max(j-1, 0)] + f[max(i-1,0)][max(j-1, 0)])/float(4.0*h*h);
            numerator = (fx*((fx*fxx) + (fy*fxy))) + (fy*((fx*fxy) + (fy*fyy)));


            laplf = fxx + fyy;
            g[i][j] = laplf - (float(numerator)/float((fx*fx + fy*fy) + (eps*eps)));
            g[i][j] = g[i][j] / float(sqrt((fx*fx + fy*fy) + (eps*eps)));
            g[i][j] = g[i][j] - (2*lambda*tempf[i][j]);
            g[i][j] = f[i][j] + (k*g[i][j]);
        
      }


#pragma omp parallel for collapse(2) shared(g, f, width, height) private(i, j)
  
      for(i=0; i<width; i++)
         for(j=0; j<height; j++)
               f[i][j] = g[i][j];

   
}

//stop timer
wtime = omp_get_wtime ( ) - wtime;

cout << "Computation Wall Time = " << wtime << endl;


// writing picture
   ofstream outfile("zebra_Deblurred.bmp", ios::out | ios::binary);
   writepicture(outfile,f,width,height);
   outfile.close();

   free_matrix(f,width,height);
}

void heat(double **newf, double **f, int N, int width, int height)
{
   int i, j, step;

   for(step = 1; step <= N; step++)
   {
      #pragma omp parallel for collapse(2) shared(newf, f, width, height) private(i, j)

      for(i = 0; i <width; i++)
         for (j=0; j<height; j++)
            newf[i][j] = (f[max(i-1,0)][j] + f[min(i+1, width-1)][j]+f[i][max(j-1,0)]+f[i][min(j+1, height-1)])/4.0;

      for(i = 0; i <width; i++)
         for (j=0; j<height; j++)
         {
            f[i][j] = newf[i][j];
         }

   }
}

//**************************************************************************
//FUNCTIONS FOR READING AND WRITING
//**************************************************************************
double **readpicture(ifstream &infile, int &width, int &height)
{
   int i, j, k;
   char junk, theformat[2];
   double rgb[3], **f;
   int num = 0;
   int upsidedown = 0;
   cout << "reading picture" << endl;

   for (i = 0; i < 2; i++)
      theformat[i] = readchar(infile,num);
   int filesize = readint(infile,4,num);
   cout << "   numbytes = " << filesize << endl;
   for (i = 0; i < 4; i++)
      junk = readchar(infile,num);
   int offset = readint(infile,4,num);
   int headerstart = num;
   int headersize = readint(infile,4,num);
   if (headersize == 12)
   {
      width = readint(infile,2,num);
      height = readint(infile,2,num);
   }
   else
   {
      width = readint(infile,4,num);
      height = readint(infile,4,num);
   }
   if (height < 0)
   {
      height = abs(height);
      upsidedown = 1;
   }
   cout << "   width = " << width << endl;
   cout << "   height = " << height << endl;
   int numcolorpanes = readint(infile,2,num);
   int bitsperpixel = readint(infile,2,num);
   if (bitsperpixel != 24)
   {
      cout << "ERROR: this program is only built for 1 byte per rgb, not a total of " 
           << bitsperpixel << " bits" << endl;
      exit(1);
   }
   for (i = num-headerstart; i < headersize; i++)
      junk = readchar(infile,num);

   f = matrix(width,height);

   if (upsidedown)
      for (j = height-1; j >= 0; j--)
      {
         for (i = 0; i < width; i++)
         {
            readrgb(infile,rgb,num);
            f[i][j] = 0.0;
            for (k = 0; k < 3; k++)
               f[i][j] += rgb[k]*rgb[k];
            f[i][j] = sqrt(f[i][j])/sqrt(3.0);
         }
         for (i = 0; i < (4-(3*width)%4)%4; i++)
            junk = readchar(infile,num);
      }
   else
      for (j = 0; j < height; j++)
      {
         for (i = 0; i < width; i++)
         {
            readrgb(infile,rgb,num);
            f[i][j] = 0.0;
            for (k = 0; k < 3; k++)
               f[i][j] += rgb[k]*rgb[k];
            f[i][j] = sqrt(f[i][j])/sqrt(3.0);
         }
         for (i = 0; i < (4-(3*width)%4)%4; i++)
            junk = readchar(infile,num);
      }

   return f;
}

int readint(ifstream& infile, int numbytes, int &num)
{
   char *temp;
   int i;
   int theint;

   temp = new char[numbytes];

   for (i = 0; i < numbytes; i++)
      infile.read(reinterpret_cast<char *>(&(temp[i])),sizeof(char));
   theint = char2int(temp,numbytes);

   delete[] temp;
   num += numbytes;

   return theint;
}

int char2int(char thechar)
{
   int i, theint, imask;
   char cmask;

   cmask = 1;
   imask = 1;
   theint = 0;
   for (i = 0; i < 8; i++)
   {
      if (thechar & cmask)
      {
         theint += imask;
      }
      cmask = cmask << 1;
      imask *= 2;
   }

   return theint;
}

int char2int(char *thechar, int numbytes)
{
   int i;
   int theint, power;

   power = 1;
   theint = 0;
   for (i = 0; i < numbytes; i++)
   {
      theint += char2int(thechar[i])*power;
      power = power*256;
   }

   return theint;
}

char readchar(ifstream& infile, int &num)
{
   char thechar;

   infile.read(reinterpret_cast<char *>(&thechar),sizeof(char));
   num++;

   return thechar;
}

void readrgb(ifstream& infile, double *rgb, int &num)
{
   char ctemp;
   int i, itemp, ijunk;

   for (i = 0; i < 3; i++)
   {
      itemp = readint(infile,1,ijunk);
      rgb[i] = static_cast<double>(itemp)/255.0;
      num++;
   }
}

void writepicture(ofstream& outfile, double **f, int width, int height)
{
   int numbytes;
   int i, j;
   cout << "writing picture" << endl;

   writechar(outfile,66);
   writechar(outfile,77);

// number of bytes
   numbytes = 54+height*(3*width+(4-(3*width)%4)%4);
   cout << "   numbytes = " << numbytes << endl;
   writeint(outfile,numbytes,4);

   writeint(outfile,0,2);
   writeint(outfile,0,2);
   writeint(outfile,54,4);
   writeint(outfile,40,4);

// width
   cout << "   width = " << width << endl;
   writeint(outfile,width,4);
// height
   cout << "   height = " << height << endl;
   writeint(outfile,height,4);

   writeint(outfile,1,2);
   writeint(outfile,24,2);
   writeint(outfile,0,4);
   writeint(outfile,16,4);
   writeint(outfile,2835,4);
   writeint(outfile,2835,4);
   writeint(outfile,0,4);
   writeint(outfile,0,4);

   for (j = 0; j < height; j++)
   {
      for (i = 0; i < width; i++)
         writergb(outfile,f[i][j],f[i][j],f[i][j]);
      for (i = 0; i < (4-(3*width)%4)%4; i++)
         writechar(outfile,0);
   }
}

char int2char(unsigned long theint)
{
   char thechar, mask;
   int i;

   mask = 1;
   thechar = 0;
   for (i = 0; i < 8; i++)
   {
      if (theint%2)
         thechar += mask;
      theint /= 2;
      mask = mask << 1;
   }

   return thechar;
}

void int2char(char *thechar, int theint, int numbytes)
{
   int temp = theint;
   int i;

   for (i = 0; i < numbytes; i++)
   {
      thechar[i] = int2char(temp%256);
      temp = temp/256;
   }
}

void writechar(ofstream& outfile, char thechar)
{
   outfile.write(reinterpret_cast<char *>(&thechar),sizeof(char));
}

void writeint(ofstream &outfile, int theint, int numbytes)
{
   char *temp = new char[numbytes];
   int i;

   int2char(temp,theint,numbytes);
   for (i = 0; i < numbytes; i++)
      outfile.write(reinterpret_cast<char *>(&(temp[i])),sizeof(char));

   delete[] temp;
}

void writergb(ofstream& outfile, double r, double g, double b)
{
   int i, irgb;
   char temp;
   char a;
   double rgb[3];

   rgb[0] = r;
   rgb[1] = g;
   rgb[2] = b;
   for (i = 0; i < 3; i++)
   {
      irgb = static_cast<int>(floor(255.0*rgb[i]+0.5));
      if (irgb < 0)
         irgb = 0;
      else if (irgb > 255)
         irgb = 255;

      temp = int2char(irgb);
      outfile.write(reinterpret_cast<char *>(&temp),sizeof(char));
   }
}

//**************************************************************************
//FUNCTIONS FOR CREATING AND REMOVING MEMORY
//**************************************************************************

double **matrix(int width, int height)
{
   double **f;
   int i;

   f = new double*[width];
   for (i = 0; i < width; i++)
      f[i] = new double[height];

   return f;
}

void free_matrix(double **f, int width, int height)
{
   int i;

   for (i = 0; i < width; i++)
      delete f[i];
   delete f;
}

int **imatrix(int width, int height)
{
   int **f;
   int i;

   f = new int*[width];
   for (i = 0; i < width; i++)
      f[i] = new int[height];

   return f;
}

void free_matrix(int **f, int width, int height)
{
   int i;

   for (i = 0; i < width; i++)
      delete f[i];
   delete f;
}