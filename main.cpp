#include <iostream>
#include "readBMP.h"
#include <math.h>
#include <stdlib.h>
#include <boost/thread.hpp>
#include <GL/glut.h>

std::vector<float> errs;

float * vis_preview = new float[20*20];
float * vis0_preview = new float[20*20];

void vis2hid_worker(const float * X,float * H,int h,int v,float * c,float * W,std::vector<int> const & vrtx)
{
  for(int t=0;t<vrtx.size();t++)
  {
    int k = vrtx[t];
    for(int j=0;j<h;j++)
    {
      H[k*h+j] = c[j]; 
      for(int i=0;i<v;i++)
      {
        H[k*h+j] += W[i*h+j] * X[k*v+i];
      }
      H[k*h+j] = 1.0f/(1.0f + exp(-H[k*h+j]));
    }
  }
}

void hid2vis_worker(const float * H,float * V,int h,int v,float * b,float * W,std::vector<int> const & vrtx)
{
  for(int t=0;t<vrtx.size();t++)
  {
    int k = vrtx[t];
    for(int i=0;i<v;i++)
    {
      V[k*v+i] = b[i]; 
      for(int j=0;j<h;j++)
      {
        V[k*v+i] += W[i*h+j] * H[k*h+j];
      }
      V[k*v+i] = 1.0f/(1.0f + exp(-V[k*v+i]));
    }
  }
}

struct RBM
{
  int h; // number hidden elements
  int v; // number visible elements
  int n; // number of samples
  float * c; // bias term for hidden state, R^h
  float * b; // bias term for visible state, R^v
  float * W; // weight matrix R^h*v
  char * X; // input data, binary [0,1], v*n
  RBM(int _h,int _v,int _n,char* _X)
  {
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = new float[h];
    b = new float[v];
    W = new float[h*v];
    constant(c,0.3f,h);
    constant(b,0.3f,v);
    constant(W,0.3f,v*h);
  }

  float norm(float * dat,int size)
  {
    float ret = 0;
    for(int i=0;i<size;i++)
    {
      ret += dat[i]*dat[i];
    }
    return sqrt(ret);
  }

  void zero(float * dat,int size)
  {
    for(int i=0;i<size;i++)
    {
      dat[i] = 0;
    }
  }

  void constant(float * dat,float val,int size)
  {
    for(int i=0;i<size;i++)
    {
      dat[i] = (-1+2*((rand()%10000)/10000.0f))*val;
    }
  }

  void add(float * A, float * dA, float epsilon, int size)
  {
    for(int i=0;i<size;i++)
    {
      A[i] += epsilon * dA[i];
    }
  }

  void cd(int nGS,float epsilon)
  {

    // CD Contrastive divergence (Hinton's CD(k))
    //   [dW, db, dc, act] = cd(self, X) returns the gradients of
    //   the weihgts, visible and hidden biases using Hinton's
    //   approximated CD. The sum of the average hidden units
    //   activity is returned in act as well.
   
    float * vis0 = new float[n*v];
    float * hid0 = new float[n*h];
    float * vis = new float[n*v];
    float * hid = new float[n*h];
    float * dW = new float[h*v];
    float * dc = new float[h];
    float * db = new float[v];

    int off = rand()%(n);
    std::cout << "offset = " << off << std::endl;
    off *= v;

    for(int i=0,size=n*v;i<size;i++)
    {
      vis0[i] = (float)X[i];
    }

    vis2hid(vis0,hid0);

    for(int i=0;i<n*h;i++)
    {
      hid[i] = hid0[i];
    }

    for (int iter = 1;iter<=nGS;iter++)
    {
      // sampling
      hid2vis(hid,vis);
      vis2hid(vis,hid);
    }
  
    zero(dW,v*h);
    zero(dc,h);
    zero(db,v);
    float err = 0;
    for(int k=0;k<n;k++)
    {
      for(int i=0;i<v;i++)
      {
        for(int j=0;j<h;j++)
        {
          dW[i*h+j] -= (vis0[k*v+i]*hid0[k*h+j] - vis[k*v+i]*hid[k*h+j]) / n;
        }
      }

      for(int j=0;j<h;j++)
      {
        dc[j] -= (hid0[k*h+j]*hid0[k*h+j] - hid[k*h+j]*hid[k*h+j]) / n;
      }

      for(int i=0;i<v;i++)
      {
        db[i] -= (vis0[k*v+i]*vis0[k*v+i] - vis[k*v+i]*vis[k*v+i]) / n;
      }

      for(int i=0;i<v;i++)
      {
        err += (vis0[k*v+i]-vis[k*v+i])*(vis0[k*v+i]-vis[k*v+i]);
      }
    }
    err = sqrt(err);
    errs.push_back(err);
    add(W,dW,-epsilon,v*h);
    add(c,dc,-epsilon,h);
    add(b,db,-epsilon,v);

    std::cout << "dW norm = " << norm(dW,v*h) << std::endl;
    std::cout << "dc norm = " << norm(dc,h) << std::endl;
    std::cout << "db norm = " << norm(db,v) << std::endl;
    std::cout << "err = " << err << std::endl;

    for(int x=0,k=0;x<20;x++)
    {
      for(int y=0;y<20;y++,k++)
      {
        vis_preview[k] = vis[off+k];
      }
    }
    for(int x=0,k=0;x<20;x++)
    {
      for(int y=0;y<20;y++,k++)
      {
        vis0_preview[k] = vis0[off+k];
      }
    }

    delete [] vis0;
    delete [] hid0;
    delete [] vis;
    delete [] hid;
    delete [] dW;
    delete [] dc;
    delete [] db;

  }

  void sigmoid(float * p,float * X,int n)
  {
    for(int i=0;i<n;i++)
    {
      p[i] = 1.0f/(1.0f + exp(-X[i]));
    }
  }

  void vis2hid(const float * X,float * H)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<int> > vrtx(8);
    for(int i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(int thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(vis2hid_worker,X,H,h,v,c,W,vrtx[thread]));
    }
    for(int thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
  }
  
  void hid2vis(const float * H,float * V)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<int> > vrtx(8);
    for(int i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(int thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(hid2vis_worker,H,V,h,v,b,W,vrtx[thread]));
    }
    for(int thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
  }

};

RBM * rbm = NULL;

void drawBox(void)
{
  std::cout << "drawBox" << std::endl;
  float max_err = 0;
  for(int k=0;k<errs.size();k++)
  {
    if(max_err<errs[k])max_err=errs[k];
  }
  glColor3f(1,1,1);
  glBegin(GL_LINES);
  for(int k=0;k+1<errs.size();k++)
  {
    glVertex3f( -1 + 2*k / (float)errs.size()
              , errs[k] / max_err
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / (float)errs.size()
              , errs[k+1] / max_err
              , 0
              );
  }
  glEnd();
  if(rbm)
  {
    float max_W = -1000;
    float min_W =  1000;
    for(int i=0,k=0;i<rbm->v;i++)
      for(int j=0;j<rbm->h;j++,k++)
      {
        if(rbm->W[k]>max_W)max_W=rbm->W[k];
        if(rbm->W[k]<min_W)min_W=rbm->W[k];
      }
    float fact_W = 1.0 / (max_W - min_W);
    float col;
    glBegin(GL_QUADS);
    float d=3e-3;
    for(int x=0;x<20;x++)
    {
      for(int y=0;y<20;y++)
      {
        for(int i=0;i<rbm->v/20;i++)
        {
          for(int j=0;j<rbm->h/20;j++)
          {
            col = 0.5f + 0.5f*(rbm->W[(i+x)*rbm->h+j+y]-min_W)*fact_W;
            glColor3f(col,col,col);
            glVertex3f(  -1+(1.15*20*i+x)/(1.15*(float)rbm->v) ,  -1+(1.15*20*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(d+-1+(1.15*20*i+x)/(1.15*(float)rbm->v) ,  -1+(1.15*20*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(d+-1+(1.15*20*i+x)/(1.15*(float)rbm->v) ,d+-1+(1.15*20*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(  -1+(1.15*20*i+x)/(1.15*(float)rbm->v) ,d+-1+(1.15*20*j+y)/(1.15*(float)rbm->h),0);
          }
        }
      }
    }
    glEnd();
  }
  {
    float d = 5e-2;
    float col;
    glBegin(GL_QUADS);
    for(int y=0,k=0;y<20;y++)
    {
      for(int x=0;x<20;x++,k++)
      {
        col = vis_preview[k];
        glColor3f(col,col,col);
        glVertex3f(    (x)/40.0f ,  -1+(20-1-y)/20.0f,0);
        glVertex3f(d/2+(x)/40.0f ,  -1+(20-1-y)/20.0f,0);
        glVertex3f(d/2+(x)/40.0f ,d+-1+(20-1-y)/20.0f,0);
        glVertex3f(    (x)/40.0f ,d+-1+(20-1-y)/20.0f,0);
      }
    }
    for(int y=0,k=0;y<20;y++)
    {
      for(int x=0;x<20;x++,k++)
      {
        col = vis0_preview[k];
        glColor3f(col,col,col);
        glVertex3f(    0.5f+(x)/40.0f ,  -1+(20-1-y)/20.0f,0);
        glVertex3f(d/2+0.5f+(x)/40.0f ,  -1+(20-1-y)/20.0f,0);
        glVertex3f(d/2+0.5f+(x)/40.0f ,d+-1+(20-1-y)/20.0f,0);
        glVertex3f(    0.5f+(x)/40.0f ,d+-1+(20-1-y)/20.0f,0);
      }
    }
    glEnd();
  }
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawBox();
  glutSwapBuffers();
}

void idle(void)
{
  usleep(1000000);
  glutPostRedisplay();
}

void init(void)
{
  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 3,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */
}

void run_rbm(int w,int h,int nw,int nh,char * dat)
{
  rbm = new RBM(w*h,w*h,nw*nh,dat);
  for(int i=0;i<1000;i++)
  {
    rbm->cd(3,.1);
  }
}

void keyboard(unsigned char Key, int x, int y)
{
  switch(Key)
  {
    case ' ':
      {
        Image img;
        ImageLoad("digits_mnist.bmp",&img);
        std::cout << img.sizeX << " " << img.sizeY << std::endl;
        int w = 20;
        int h = 20;
        int nw = img.sizeX / w;
        int nh = img.sizeY / h;
        int woff = 0;
        int hoff = 0;
        char ch = 0;
        int disp = img.sizeX;
        int off = 0;
        char * dat = new char[img.sizeX*img.sizeY];
        for(int J=0,k=0;J<nh;J++)
        for(int I=0;I<nw;I++)
        {
          for(int j=0;j<1;j++)
          {
            for(int i=0;i<w;i++,k++)
            {
              dat[k] = (i==J/5);
            }
          }
          for(int j=1;j<h;j++)
          {
            for(int i=0;i<w;i++,k++)
            {
              dat[k] = ((img.data[3*((img.sizeY-1-(J*h+j))*disp+(I*w+i+off))  ]&&
                         img.data[3*((img.sizeY-1-(J*h+j))*disp+(I*w+i+off))+1]&&
                         img.data[3*((img.sizeY-1-(J*h+j))*disp+(I*w+i+off))+2] < 0)?1:0);
            }
          }
        }
        boost::thread * thr ( new boost::thread(run_rbm,w,h,nw,nh,dat) );
        break;
      }
    case 27:
              exit(1);
              break;
  };
}

int main(int argc,char ** argv)
{
  srand(time(0));
  std::cout << "Press space to start..." << std::endl;
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("Boltzmann Machine");
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutKeyboardFunc(keyboard);
  init();
  glutMainLoop();
  return 0;
}

