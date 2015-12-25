#include <openrave-core.h>
#include <vector>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <sstream>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <visp/vpDebug.h>

#include <visp/vpImage.h>
#include <visp/vpImageIo.h>
#include <visp/vpImageTools.h>

#include <visp/vpCameraParameters.h>
#include <visp/vpTime.h>
#include <visp/vpRobotCamera.h>

#include <visp/vpMath.h>
#include <visp/vpHomogeneousMatrix.h>
#include <visp/vpDisplayGTK.h>
#include <visp/vpDisplayGDI.h>
#include <visp/vpDisplayOpenCV.h>
#include <visp/vpDisplayD3D.h>
#include <visp/vpDisplayX.h>

#include <visp/vpFeatureLuminance.h>
#include <visp/vpParseArgv.h>

#include <visp/vpImageSimulator.h>
#include <stdlib.h>
#define  Z             1
#define gsigma 4
#define gsz 13
#define cellsz 16
#define Nbins 4
#include <visp/vpParseArgv.h>
#include <visp/vpIoTools.h>
#include <visp/vpImageFilter.h>
extern "C" {
  #include <../vlfeat-0.9.18/vl/generic.h>
  #include <../vlfeat-0.9.18/vl/hog.h>
}

using namespace OpenRAVE;
using namespace std;
using namespace cv;
static int viewer_done=0;

void getQuat(float rx, float ry, float rz, float (&quat)[4]) {
	
	float heading=ry;
	float attitude=-rx;//rz
	float bank=-rz;//rx
    // Assuming the angles are in radians.
    float c1 = cos(heading/2);
    float s1 = sin(heading/2);
    float c2 = cos(attitude/2);
    float s2 = sin(attitude/2);
    float c3 = cos(bank/2);
    float s3 = sin(bank/2);
    float c1c2 = c1*c2;
    float s1s2 = s1*s2;
    float w =c1c2*c3 - s1s2*s3;
  	float x =c1c2*s3 + s1s2*c3;
	float y =s1*c2*c3 + c1*s2*s3;
	float z =c1*s2*c3 - s1*c2*s3;
	quat[0]=w;
	quat[1]=x;
	quat[2]=y;
	quat[3]=z;
	//cout<<"Quat(i)="<<quat[0]<<","<<quat[1]<<","<<quat[2]<<","<<quat[3]<<","<<endl;
  }

void vpImage_to_mat(vpImage<unsigned char> &src , Mat &dst){
	for (unsigned int i = 0; i < src.getRows(); ++i) {
        for (unsigned int j = 0; j < src.getCols(); ++j) {
            // Assuming one channel ...
            dst.at<uchar>(i, j)= src[i][j];
        }
    }
}

void mat_to_vpImage(Mat &src , vpImage<unsigned char> &dst){
	for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Assuming one channel ...
            dst[i][j] = src.at<uchar>(i, j);
            
        }
    }
}

/*input: the image that you want rotated.
output: the Mat object to put the resulting file in.
alpha: the rotation around the x axis
beta: the rotation around the y axis
gamma: the rotation around the z axis (basically a 2D rotation)
dx: translation along the x axis
dy: translation along the y axis
dz: translation along the z axis (distance to the image)
f: focal distance (distance between camera and image, a smaller number exaggerates the effect)
color: defaulut pixel color for interpolation 
*/
void rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f, int color){

    
    //output=Mat(Size(input.cols,input.rows),input.type(),255);
    
    
    alpha = (alpha - 90.)*CV_PI/180.;

    beta = (beta - 90.)*CV_PI/180.;

    gamma = (gamma - 90.)*CV_PI/180.;

    // get width and height for ease of use in matrices

    double w = (double)input.cols;

    double h = (double)input.rows;

    // Projection 2D -> 3D matrix

    Mat A1 = (Mat_<double>(4,3) <<

              1, 0, -w/2,

              0, 1, -h/2,

              0, 0,    0,

              0, 0,    1);

    // Rotation matrices around the X, Y, and Z axis
	
    Mat RX = (Mat_<double>(4, 4) <<

              1,          0,           0, 0,

              0, cos(alpha), -sin(alpha), 0,

              0, sin(alpha),  cos(alpha), 0,

              0,          0,           0, 1);

    Mat RY = (Mat_<double>(4, 4) <<

              cos(beta), 0, -sin(beta), 0,

              0, 1,          0, 0,

              sin(beta), 0,  cos(beta), 0,

              0, 0,          0, 1);

    Mat RZ = (Mat_<double>(4, 4) <<

              cos(gamma), -sin(gamma), 0, 0,

              sin(gamma),  cos(gamma), 0, 0,

              0,          0,           1, 0,

              0,          0,           0, 1);

    // Composed rotation matrix with (RX, RY, RZ)

    Mat R = RX * RY * RZ;

    // Translation matrix

    Mat T = (Mat_<double>(4, 4) <<

             1, 0, 0, dx,

             0, 1, 0, dy,

             0, 0, 1, dz,

             0, 0, 0, 1);

    // 3D -> 2D matrix
	
    Mat A2 = (Mat_<double>(3,4) <<

              f, 0, w/2, 0,

              0, f, h/2, 0,

              0, 0,   1, 0);

    // Final transformation matrix
	
    Mat trans = A2 * (T * (R * A1));

    // Apply matrix transformation
	
    warpPerspective(input, output, trans, input.size(), INTER_LANCZOS4,BORDER_CONSTANT,255);
	//cv::namedWindow( "Display window1", cv::WINDOW_AUTOSIZE );
	//cv::imshow( "Display window1", output ); 
	//cv::waitKey(0); 
  }

class OpenRAVECamera{
public:
    OpenRAVECamera(SensorBasePtr psensor)
    {
        pcamera=psensor;
        pdata = boost::static_pointer_cast<SensorBase::CameraSensorData>(pcamera->CreateSensorData(SensorBase::ST_Camera));
        geom = *boost::static_pointer_cast<SensorBase::CameraGeomData>(pcamera->GetSensorGeometry(SensorBase::ST_Camera));
        img = cvCreateImage(cvSize(geom.width,geom.height),IPL_DEPTH_8U,3);
        KK=geom.KK;
    }
    virtual ~OpenRAVECamera() {
        cvReleaseImage(&img);
    }
    SensorBasePtr pcamera;
    SensorBase::CameraGeomData geom;
    boost::shared_ptr<SensorBase::CameraSensorData> pdata;
    IplImage* img;
    SensorBase::CameraIntrinsics KK;
};

void SetViewer(EnvironmentBasePtr penv, const string& viewername){
    ViewerBasePtr viewer = RaveCreateViewer(penv,viewername);
    BOOST_ASSERT(!!viewer);
    // attach it to the environment:
    penv->Add(viewer);
    // finally call the viewer's infinite loop (this is why a separate thread is needed)
    bool showgui = true;
    viewer->main(showgui);
    
    viewer_done=1;
}

void getCameraimage(EnvironmentBasePtr penv){
	
	size_t ienablesensor = 1;
    // get all the sensors, this includes all attached robot sensors
    std::vector<SensorBasePtr> sensors;
    penv->GetSensors(sensors);
    boost::shared_ptr<OpenRAVECamera> vcamera;
    
    	if( sensors[0]->Supports(SensorBase::ST_Camera) ) {
                sensors[0]->Configure(SensorBase::CC_PowerOn);
                //sensors[0]->Configure(SensorBase::CC_RenderDataOn);
                vcamera=boost::shared_ptr<OpenRAVECamera>(new OpenRAVECamera(sensors[0]));
		}
	
	//cvNamedWindow("image", CV_WINDOW_AUTOSIZE);	
    
					//VISP image conversion		
					vpImage<unsigned char> hogimg(240,320,0);
					vpImage<unsigned char> I(240,320,0) ;
					vpImage<unsigned char> Id(240,320,0) ;
					vpImage<unsigned char> Idiff(240,320,0) ;
					
					//initialize displays
					
					#if defined VISP_HAVE_X11
						vpDisplayX d,d1;
					#elif defined VISP_HAVE_GDI
						vpDisplayGDI d,d1;
					#elif defined VISP_HAVE_GTK
						vpDisplayGTK d,d1;
					#endif
					#if defined(VISP_HAVE_X11) || defined(VISP_HAVE_GDI) || defined(VISP_HAVE_GTK)
						
						d.init(I, 680+20, 10, "Photometric visual servoing : s") ;
						d1.init(Idiff, 680+20+(int)I.getWidth(), 10, "photometric visual servoing : s-s* ") ;
						
					#endif
	
					
	int iter=0;	
	int zinit=0.5;
	std::vector<RobotBasePtr> vrobots;
	penv->GetRobots(vrobots);
	Transform tinit = vrobots.at(0)->GetTransform();
	tinit.trans.z-=zinit;
	vrobots.at(0)->SetTransform(tinit);
	
    while(viewer_done==0){
			
	        // read the camera data and save the image
            
                vcamera->pcamera->GetSensorData(vcamera->pdata);
                if( vcamera->pdata->vimagedata.size() > 0 ) {
                    char* imageData = vcamera->img->imageData;
                    uint8_t* src = &vcamera->pdata->vimagedata.at(0);
                    for(int i=0; i < vcamera->geom.height; i++, imageData += vcamera->img->widthStep, src += 3*vcamera->geom.width) {
                        for(int j=0; j<vcamera->geom.width; j++) {
                            // opencv is bgr while openrave is rgb
                            imageData[3*j] = src[3*j+2];
                            imageData[3*j+1] = src[3*j+1];
                            imageData[3*j+2] = src[3*j+0];
                        }
                    }
                    
                    IplImage* grayframe = cvCreateImage(cvGetSize(vcamera->img), IPL_DEPTH_8U, 1);
					cvCvtColor(vcamera->img, grayframe, CV_RGB2GRAY);
					Mat mat_img(grayframe);
					
					string filename = str(boost::format("camera%d.jpg")%0);
                    //RAVELOG_INFO(str(boost::format("saving image %s")%filename));
					//cvSaveImage(filename.c_str(),grayframe);
					
					cv::Mat Id_mat(mat_img.rows/2, mat_img.cols/2, CV_8UC1); 
					cv::Mat resized(mat_img.rows/2, mat_img.cols/2, CV_8UC1);
					
					resize(mat_img,resized,resized.size(), 0, 0, cv::INTER_CUBIC); 
					mat_to_vpImage(resized,I);
					//mat_to_vpImage(resized,Img);
					
				    
				    
				if(iter==0)
					{
						cout<<vcamera->geom.KK.fx<<endl;
						
						rotateImage(resized, Id_mat, 90, 90, 90, 30, 0, 0.64-zinit, 0.64, 255);
						mat_to_vpImage(Id_mat,Id);
					}	
						//display images
						vpDisplay::display(I);
						vpDisplay::flush(I);
				
			
				
				Idiff = Id ;
				vpImageTools::imageDifference(I,Id,Idiff) ;
				
				
				
					vpDisplay::display(Idiff) ;
					vpDisplay::flush(Idiff) ;
				if(iter==0)
							{
				std::cout << "Click in the image to continue..." << std::endl;
				vpDisplay::getClick(Idiff) ;
				}
					
					vpDisplay::display(I) ;
					vpDisplay::flush(I) ;
					vpDisplay::display(Idiff) ;
					vpDisplay::flush(Idiff) ;
					
					
			//vpCameraParameters cam(vcamera->geom.KK.fx, vcamera->geom.KK.fy, 160, 120);
			vpCameraParameters cam(vcamera->geom.KK.fx, vcamera->geom.KK.fy, vcamera->geom.KK.cx/2, vcamera->geom.KK.cy/2);
			//cout<<cam.get_px()<<","<<cam.get_py()<<endl;
			//cout<<cam.u0<<","<<cam.v0<<","<<cam.inv_px<<","<<cam.inv_py<<endl;
			vpFeatureLuminance sI ;		
            sI.init( I.getHeight(), I.getWidth(), Z) ;
			sI.setCameraParameters(cam) ;
			sI.buildFrom(I) ;
            
            // desired visual feature built from the image
			vpFeatureLuminance sId ;
			sId.init(Id.getHeight(), Id.getWidth(),  Z) ;
			sId.setCameraParameters(cam) ;
			sId.buildFrom(Id) ;
			
			
			// Matrice d'interaction, Hessien, erreur,...
    vpMatrix Lsd;   // matrice d'interaction a la position desiree
    vpMatrix Hsd;  // hessien a la position desiree
    vpMatrix H ; // Hessien utilise pour le levenberg-Marquartd
    vpColVector error ; // Erreur I-I*

    // Compute the interaction matrix
    // link the variation of image intensity to camera motion

    // here it is computed at the desired position
    sId.interaction(Lsd,Id) ;


    // Compute the Hessian H = L^TL
    Hsd = Lsd.AtA() ;

    // Compute the Hessian diagonal for the Levenberg-Marquartd
    // optimization process
    unsigned int n = 6 ;
    vpMatrix diagHsd(n,n) ;
    diagHsd.eye(n);
    for(unsigned int i = 0 ; i < n ; i++) diagHsd[i][i] = Hsd[i][i];



    // ------------------------------------------------------
    // Control law
    double lambda ; //gain
    vpColVector e ;
    vpColVector v ; // camera velocity send to the robot


    // ----------------------------------------------------------
    // Minimisation

    double mu ;  // mu = 0 : Gauss Newton ; mu != 0  : LM
    double lambdaGN;


    mu       =  0.01;//(0.01 working)10000
    
    lambda   = 3;//(30 working)300000
    lambdaGN = 30;

	// ----------------------------------------------------------
    //int iter   = 1;
    int iterGN = 900000 ; // swicth to Gauss Newton after iterGN iterations

    double normeError = 0;
    
    // compute current error
      sI.error(sId,error) ;

      normeError = (error.sumSquare());
      // Compute the levenberg Marquartd term
        
          H = ((mu * diagHsd) + Hsd).inverseByLU();
        
        //	compute the control law
        e = H * Lsd.t() *error ;
    
        v = - lambda*e;
        //v[0]=0;v[1]=0;v[2]=0.1;v[3]=0;v[4]=0;v[5]=0;
        
      	//std::cout <<"|e| = " << normeError << ", |Tc| = " << sqrt(v.sumSquare()) << ", iter = " << iter << std::endl;
      	std::cout << normeError << "," << sqrt(v.sumSquare()) << "," << iter <<","<<v[0]<<","<<v[1]<<", "<<v[2]<<","<<v[3]<<","<<v[4]<<","<<v[5]<<endl;
      	
      	std::ostringstream ss;
		ss << std::setw(5) << std::setfill('0') << iter << "\n";
		std::string s2(ss.str());
      	//imwrite("result_img/cam/img"+s2+".jpg",resized);
      	//imwrite("result_img/error/img"+s2+".jpg",outimg);
      	
      	if(normeError<40000000)	viewer_done=1;
      	iter++;
      	
        //cout<<v<<endl;
      
            //move camera
            
            if( vrobots.size() > 0 ) {
				//RAVELOG_INFO("moving the robot a little\n");
				 
				Transform t = vrobots.at(0)->GetTransform();
				Transform tf,td;
				
				tf.trans.x = t.trans.x+v[0];
				tf.trans.y = t.trans.y+v[1];
				tf.trans.z = t.trans.z+v[2];
				float quat[4];
				getQuat(v[3],v[4],v[5],quat);
				//getQuat(0.01,0,0,quat);
				td.rot.w =quat[0];
				td.rot.x =quat[1];
				td.rot.y =quat[2];
				td.rot.z =quat[3];
				
				//Quat addition
				tf.rot.w =t.rot.w*td.rot.w-t.rot.x*td.rot.x-t.rot.w*td.rot.y-t.rot.z*td.rot.z;
				tf.rot.x =t.rot.w*td.rot.x+t.rot.x*td.rot.w+t.rot.y*td.rot.z-t.rot.z*td.rot.y;
				tf.rot.y =t.rot.w*td.rot.y-t.rot.x*td.rot.z+t.rot.y*td.rot.w+t.rot.z*td.rot.x;
				tf.rot.z =t.rot.w*td.rot.z+t.rot.x*td.rot.y-t.rot.y*td.rot.x+t.rot.z*td.rot.w;
				
				vrobots.at(0)->SetTransform(tf);
			}
            boost::this_thread::sleep(boost::posix_time::milliseconds(20));	
            
            }
		
		boost::this_thread::sleep(boost::posix_time::milliseconds(50));
		}
            
			cvDestroyWindow("image");
	
	

}	

int main(int argc, char ** argv)
{
    
    //int num = 1;
    //string scenefilename = "/home/harit/openrave_data/data/cup_env/tridof.cup001.xml";
    string scenefilename = "/home/harit/openrave_data/data/cup_env/tridof.cup003.xml";
    string viewername = "qtcoin";
    RaveInitialize(true); // start openrave core
    EnvironmentBasePtr penv = RaveCreateEnvironment(); // create the main environment
    RaveSetDebugLevel(Level_Debug);
    boost::thread thviewer(boost::bind(SetViewer,penv,viewername));
    penv->Load(scenefilename); // load the scene
    
    boost::thread getcam(boost::bind(getCameraimage,penv));
    
    while(viewer_done==0){
    //sleep(5);
    boost::this_thread::sleep(boost::posix_time::milliseconds(5000));	
    
	}
    thviewer.join(); // wait for the viewer thread to exit
    getcam.join();   // wait for the viewer thread to exit
    penv->Destroy(); // destroy
    
    return 0;
}
