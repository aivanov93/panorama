// g++ panorama.cpp -I ../include -L ../bin -rdynamic -lHalide -lpthread -ldl  -lpng  -larmadillo -o panorama && LD_LIBRARY_PATH=../bin ./panorama

#include <Halide.h>

#include <stdio.h>
#include <math.h>
#include <sstream>
#include <string>

#include "armadillo"
//using namespace arma;

using namespace Halide;
#include "../apps/support/image_io.h"


timeval t1, t2; float timex;
#define begin_timing gettimeofday(&t1, NULL); 
#define end_timing  gettimeofday(&t2, NULL); timex = (t2.tv_sec - t1.tv_sec)*1.0f + (t2.tv_usec - t1.tv_usec) / 1000000.0f ; std::cout<<timex<<"s \n";

Image<float> calculate_gaussian(double sigmaD){
	int sigma=(int)floor(sigmaD*3);
	Image<float> convolution(sigma*2+1,sigma*2+1); float sum=0;


	for (int i=-sigma; i<sigma+1; i++){
		for (int j=-sigma; j<sigma+1; j++){
			float r=i*i+j*j;
			convolution(i+sigma,j+sigma)=std::exp(-r/(2*sigmaD*sigmaD));
			sum+=convolution(i+sigma,j+sigma);
		}
	}
	float sum1=0;
	for (int i=0; i<sigma*2+1; i++){
		for (int j=0; j<sigma*2+1; j++){
			convolution(i,j)/=sum;
			sum1+=convolution(i,j);
		}
	}


	return convolution;
}

Func convolve(Func input, Image<float> & convolution,  int grayscale){

	RDom r(convolution);
	int w=convolution.width(), h=convolution.height();
	Var x,y,c,xo,yo,xi,yi,tile_index;
	Func blur("convolve");
	if (grayscale==0){
		blur(x, y,c) =sum(convolution(r.x, r.y) * input(x + r.x - w/2, y + r.y - h/2,c));		
	} else {
		blur(x, y) =sum(convolution(r.x, r.y) * input(x + r.x - w/2, y + r.y - h/2));
	}
	/*0.47s blur.compute_root().tile(x,y,xo,yo,xi,yi,300,300).fuse(xo,yo,tile_index).parallel(tile_index).vectorize(xi,16);/**/
	/*0.37s*/ 
	 Target target = get_target_from_environment();
	if (target.features & (Target::CUDA | Target::OpenCL)) {
		std::cout<<"found gpu \n";
		blur.cuda_tile(x,y,20,20).compute_root(); }
	else {
		blur.compute_root().vectorize(x,16).parallel(y);
	}
	return blur;
}

Func blur_gaussian(Func im, float sigma, int grayscale){
	Image<float> gaussian=calculate_gaussian(sigma);
	return convolve(im,gaussian, grayscale);
}

arma::vec dot_prod_number(double x, double y, arma::mat H1){
	arma::fmat H=arma::conv_to<arma::fmat>::from(H1);
    float in [3]={0.0f,0.0f,1.0f}; in[0]=y; in[1]=x;
    float out [3]={0.0f,0.0f,0.0f};
    for (int i=0; i<3;i++){
        for (int j=0; j<3; j++){
            out[i]+=H(i,j)*in[j];
        };
    };
    arma::vec p(2);
    p(0)=(int)floor(out[0]/out[2]);
    p(1)=(int)floor(out[1]/out[2]);
    return p;
};


Expr dot_prod_expr(Expr x, Expr y, arma::mat H1,int ind){
	arma::fmat H=arma::conv_to<arma::fmat>::from(H1);
    Expr in [3]={0.0f,0.0f,1.0f}; in[0]=y; in[1]=x;
    Expr out [3]={0.0f,0.0f,0.0f};
    for (int i=0; i<3;i++){
        for (int j=0; j<3; j++){
        	float r=H(i,j);
            out[i]+=r*in[j];
        };
    };
    if (ind==1) {
        return cast<int>(out[0]/out[2]);
    }
    else {
        return cast<int>(out[1]/out[2]);
    };
};

Expr dotPf(Expr x, Expr y, arma::mat H1,int ind){
	arma::fmat H=arma::conv_to<arma::fmat>::from(H1);
    Expr in [3]={0.0f,0.0f,1.0f}; in[0]=y; in[1]=x;
    Expr out [3]={0.0f,0.0f,0.0f};
    for (int i=0; i<3;i++){
        for (int j=0; j<3; j++){
        	float r=H(i,j);
            out[i]+=r*in[j];
        };
    };
    if (ind==1) {
        return out[0]/out[2];
    }
    else {
        return out[1]/out[2];
    };
};

arma::mat calculate_homography( int  point_list1 [4][2], int  point_list2 [4][2]){
	arma::mat A(9,9); A.fill(0.0);
	A(8,8)=1;
	//LeastSquaresSolver<3, 2> solver;
	for (int i=0; i<4; i++){
		int y=point_list1[i][0],x=point_list1[i][1],y1=point_list2[i][0],x1=point_list2[i][1];
		int yy[2], xx[3]; xx[0]=x; xx[1]=y; xx[2]=1; yy[0]=x1; yy[1]=y1;
		//solver.addCorrespondence(xx,yy);
		A(2*i,0)=y;     A(2*i,1)=x;     A(2*i,2)=1.0f;   A(2*i,3)=0.0f;   A(2*i,4)=0.0f;   A(2*i,5)=0.0f;   A(2*i,6)=-y*y1;   A(2*i,7)=-y1*x;   A(2*i,8)=-y1;
		A(2*i+1,0)=0.0f; A(2*i+1,1)=0.0f; A(2*i+1,2)=0.0f; A(2*i+1,3)=y;   A(2*i+1,4)=x;   A(2*i+1,5)=1.0f;   A(2*i+1,6)=-x1*y; A(2*i+1,7)=-x1*x; A(2*i+1,8)=-x1;
		
		
	}
	//double sol[3][3];
	//solver.solve(sol);
	arma::vec B(9); B.fill(0);  B(8)=1;
	arma::vec X=arma::solve(A,B);
	arma::mat H(3,3);
	H<<X(0)<<X(1)<<X(2)<<arma::endr
	 <<X(3)<<X(4)<<X(5)<<arma::endr
	 <<X(6)<<X(7)<<X(8)<<arma::endr;
	// std::cout<<"correct one "<< H<<"\n";
	// std::cout<<"our one"<<"\n"<< sol[0][0]<<" "<< sol[0][1]<<" "<< sol[0][2]<<"\n"<< sol[1][0]<<" "<< sol[1][1]<<" "<< sol[1][2]<<"\n"<< sol[2][0]<<" "<< sol[2][1]<<" "<< sol[2][2]<<" ";
	return H;
}


arma::mat compute_transformed_bbox(const Image<uint8_t> im, arma::mat H){
	arma::mat H1=arma::inv(H);
	int xmin=0,xmax=0.,ymin=0,ymax=0;
	int fory [4]={0,0, im.height()-1, im.height()-1}, forx [4]={0, im.width()-1, 0, im.width()-1};
	for (int i=0; i<4; i++){
			int y=fory[i]; int x=forx[i];
			arma::vec p=dot_prod_number(x,y,H1);
			int y1=p(0); int x1=p(1);
			xmin=std::min(xmin,x1);
			xmax=std::max(xmax,x1);
			ymin=std::min(ymin,y1);
			ymax=std::max(ymax,y1);
	}
	arma::mat M;
	M<<ymin<<xmin<<arma::endr<<
	    ymax<<xmax<<arma::endr;
	return M;
}

arma::mat bbox_union(arma::mat B1, arma::mat B2){
	int ymin=std::min(B1(0,0),B2(0,0));
	int xmin=std::min(B1(0,1),B2(0,1));
	int ymax=std::max(B1(1,0),B2(1,0));
	int xmax=std::max(B1(1,1),B2(1,1));
	arma::mat M;
	M<<ymin<<xmin<<arma::endr<<
	    ymax<<xmax<<arma::endr;
	return M;
}



arma::mat translate(arma::mat bounding_box){
	int tx=-bounding_box(0,1);
	int ty=-bounding_box(0,0);
	arma::mat M;
	M<<1<<0<<ty<<arma::endr
	 <<0<<1<<tx<<arma::endr
	 <<0<<0<<1<<arma::endr;
	return M;
}

Image<uint8_t> apply_homography(const Image<uint8_t>& im1, const Image<uint8_t>& im2, arma::mat H){
	Expr w2=im2.width(), h2=im2.height();
    Var x("x"),y("y"),c("c");
	Func apply("homo"), Final("Final");
	
	apply(x,y,c)=select( ( (dot_prod_expr(x,y,H,0)<0) || (dot_prod_expr(x,y,H,0)>w2-1) || (dot_prod_expr(x,y,H,1)<0) || (dot_prod_expr(x,y,H,1)>h2-1) ), im1(x,y,c),im2(clamp(dot_prod_expr(x,y,H,0),0,w2-1), clamp(dot_prod_expr(x,y,H,1),0,h2-1),c));
	Image<uint8_t> output = apply.realize(im1.width(), im1.height(),im1.channels());
	return output;
}


Image<uint8_t> stitch( Image<uint8_t>& im1, Image<uint8_t>& im2, arma::mat & H){
	arma::mat b2=compute_transformed_bbox(im2, H);
	arma::mat b1;
	b1<<0<<0<<arma::endr
	  <<im1.height()-1<<im1.width()-1<<arma::endr;
	arma::mat box=bbox_union(b1,b2);
	Image<uint8_t> black(box(1,1)-box(0,1),box(1,0)-box(0,0),3);
	arma::mat T=translate(box);
	T(0,2)*=-1;
	black=apply_homography(black, im1,  T);
	arma::mat HT=H*T;
	return apply_homography(black, im2,  HT);
}



Func calculate_tensor( Image<uint8_t>& im, double sigma_g=1, double factor_sigma=4){
	Var x,y,c,xo,yo,xi,yi,tile_index;
	Func clamped,tensor_im, tensor,luminance("lum"), luminance_blurred("luminance blurred"), gradient_x("gx"), gradient_x1,gradient_y("gy"), tensor_blurred("tensor_blurred");
	int w=im.width(), h=im.height();

	//calculate luminance
	clamped(x,y,c)=im(clamp(x, 0, w-1), clamp(y, 0, h-1),c);
	luminance(x,y)=clamped(x,y,0)*0.3f+clamped(x,y,1)*0.6f+clamped(x,y,2)*0.1f;
		
	//blur it
	luminance_blurred=blur_gaussian(luminance,sigma_g,1);
	
	//calculate the gradient
	gradient_x(x,y)=luminance_blurred(max(x-1,0),y)-luminance_blurred(min(x+1,w-1),y);
	gradient_y(x,y)=luminance_blurred(x,max(y-1,0))-luminance_blurred(x,min(y+1,h-1));
		
	//calculate the tensor
	tensor(x,y,c)=select((c==0),gradient_x(x,y)*gradient_x(x,y), select((c==1),gradient_x(x,y)*gradient_y(x,y),gradient_y(x,y)*gradient_y(x,y)) );
	tensor.compute_root().parallel(y).vectorize(x,16);//.tile(x,y,xo,yo,xi,yi,300,300).fuse(xo,yo,tile_index).parallel(tile_index);
	
	//blur the tensor	
	tensor_blurred=blur_gaussian(tensor,sigma_g*factor_sigma,0);
	
	//schedule everything
	luminance.compute_root();	
	//gradient_x.compute_root().parallel(y);
	//gradient_x.compute_at(tensor,y);
	//gradient_y.reorder(y,x).compute_at(tensor,x);
	
	tensor.reorder(c,x,y).compute_root();
	
	return tensor_blurred;
}

/*
 * 	begin_timing
		tensor.realize(w,h,3);
		std::cout<<"tensor "; end_timing
	
	*/
Image<int> harris_corners(Image<uint8_t>& im, int max_diam=10, float k=0.15, double sigma_g=1, double factor=4,  double boundary_size=5){
	
	Var x("x"),y("y"),c("c"),yo,yi,xo,xi,tile;
	Func tensor("tensor"), corner_response("CornerResponse"),show_corners("showCorners"),maximum_window_x("maxWindowX"),maximum_window("maxWindow");
	int w=im.width(), h=im.height(), number_corners=0;
	RDom r(-max_diam/2, max_diam);
	Image<uint8_t> detected_corners;
	
	//get the tensor
	tensor=calculate_tensor(im,sigma_g,factor);
		
	//calcullate Corner response
	corner_response(x,y)=tensor(x,y,0)*tensor(x,y,2)-tensor(x,y,1)*tensor(x,y,1)-k*(tensor(x,y,0)+tensor(x,y,2))*(tensor(x,y,0)+tensor(x,y,2));
	
	
	//find the maximum value in each window
	maximum_window_x(x,y)=maximum(corner_response(clamp(x+r,0,w-1),y));
	maximum_window(x,y)=maximum(maximum_window_x(x,clamp(y+r,0,h-1)));
		
		
	//find the corners(local maximas).
	show_corners(x,y,c)=select((corner_response(x,y)==maximum_window(x,y)), cast<uint8_t>(255), cast<uint8_t>(im(x,y,c)*0.5f));
    
    //schedule everything
	show_corners.compute_root();
	tensor.compute_root();
	
	corner_response.compute_root().parallel(y).vectorize(x,16);
	
	maximum_window.reorder(x,y).compute_root();
	maximum_window_x.compute_root();//store_at(maximum_window,x).compute_at(maximum_window,x).vectorize(x,4);
	
	//show the corners
	//begin_timing;
	detected_corners=show_corners.realize(im.width(), im.height(),im.channels());	
	//end_timing;	
	//printf("corners took \n");
	
	//calculate the number of corners
	for (int i=boundary_size; i<w-boundary_size; i++){
		for (int j=boundary_size; j<h-boundary_size; j++){
			if (detected_corners(i,j,0)==255) {
				number_corners++;
			}
		}
	}
	
	Image<int> corners(number_corners,2);
	//save the coordinates of the corners in "corners"
	number_corners=-1;
	for (int i=boundary_size; i<w-boundary_size; i++){
		for (int j=boundary_size; j<h-boundary_size; j++){
			if (detected_corners(i,j,0)==255) {
				number_corners++;
				corners(number_corners,0)=i;
				corners(number_corners,1)=j;
			}
		}
	}
	
	printf("calculated corners %d \n", number_corners);
	return corners;
}


Func compute_features(Image<uint8_t> im, Image<int> corners, float sigma_blur_descriptor=3, int radius_descriptor=10){
    Var x,y,c,number;
	Func input,luminance,descriptor("descriptor"),mean("mean"),st_dev("stDev"),normalized_descriptor("normalizedDescriptor");
	int number_corners=corners.width();
	float area=(radius_descriptor*2.0+1)*(radius_descriptor*2.0+1);
	RDom r(0, radius_descriptor*2+1, 0, radius_descriptor*2+1);
	
	
	
	input(x,y,c)=im(clamp(x,0,im.width()-1),clamp(y,0,im.height()-1),c);
	
	//calculate luminance
	luminance(x,y)=input(x,y,0)*0.3f+input(x,y,1)*0.6f+input(x,y,2)*0.1f;
	
	//blur it
	Func luminance_blurred=blur_gaussian(luminance,sigma_blur_descriptor,1);
   
   //calculate desciptors
	descriptor(number,x,y)=luminance_blurred(clamp(corners(number,0)+x-radius_descriptor,0,im.width()), clamp(corners(number,1)+y-radius_descriptor,0,im.height()));
	
	//calculate mean and standard deviation
	mean(number)=sum(descriptor(number,r.x,r.y))/area;
	st_dev(number)=sqrt(sum((descriptor(number,r.x,r.y)-mean(number))*(descriptor(number,r.x,r.y)-mean(number)))/area);

	//normalize descriptors
	normalized_descriptor(number,x,y)=(mean(number)-descriptor(number,x,y))/st_dev(number);
	
	//schedule everything
	descriptor.compute_root();
	mean.compute_root(); 
	st_dev.compute_root();
    luminance_blurred.compute_root();
	normalized_descriptor.compute_root();
	
	return normalized_descriptor;
}


void visualize_features(Image<uint8_t> im1, Image<float> listFeatures, Image<int> cornerL, int radiusDescriptor=4){
	Func f; Var x,y,c;
	f(x,y,c)= cast<uint8_t>(im1(x,y,c)*0.5f);
	Image<uint8_t> lessBright=f.realize(im1.width(), im1.height(), im1.channels());
	for (int i=0; i<listFeatures.width(); i++){
		for (int j=cornerL(i,0)-radiusDescriptor; j<cornerL(i,0)+radiusDescriptor+1;j++){
			for (int k=cornerL(i,1)-radiusDescriptor; k<cornerL(i,1)+radiusDescriptor+1;k++){
							int xx=j-(cornerL(i,0)-radiusDescriptor); int yy=k-(cornerL(i,1)-radiusDescriptor);
							lessBright(j,k,0)=(int)floor(-listFeatures(i,xx,yy)*0.5);
							lessBright(j,k,1)=(int)floor(listFeatures(i,xx,yy)*0.5);
						}
		}
	}
//	save(lessBright,"check_features.png");
}

Image<int> find_correspondences(Func features_1, Func features_2, const Image<int>& cornerL1, const Image<int>& cornerL2, float radius=21, float threshold=2.0){
	int w1=cornerL1.width(); int w2=cornerL2.width();
	std::vector< std::pair<int,int> > correspondences; int numberCorrespondences=0, best1=0, best2=0;
	Image<float> leastSquaresDif(w1,w2);
	RDom r(0, radius, 0, radius);
	Func leastSquares; Var x,y,xo,yo,xi,yi,tile_index;
	
	//calculate the sqaure difference between any two descriptors	
	leastSquares(x,y)=sum( (features_1(x,r.x,r.y)-features_2(y,r.x,r.y))*(features_1(x,r.x,r.y)-features_2(y,r.x,r.y)) );
	
	leastSquares.parallel(y).vectorize(x,16);
	//1.05 leastSquares.compute_root().tile(x,y,xo,yo,xi,yi,20,20).fuse(xo,yo,tile_index).parallel(tile_index).vectorize(xi,16);/**/
	
	//features_1.compute_at(leastSquares, y);
	
	leastSquaresDif=leastSquares.realize(w1,w2);
	
	//for each descriptor find the best match checking the second neighbor
	for (int i=0; i<w1; i++){
        float minim1=9000000.0f;
       	float minim2=9000000.0f;
        for (int j=0; j<w2; j++){
        	if (leastSquaresDif(i,j)<minim2){
                if (leastSquaresDif(i,j)<minim1){
                    minim1=leastSquaresDif(i,j);
                    best1=j;
                }else{
                    minim2=leastSquaresDif(i,j);
                    best2=j;
                }
            }
        }
            
        if (minim1==0.0f){
        	correspondences.push_back(std::make_pair(i,best1));
        } else if (1.0f*minim2/(minim1*1.0f)>threshold){
        	correspondences.push_back(std::make_pair(i,best1));
        }

	}
	Image<int> output(correspondences.size(),2,2);
	for (int i=0;i<correspondences.size();i++){
		output(i,0,0)=cornerL1(correspondences[i].first,0);
		output(i,0,1)=cornerL1(correspondences[i].first,1);
		output(i,1,0)=cornerL2(correspondences[i].second,0);
		output(i,1,1)=cornerL2(correspondences[i].second,1);		
	}
	printf("found correspondences %d \n", numberCorrespondences);
    return output;

}




std::vector<int> generate4(int n){
    std::vector<int> result(4);
	for(int j = 0; j < 4; j++){
	    int r;
	    int repeated=0;
	    do{
	    	r = rand()%n;
	    	repeated=0;
	    	for (int k=0; k<j;k++){
	    		if (r==result[k]) {repeated=1;};
	    	}
	   } while (repeated==1);
	   result[j] = r;
	}
	return result;
}

Expr difference(const arma::mat& M, const Expr& y, const Expr& x,const Expr& y1, const Expr& x1){
	Expr x0=dotPf(x,y,M,0), y0=dotPf(x,y,M,1);
    return (y0-y1)*(y0-y1)+(x0-x1)*(x0-x1);
}


arma::mat ransac(Image<int> listOfCorrespondences, int Niter=10000, float epsilon=4.0){
    int maxim=0;
    arma::mat bestH;
    int numberCorrespondences=listOfCorrespondences.width();
    printf("%d\n",  numberCorrespondences);
    for (int i=0; i<Niter;i++){
            std::vector<int> result=generate4(numberCorrespondences);
			int point_list1 [4][2]={{listOfCorrespondences(result[0],0,1), listOfCorrespondences(result[0],0,0)},
								   {listOfCorrespondences(result[1],0,1), listOfCorrespondences(result[1],0,0)},
								   {listOfCorrespondences(result[2],0,1), listOfCorrespondences(result[2],0,0)},
								   {listOfCorrespondences(result[3],0,1), listOfCorrespondences(result[3],0,0)}};
			int point_list2 [4][2]={{listOfCorrespondences(result[0],1,1), listOfCorrespondences(result[0],1,0)},
								   {listOfCorrespondences(result[1],1,1), listOfCorrespondences(result[1],1,0)},
								   {listOfCorrespondences(result[2],1,1), listOfCorrespondences(result[2],1,0)},
	        					   {listOfCorrespondences(result[3],1,1), listOfCorrespondences(result[3],1,0)}};
            arma::mat M=calculate_homography(point_list1,point_list2);

            int numInliers=0;
            Func inlier; Var j;
           //inlier(j)=select(difference(M,listOfCorrespondences(j,0,1),listOfCorrespondences(j,0,0),listOfCorrespondences(j,1,1),listOfCorrespondences(j,1,0))<epsilon*epsilon, 1,0);
          // inlier.parallel(j);
          // Image<int> inliers=inlier.realize(numberCorrespondences);
            for (int k=0; k<numberCorrespondences; k++){
            	arma::vec p=dot_prod_number(listOfCorrespondences(k,0,0),listOfCorrespondences(k,0,1),M);
            	float dif=(listOfCorrespondences(k,1,0)-p(1))*(listOfCorrespondences(k,1,0)-p(1))+(listOfCorrespondences(k,1,1)-p(0))*(listOfCorrespondences(k,1,1)-p(0));

            	if (dif<epsilon*epsilon) {numInliers++;};
            }

            if (numInliers>maxim){
                maxim=numInliers;
                bestH=M;
            }
            //printf("%d %d\n", i,maxim );
    }
    printf("ggg? \n");
    return bestH;
}

 std::string NumberToString ( int Number )
  {
     std::ostringstream ss;
     ss << Number;
     return ss.str();
  }
  
void VisualizeCorrespondences(Image<uint8_t> im1, Image<uint8_t> im2, Image<int> correspondences){
	Func f("ff"),g("gg"); Var x,y,c;

	for (int i=0; i<correspondences.width();i++){
		Image<uint8_t> im1copy(im1.width(),im1.height(),im1.channels());
		Image<uint8_t> im2copy(im2.width(),im2.height(),im2.channels());

		for (int t=0; t<im1.width();t++){
			for (int j=0; j<im1.height(); j++){
				for (int k=0; k<im1.channels();k++){
					im1copy(t,j,k)=im1(t,j,k)*1;
					im2copy(t,j,k)=im2(t,j,k)*1;
				}
			}
		}

		int x1=correspondences(i,0,0), y1=correspondences(i,0,1), x2=correspondences(i,1,0), y2=correspondences(i,1,1);
		im1copy(x1,y1,0)=255; im1copy(x1-1,y1,0)=255; im1copy(x1+1,y1,0)=255; im1copy(x1,y1-1,0)=255; im1copy(x1,y1+1,0)=255; im1copy(x1-1,y1-1,0)=255;
		im1copy(x1,y1,1)=0;   im1copy(x1-1,y1,1)=0;   im1copy(x1+1,y1,1)=0;   im1copy(x1,y1-1,1)=0;   im1copy(x1,y1+1,1)=0;   im1copy(x1-1,y1-1,1)=0;
		im1copy(x1,y1,2)=0;   im1copy(x1-1,y1,2)=0;   im1copy(x1+1,y1,2)=0;   im1copy(x1,y1-1,2)=0;   im1copy(x1,y1+1,2)=0;   im1copy(x1-1,y1-1,2)=0;
		im2copy(x2,y2,0)=255; im2copy(x2-1,y2,0)=255; im2copy(x2+1,y2,0)=255; im2copy(x2,y2-1,0)=255; im2copy(x2,y2+1,0)=255; im2copy(x2-1,y2-1,0)=255;
		im2copy(x2,y2,1)=0;   im2copy(x2-1,y2,1)=0;   im2copy(x2+1,y2,1)=0;   im2copy(x2,y2-1,1)=0;   im2copy(x2,y2+1,1)=0;   im2copy(x2-1,y2-1,1)=0;
		im2copy(x2,y2,2)=0;   im2copy(x2-1,y2,2)=0;   im2copy(x2+1,y2,2)=0;   im2copy(x2,y2-1,2)=0;   im2copy(x2,y2+1,2)=0;   im2copy(x2-1,y2-1,2)=0;
		std::string n=NumberToString(i);
		save(im1copy, "features/"+n+"-im1.png");
		save(im2copy, "features/"+n+"-im2.png");
	}
}

void auto_stitch(Image<uint8_t>& im1, Image<uint8_t>& im2, Image<uint8_t>& destination){
	
	begin_timing
	
	Image<int> corners1=harris_corners(im1, 10, 0.15, 1, 5,  5);
	std::cout<<"corners in main"; end_timing;
	
	Image<int> corners2=harris_corners(im2, 10, 0.15, 1, 5,  5);
	
	begin_timing
	
	Func features1=compute_features(im1,corners1);
	Func features2=compute_features(im2, corners2);

	std::cout<<"features"; end_timing;
	
	begin_timing
	Image<int> correspondences=find_correspondences(features1,features2, corners1, corners2);
	std::cout<<"correspondences"; end_timing;
	

	arma::mat H=ransac(correspondences,10000,4);

	destination=stitch(im1,im2,H);
	
}

int main(int argc, char **argv) {
	srand(time(NULL));
	Image<uint8_t> im1 = load<uint8_t>("images/iphone1.png");
	Image<uint8_t> im2 = load<uint8_t>("images/iphone2.png");

	Image<uint8_t> output;
	auto_stitch(im1,im2, output);
	save(output, "stitch2.png");

	printf("Success!\n");
	return 0;
}
