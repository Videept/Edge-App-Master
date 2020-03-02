// compile using: make server (after installing cuda libraries and if necessary changing Makefile to point to them). 

// Changelog:
// Added resizing of network input (no need to relearn weights since only used convolutional layers). DL 28/8/18
// Updated HTTP interface to allow client to specify rotation to be applied to image before detection. Cleaned u JSON response format.  DL 5/9/18
// Faster jpeg and rotate code
// Now supports UDP.  DL 15/9/18
// Allows raw YUV image input now (not just JPEG).  DL 20/9/18
// Timing info added to json.  DL 23/9/18 
// Changed to use a single TCP connection for a client to send multipler images, to avoid syn-synack overhead.  A hack just now, and means server can currently only accept one user at a time.  DL 3/5/19

#define VERSION "1.7"

#define DEFAULT_CONFIG_MODEL "darknet/cfg/yolov3.cfg"
#define DEFAULT_MODEL_WEIGHTS "darknet/yolov3.weights"
#define DEFAULT_MODEL_NAMES "darknet/data/coco.names"
#define DEFAULT_DIM 608 // default input size to network 608x608
#define DEFAULT_PORT 8000
#define MAXLEN 1024000 // 1MB, max POST image size
#define BUFFER_SIZE 4096 // max line size of HTTP request
#define RECV_TIMEOUT 20000 // timeout in us (used to abort connection on packet loss)

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
#include <string.h>

#include "darknet/include/darknet.h"

#ifdef LIBJPEG // faster option using tweaked libjpeg-turbo
  #include <jpeglib.h>
#else
  #define STB_IMAGE_IMPLEMENTATION
  #ifdef __ANDROID__  // ndk-build compiles statically against darknet
  #else
    #include "darknet/src/stb_image.h"
  #endif
#endif

#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#include <pthread.h>

// global vars, easier to use within thread
int active=0;  // indicates whether GPU is busy or not
pthread_mutex_t active_mutex = PTHREAD_MUTEX_INITIALIZER;
network *net;           // neural net
char **names;           // class labels
int verbose=0;          // debugging level
int save_to_file=0;     // indicates whether received images are to be dumped out to file
int count=0;            // counts number of images processed

// struct for passing parameters to thread
typedef struct Params {
   int session_fd;  // socket to receive data on (will be pipe when a UDP connection)  
   int send_fd; // socket to send response on
   struct sockaddr_in si_active; // client address for connection, NULL for TCP
   int slen;  // set to 0 fot TCP
   void* ptr; // for debugging
} Params;

// struct for reassembly buffer
#define REASSEMBLY_SIZE 1024
typedef struct reassembly_info {
  int nxt_pkt_index; // index of next in-order packet
  int highest_pkt_index; 
  char* reassembly_buf[REASSEMBLY_SIZE]; // out of order packets
  int reassembly_buf_len[REASSEMBLY_SIZE]; // size of each packet
} reassembly_info;
void dump_reassembly_state(reassembly_info *r_info);

#define DEBUG_JSON(args ...) if (verbose&16) printf(args)
#define DEBUG_UDP(args ...) if (verbose&8) printf(args)
#define DEBUG_HTTP(args ...) if (verbose&4) printf(args)
#define DEBUG_JPG(args ...) if (verbose&2) printf(args)
#define DEBUG_TIME(args ...) if (verbose) printf(args)
#define ERR(args ...) do{fprintf(stderr,"ERROR: "); fprintf(stdout, args);}while(0)
#define WARN(args ...) do{fprintf(stderr,"WARNING: "); fprintf(stdout, args);}while(0)
#define INFO(args ...) if (verbose) fprintf(stdout, args)
#define TICK(X) clock_t (X) = clock()
#define TOCK(X,Y)  (double)((X) - (Y)) / CLOCKS_PER_SEC
#define NOW clock()

void usage(char *progname) {
  char* usage_str =
  "     Usage: %s v%s\n"
  "          -m    sets file containing model config\n"
  "          -w    sets file containing model weights\n"
  "          -n    sets file containing class names\n"
  "          -d    sets input size of network\n"
  "          -p    sets port for server to listen on\n"
  "          -v    print extra diagnostic output\n"
  "          -s    saves each received image to a file (named img_<count>.jpg)\n"
  "          -h    prints this message\n";
  printf(usage_str, progname, VERSION);
}

int read_line(int fd, char* inbuf, size_t *inbuf_used, char* line) {
  //read from socket until hit next newline. fine for both TCP and UDP sockets.
  int i=0;
  size_t read_posn=0;
  while (i < BUFFER_SIZE) {
    if (read_posn == *inbuf_used) {
      // read from socket
      // TO DO: check that packet is from expected source IP/port (might be interleaved with a new request for example)
      ssize_t rv = recv(fd, (void*)&inbuf[*inbuf_used], BUFFER_SIZE - *inbuf_used, 0);
      if (rv == 0) {
        WARN("HTTP connection closed.\n");
        return -1;
      }
      if (rv < 0) {
        if (errno == EAGAIN) { 
           WARN("HTTP connection timeout\n"); 
        } else {
           ERR("HTTP connection error: %s\n",strerror(errno));
        }
        return -1;
      }
      *inbuf_used += rv;
    }
    line[i++] = inbuf[read_posn++]; // advance read position within buffer
    if (line[i-1]=='\n') break; // have hit a newline, stop
  }
  if (i==BUFFER_SIZE) {
    ERR("HTTP request input line larger than %d.\n",BUFFER_SIZE);
    // could send "413 Entity Too Large" response back to client
    return -1;
  }
  line[i]='\0'; // terminate line as string, makes for easier printing when debugging
  // shift buffer contents so next line starts at posn 0
  memmove(inbuf,inbuf+read_posn,*inbuf_used-read_posn);
  *inbuf_used -= read_posn;

  return i;
}

int get_post_data(int fd, char* post_data, int *len, int *out_format, int *rotation, 
                  int *isYUV, int *w, int *h) {
  // extract POST data (the image to be processed) from http request
  size_t inbuf_used = 0;
  char inbuf[BUFFER_SIZE], line[BUFFER_SIZE];
  int lines_read=0, content_length=0, res;
  // parse HTTP request and headers
  while ( (res=read_line(fd, inbuf, &inbuf_used, line))>0) {
    DEBUG_HTTP("%s",line);
    lines_read++;
    if (lines_read==1) {
       // first line is HTTP request
       char req[BUFFER_SIZE]; memset(req,0,BUFFER_SIZE);
       sscanf(line,"POST %s HTTP/", req);
       DEBUG_HTTP("req: %s\n",req);
       // separate request from parameters, if any
       size_t posn = strcspn(req, "?");
       req[posn]='\0';
       char* paras = &req[posn+1];
       if (strcmp("/api/edge_app",req)==0) {
          *out_format=0;
       } else if (strcmp("/dummy",req)==0) {
	  return -1; // dummy req, bail
       } else if (strcmp("/detect",req)==0) {
          *out_format=1;
       } else if (strcmp("/api/edge_app2",req)==0) { // NG format!
          *out_format=2;
          if (strlen(paras)>0) {
            char* lasts;
            char* para=strtok_r(paras,"& ",&lasts);
            //printf("para=%s\n",para);
            while (para!=NULL) {
               char* name = strtok(para,"=");
               char* val = strtok(NULL," ");
               //printf("%s %s\n",name,val);
               para=strtok_r(NULL,"& ",&lasts);
               //printf("para=%s\n",para);
               if (strcmp(name, "r")==0) { *rotation=atoi(val); continue;}
               if (strcmp(name, "w")==0) {*w=atoi(val); continue;}
               if (strcmp(name, "h")==0) {*h=atoi(val); continue;}
               if (strcmp(name, "isYUV")==0) {*isYUV=atoi(val); continue;}
            }
            DEBUG_HTTP("rotate: %d\n", *rotation);
            DEBUG_HTTP("yuv: %d\n", *isYUV);
            DEBUG_HTTP("w: %d\n", *w);
            DEBUG_HTTP("h: %d\n", *h);
          }
       } else {
         ERR("Invalid request: %s",line);
         return -1;
       }
       continue; 
    } 
    // parse HTTP headers to get content-length (length of POST body)
    if(line[0]=='\r' && line[1] == '\n') {
       // an empty line, have reached end of request and headers
       break;
    }
    size_t posn = strcspn(line, ":");  // extract the header name
    char *header_name = line; header_name[posn]='\0';
    posn++; while (line[posn] == ' ') {posn++;}
    size_t value_len = strcspn(line+posn, "\r\n"); // header value
    char *header_value = line+posn; header_value[value_len]='\0';
    DEBUG_HTTP("header: %s %s\n", header_name, header_value);
    if (strcmp("Content-Length",header_name)==0) {
      content_length = atoi(header_value);
      DEBUG_HTTP("content-length=%d\n",content_length);
    }
  }
  if (res<0) {
    return -1;
  }
  if (content_length ==0) {
    ERR("HTTP request has no POST data\n");
    return -1;
  }
  // now read the post data ...
  // copy over the part already read into buffer
  int extra = (int) inbuf_used;
  memcpy(post_data,inbuf,extra);
  ssize_t bytes=extra;
  // and now read the rest from socket
  ssize_t rv=0; 
  while (bytes < content_length) {
    rv = recv(fd, (void*)&post_data[bytes], MAXLEN - bytes, 0);
    if (rv < 0) {
       if (errno == EAGAIN)
          INFO("Timeout when reading POST data %d/%d (packet loss ?)\n", bytes, content_length);
       else 
          ERR("Problem when reading POST data: %s\n",strerror(errno));
    }
    if (rv<=0) break;
    bytes+=rv;
    //DEBUG_HTTP("received %d/%d/%d POST data\n", rv, bytes, content_length);
  }
  if (rv<0) {
    return -1;
  }
  if (bytes < content_length) {
    ERR("POST data ended early, got %d but expected %d\n",(int)bytes,content_length);
    return -1;
  }
  *len = content_length;
  //post_data[*len+1]='\0';printf("%s\n",post_data);
  return 0;
}

network* init(char* cfgfile, char* weightfile, int w, int h) {
  // load config files
  network *net = load_network(cfgfile, weightfile, 0);
  set_batch_network(net, 1);
  if ((net->w != w) || (net->h !=h)) {
    DEBUG_JPG("resizing from %dx%d to %dx%d\n",net->w, net->h, w, h);
    TICK(start_resize);
    resize_network(net,w,h);
    DEBUG_TIME("time to resize: %f ms\n",TOCK(NOW,start_resize)*1000);    
  }
  return net;
}

#ifdef LIBJPEG
unsigned char* libjpg_load_from_memory(unsigned char *buff, int len, int rotation, int net_w, int net_h,
                      int *w, int*h, int *c, float *scale) {
    // call libjpeg-turbo to decode image pointed to by buff
  
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPARRAY buffer;
    int row_stride;
    unsigned char* bbuf;

    // initialise jpeg decoder and extract image width and height from header ...
    cinfo.err = jpeg_std_error( &jerr );
    jpeg_create_decompress( &cinfo );
    jpeg_mem_src( &cinfo, buff, len );
    jpeg_read_header( &cinfo, 1 );
    jpeg_calc_output_dimensions(&cinfo);

    // calc image width and height after rotation is applied, we'll scale and trim image while
    // decoding so that it fits into net_w by net_h box *after* rotation
    int dst_w=cinfo.output_width, dst_h=cinfo.output_height;
    if ((rotation%180==90) || (rotation%180==-90)) {
       dst_w=cinfo.output_height; dst_h = cinfo.output_width;
    }

    // assuming net_w=net_h to simplify the following.  so when image w>h its enough to scale it so that its width 
    // is less than net_w since that automatically ensures its height is less than net_h.  libjpeg-turbo allows image
    // scaling of the form num/8 where num is an integer between 1 and 15.

    if (dst_w >= dst_h) { // landscape image, we need to scale so that width fits inside net_w wide box
       int i;
       for (i=15;i>0;i--) {
          if (dst_w*i/8 <= net_w) break;
       }
       cinfo.scale_num=i; cinfo.scale_denom=8; 
       *scale=cinfo.scale_num*1.0/cinfo.scale_denom;
    } else { //portrait, need to scale so that height <= net_h
       int i;
       for (i=15;i>0;i--) {
          if (dst_h*i/8 <= net_h) break;
       }
       cinfo.scale_num=i; cinfo.scale_denom=8; 
       *scale=cinfo.scale_num*1.0/cinfo.scale_denom;
    }
    DEBUG_JPG("dst_w=%d,dst_h=%d,scale=%0.1f,scaled_w=%.1f,scaled_h=%.1f, net_w/h=%d\n",
               dst_w,dst_h,*scale,dst_w*(*scale),dst_h*(*scale), net_w);
  
    // set decoder parameters -- try to be fast!
    cinfo.dct_method=JDCT_FASTEST;
    cinfo.do_fancy_upsampling=0;
    jpeg_start_decompress( &cinfo );
  
    // now do the actual jpeg decoding ...
    row_stride = cinfo.output_width * cinfo.output_components;
    unsigned char* data = (unsigned char*)malloc(
              cinfo.output_width * cinfo.output_height *  cinfo.output_components);
    while( cinfo.output_scanline < cinfo.output_height ) {
       bbuf = ( ( data + ( row_stride * cinfo.output_scanline ) ) );
       buffer = &bbuf;
       jpeg_read_scanlines( &cinfo, buffer, 1 );
    }
    jpeg_finish_decompress( &cinfo );
    *w= cinfo.output_width; // image width
    *h= cinfo.output_height; // image height
    *c= cinfo.output_components;  // number of channels (3 for color image)
    jpeg_destroy_decompress( &cinfo );
    return data;
}
#endif

inline float u8tofloat(uint8_t x){
   // see http://lolengine.net/blog/2011/3/20/understanding-fast-float-integer-conversions
   union { float f; uint32_t i; } u; u.f = 32768.0f; u.i |= x;
   return u.f - 32768.0f;
}

int load_image_mem(unsigned char *buff, int len, int rotation, int net_w, int net_h, 
                                unsigned char** rgb_data, int *w, int *h, int *c, float *scale) {
    // try to decode contents of buff as jpeg image and rescale to width net_w pixels.
    // scale is set to scaling applied 

#ifdef LIBJPEG
    // decode jpeg and scale to fit within net_w by net_h box after rotation applied (but rotation not yet
    // applied of course)
    *rgb_data = libjpg_load_from_memory(buff, len, rotation, net_w, net_h, w, h, c, scale);
#else
    *rgb_data = stbi_load_from_memory(buff, len, w, h, c, 3);
    if (!rgb_data) {
        ERR("Cannot load image, STB Reason: %s\n", stbi_failure_reason());
        return -1;
    }
    *scale=1.0; 
#endif
   return 0;
}

void rotate_and_convert(unsigned char *rgb_data, int w, int h, int c, int rotation, int net_w, int net_h,
                        image *im, float *scale, int *pad_w, int*pad_h) { 
    // apply any requested rotation and rearrange pixels to bitmap format used by yolo 
    // (we assume that net_w=net_h).  we keep everything as integers for now as helps
    // with CPU caching (image is 4 times smaller than when converted to floats) 
    int i,j,k,dst_index,src_index;

    DEBUG_JPG("Converting to yolo format and rotating image by %d ...\n", rotation);
    DEBUG_JPG("rot=%d\n",rotation);
    unsigned char* yolo_data = malloc(w*h*c);
    switch (rotation) {
      case 90:
         for(k = 0; k < c; ++k){
         for(i = 0; i < w; i++){
            dst_index = h-1-0 + h*i + w*h*k;
            src_index = k + c*i + c*w*0;
            for(j = 0; j < h; ++j){
               yolo_data[dst_index] = rgb_data[src_index];
               dst_index--;
               src_index+=c*w;
            }
         }
         }
         break;
      case 180:
         for(k = 0; k < c; ++k){
         for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
               src_index = k + c*i + c*w*j;
               dst_index = (w-1-i) + w*(h-1-j) + w*h*k;
               yolo_data[dst_index] = rgb_data[src_index];
            }
         }
         }
         break;
      case 270:
      case -90:
         for(k = 0; k < c; ++k){
            for(i = 0; i < w; ++i){
              src_index = k + c*i + c*w*(h-1);
              dst_index = (h-1)+h*(w-1-i)+w*h*k;
              for(j = h-1; j >= 0; j--){ // runs x3 faster when count down rather than up (caching I guess ...)
                 //src_index = k + c*i + c*w*j;
                 //dst_index = j+h*(w-1-i)+w*h*k;
                 yolo_data[dst_index] = rgb_data[src_index];
                 dst_index--; src_index-=c*w;
            }
        }
        }
         break;
      case 0:
      default:
        for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            src_index = k + c*(w-1) + c*w*j;
            dst_index = (w-1) + w*j + w*h*k;
            for(i = w-1; i >=0; i--){ // runs x3 faster when count down rather than up (caching I guess ...)
               //src_index = k + c*i + c*w*j;
               //dst_index = i + w*j + w*h*k;
               yolo_data[dst_index] = rgb_data[src_index];
               dst_index--; src_index-=c;
            }
        }
       }
   }

   int w_rot=w, h_rot=h;
   if ((rotation%180==90) || (rotation%180==-90)) {
         w_rot=h; h_rot=w; 
   }
   int size = w_rot>h_rot ? w_rot : h_rot;
   if (size < net_w) size=net_w;
   *pad_w = (size-w_rot)/2; *pad_h = (size-h_rot)/2;
   *im = make_image(size, size, c);
   for(k = 0; k < c; ++k){
      for(j = 0; j < h_rot; ++j){
         dst_index = (w_rot-1+*pad_w) + size*(j+*pad_h) + size*size*k;
         src_index = (w_rot-1)+w_rot*j+w_rot*h_rot*k;
         for(i = w_rot-1; i >=0; i--){
            //im->data[dst_index] = (float)yolo_data[src_index]/255.;
            im->data[dst_index] =  u8tofloat(yolo_data[src_index]);
            dst_index--; src_index--;
         }
      }
   }
   free(rgb_data); free(yolo_data);

   // resize image if necessary (will never be called if have used libjpeg)
   if (size > net_w || size > net_h) { // only scale down, not up (is this ok ?) 
      DEBUG_JPG("Resizing image from w:%d h:%d to w:%d h:%d\n",im->w,im->h,net_w,net_h);
      // this call is slooow ...
      image resized = resize_image(*im, net_w, net_h);
      free_image(*im);
      *im = resized;
      *scale = net_w*1.0/size;
    }
  DEBUG_JPG("done\n");
}

uint8_t clamp(int16_t value) {
   return value<0 ? 0 : (value>255 ? 255 : value);
}

void yuv2rgb(uint8_t y, uint8_t u, uint8_t v, uint8_t *r, uint8_t *g, uint8_t *b) {
   // yuvimage conversion values (from wikipedia):
   // int rTmp = yValue + (1.370705 * (vValue-128));
   // int gTmp = yValue - (0.698001 * (vValue-128)) - (0.337633 * (uValue-128));
   // int bTmp = yValue + (1.732446 * (uValue-128));
   int y1192 = 1192 * y;
   int r_tmp = (y1192 + 1634 * (v-128))/1192;
   int g_tmp = (y1192 - 833 * (v-128) - 400 * (u-128))/1192;
   int b_tmp = (y1192 + 2066 * (u-128))/1192;
 
   *r =clamp(r_tmp);  *g =clamp(g_tmp);  *b =clamp(b_tmp);
}

void convertYUVtoRGB(unsigned char *yuv, int len, int w, int h, float scale, unsigned char **rgb) {
   // really NV21 to RGB
   // see https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420p_(and_Y%E2%80%B2V12_or_YV12)_to_RGB888_conversion
   int i,j,k;
   *rgb = malloc(w*h*3);
   float hcount=0, wcount=0;   
   for (j=0; j<h; j++) {
      wcount=0;
      for (i=0; i<w; i++) {
         uint8_t y = yuv[j * w  + i];
         uint8_t u = yuv[(j / 2) * w  + (i/2)*2  + 1 + w*h];
         uint8_t v = yuv[(j / 2) * w  + (i/2)*2  + w*h];
         int dst=(int)(hcount)*(int)(w*scale)*3 + (int)(wcount)*3;
         yuv2rgb(y, u, v, &(*rgb)[dst], &(*rgb)[dst+1], &(*rgb)[dst+2]);   
         wcount+=scale;
      }
      hcount += scale;
   }
   /*
   // for debugging .. dump out bitmap as .ppm file so we can look at it
   int ww=(int)(w*scale), hh=(int)(h*scale);
   FILE *f = fopen("img.ppm", "w");
   fprintf(f,"P3\n%d %d\n255\n",ww,hh);
   for (j=0; j<hh; j++) {
      for (i=0; i<ww; i++) {
         fprintf(f,"%d %d %d ",(int)(*rgb)[j*ww*3+i*3],(int)(*rgb)[j*ww*3+i*3+1],(int)(*rgb)[j*ww*3+i*3+2]);
      }
      fprintf(f,"\n");
   }
   fclose(f);
   */
}

void display_detections(detection *dets, int num, float thresh, char **names, int classes) {
    // print classification output
    int i,j;
    if (!verbose) return;
    for(i = 0; i < num; ++i){
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
    }
}

void close_session(int *session_fd, int send_fd, struct sockaddr_in* si_active, int slen,
                   char* msg, int len) {
   char* buf = "[]";
   if (msg==NULL) {
      DEBUG_UDP("closing connection, sending empty pkt\n");
      msg=buf; len=2; // send empty response.
   }
   sendto(send_fd,msg,len,0,(struct sockaddr*)si_active,slen);
   if (slen>0) { //its UDP
      // send a couple more, in case of loss (extras will be ignored by client)
      sendto(send_fd,msg,len,0,(struct sockaddr*)&si_active,slen);
      sendto(send_fd,msg,len,0,(struct sockaddr*)&si_active,slen);
   } else { // try to flush tcp sock
      int nodelay=1;
      if (setsockopt(send_fd,IPPROTO_TCP,TCP_NODELAY,&nodelay,sizeof(nodelay))<0) {
        WARN("Failed to set TCP_NODELAY socket option");
       }
   }
   // close the incoming link.  
   // NB for TCP connections incoming and outgoing links (session_fd, send_fd) are the same
   // but for UDP outgoing send_fd link is separate and may be shared by other connections
   // so important not to close it
   if (*session_fd != -1) { close(*session_fd); *session_fd=-1;}
}

void flagGPUfree() {
   // flag that GPU is now available
   pthread_mutex_lock(&active_mutex);
   active = 0; // flag GPU as free
   pthread_mutex_unlock(&active_mutex);
}

void* handle_connection(void* params) {
  // read HTTP request and process it.  handles both udp and tcp connections
  Params *p = (Params*)params;

  // take a copy of passed parameters, so caller is free to reuse params object 
  int session_fd = p->session_fd;
  int send_fd = p->send_fd;
  int slen = p->slen;
  int isTCP = (slen==0);
  struct sockaddr_in *si_active, si_active_buf;
  if (isTCP) // TCP 
     si_active=NULL;
  else { // UDP
     memcpy(&si_active_buf,&p->si_active,slen);
     si_active=&si_active_buf;
  }
  reassembly_info* r_info = (reassembly_info*)p->ptr;
 
  // read from socket image to be processed
start: // nasty temporary goto hack for tcp
  do {} while (0);// dummy
  TICK(starttime);
  char post_data[MAXLEN];
  int len=-1;
  int out_format=0, rotation=0, isYUV=0, w=0, h=0, c=3;
  if (get_post_data(session_fd, post_data, &len, &out_format, &rotation, &isYUV, &w, &h)<0) {
    if (r_info) dump_reassembly_state(r_info); // for debugging
    close_session(&session_fd,send_fd,si_active,slen,NULL,0); 
    flagGPUfree();
    pthread_exit(NULL);
  }

  if (save_to_file) {
    // dump received image out to a file (used for debugging)
    FILE *f; char fname[1024];
    sprintf(fname, "img_%d.jpg",count); count++;
    f = fopen(fname,"wb");
    fwrite(post_data,len,1,f);
    fclose(f);
  }
  
  // decode image to get bitmap in yolo format
  TICK(starttime_decode);
  float scale=1.0; int pad_w=0, pad_h=0;
  image im;
  unsigned char* rgb_data;
  if (!isYUV) { // parse JPEG
    if (load_image_mem((unsigned char*)post_data,(int)len,rotation,net->w,net->h,&rgb_data,&w,&h,&c,&scale)<0){
      close_session(&session_fd,send_fd,si_active,slen,NULL,0);
      flagGPUfree();
      pthread_exit(NULL);
    }
  } else {
     // convert YUV to RGB
     if (w*h*3/2 != len) {
        WARN("POST YUV data len %d does not match supplied image size w=%d, h=%d, c=%d\n",len,w,h,c);
        close_session(&session_fd,send_fd,si_active,slen,NULL,0);
        flagGPUfree();
        pthread_exit(NULL);
     }
     int dst_w=w, dst_h=h;
     if ((rotation%180==90) || (rotation%180==-90)) {
        dst_w=h; dst_h = w;
     }
     scale = dst_w>dst_h ? net->w*1.0/dst_w : net->h*1.0/dst_h;
     if (scale>1.0) scale=1.0;
     convertYUVtoRGB((unsigned char*)post_data, len, w, h, scale, &rgb_data);
     w=w*scale; h=h*scale;
  };
  DEBUG_JPG("w=%d, h=%d, net_w=%d, net_h=%d\n", w, h, net->w, net->h);
 
  TICK(starttime_rot);
  rotate_and_convert(rgb_data, w, h, c, rotation, net->w, net->h, &im, &scale, &pad_w, &pad_h);

  // finally call yolo to do the object detection
  TICK(starttime_yolo);
  network_predict(net, im.data);
  int nboxes = 0;
  float thresh=.5, hier_thresh=.5;
  detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, &nboxes);

  flagGPUfree();
  free_image(im);

  TICK(starttime_results);
  // build the response to send back to client ...
  int classes=0;
  // construct json response
  int json_size = BUFFER_SIZE;
  char* json=malloc(json_size);
  if (out_format>1) { // new format
     strcpy(json,"{");
     strcat(json,"\"results\": [");
  } else
     strcpy(json,"[");

  if (nboxes>0) {
    classes=dets[0].classes;
    float nms=.45;
    do_nms_sort(dets, nboxes, classes, nms);  
    //display_detections(dets, nboxes, thresh, names, classes);
    
    int i,j,count=0;
    for(i = 0; i < nboxes; ++i){
      for(j = 0; j < classes; ++j){
        if (dets[i].prob[j] > thresh){
          if (count) {strcat(json,",");}
          char tuple[BUFFER_SIZE];
          int x,y;
          x=(int)(dets[i].bbox.x-pad_w)/scale;
          y=(int)(dets[i].bbox.y-pad_h)/scale;
          int w=(int)dets[i].bbox.w/scale;
          int h=(int)dets[i].bbox.h/scale;
          float p=dets[i].prob[j];
          switch (out_format) {
          case 0: // victor's jsonpickle format ...
              sprintf(tuple,"{\"py/tuple\": [\"%s\", %f, {\"py/tuple\": [%d,%d,%d,%d]}] }",
                       names[j],p,x,y,w,h);
             break;
          case 1: // darragh's android json format ...
             sprintf(tuple,"{\"topleft\": {\"y\": %d, \"x\": %d}, \"confidence\": %f, \"bottomRight\": {\"y\": %d, \"x\": %d}, \"label\": \"%s\"}",y-h/2,x-w/2,p,y+h/2,x+w/2,names[j]);
             break;
          default: // new improved JSON format
             sprintf(tuple,"{\"title\": \"%s\", \"confidence\": %f, \"x\": %d, \"y\": %d, \"w\": %d, \"h\": %d}",
                       names[j],p,x,y,w,h);
          }
          if (strlen(json)+strlen(tuple) > json_size) {
             // need to expand json buffer
             json_size+=BUFFER_SIZE;
             json = realloc(json, json_size);
          }
          strcat(json,tuple);
          count++;
        }
      }
    }
  }
  strcat(json,"], ");
  
  char json_timing[BUFFER_SIZE];
  sprintf(json_timing,"\"server_timings\": {\"size\": %d, \"r\": %.1f, \"jpg\": %.1f, \"rot\": %.1f, \"yolo\": %.1f, \"json\": %.1f, \"tot\": %.1f}",
             len, 
             TOCK(starttime_decode,starttime)*1000,
             TOCK(starttime_rot,starttime_decode)*1000,
             TOCK(starttime_yolo,starttime_rot)*1000,
             TOCK(starttime_results,starttime_yolo)*1000, 
             TOCK(NOW,starttime_results)*1000,
             TOCK(NOW,starttime)*1000);
  DEBUG_TIME("%s\n", json_timing);

  if (strlen(json)+strlen(json_timing) > json_size) {
    // need to expand json buffer
    json_size+=BUFFER_SIZE;
    json = realloc(json, json_size);
  }
  strcat(json,json_timing);

  // strcat(json_final,"]"); // printf("%s, %d\n",json,strlen(json));
  if (out_format>1) // new format
     strcat(json,"}");
  free_detections(dets, nboxes);
  
  // send the response and shut down connection
  int res;
  if (isTCP) { //TCP, send HTTP response headers for backward compatibility
    char* header=malloc(json_size+BUFFER_SIZE);
    sprintf(header,"HTTP/1.1 200 OK\nContent-Type: application/json\nConnection: close\nContent-Length: %d\n\n%s\n",(int)strlen(json),json);
    DEBUG_JSON("%s\n", header);
    //close_session(&session_fd,send_fd,si_active,slen,header,strlen(header));
    int temp_fd=-1; // keep session open
    close_session(&temp_fd, send_fd, si_active, slen, header, strlen(header));
    free(header); free(json);
    goto start; // reuse tcp connection until client resets it.  nasty hack !
  } else {// UDP, don't bother with HTTP headers
    DEBUG_JSON("%s\n", json);
    close_session(&session_fd,send_fd,si_active,slen,json,strlen(json));
    free(json);
  }

  pthread_exit(NULL);
}

void* accept_tcp(void* param) {
  // listen on TCP port for http connections
  int port = *(int*)param;

  int server_fd=socket(AF_INET,SOCK_STREAM,0);
  if (server_fd==-1) {
    ERR("Failed to create server TCP socket: %s\n",strerror(errno));
    exit(-1);
  }
  int reuseaddr=1;
  setsockopt(server_fd,SOL_SOCKET,SO_REUSEADDR,&reuseaddr,sizeof(reuseaddr));

  struct sockaddr_in addr;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr))==-1) {
    ERR("Failed to bind TCP socket: %s\n",strerror(errno));
    exit(-1);
  }
  if (listen(server_fd,SOMAXCONN)) {
    ERR("Failed to listen for TCP connections: %s\n",strerror(errno));
    exit(-1);
  }
  printf("Listening on TCP port %d\n",port);

  // loop indefinitely and wait for HTTP connections
  while (1) {
    int session_fd=accept(server_fd,0,0);
    if (session_fd==-1) {
      ERR("Failed to accept TCP connection: %s\n",strerror(errno));
      continue;
    }

    // GPU can run only one darknet process at a time, so we need to serialise web requests.
    pthread_t thread_id; 
    pthread_mutex_lock(&active_mutex);
    Params p;
    if (active) {
       pthread_mutex_unlock(&active_mutex);
       close_session(&session_fd,session_fd,NULL,0,NULL,0);
    } else {
       active=1; // flag GPU as busy.
       pthread_mutex_unlock(&active_mutex);
       p.session_fd = session_fd; p.send_fd=session_fd; p.slen=0; p.ptr=0; 
       if( pthread_create( &thread_id , NULL , handle_connection, (void*) &p) < 0) {
         ERR("Failed to create thread\n");
         close_session(&session_fd,session_fd,NULL,0,NULL,0);
         active=0; // flag GPU as free. no need for mutex here.
       }
    }
  }

  close(server_fd);
}

int get_pkt_index(unsigned char* buf) {
   // read packet index out of our header (first 2 bytes of UDP payload)
   // (for in-order delivery book-keeping)
   return  ((int)buf[1])*256+(int)buf[0];
}

void dump_reassembly_state(reassembly_info *r_info) {
   INFO("next=%d, highest=%d, buffered=", r_info->nxt_pkt_index,r_info->highest_pkt_index);
   int i;
   for (i=r_info->nxt_pkt_index; i< r_info->nxt_pkt_index+REASSEMBLY_SIZE; i++) {
      if (i > r_info->highest_pkt_index) break;
      if (r_info->reassembly_buf[i%REASSEMBLY_SIZE]) INFO("%d ",i);
   }
   INFO("\n");
}

void init_reassembly_state(reassembly_info *r_info) {
    r_info->nxt_pkt_index=0;
    r_info->highest_pkt_index=0;
    memset(r_info->reassembly_buf,0,REASSEMBLY_SIZE*sizeof(char*)); // initialise reassembly buffer
}

void clear_reassembly_state(reassembly_info *r_info) {
    int i;
    for (i=r_info->nxt_pkt_index+1; i<=r_info->highest_pkt_index; i++) {
       if (r_info->reassembly_buf[i%REASSEMBLY_SIZE]!=NULL) {
          free(r_info->reassembly_buf[i%REASSEMBLY_SIZE]);
          r_info->reassembly_buf[i%REASSEMBLY_SIZE]=NULL;
       }
    }
    r_info->nxt_pkt_index=0;
    r_info->highest_pkt_index=0;
    memset(r_info->reassembly_buf,0,REASSEMBLY_SIZE*sizeof(char*)); // initialise reassembly buffer
}

void process_pkt_inorder(char* buf, int len, int fd, int send_fd, reassembly_info *r_info) {
   // process incoming UDP packets, buffering as needed to ensure in-order delivery
   if (get_pkt_index(buf) > r_info->highest_pkt_index) r_info->highest_pkt_index=get_pkt_index(buf);
   if (get_pkt_index(buf) == r_info->nxt_pkt_index) {
      // in-order packet, send it on to handler
      DEBUG_UDP("forwarding %d ",r_info->nxt_pkt_index);
      if (send(fd,buf+2,len-2,0)<0) {
         ERR("Pipe error: %s\n",strerror(errno));
      }
      r_info->nxt_pkt_index++;
      // try to send any buffered packets
      while (r_info->reassembly_buf[r_info->nxt_pkt_index%REASSEMBLY_SIZE]!=NULL) {
         DEBUG_UDP(" %d ",r_info->nxt_pkt_index);
         send(fd,
            r_info->reassembly_buf[r_info->nxt_pkt_index%REASSEMBLY_SIZE]+2,
            r_info->reassembly_buf_len[r_info->nxt_pkt_index%REASSEMBLY_SIZE]-2,0);
         char* ptr = r_info->reassembly_buf[r_info->nxt_pkt_index%REASSEMBLY_SIZE];
         r_info->reassembly_buf[r_info->nxt_pkt_index%REASSEMBLY_SIZE]=NULL;
         free(ptr); // we cache the ptr in case a recv() timeout happens between the free() and setting to NULL
         r_info->nxt_pkt_index++;
       }
       DEBUG_UDP("\n");
    } else {  
       if (get_pkt_index(buf) > r_info->nxt_pkt_index+REASSEMBLY_SIZE) {
          // have run out of buffer space, drop packet and reset connection
          //close_session(fd,send_fd,&si_other,slen,NULL,0);
       } else if (get_pkt_index(buf) > r_info->nxt_pkt_index) {
          // out of order packet, buffer it for now
          if (r_info->reassembly_buf[get_pkt_index(buf)%REASSEMBLY_SIZE]!=NULL) {
             WARN("Duplicate packet, index=%d, nxt=%d\n",get_pkt_index(buf),r_info->nxt_pkt_index);
          } else {
             r_info->reassembly_buf[get_pkt_index(buf)%REASSEMBLY_SIZE] = buf;
             r_info->reassembly_buf_len[get_pkt_index(buf)%REASSEMBLY_SIZE] = len;
             return; // don't free(buf)
          }
       } else {
          WARN("Late packet, index=%d, nxt=%d\n",get_pkt_index(buf),r_info->nxt_pkt_index);
       }
   }
   free(buf);
}

void* accept_udp(void* param) {
  // listen on UDP port for connections
  int port = *(int*)param;

  int server_fd=socket(AF_INET,SOCK_DGRAM,0);
  if (server_fd==-1) {
    ERR("Failed to create server UDP socket: %s\n",strerror(errno));
    exit(-1);
  }
  int reuseaddr=1;
  setsockopt(server_fd,SOL_SOCKET,SO_REUSEADDR,&reuseaddr,sizeof(reuseaddr));

  struct sockaddr_in addr;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr))==-1) {
    ERR("Failed to bind UDP socket: %s\n",strerror(errno));
    exit(-1);
  }

  int pipefd[2]; // pipe for multiplexing UDP connections
  pipefd[0]=-1; pipefd[1]=-1;
  struct sockaddr_in si_active; // the client address of the current active connection
  // reassembly buffer for in-order delivery of UDP payloads
  reassembly_info r_info; 
  init_reassembly_state(&r_info);

  printf("Listening on UDP port %d\n",port);

  // loop indefinitely and wait for connections
  while (1) {
    #define BUFLEN 1500
    char *buf = malloc(BUFLEN);
    struct sockaddr_in si_other;
    int slen=sizeof(si_other);
    int res=recvfrom(server_fd,buf,BUFLEN,0,(struct sockaddr *) &si_other, &slen);
    char str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET,(struct inaddr *) &si_other.sin_addr, str, INET_ADDRSTRLEN);
    DEBUG_UDP("received UDP %d from %s: %d\n",res, str, ntohs(si_other.sin_port));
    if (res==-1) {
      ERR("Failed to accept UDP connection: %s\n",strerror(errno));
      free(buf);
      continue;
    }

    // GPU can run only one darknet process at a time, so we need to serialise web requests.
    pthread_t thread_id;
    pthread_mutex_lock(&active_mutex);
    Params p;
    if (active) {
       pthread_mutex_unlock(&active_mutex);
       // if packet is from the active connection we forward it to the handler thread via pipe
       if ((si_other.sin_addr.s_addr == si_active.sin_addr.s_addr) && (si_other.sin_port == si_active.sin_port)) {
          DEBUG_UDP("index=%d/%d\n",get_pkt_index(buf),r_info.nxt_pkt_index);
          process_pkt_inorder(buf,res,pipefd[0],server_fd,&r_info);
       } else {
          // end connection if GPU busy, might be nicer to send a message
          DEBUG_UDP("Resetting UDP connection (pkt_index=%d)\n",get_pkt_index(buf));
          if (get_pkt_index(buf)==0) DEBUG_UDP("%60.60s\n",buf);
          int fd=-1;
          close_session(&fd,server_fd,&si_other,slen,NULL,0);
          free(buf);
       }
    } else {
       active=1; // flag GPU as busy.
       pthread_mutex_unlock(&active_mutex);
       if (get_pkt_index(buf) > 0) {
         WARN("First packet %d of request is out of order (old connection? packet loss?)\n",get_pkt_index(buf));
         int fd=-1;
         close_session(&fd,server_fd,&si_other,slen,NULL,0);
         active=0; free(buf);
         continue;
       }
       // create pipe for multiplexing UDP connections
       if (pipefd[0]!=-1) {close(pipefd[0]); pipefd[0]=-1;}
       if (pipefd[1]!=-1) {close(pipefd[1]); pipefd[1]=-1;}
       if (socketpair(AF_UNIX,SOCK_STREAM,0,pipefd)<0) {
          ERR("Failed to create UDP pipe: %s\n", strerror(errno));
          int fd=-1;
          close_session(&fd,server_fd,&si_other,slen,NULL,0);
          active=0; free(buf);
          continue;
       }
       struct timeval timeout;
       timeout.tv_sec = 0; timeout.tv_usec = RECV_TIMEOUT; // timeout of inter-packet delay greater than this
       if (setsockopt (pipefd[1], SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)) < 0) {
          ERR("Pipe setsockopt failed\n");
          close_session(&pipefd[1],server_fd,&si_other,slen,NULL,0);
          active=0; close(pipefd[0]); pipefd[0]=-1; free(buf);
          continue;
       }

       memcpy(&si_active, &si_other, slen); // keep a note of the client address for connection
       p.session_fd = pipefd[1]; p.send_fd=server_fd; p.si_active=si_other; p.slen=slen;
       p.ptr = &r_info; // for debugging
       if( pthread_create( &thread_id , NULL , handle_connection, (void*) &p) < 0) {
         ERR("Failed to create thread\n");
         close_session(&pipefd[1],server_fd,&si_other,slen,NULL,0);
         active=0; close(pipefd[0]); pipefd[0]=-1; free(buf); 
         continue;
       }

       // finally, we pass the request on to the handler via the pipe
       send(pipefd[0],buf+2,res-2,0); // the +2 is because 1st two bytes of payload are pkt index
       free(buf);
       clear_reassembly_state(&r_info); // initialise reassembly buffer
       r_info.nxt_pkt_index=1; // index of next in-order packet
       DEBUG_UDP("index=%d\n",0);
    }
  }

  close(server_fd);
}

int main(int argc, char **argv) {

  char *model_file = DEFAULT_CONFIG_MODEL;
  char *weights_file = DEFAULT_MODEL_WEIGHTS;
  char *names_file = DEFAULT_MODEL_NAMES;
  int w = DEFAULT_DIM, h = DEFAULT_DIM;
  int port = DEFAULT_PORT;
  char c;
  while ((c = (char)getopt(argc, argv,"p:m:w:n:v::hd:sd:")) != EOF) {
    switch(c) {
      case 'd':
        // set input size of network
        w = atoi(optarg); h=w;
        if (w%32 || h%32) {
           ERR("network width %d and height %d must be a multiple of 32\n",w,h);
           exit(-1);
        } 
        break;
      case 'p':
        port = atoi(optarg);
        if(port < 0 || port > 65535) {
          ERR("Invalid port %d\n", port);
          exit(-1);
        }
        break;
      case 'm':
        model_file = optarg;
        break;
      case 'w':
        weights_file = optarg;
        break;
      case 'n':
        names_file = optarg;
        break;
      case 's':
        save_to_file = 1;
        break;
      case 'v':
        if (optarg) {
           verbose = atoi(optarg);
        } else {
          verbose = 1;
        }
        break;
      case 'h':
        usage(basename(argv[0]));
        exit(0);
      default:
        exit(-1);

    }
  }

  //gpu_index=-1; // disable use of gpu
#ifdef GPU
  cuda_set_device(gpu_index);
#endif

  net = init(model_file, weights_file, w, h);
  names = get_labels(names_file);

  // create thread to listen for TCP http connections
  pthread_t tcp_thread;
  pthread_create(&tcp_thread,NULL,accept_tcp,(void*)&port);

  // and listen for UDP connectiosn on current thread
  accept_udp(&port);

  return 0;
}

