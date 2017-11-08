#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "blas.h"
#include "detection.h"

int predict(network * net,image im,float thresh,char* nameslist, detection* dcts){
      int j;
	  char ** names = get_labels(nameslist);
      image sized = letterbox_image(im, net->w, net->h);
      float hier_thresh = thresh;
      float nms=.3;
      //image sized = resize_image(im, net->w, net->h);
      //image sized2 = resize_max(im, net->w);
      //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
      //resize_network(&net, sized.w, sized.h);
      layer l = net->layers[net->n-1];

      box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
      float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
      for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
      float **masks = 0;
      if (l.coords > 4){
          masks = calloc(l.w*l.h*l.n, sizeof(float*));
          for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = calloc(l.coords-4, sizeof(float *));
      }

      float *X = sized.data;
      network_predict(*net, X);
      get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
      if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

      return  get_detections(l.w * l.h * l.n, thresh, boxes, probs, names,l.classes,dcts);
}






void train_detector_i(network * net,char **paths,int size)
{
    printf("tamanho %i\n",size);
    char *backup_directory = "backup/";
    int i;
    int ngpus = 1;
    srand(time(0));
    float avg_loss = -1;

    srand(time(0));
    int seed = rand();
    srand(time(0));

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;
   
    load_args args = get_base_args(*net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 8;

    #ifdef GPU
        cuda_set_device(0);
    #endif

    pthread_t load_thread = load_data(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(*net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(*net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            resize_network(net, dim, dim);

        }
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;

        loss = train_network(*net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        i = get_current_batch(*net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(*net), loss, avg_loss, get_current_rate(*net), sec(clock()-time), i*imgs);
        if(i%100==0){
            char buff[256];
            save_weights(*net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            save_weights(*net, buff);
        }
        free_data(train);
    }
    char buff[256];
    save_weights(*net, buff);
}
