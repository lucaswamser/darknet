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





void train_classifier(network * net)
{

    int imgs = net->batch * net->subdivisions * 1;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.threads = 64;
    args.hierarchy = net.hierarchy;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;

        loss = train_network(net, train);

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net.seen);
        free_data(train);
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}
