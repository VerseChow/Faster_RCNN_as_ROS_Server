from obj_det_sys import *

progress_coco_class = ('__background__', 'table', 'tide', 'downy', 'clorox', 'coke', 'cup')
progress_pascal_class = ('__background__', 'table', 'tide', 'downy', 'clorox', 'coke', 'cup')
# coco_class = ('__background__', 'person', 'bicycle', 'car','motorcycle','airplane','bus','train', 'truck', 'boat','traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench','bird','cat','dog','horse','sheep','cow', 'elephant','bear','zebra','giraffe','hat','umbrella', 'handbag','tie','suitcase', 'frisbee','skis','snowboard','sports ball','kite', 'baseball bat','baseball glove','skateboard','surfboard','tennis racket', 'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich', 'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant', 'bed','dining table','window','tv','laptop','mouse','remote','keyboard','cell phone','microwave', 'oven', 'sink','refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','tooth brush')
coco_class = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
          'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
          'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
          'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush')
complete_class =  ('__background__', 'chair', 'table', 'lobby_chair', 'lobby_table_small', 'lobby_table_large',
    'tide', 'spray_bottle_a', 'waterpot', 'sugar', 'red_bowl', 'clorox', 'shampoo',
    'downy', 'salt', 'toy', 'detergent', 'scotch_brite', 'blue_cup', 'ranch')
iros_coco_class = ('__background__', 'apple', 'bowl', 'cereal', 'coke', 'cup', 'milk', 'pringle')
progressiros_class = ('__background__',
                       'apple', 'bowl', 'cereal', 'coke', 'cup', 'milk', 'pringle', 'table', 'shampoo',
                       'alumn_cup', 'dispenser', 'loofah', 'rack')

NETS = {'progress_coco': ('progress_coco', 'coco_vgg16.5objects1table', 0, progress_coco_class),
        'progress_pascal': ('progress_pascal', 'vgg16_faster_rcnn.caffemodel.5objects_1table', 0, progress_pascal_class),
	'coco': ('coco', 'coco_vgg16_faster_rcnn_final.caffemodel', 1, coco_class),
	'iros_coco': ('iros_coco', 'vgg16_faster_rcnn_iter_10000.iros_2.7objects', 0, iros_coco_class),
    'progressiros': ('progressiros', 'vgg16_faster_rcnn_iter_63020.iros.13obj', 0, progressiros_class)}
        # shuffle_2/vgg16_faster_rcnn_iter_3000.caffemodel
        # vgg16_faster_rcnn_iter_63020.iros.13obj
        # vgg16_faster_rcnn_iter_18000.iros.13obj.confused & 2000
        # vgg16_faster_rcnn_iter_43020.iros.13obj
        # key: (name of folder, model name, pre-trained or not, tuples of class)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='coco')

    args = parser.parse_args()

    return args

def object_detector(args):
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0], 'VGG16',
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    cfg.PRETRAINED = NETS[args.demo_net][2]
    cfg.CLASSES = NETS[args.demo_net][3]

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
        print 'cpu'
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    obj_detsys = obj_detection_system(net, gpu_flag = (not args.cpu_mode), gpu_device = args.gpu_id)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    return obj_detsys


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    obj_detsys = object_detector(args)

    # Warmup on a dummy image
    '''
    im = 128 * np.ones((224, 224, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    '''
    obj_detsys.bbox_detection_server()
    '''
    im_names = sorted(glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg')))
    if not os.path.exists('./results'):
        os.makedirs('./results')

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

    for im_path in im_names:
        im_name = os.path.basename(im_path)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        im = cv2.imread(im_path)


        img = obj_detsys.demo(im_name)
        print img
        cv2.imshow('image',img)
        cv2.waitKey(25)
    '''

    #plt.show()





