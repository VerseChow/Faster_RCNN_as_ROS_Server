from obj_det_sys import *


NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_50000.caffemodel')}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

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
    
    # Warmup on a dummy image
    '''
    im = 128 * np.ones((224, 224, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    '''
    #obj_detsys.bbox_detection_server()
    im_names = sorted(glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg')))
    if not os.path.exists('./results'):
        os.makedirs('./results')

    for im_path in im_names:
        im_name = os.path.basename(im_path)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        im = cv2.imread(im_path)
        obj_detsys.demo(im_name)
    
    #plt.show()
    



    
