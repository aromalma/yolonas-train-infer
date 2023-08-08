from config import config
from super_gradients.training import models
from opencv_draw_annotation import draw_bounding_box
import cv2
import numpy as np
import time
from tqdm import tqdm


video_path='' # give videopath here

best_model = models.get('yolo_nas_m',
                        num_classes=len(config['dataset_params']['classes']),
                        checkpoint_path=f"{config['CHECKPOINT_DIR']}/{config['experiment_name']}/ckpt_best.pth").cuda()
                        
# best_model = models.get('yolo_nas_m', 
#                    #num_classes=len(dataset_params['classes']), 
#                    pretrained_weights="coco",
#                    ).cuda()


cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# while True:
for __ in tqdm(range(0,length,1)):
    ret, im = cap.read()
    if not ret :
        break
    #t=time.time()

    dets=next(iter(best_model.predict(images=im,iou=0.7, conf=0.7))).prediction

    # print(1/(time.time()-t),end="")

    dets=np.concatenate([dets.bboxes_xyxy,dets.confidence.reshape(-1,1),dets.labels.reshape(-1,1)],axis=1)

    tracker_outputs=dets
    

    for z in tracker_outputs:
        draw_bounding_box(im,(z[0],z[1],z[2],z[3]),labels=[config['dataset_params']['classes'][int(z[-1])]],color='green')
    cv2.imshow('out',im[::2,::2])
    if cv2.waitKey(1) & 0xff ==27:
        break
cap.release()
cv2.destroyAllWindows()