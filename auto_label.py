import cv2
from PIL import Image
import sys
import torch
import os 


sys.path.append("grounding_dino/")

from grounding_dino.groundingdino.util.inference import load_model, predict, annotate
import grounding_dino.groundingdino.datasets.transforms as T


TEXT_PROMPT = "hand"

image_path = "tao/dataset/test_images/"+TEXT_PROMPT+"/"
label_path = "tao/dataset/labels/"+TEXT_PROMPT+"/"
bbox_image_path = "tao/dataset/bbox_images/"+TEXT_PROMPT+"/"

# 폴더가 없으면 생성 
if not os.path.exists(image_path):
    os.makedirs(image_path)
if not os.path.exists(label_path):
    os.makedirs(label_path)
if not os.path.exists(bbox_image_path):
    os.makedirs(bbox_image_path)

# model parameter 
cap = cv2.VideoCapture(0)

model = load_model("grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", "grounding_dino/weights/groundingdino_swint_ogc.pth")
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

find_hand = False
num = 0
label_str_temp = ""
while True:
    # read frame
    ret, frame = cap.read()
    if ret == False:
        print("Error reading frame")
        break
    
    # cv2 to PIL
    image_source = Image.fromarray(frame).convert("RGB")

    # PIL to tensor
    image_transformed, _ = transform(image_source, None)

    # predict
    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    h, w, _ = frame.shape
    boxes = boxes * torch.Tensor([w, h, w, h])

    find_obj = False
    label_str_temp = ""
    bbox_frame = frame.copy()
    for box in boxes:
        x, y, w, h = box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        find_obj = True
        
        label_str_temp += TEXT_PROMPT + " 0.00 0 0.00 " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " "+ "0.00 0.00 0.00 0.00 0.00 0.00 0.00 \n"
        cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if find_obj:
        print(TEXT_PROMPT+"found")
        label_str = str(num) + ".txt"
        label_file = open(label_path + label_str, "w")
        label_file.write(label_str_temp)
        label_file.close()
        cv2.imwrite(image_path + str(num) + ".jpg", frame)
        cv2.imwrite(bbox_image_path + str(num) + ".jpg", bbox_frame)
        num += 1
    else:
        print(TEXT_PROMPT+"not found")

    cv2.imshow('frame', bbox_frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # annotated_frame = annotate(image_source=frame, boxes=boxes, logits=logits, phrases=phrases)
    # out_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow('frame', out_frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break