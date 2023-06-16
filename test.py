import torchvision 
import torch
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as transforms
num_classes = 3
# the probability of the box predicted by the network that belongs to each class
threshold = 0.5
if __name__ == "__main__":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
    model_weights = torch.load("model_200_02.pth", map_location=torch.device("cpu"))
    # get the number of in_features.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # print(model_weights['model_state_dict'])
    model.load_state_dict(model_weights['model_state_dict'])
    # read image: 
    model.eval()
    image = cv2.imread("data/oct_images/00000.oct.png")
    # Define the transformation before passing to the model for forward inference. 
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    image = transform(image)
    # add batch dimension
    image = torch.unsqueeze(image, 0)

    with torch.no_grad(): 
        output = model(image.to('cpu'))
    
    boxes = output[0]['boxes']
    scores = output[0]['scores']
    boxes = boxes[scores>threshold]
    image2 = cv2.imread("data/oct_images/00000.oct.png")
    overlay = image2.copy()
    region_image = image2.copy()
    for i, item in enumerate(boxes): 
        cv2.rectangle(overlay, (int(item[0]),0), (int(item[2]), 1024), (0,0,255), 2, lineType=cv2.LINE_AA)
        cv2.rectangle(overlay, (int(item[0]),0), (int(item[2]), 1024), (0,0,255), -1)

        cv2.rectangle(image2, (int(item[0]),int(item[1])), (int(item[2]), int(item[3])), (0,0,255), 2, lineType=cv2.LINE_AA)

    # transparent level alpha
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, region_image, 1-alpha, 0, region_image)
    cv2.imshow("result_box", image2)
    cv2.imshow("image_region", region_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()