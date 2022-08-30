import torch
from torchvision import transforms
from model_v3 import mobilenet_v3_small as creat_model
from crop_pit import crop_pit
from PIL import Image
import cv2
import sys

def predict(path1,path2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])
    # imgs_data = './example/1.jpg'
    # imgs_data = './example/2.jpg'
    # imgs_data = 'D:/pit/4.jpg'

    img = cv2.imread(path1)
    if len(img) == 0:
        raise Exception("图像输入错误")
    #
    # # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # # faces = face_cascade.detectMultiScale(img, 1.1, 5)
    # # if len(faces):
    # #         for (x, y, w, h) in faces:
    # #             if w >= 224 and h >= 224:
    # #                 X = int(x)
    # #                 W = min(int(x + w), img.shape[1])
    # #                 Y = int(y)
    # #                 H = min(int(y + h), img.shape[0])
    # #                 img = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
    # # else:
    # #     raise Exception("未检测到人脸，请重试！")
    # # create model
    try:
        model_complexion = creat_model(num_classes=5).to(device)
        model_gloss = creat_model(num_classes=2).to(device)
        model_lip = creat_model(num_classes=4).to(device)
        model_eye = creat_model(num_classes=2).to(device)
        model_cheek = creat_model(num_classes=2).to(device)
        model_eyecheek = creat_model(num_classes=2).to(device)
        # load model weights
        weights_complexion_path = "./complexion/mobilenet_v3_small_checkpoints/153_0.80686_mobilenet_v3_small.pth"
        weights_lip_path = "./lip/mobilenet_v3_small_checkpoints/284_0.76173_mobilenet_v3_small.pth"
        weights_eye_path = "./local/eye_socket_mobilenet_v3_small_checkpoints/201_0.78678_mobilenet_v3_small.pth"
        weights_cheek_path = "./local/cheek_mobilenet_v3_small_checkpoints/301_0.85455_mobilenet_v3_small.pth"
        weights_eyecheek_path = "./local/eye_socket_cheek_mobilenet_v3_small_scheckpoints/225_0.81042_mobilenet_v3_small.pth"
        weights_gloss_path = "./gloss/mobilenet_v3_small_checkpoints/99_0.82394_mobilenet_v3_small.pth"
        model_complexion.load_state_dict(torch.load(path2+weights_complexion_path, map_location=device))
        model_complexion.eval()
        model_gloss.load_state_dict(torch.load(path2+weights_gloss_path, map_location=device))
        model_gloss.eval()
        model_lip.load_state_dict(torch.load(path2+weights_lip_path, map_location=device))
        model_lip.eval()
        model_eye.load_state_dict(torch.load(path2+weights_eye_path, map_location=device))
        model_eye.eval()
        model_cheek.load_state_dict(torch.load(path2+weights_cheek_path, map_location=device))
        model_cheek.eval()
        model_eyecheek.load_state_dict(torch.load(path2+weights_eyecheek_path, map_location=device))
        model_eyecheek.eval()
    except:
        return '模型调用错误'
    # #
    try:
        with torch.no_grad():
            images_lip, images_lift_eye, images_lift_cheek, images_lift_eyecheek = crop_pit(img)
            img = Image.fromarray(img)
            img_complexion = data_transform(img)
            img_complexion = torch.unsqueeze(img_complexion, dim=0)
            output_complexion = torch.squeeze(model_complexion(img_complexion.to(device))).cpu()
            predict_complexion = torch.softmax(output_complexion, dim=0)
            predict_cla_complexion = torch.argmax(predict_complexion).numpy()

            img_gloss = data_transform(img)
            img_gloss = torch.unsqueeze(img_gloss, dim=0)
            output_gloss = torch.squeeze(model_gloss(img_gloss.to(device))).cpu()
            predict_gloss = torch.softmax(output_gloss, dim=0)
            predict_cla_gloss = torch.argmax(predict_gloss).numpy()

            img_lip = data_transform(images_lip)
            img_lip = torch.unsqueeze(img_lip, dim=0)
            output_lip = torch.squeeze(model_lip(img_lip.to(device))).cpu()
            predict_lip = torch.softmax(output_lip, dim=0)
            predict_cla_lip = torch.argmax(predict_lip).numpy()

            img_eye = data_transform(images_lift_eye)
            img_eye = torch.unsqueeze(img_eye, dim=0)
            output_eye = torch.squeeze(model_eye(img_eye.to(device))).cpu()
            predict_eye = torch.softmax(output_eye, dim=0)
            predict_cla_eye = torch.argmax(predict_eye).numpy()

            img_cheek = data_transform(images_lift_cheek)
            img_cheek = torch.unsqueeze(img_cheek, dim=0)
            output_cheek = torch.squeeze(model_cheek(img_cheek.to(device))).cpu()
            predict_cheek = torch.softmax(output_cheek, dim=0)
            predict_cla_cheek = torch.argmax(predict_cheek).numpy()

            img_eyecheek = data_transform(images_lift_eyecheek)
            img_eyecheek = torch.unsqueeze(img_eyecheek, dim=0)
            output_eyecheek = torch.squeeze(model_eyecheek(img_eyecheek.to(device))).cpu()
            predict_eyecheek = torch.softmax(output_eyecheek, dim=0)
            predict_cla_eyecheek = torch.argmax(predict_eyecheek).numpy()

            '''
                "0": "Red",
                "1": "Yellow",
                "2": "black",
                "3": "red_yellow",
                "4": "white"
            '''
            p = [(0,) for i in range(17)]
            color = [('红',), ('黄',), ('黑',), ('红黄隐隐',), ('白',), ('有光泽',), ('光泽淡',),
                     ('青紫',), ('暗红',), ('口红',), ('红',), ('黑红',), ('非黑红',), ('黑',),
                     ('非黑',), ('红',), ('非红',)]

            #color = [('红',p[0]),('黄',p[1]),('黑',p[2]),('红黄隐隐',p[3]),('白',p[4]),('有光泽',p[5]),('光泽淡',p[6]),('青紫',p[7]),('暗红',p[8]),('口红',p[9]),('红',p[10]),('黑红',p[11]),('非黑红',p[12]),('黑',p[13]),('非黑',p[14]),('红',p[15]),('非红',p[16])]
            if predict_cla_complexion == 0:
                p[0] = (round(predict_complexion[predict_cla_complexion].item(),4),)
                color[0] = color[0] + p[0]
                #return ('面色:红', predict_complexion[predict_cla_complexion].numpy())
            elif predict_cla_complexion == 1:

                p[1] = (round(predict_complexion[predict_cla_complexion].item(),4),)
                color[0] = color[1] + p[1]
                #return('面色:黄', predict_complexion[predict_cla_complexion].numpy())
            elif predict_cla_complexion == 2:

                p[2] = (round(predict_complexion[predict_cla_complexion].item(),4),)
                color[0] = color[2] + p[2]
                #return('面色:黑', predict_complexion[predict_cla_complexion].numpy())
            elif predict_cla_complexion == 3:

                p[3] = (round(predict_complexion[predict_cla_complexion].item(),4),)
                color[0] = color[3] + p[3]
                #return('面色:红黄隐隐', predict_complexion[predict_cla_complexion].numpy())
            elif predict_cla_complexion == 4:
                p[4] = (round(predict_complexion[predict_cla_complexion].item(),4),)
                color[0] = color[4] + p[4]


                #return('面色:白', predict_complexion[predict_cla_complexion].numpy())
            '''
                 "0": "gloss",
                 "1": "little_gloss"
            '''
            if predict_cla_gloss == 0:

                p[5] = (round(predict_gloss[predict_cla_gloss].item(),4),)
                color[5] = color[5] + p[5]
                #print('面部：有光泽', predict_gloss[predict_cla_gloss].numpy())
            elif predict_cla_gloss == 1:

                p[6] = (round(predict_gloss[predict_cla_gloss].item(),4),)
                color[5] = color[6] + p[6]
                #print('面部：光泽淡', predict_gloss[predict_cla_gloss].numpy())
            '''
                "0": "cyan",
                "1": "dark_red",
                "2": "lipstick",
                "3": "red"
            '''
            if predict_cla_lip == 0:

                p[7] = (round(predict_lip[predict_cla_lip].item(),4),)
                color[7] = color[7] + p[7]
               # print('唇色：青紫', predict_lip[predict_cla_lip].numpy())
            elif predict_cla_lip == 1:

                p[8] = (round(predict_lip[predict_cla_lip].item(),4),)
                color[7] = color[8] + p[8]
                #print('唇色：暗红', predict_lip[predict_cla_lip].numpy())
            elif predict_cla_lip == 2:

                p[9] = (round(predict_lip[predict_cla_lip].item(),4),)
                color[7] = color[9] + p[9]
                #print('唇色：口红', predict_lip[predict_cla_lip].numpy())
            elif predict_cla_lip == 3:

                p[10] = (round(predict_lip[predict_cla_lip].item(),4),)
                color[7] = color[10] + p[10]
                #print('唇色：红', predict_lip[predict_cla_lip].numpy())
            '''
                "0": "black",
                "1": "other"
            '''
            if predict_cla_eye == 0:

                p[13] = (round(predict_eye[predict_cla_eye].item(),4),)
                color[13] = color[13] + p[13]
                #print('眼眶色：黑', predict_eye[predict_cla_eye].numpy())
            elif predict_cla_eye == 1:

                p[14] = (round(predict_eye[predict_cla_eye].item(),4),)
                color[13] = color[14] + p[14]
                #print('眼眶色：非黑', predict_eye[predict_cla_eye].numpy())
            '''
                "0": "other",
                "1": "red"
            '''
            if predict_cla_cheek == 0:

                p[15] = (round(predict_cheek[predict_cla_cheek].item(),4),)
                color[15] = color[15] + p[15]
                #print('两颧色：非红', predict_cheek[predict_cla_cheek].numpy())
            elif predict_cla_cheek == 1:

                p[16] = (round(predict_cheek[predict_cla_cheek].item(),4),)
                color[15] = color[16] + p[16]
                #print('两颧色：红', predict_cheek[predict_cla_cheek].numpy())
            '''
                 "0": "black-red",
                 "1": "other"
            '''
            if predict_cla_eyecheek == 0:

                p[11] = (round(predict_eyecheek[predict_cla_eyecheek].item(),4),)
                color[11] = color[11] + p[11]
                #print('眼眶-两颧：黑红', predict_eyecheek[predict_cla_eyecheek].numpy())
            elif predict_cla_eyecheek == 1:

                p[12] = (round(predict_eyecheek[predict_cla_eyecheek].item(),4),)
                color[11] = color[12] + p[12]
                #print('眼眶-两颧：非黑红', predict_eyecheek[predict_cla_eyecheek].numpy())
    except:
        return '人脸关键点获取错误'

    # color = [i for i in range(16)]
    result = {'面色': color[0], '面部光泽': color[5], '唇色': color[7], '眼眶色': color[13], '两颧色': color[15], '眼眶-两颧色': color[11]}
    #print(result)
    return result

# result= predict(str(sys.argv[1]),str(sys.argv[2]))
result = predict('./example/3.jpg','')
print(result)






