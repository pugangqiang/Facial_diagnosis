import face_recognition
from PIL import Image

def crop_pit(imgs):
            up = [0,0,0,0]
            down = [0,0,0,0]
            left = [0,0,0,0]
            right = [0,0,0,0]
            face_marks = face_recognition.face_landmarks(imgs, None, "large")
            if face_marks == []:
               raise Exception("未检测到人脸，请重试！")

            for j, dict_ in enumerate(face_marks):
                for key, value in dict_.items():
                    if key == 'top_lip':
                        up[0] = value[2][1]
                        left[0] = value[0][0]
                    if key == 'bottom_lip':
                        down[0] = value[3][1]
                        right[0] = value[0][0]
                    if key == 'left_eyebrow':
                        up[1] = value[0][1]
                        left[1] = value[0][0]
                        up[3] = value[0][1]
                        left[3] = value[0][0]
                    if key == 'nose_bridge':
                        down[1] = value[2][1]
                        right[1] = value[2][0]
                    if key == 'chin':
                        up[2] = value[1][1]
                        down[2] = value[3][1]
                        left[2] = value[3][0]
                        down[3] = value[3][1]
                    if key == 'top_lip':
                        right[2] = value[0][0]
                    if key == 'nose_tip':
                        right[3] = value[0][0]

            images_lip = Image.fromarray(imgs[up[0]:down[0], left[0]:right[0]])
            images_lift_eye = Image.fromarray(imgs[up[1]:down[1], left[1]:right[1]])
            images_lift_cheek = Image.fromarray(imgs[up[2]:down[2], left[2]:right[2]])
            images_lift_eyecheek = Image.fromarray(imgs[up[3]:down[3], left[3]:right[3]])

            return images_lip,images_lift_eye,images_lift_cheek,images_lift_eyecheek,