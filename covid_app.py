import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class Load_var:

    def __init__(self,img):
        
        self.img_mat = cv2.imread(img)
        self.model = load_model("D:/CNN_Projects/Covid/2020_covid.h5")



class Predict_image(Load_var):

    def __init__(self, img , size):
        
        super().__init__(img)
        self.image = self.img_mat
        self.cnn_model = self.model
        self.mat_size = size

    def predict_output(self):

        self.image_ = cv2.resize(self.image , (self.mat_size,self.mat_size))
        self.image_ = self.image_ / 255.0
        img_arr = image.img_to_array(self.image_)
        img_res = np.expand_dims(img_arr, axis = 0)
        usr_prd = np.argmax(self.cnn_model.predict(img_res))
        return usr_prd


class Diplay_result(Load_var):

    def __init__(self, img,image_size,display_img_size):

        super().__init__(img)
        
        self.dis_img = display_img_size
        pr = Predict_image(img,image_size)
        self.user_image = self.img_mat
        self.result = pr.predict_output()
        

    def display(self):

        self.user_image = cv2.resize(self.user_image , (self.dis_img,self.dis_img))
        if self.result == 0:
            cv2.putText(self.user_image,"Covid Negative -",(80,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),6)
        elif self.result == 1:
            cv2.putText(self.user_image,"Covid Positive +",(60,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),6)

        cv2.imshow("covid",self.user_image)
        cv2.moveWindow("covid", 420,60)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":

    dis = Diplay_result("D:/CNN_Projects/Covid/normal_1.jpg" , image_size=48 , display_img_size=500)
    dis.display()
        
