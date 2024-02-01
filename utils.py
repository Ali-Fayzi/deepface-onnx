import os 
import cv2 
import gdown
import onnxruntime
import numpy as np 
from config import find_verification_model_target_size, find_threshold, find_analyis_model_target_size
from scipy.spatial.distance import cosine , euclidean
def check_file(model_name):
    urls = {
        "arcface"   :   "https://drive.google.com/uc?id=1E-tsjH_6nBRez2rUw1GNVGFA1UZHljnq",
        "deepid"    :   "https://drive.google.com/uc?id=1Jm1r89TaS5bU0jnyTrmdnkxFNVWWouag",
        "facenet128":   "https://drive.google.com/uc?id=1uAI_fy0JBT4QlAhCBMuRkd6wcpqvJyuV",
        "facenet512":   "https://drive.google.com/uc?id=1x5Tct5hOCIOve229LNRrk2axWQFwojwU",
        "openface"  :   "https://drive.google.com/uc?id=1Xe5WfTH4F0lxsz11izJloeyAZ3-puSpK",
        "vggface"   :   "https://drive.google.com/uc?id=1-0hl18ixXMUp-lXnjeI34IpLHtO9sA0b",
    
        "age"       :   "https://drive.google.com/uc?id=1z4dV2qePKsfYYAB8Zq5z368WLIJ-0vSQ",
        "emotion"   :   "https://drive.google.com/uc?id=1ZhvPBZ1mM9VUKJ9iWIWtcErgd3QaO2Ew",
        "gender"    :   "https://drive.google.com/uc?id=1--mRjn1yay65IzoQ0PNfzJiU7hBfdhAi",
        "race"      :   "https://drive.google.com/uc?id=1-1PmcjZ_e6FVzqtQgX2YZ4DJXdR9ItcM",
    }
    if model_name.lower() in urls:
        os.makedirs("./models",exist_ok=True)
        file_path = os.path.join("./models",model_name+".onnx")
        if not os.path.exists(file_path):
            gdown.download(urls[model_name.lower()], file_path, quiet=False)
            print("DONWLOAD DONE!")
        return file_path 
    else:
        raise ValueError(f"unimplemented model name - {model_name}")
class DeepFace:
    def __init__(self,model_name):
        self.model_name = model_name
        self.model,self.input_name,self.output_name = self.load_model()

    def load_model(self):
        file_path = check_file(model_name=self.model_name)
        try:
            session = onnxruntime.InferenceSession(file_path)
            session.get_inputs()[0].shape
            session.get_inputs()[0].type
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            return session, input_name, output_name
        except :
            return None  
    
    def preprocess(self,input):
        target_size = find_verification_model_target_size(self.model_name)
        if len(input.shape) == 4:
            input = input[0]
        if len(input.shape) == 3:
            input = cv2.resize(input, target_size)
            input = np.expand_dims(input, axis=0)
            if input.max() > 1:
                input = (input.astype(np.float32) / 255.0).astype(np.float32)
        return input 
    def predict(self,input):
        input  = self.preprocess(input)
        result = self.model.run([self.output_name], {self.input_name: input}) 
        return result 
    
    def verify(self,input1,input2,distance_metric):
        source_representation = self.predict(input1)[0][0]
        target_representation = self.predict(input2)[0][0]

        threshold = find_threshold(self.model_name,distance_metric)
        if distance_metric == "cosine":
            distance = cosine(source_representation, target_representation)
        elif distance_metric == "euclidean":
            distance = euclidean(source_representation, target_representation)
        else:
            raise ValueError(f"invalid distance metric passes - {distance_metric}")
        if distance < threshold:
            return True, threshold, distance 
        return False, threshold, distance 




class FaceAnalysis:

    def __init__(self,model_name):
        self.model_name = model_name
        self.model,self.input_name,self.output_name = self.load_model()
    
    def load_model(self):
        file_path = check_file(model_name=self.model_name)
        try:
            session = onnxruntime.InferenceSession(file_path)
            session.get_inputs()[0].shape
            session.get_inputs()[0].type
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            return session, input_name, output_name
        except :
            return None 


    def preprocess(self,input):
        target_size = find_analyis_model_target_size(self.model_name)
        if len(input.shape) == 4:
            input = input[0]
        if len(input.shape) == 3:
            if self.model_name == "emotion":
                input = cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
            input = cv2.resize(input, target_size)
            input = np.expand_dims(input, axis=0)
            if self.model_name == "emotion":
                input = np.expand_dims(input, axis=3)
            if input.max() > 1:
                input = (input.astype(np.float32) / 255.0).astype(np.float32)
            
        return input 
    
    def postprocess(self,output):
        if self.model_name == "emotion":
            labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            return labels[np.argmax(output)]
        elif self.model_name == "age":
            output_indexes = np.array(list(range(0, 101)))
            apparent_age = np.sum(output * output_indexes)
            return int(apparent_age)
        elif self.model_name == "gender":
            labels = ["Woman", "Man"]
            return labels[np.argmax(output)] 
        elif self.model_name == "race":
            labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]
            return labels[np.argmax(output)] 
        else:
            raise ValueError(f"unimplemented model name - {self.model_name}")

    def predict(self,input):
        input  = self.preprocess(input)
        result = self.model.run([self.output_name], {self.input_name: input}) 
        result = self.postprocess(result)
        return result 


    
    