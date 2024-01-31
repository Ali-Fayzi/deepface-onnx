import cv2 
from utils import DeepFace 
from matplotlib import pyplot as plt 

if __name__ == "__main__":
    models = [
        "arcface",   
        "deepid",    
        "facenet128",
        "facenet512",
        "openface", 
        "vggface",   
    ]
    deepface = DeepFace(model_name=models[2])
    input1 = cv2.imread("./test_images/bradpitt_1.png")
    input2 = cv2.imread("./test_images/bradpitt_2.png")
    result, threshold, distance = deepface.verify(input1,input2,"cosine")
    plt.title(f"Verification Result:{result}\nThreshold:{threshold:.2}\nDistance:{distance:.2}")
    plt.subplot(1,2,1)
    plt.imshow(input1[...,::-1])
    plt.subplot(1,2,2)
    plt.imshow(input2[...,::-1])
    plt.show()
    