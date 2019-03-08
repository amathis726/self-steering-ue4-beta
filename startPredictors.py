import os
from fastai import *
from fastai.vision import *
import pyperclip

defaults.device = torch.device('cpu')

#Variables and function needed for image segmentation learner
codes = np.loadtxt('./data/imageseg/codes.txt', dtype=str); codes
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

# learn_imageSeg = load_learner('.', 'export_imageSeg.pkl')
learn_steerPred = load_learner('.', 'export_steerPredict.pkl')

path = Path('./captures')
fname = "capture.jpg"

steerVal = 0.0
pyperclip.copy(os.path.abspath('captures'))
printMessage = True

try:
    if os.path.isfile("./captures/"+fname):
        os.remove("./captures/"+fname)

    pyperclip.copy(os.path.abspath('captures'))

    while True:
        while True:
            #Wait until an image is captured from Unreal
            if os.path.isfile("./captures/"+fname):
                printMessage = True
                break
            elif printMessage == True:
                print('\nWaiting for capture...')
                printMessage = False

        while True:
            #Open image then delete it from hd
            try:
                img = open_image(path/fname)
        #       os.remove("./captures/"+fname)

            except OSError:
                print('Warning: Cannot open file. Continuing...')
                printMessage = True
                break
            else:
                if printMessage == True:
                    print('\n' + fname + ' found! Making predictions...\nPress Cntl-C to quit...')
                    printMessage = False
                #Get image segmentation prediction, convert to grayscale image that steer predictor was trained with
        #       predImSeg_class = learn_imageSeg.predict(img)[0]
        #       x = image2np(predImSeg_class.data*28).astype(np.uint8)
        #       test = PIL.Image.fromarray(x).convert('RGB')
        #       test = pil2tensor(test, dtype=np.float32)
        #       test.div_(255)

                #Get steer prediction
                predSteer_class = learn_steerPred.predict(img)[0]
                # print(predSteer_class.data[0])
                steerVal = float(predSteer_class.data[0])
                pyperclip.copy(steerVal)
                
except (KeyboardInterrupt, SystemExit):
    print('Exiting...')

        