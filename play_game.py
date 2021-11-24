import time
from PIL import Image
from mss import mss
import keyboard
import numpy as np
from keras.models import model_from_json


frame = {"top":145, "left":724, "width":124, "height":166} # Borders of the screenshot
ss_manager = mss()  # We are using mss() for taking a screenshot
is_exit = False     # A variable for stopping the program
my_timer = 0        # A variable which store the time passed

width = 80     # Width of all images
height = 75    # Height of all images


# A function for go down in the game
def down():
    keyboard.release("right")
    keyboard.release(keyboard.KEY_UP)
    keyboard.press(keyboard.KEY_DOWN)


# A function for go up in the game
def up():
    keyboard.release("right")
    keyboard.release(keyboard.KEY_DOWN)
    keyboard.press(keyboard.KEY_UP)


# A function for go right in the game
def right():
    keyboard.release(keyboard.KEY_UP)
    keyboard.release(keyboard.KEY_DOWN)
    keyboard.press("right")


# A function for stopping the program
def exit():
    global is_exit
    is_exit = True


# MAIN PROGRAM
if __name__ == '__main__':
    keyboard.add_hotkey("esc", exit)    # If user clik the 'esc', the program will stop

    # Load the model and weights
    model = model_from_json(open("model.json","r").read())
    model.load_weights("weights.h5")

    while True:
        if is_exit == True:
            keyboard.release("right")
            keyboard.release(keyboard.KEY_DOWN)
            keyboard.release(keyboard.KEY_UP)
            break

        screenshot = ss_manager.grab(frame)
        image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        grey_image = image.convert("L")                       # Convert RGB image to grey_scale image
        a_img = np.array(grey_image.resize((width, height)))  # Resize the grey image and convert it to numpy array
        img = a_img / 255                                     # Normalize the image array
            
        X = np.array([img])                                 # Convert list X to numpy array
        X = X.reshape(X.shape[0], width, height, 1)         # Reshape the X
        prediction = model.predict(X)                       # Get prediction by using the model
        
        result = np.argmax(prediction)  # Convert one-hot prediction to the number
        print("--------------------------")

        if result == 1:     # go right
            right()
            print("right")
        elif result == 0:   # go down
            down()
            print("down")
        elif result == 2:   # go up
            up()
            print("up")
        
        time.sleep(0.000000000001)
