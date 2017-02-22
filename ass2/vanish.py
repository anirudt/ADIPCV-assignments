import Tkinter
from PIL import Image, ImageTk
from sys import argv
from Tkinter import NW, FLAT, TOP
import cPickle as pickle
import pdb

X, X_dash = [], []
NUM_CLICK_EVENTS = 0
WIDTH, HEIGHT = 0, 0


window = Tkinter.Tk(className="bla")

image1 = Image.open(argv[1] if len(argv) >=2 else "bla2.png")

image1 = image1.resize((image1.size[0] // 2, image1.size[1]//2), Image.ANTIALIAS)

WIDTH, HEIGHT = image1.size

print image1.size
canvas = Tkinter.Canvas(window, width=image1.size[0], height=image1.size[1])
canvas.pack()

help = ""
canvas.create_text(50, 10, text="Hello World")

image_tk1 = ImageTk.PhotoImage(image1)
canvas.create_image(image1.size[0]//2, image1.size[1]//2, image=image_tk1)

def buttonCallBack():
    print "Button pressed"
    with open('data_vl', 'wb') as g:
        pickle.dump(X, g)
    window.destroy()
    pass


def callback(event):
    global WIDTH, HEIGHT, NUM_CLICK_EVENTS
    X.append([event.x*2, event.y*2, 1])
    print event.x*2, event.y*2
    if NUM_CLICK_EVENTS == 8:
        return

    NUM_CLICK_EVENTS += 1

b1 = Tkinter.Button(window, width=100, height=100, text="Compute!", command=buttonCallBack)
b1.configure(width=10, height=10, activebackground="#33B5E5", relief=FLAT)
b1.pack(side=TOP)

canvas.bind("<Button-1>", callback)
Tkinter.mainloop()
