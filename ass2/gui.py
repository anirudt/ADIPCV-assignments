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
image2 = Image.open(argv[2] if len(argv) >=3 else "bla2.png")

image1 = image1.resize((image1.size[0] // 2, image1.size[1]//2), Image.ANTIALIAS)
image2 = image2.resize((image2.size[0] // 2, image2.size[1]//2), Image.ANTIALIAS)

WIDTH, HEIGHT = image1.size

print image1.size, image2.size
canvas = Tkinter.Canvas(window, width=image1.size[0]+image2.size[0], height=image1.size[1])
canvas.pack()
image_tk1 = ImageTk.PhotoImage(image1)
image_tk2 = ImageTk.PhotoImage(image2)
canvas.create_image(image1.size[0]//2, image1.size[1]//2, image=image_tk1)
canvas.create_image(image2.size[0]+image2.size[0]//2, image2.size[1]//2, image=image_tk2)

def buttonCallBack():
    print "Button pressed"
    with open('data', 'wb') as g:
        pickle.dump((X, X_dash), g)
    window.destroy()
    pass


def callback(event):
    global WIDTH, HEIGHT, NUM_CLICK_EVENTS
    print WIDTH, HEIGHT
    print event.x, event.y
    
    if NUM_CLICK_EVENTS % 2 == 0:
        print "clicked at: ", event.x, HEIGHT-event.y
        X.append([event.x, HEIGHT-event.y])
    else:
        w = event.x - WIDTH
        h = HEIGHT-event.y
        print "clicked at: ", w, h
        X_dash.append([event.x-WIDTH, HEIGHT-event.y])

    NUM_CLICK_EVENTS += 1

    

b1 = Tkinter.Button(window, width=100, height=100, text="Compute!", command=buttonCallBack)
b1.configure(width=10, height=10, activebackground="#33B5E5", relief=FLAT)
b1.pack(side=TOP)

canvas.bind("<Button-1>", callback)
Tkinter.mainloop()
