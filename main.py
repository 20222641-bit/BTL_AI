# main.py
from tkinter import Tk
from Controller.ctl import Controller
from GUI.mainview import Mainview

if __name__ == "__main__":
    root = Tk()
    ctl = Controller(model_path=r"runs\detect\train2\weights\best.pt")
    app = Mainview(root, ctl)
    root.mainloop()
