import tkinter as tk
import cv2
import numpy as np
from PIL import ImageGrab, ImageTk, Image
from core.models import Model

HEIGHT = 600
WIDTH = 600


def predict_digits(img, model):
    ret, img = cv2.threshold(img, 127, 255, 0)
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imwrite('img.jpg', img)

    imgs = (img.reshape(-1).astype('float32') - 127.5) / 127.5
    res = model.predict(imgs)
    return res[0]


class App(tk.Tk):
    def __init__(self, model):
        tk.Tk.__init__(self)
        self.model = model

        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT, bg="white", cursor="cross")
        self.labels = [tk.Label(self, text=str(i) + "..", font=("Helvetica", 15)) for i in range(10)]
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W, rowspan=11)
        [self.labels[i].grid(row=i, column=2, columnspan=2) for i in range(10)]
        self.classify_btn.grid(row=10, column=2, pady=2, padx=2,)
        self.button_clear.grid(row=10, column=3, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

        self.mainloop()

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()

        pil_image = ImageGrab.grab((x0, y0, x1, y1))
        open_cv_image = np.array(pil_image)

        res = predict_digits(open_cv_image, self.model)

        [self.labels[i].configure(text=str(i) + ', ' + str(int(res[i] * 100)) + '%') for i in range(10)]

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 20
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
