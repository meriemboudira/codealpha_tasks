"""Draw characters, words or sentences and get live predictions.

    python app.py --model models/emnist_balanced.pt
"""
import argparse
import tkinter as tk

from PIL import Image, ImageDraw

from src.predict import load_model, predict
from src.word import recognise_text

W, H, BRUSH = 760, 240, 7


class App:
    def __init__(self, root, model, classes):
        self.model, self.classes = model, classes
        root.title("Handwritten Character Recognition")

        self.canvas = tk.Canvas(root, width=W, height=H, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.start)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.recognise())

        self.word_mode = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Word / sentence mode", variable=self.word_mode,
                       command=self.recognise).grid(row=1, column=0)
        tk.Button(root, text="Clear", command=self.clear, width=12).grid(row=1, column=1, pady=5)
        tk.Button(root, text="Recognise", command=self.recognise, width=12).grid(row=1, column=2)

        self.result = tk.Label(root, text="Write something", font=("Helvetica", 22), justify="left")
        self.result.grid(row=2, column=0, columnspan=3, pady=10)
        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (W, H), 0)
        self.pen = ImageDraw.Draw(self.img)
        self.last = None
        self.result.config(text="Write something")

    def start(self, e):
        self.last = (e.x, e.y)

    def draw(self, e):
        x0, y0 = self.last or (e.x, e.y)
        self.canvas.create_line(x0, y0, e.x, e.y, fill="white", width=BRUSH * 2,
                                capstyle=tk.ROUND, smooth=True)
        self.pen.line([x0, y0, e.x, e.y], fill=255, width=BRUSH * 2)
        self.pen.ellipse([e.x - BRUSH, e.y - BRUSH, e.x + BRUSH, e.y + BRUSH], fill=255)
        self.last = (e.x, e.y)

    def recognise(self):
        try:
            if self.word_mode.get():
                text, _ = recognise_text(self.model, self.classes, self.img)
                self.result.config(text=f"Text:  {text}")
            else:
                top = predict(self.model, self.classes, self.img, k=3)
                lines = [f"{c}   {p * 100:5.1f}%" for c, p in top]
                lines[0] = "Prediction:  " + lines[0]
                self.result.config(text="\n".join(lines))
        except ValueError:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mnist.pt")
    a = ap.parse_args()
    model, classes = load_model(a.model)
    root = tk.Tk()
    App(root, model, classes)
    root.mainloop()
