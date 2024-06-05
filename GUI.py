import tkinter as tk
import subprocess
import tkinter.ttk as ttk


def open_classification_script():
    subprocess.Popen(["python", "classification.py"])


def open_boundaries():
    subprocess.Popen(["python", "boundaries.py"])


def open_vanishing():
    subprocess.Popen(["python", "vanishing.py"])


def open_layering():
    subprocess.Popen(["python", "layering.py"])


root = tk.Tk()
root.title("")
root.resizable(False, False)
root.wm_attributes('-toolwindow', 'True')

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", padding=10, font=('Helvetica', 12))
root.configure(background="#f0f0f0")

main_frame = ttk.Frame(root, padding=20)
main_frame.pack(fill=tk.BOTH, expand=True)

title_label = ttk.Label(main_frame, text="Main GUI", font=('Helvetica', 16, 'bold'))
title_label.pack(pady=10)

buttons = [
    ("Open Classifier", open_classification_script),
    ("Open Boundaries", open_boundaries),
    ("Open Vanishing", open_vanishing),
    ("Open Layering", open_layering),
]

for text, command in buttons:
    button = ttk.Button(main_frame, text=text, command=command)
    button.pack(fill=tk.X, pady=5)

root.mainloop()
