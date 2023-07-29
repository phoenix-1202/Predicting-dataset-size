import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter.font import BOLD, Font, nametofont
from PIL import ImageTk, Image
from back.main.utils import get_all_files_from
from front.pretty_print_path import shorten_path


def _get_suffix_image(n):
    if n == 1:
        return ""
    return "s"
    # suffix = "ий"
    # if 10 < n % 100 < 20:
    #     suffix = "ий"
    # elif n % 10 == 1:
    #     suffix = "ие"
    # elif 1 < n % 10 < 5:
    #     suffix = "ия"_get_suffix_image
    # return suffix


def _get_suffix_file(n):
    if n == 1:
        return ""
    return "s"
    # suffix = "ов"
    # if 10 < n % 100 < 20:
    #     suffix = "ов"
    # elif n % 10 == 1:
    #     suffix = ""
    # elif 1 < n % 10 < 5:
    #     suffix = "а"
    # return suffix


class GUI:
    def __init__(self, predictor):
        self._window = tk.Tk()
        self._window.title('Predicting train dataset size')
        self._window.geometry('700x400')
        self._window.minsize(700, 400)
        self._window.maxsize(700, 400)
        
        self._inner_frame = tk.Frame(self._window, borderwidth=25)
        self._inner_frame.pack(fill="both", expand=True)
        self._inner_frame.columnconfigure(0, weight=6)
        self._inner_frame.columnconfigure(1, weight=6)

        self._font = Font(self._inner_frame, weight=BOLD)

        self._folder_icon = ImageTk.PhotoImage(Image.open("front/folder-icon.png").resize((80, 80)))
        
        self._images_folder = tk.StringVar()
        self._labels_folder = tk.StringVar()
        self._error = tk.StringVar()
        self._result = tk.StringVar()
        self._map50 = tk.StringVar()

        self._predictor = predictor
        
    def draw(self):
        menubar = tk.Menu(self._window)
        file = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=file)
        file.add_command(label='Info', command=self._openNewWindow)

        image_canvas = tk.Canvas(self._inner_frame, width=80, height=80)
        image_canvas.create_image(0, 0, anchor=tk.NW, image=self._folder_icon)
        image_canvas.grid(row=0, column=0)
        image_folder_label = tk.Label(self._inner_frame, textvariable=self._images_folder)
        image_folder_label.grid(row=1, column=0)
        image_folder_button = tk.Button(self._inner_frame, text='Images folder', command=self._browse_images_button)
        image_folder_button.grid(row=2, column=0)
    
        label_canvas = tk.Canvas(self._inner_frame, width=80, height=80)
        label_canvas.create_image(0, 0, anchor=tk.NW, image=self._folder_icon)
        label_canvas.grid(row=0, column=1)
        label_folder_label = tk.Label(self._inner_frame, textvariable=self._labels_folder)
        label_folder_label.grid(row=1, column=1)
        label_folder_button = tk.Button(self._inner_frame, text='Labels folder', command=self._browse_labels_button)
        label_folder_button.grid(row=2, column=1)
        
        error_label = tk.Label(self._inner_frame, textvariable=self._error, fg='#ff1100')
        error_label.grid(row=3, columnspan=2, pady=10)
    
        map_label = tk.Label(self._inner_frame, text="Target mAP50:")
        map_label.grid(row=4, column=0, pady=10, padx=10, sticky='e')
        map_entry = tk.Entry(self._inner_frame, textvariable=self._map50)
        map_entry.grid(row=4, column=1, pady=10, padx=10, sticky='w')
    
        button_compute = tk.Button(self._inner_frame, text="Calculate result", command=self._compute_button)
        button_compute.grid(row=5, columnspan=2, pady=10)
    
        result_label = tk.Label(self._inner_frame, textvariable=self._result, font=self._font)
        result_label.grid(row=6, columnspan=2, pady=20)

        self._window.config(menu=menubar)

        self._window.mainloop()
        
    _MAX_LENGTH = 22

    def _browse_images_button(self):
        filename = filedialog.askdirectory()
        if len(filename) == 0:
            return
        no_images = len(get_all_files_from(filename, self._predictor.image_formats))
        suffix = _get_suffix_image(self._predictor.number_of_images)
        if no_images < self._predictor.number_of_images:
            self._error.set(f'Folder must contain at least {self._predictor.number_of_images} image{suffix} in format'
                            + ', '.join(self._predictor.image_formats))
        else:
            self._predictor.add_images(filename)
            self._images_folder.set(shorten_path(filename, self._MAX_LENGTH))
            self._error.set('')

    def _browse_labels_button(self):
        filename = filedialog.askdirectory()
        if len(filename) == 0:
            return
        no_images = len(get_all_files_from(filename, self._predictor.labels_formats))
        if no_images < self._predictor.number_of_images:
            suffix = _get_suffix_file(self._predictor.number_of_images)
            self._error.set(f'Folder must contain at least {self._predictor.number_of_images} .txt file{suffix}')
        else:
            self._predictor.add_labels(filename)
            self._labels_folder.set(shorten_path(filename, self._MAX_LENGTH))
            self._error.set('')

    def _compute_button(self):
        try:
            map50 = float(self._map50.get())
        except ValueError:
            self._error.set(f'No mAP50 value')
            return
        if map50 > 1 or map50 < 0:
            self._error.set(f'Incorrect mAP50 value')
            return
        self._predictor.set_map50(map50)
        if self._predictor.is_ready_to_predict():
            result = self._predictor.predict()
            suffix = _get_suffix_image(result)
            self._result.set(f'Predicted train dataset size: {result} image{suffix}')
        else:
            self._error.set(f'Not enough data')

    def _openNewWindow(self):
        color = self._window.cget('bg')
        newWindow = tk.Toplevel(self._window)
        newWindow.title("Info")
        newWindow.geometry("500x200")
        newWindow.minsize(500, 200)
        newWindow.maxsize(500, 200)
        readonly = tk.Text(newWindow, font=nametofont("TkDefaultFont"), bg=color, wrap=tk.WORD)
        readonly.insert('end', "This application allows you to estimate the required training set size by " +
                               "given images. To do this, you need to enter sample images and labels " +
                               "in YOLOv8 format. You also need to specify the desired mAP50 (ranged from 0 to 1 " +
                               "where 1 corresponds to the best performance of the detection algorithm). " +
                               "In practice, a good value for mAP50 is 0.5 and above.")
        readonly.configure(state='disabled')
        readonly.bind("<Key>", lambda e: "break")
        readonly.pack()

