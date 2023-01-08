import tkinter
import customtkinter
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import pandas as pd
import seaborn as sn
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()
        # configure window
        self.title("Colon ")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create left frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Colon Cancer",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Input your data",
                                                        command=self.browse_folder)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Upload model",
                                                        command=self.select_model)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="data spilt",
                                                        command=self.spilt_data)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("calculations")
        self.tabview.add("calculation")

        self.tabview.tab("calculations").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("calculation").grid_columnconfigure(0, weight=1)

        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("calculations"), text="calculate accuracy ",
                                                           command=self.calculate_accuracy)
        self.string_input_button.grid(row=0, column=0, padx=20, pady=(10, 10))

        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("calculations"),
                                                           text="calculate confusion matrix",
                                                           command=self.confusion_matrix)
        self.string_input_button.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("calculations"), text="",
                                                           command=self.confusion_matrix)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        # create right frame

        # self.radiobutton_frame = customtkinter.CTkFrame(self)
        # self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # set default values

        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

        self.textbox.insert("0.0", "")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def spilt_data(self):
        global batch_size, img_height, img_width, train_datagen, train_generator, train_datagen, validation_generator, validation_splitt
        validation_split_dialog = customtkinter.CTkInputDialog(
            text="input train and test ratio (if you do not need any train data enter 0.0)",
            title="split data")
        validation_splitt = float(validation_split_dialog.get_input())

        batch_size_dialog = customtkinter.CTkInputDialog(
            text="input batch size (for confusion matrix enter 0",
            title="batch size")
        batch_size = int(batch_size_dialog.get_input())
        img_height, img_width = (224, 224)
        train_datagen = ImageDataGenerator(
            horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=1,
            class_mode='categorical')

        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            validation_split=validation_splitt)  # set validation split

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')  # set as training data

        validation_generator = train_datagen.flow_from_directory(
            train_data_dir,  # same directory as training data
            target_size=(img_height, img_width),
            batch_size=1,
            class_mode='categorical',
            subset='validation')

    def confusion_matrix(self):
        # set as validation dat
        model = tf.keras.models.load_model('{}'.format(modell))
        filenames = train_generator.filenames
        nb_samples = len(train_generator)
        y_prob = []
        y_act = []
        train_generator.reset()
        for _ in range(nb_samples):
            X_test, Y_test = train_generator.next()
            y_prob.append(model.predict(X_test))
            y_act.append(Y_test)
        predicted_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_prob]
        actual_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_act]

        out_df = pd.DataFrame(np.vstack([predicted_class, actual_class]).T, columns=['predicted_class', 'actual_class'])
        confusion_matrix = pd.crosstab(out_df['actual_class'], out_df['predicted_class'], rownames=['Actual'],
                                       colnames=['Predicted'])

        sn.heatmap(confusion_matrix, cmap='Blues', annot=True, fmt=' d')
        plt.show()
        print('test accuracy: {}'.format((np.diagonal(confusion_matrix).sum() / confusion_matrix.sum().sum() * 100)))
        return 0

    def calculate_accuracy(self):
        model = tf.keras.models.load_model('{}'.format(modell))
        test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
        print("\nTest accuracy: ", test_acc)

    def select_model(self):
        global modell
        # Open a file dialog and get the selected file
        modell = filedialog.askopenfilename()

    def browse_folder(self):
        # Open a file dialog and get the selected folder
        global train_data_dir
        train_data_dir = filedialog.askdirectory()
        print(train_data_dir)

    def Upload_image(self):
        print("HELLO")

    def LoadResNetModel(self):
        global ResNet
        ResNet = tf.keras.models.load_model('Path')


if __name__ == "__main__":
    app = App()
    app.mainloop()
