from array import array
from asyncio.windows_events import NULL
from cmath import log
from operator import index
from pathlib import Path
from pickle import NONE
import tkinter as tk
import customtkinter as customTkinter
from tkinter import ANCHOR, BROWSE, CENTER, YES, ttk
from tkinter.messagebox import NO
from turtle import onclick, width
import pandas as pd
import numpy as np
import classifier as clf
from tkinter import W, IntVar, StringVar, filedialog as fd
import matplotlib.pyplot as plt
import tach_file
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import os   

np.seterr(divide='ignore', invalid='ignore')

# Configure windows
window = customTkinter.CTk()

customTkinter.set_appearance_mode("light")  # Modes: system (default), light, dark
customTkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

window.title('Project 1')
window.geometry('925x500+300+200')
# window.configure(bg='#fff')
window.resizable(False, False)


headerFrame = tk.Frame(window, borderwidth=1,  relief='solid', padx=16, pady=4)
headerFrame.place(x=0,y=0, relwidth=1, relheight=0.07)

# Configure Input Frame
inputFrame = tk.Frame(window, borderwidth=1, relief='solid', padx=16, pady=16)
inputFrame.place(x=0, rely=0.07, relheight=1, relwidth=0.65)


propFrame = tk.Frame(master=inputFrame)

# Configure Output Frame
outputFrame = tk.Frame(window, borderwidth=1, relief='solid', padx=16, pady=16)
outputFrame.place(relx=0.65, rely=0.07, relheight=1, relwidth=0.35)


footerFrame = tk.Frame(window, borderwidth=1, relief='solid', padx=16, pady=4)
footerFrame.place(x=0, rely=(1-0.07), relwidth=1, relheight=0.07, )


def openFile():
    global filename
    filename = fd.askopenfilename(filetypes=[("Data files",
                                              ["*.xlsx", "*.csv", "*xls"])]).replace('\\', '/')
    global excelFile

    if filename[-5:] == ".xlsx" or filename[-4:] == ".xls":
        fileLabel.configure(text="File: " + filename.split('/')[-1])
        excelFile = pd.read_excel(filename)
        loadData()
    elif filename[-4:] == ".csv":
        fileLabel.configure(text="File: " + filename.split('/')[-1])
        excelFile = pd.read_csv(filename)
        loadData()
    else:
        fileLabel.configure(text="Not a correct data file")


do_label = ['Euler', 'Hamming (2 hàm)', 'Hamming (3 hàm)', 'Mahanta', 'Ngân']
do_values = ['eu', 'ha2', 'ha3', 'ma', 'ng']

vars = []
doDoVal = IntVar()


def loadData():
    for item in propFrame.winfo_children():
        item.destroy()
    global columns, l_values
    columns = list(excelFile.columns.values)
    l_values = (
        list(dict.fromkeys(list(excelFile[excelFile.columns.values[-1]].values))))

    loadTable(excelFile, columns)

    def selectColumns(index, value):
        if (columns[index] == value):
            columns[index] = None
        else:
            columns[index] = value
        
        loadTable(excelFile, [item for item in columns if item != None])

    def selectLValues(index, value):
        if (l_values[index] == value):
            l_values[index] = None
        else:
            l_values[index] = value

    columnsFrame =tk.Frame(inputFrame)
    columnsFrame.place(relwidth=1, rely=0.76)

    # scrollbar = ttk.Scrollbar(columnsFrame, orient=tk.HORIZONTAL)
    # scrollbar.pack(side='bottom')
    for index, prop in enumerate(columns[:-1]):
        vars.append(StringVar())
        vars[-1].set(1)
        radioButton = customTkinter.CTkCheckBox(columnsFrame, text=columns[index], variable=vars[-1], checkbox_width=18, checkbox_height=18, 
                                     command=lambda value=columns[index], index=index: selectColumns(index, value))
        radioButton.pack(side='left')
    
    doDoFrame =tk.Frame(inputFrame)
    doDoFrame.place(relwidth=1, rely=0.82)
    for index, doValue in enumerate(do_label):
        radioButton = customTkinter.CTkRadioButton(
            doDoFrame, text=doValue, variable=doDoVal, value=(index + 1), radiobutton_height=18, radiobutton_width=18, width=100)
        radioButton.pack(side='left')


def loadTable(excelFile, columns):
    tree = ttk.Treeview(inputFrame, columns=columns, show='headings')


    isTrain = columns.count('Prediction') > 0
    for index, item in enumerate(columns):
        tree.heading(item, text=item, anchor=CENTER)
        tree.column('#' + str(index), width=10, stretch=YES)
    tree.column('#' + str(len(columns)), width=0)

    
    showData = excelFile[columns].values

    for data in showData:
        if (isTrain and data[-1] != data[-2]):
            tree.insert('', tk.END, values=list(data), tags="wrongPrediction")
        else:
            tree.insert('', tk.END, values=list(data))

    if (isTrain):
        tree.tag_configure('wrongPrediction', background='#f2ec6d')

    tree.grid(row=0, column=0, sticky='we')
    tree.place(y=0, relwidth=1.0, relheight=0.56, relx=0)


def processingData():
    selectedColumns = [item for item in columns[:-1] if item != None]
    selectedLValues = [item for item in l_values if item != None]
    selectedDoDo = do_values[doDoVal.get() - 1]

    train_data = excelFile[selectedColumns].values
    w0 = 1 / len(selectedColumns)
    print(len(selectedColumns))
    weight_train = clf.weight_train(w0, train_data)

    classes = np.asarray(selectedLValues)
    L_train = excelFile.to_numpy()[:, -1]
    center = clf.center_find(train_data, L_train, classes)
    doDo = do_values[doDoVal.get() - 1]
    w = clf.early_stopping(weight_train, L_train, train_data, center, classes, doDo)
    
    L_pred = clf.labeling(train_data, center, classes, w, doDo)  

    accuracy = clf.accuracy(L_pred, L_train)
    x = [item for item in columns if item != None]
    x.append(('Prediction'))
    loadTable(pd.concat([excelFile, pd.Series(
        L_pred, name='Prediction')], axis=1), list(x))

    accuracyLabel.configure(text="Accuracy: " + str(round(accuracy, 5)))

    weightStr = ""
    for index in range(len(weight_train[-1])):
        weightStr = weightStr + "      "  + str(selectedColumns[index]) + ": " + str(round(weight_train[-1][index], 5)) + "\n" 

    weightResult.configure(text=weightStr)

    x = np.arange(weight_train.shape[0])
    y = np.zeros_like(x, dtype = 'float')
    for i in range(x.shape[0]):
        y[i] = clf.accuracy(clf.labeling(train_data, center, classes, weight_train[i], doDo), L_train)

    # # plt.xlabel("Iteration")
    # # plt.ylabel("Accuracy")
    
    fig = Figure(figsize=(3.6,3.4), dpi=100)
    fig.add_subplot(111).plot(x,y)
    # fig
    canvas = FigureCanvasTkAgg(fig, master=outputFrame)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().place(rely=0.36, relx=0.13)
    # exportExcelFile = pd.DataFrame(weight_train)

    # exportExcelFile.to_excel('pandas_to_excel.xlsx', sheet_name='Export Data')

# print(excelFile)
def splti_data():
    print("use spllit")
    dfData = excelFile
    listColumns = dfData.columns.to_list()
    train = pd.DataFrame()
    validate = pd.DataFrame()
    test = pd.DataFrame()
    columnsLabelName = listColumns[len(listColumns) - 1]
    listLabel = dfData[columnsLabelName].unique().tolist()

    for y in listLabel:
        dfy = dfData[dfData[columnsLabelName]==y]
        train1, validate1, test1 = tach_file.train_validate_test_split(dfy)
        train = pd.concat([train,train1])
        validate = pd.concat([validate,validate1])
        test = pd.concat([test,test1])

    path = Path(filename)
    train.to_csv(f'Train_'+path.name, index = False)
    validate.to_csv(f'Validate_'+path.name, index = False)
    test.to_csv(f'Test_'+path.name, index = False)
    print("Done")
  

buttonChooseFile = customTkinter.CTkButton(headerFrame, text="Choose File!",
                             command=openFile, )
buttonChooseFile.place(relx=0.05, rely=0.03)

fileLabel = customTkinter.CTkLabel(headerFrame, text="")
fileLabel.place(relx=0.25, rely=0.04)

buttonProcess = customTkinter.CTkButton(footerFrame, text="Process", command=processingData)
buttonProcess.place(relx=0.45, rely=0.03)


accuracyLabel = customTkinter.CTkLabel(outputFrame, text='Accuracy: ')
accuracyLabel.place(rely=0.01)


weightLabel = customTkinter.CTkLabel(outputFrame, text='Weight: ', )
weightLabel.place(rely=0.06)


weightResult = customTkinter.CTkLabel(outputFrame, text='', )
weightResult.place(rely=0.11)

splitbtn = customTkinter.CTkButton(footerFrame, text="Split", command=splti_data)
splitbtn.place(x = 0, y =0)

window.mainloop()
