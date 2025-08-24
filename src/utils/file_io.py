import pandas as pd
from tkinter import Tk, filedialog

def choose_file_dialog(file_types):
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=file_types)
    return file_path

def save_results_to_csv(results, path):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    return path
