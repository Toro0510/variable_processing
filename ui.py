import sys
from PyQt5.QtWidgets import QApplication, QWidget ,QMainWindow
from untitled import Ui_Form
import main_func
import pandas as pd

class Demo(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.toolButton.clicked.connect(self.get_filename_path)
        self.toolButton_2.clicked.connect(self.get_directory_path)

        self.pushButton.clicked.connect(self.on_button_clicked)


    def on_button_clicked(self):
        filename_path = self.label.text()
        directory_path = self.label_2.text()
        self.main_loop(filename_path, directory_path)


    def main_loop(self, filename_path, directory_path):

        self.printf('read file....')
        df = pd.read_excel(filename_path)
        df = main_func.make_M1target(df, 'overdue_days_correct')

        self.printf('variable classification....')
        var_list = main_func.make_var_list(df)
        var_classification_dict, classification_error_dict = main_func.var_classification(df, var_list)

        self.printf('variable analyzing....')
        variable_list, analyzing_error_dict = main_func.variable_analyzing(df, var_classification_dict)
        result_dataFrame = main_func.sort_report_list(variable_list)

        self.printf('output....')
        result_dataFrame.to_excel(directory_path + '\\' + 'result.xlsx')

        error_dict = dict(classification_error_dict, **analyzing_error_dict)
        self.printf('classification_error_dict:')
        self.printf(str(error_dict))

        self.printf('draw corr heatmap....')
        main_func.corr_heatmap(df, result_dataFrame, 0.02, directory_path + '\\' + 'corr_heatmap.png')

        self.printf('done!')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())