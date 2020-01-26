
import sys
import logging

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit

from p4p.client.PyQt import Context

_log = logging.getLogger(__name__)

class Demo(QWidget):
    def __init__(self, pvname):
        QWidget.__init__(self)
        self._pvname = pvname
        self.initUI()

        self._put = None # in-progress put

        self.pvname.setText(pvname)

        self.ctxt = Context('pva', parent=self)
        self._S = self.ctxt.monitor(pvname, str, notify_disconnect=True, limitHz=1.0)
        self._S.update.connect(self.value.setText)
        self._S.error.connect(self.error.setText)

        self.edit.editingFinished.connect(self.doPut)

    def initUI(self):
        layout = QVBoxLayout()

        self.pvname = QLabel(self)
        self.pvname.setText("<name>")
        layout.addWidget(self.pvname)

        self.value = QLabel(self)
        self.value.setText("<Connecting>")
        layout.addWidget(self.value)

        self.edit = QLineEdit(self)
        layout.addWidget(self.edit)

        self.error = QLabel(self)
        self.error.setText("<Connecting>")
        layout.addWidget(self.error)

        self.setLayout(layout)
        self.setGeometry(200, 200, 250, 150)

    def doPut(self):
        try:
            self._put = self.ctxt.put(self._pvname, self.edit.text())

            self._put.success.connect(self.doPutOk)
            self._put.error.connect(self.doPutFail)

        except:
            _log.exception('put')

    def doPutOk(self):
        self.error.setText("Put complete")

    def doPutFail(self, err):
        self.error.setText("Put err : "+err)

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    demo = Demo(app.arguments()[1])
    demo.show()
    sys.exit(app.exec_())
