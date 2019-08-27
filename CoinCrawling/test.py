from PyQt4.QtGui import *
import sys

class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.list = QListWidget(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.list)

    def addListItem(self, text):
        item = QListWidgetItem(text)
        self.list.addItem(item)
        widget = QWidget(self.list)
        button = QToolButton(widget)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(button)
        self.list.setItemWidget(item, widget)
        button.clicked[()].connect(
            lambda: self.handleButtonClicked(item))

    def handleButtonClicked(self, item):
        print(item.text())

if __name__ == '__main__':

    coin_list=['Bitcoin(BTC)','XRP(XRP)','Ethereum(ETH)','Stellar(XLM)','Tether(USDT)','BitcoinCash(BCH)','EOS(EOS)','BitcoinSV(BSV)','Litecoin(LTC)','TRON(TRX)']
    app = QApplication(sys.argv)
    window = Window()
    for label in coin_list:
        window.addListItem(label)
    window.setGeometry(500, 300, 300, 200)
    window.show()
    sys.exit(app.exec_())