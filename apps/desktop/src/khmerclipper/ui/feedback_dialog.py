from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout

from khmerclipper.ui.fonts import khmer_font


class FeedbackDialog(QDialog):
    """Collects a free-text message and an optional reply email for a feedback report."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Send Feedback")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("What would you like to tell the KhmerClipper developer?"))
        self.message = QPlainTextEdit()
        self.message.setFont(khmer_font(12))
        layout.addWidget(self.message)
        self.email = QLineEdit()
        self.email.setPlaceholderText("Your email (if you'd like a reply)")
        layout.addWidget(self.email)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.send_button = buttons.addButton("Send", QDialogButtonBox.ButtonRole.AcceptRole)
        self.send_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.message.textChanged.connect(
            lambda: self.send_button.setEnabled(bool(self.message.toPlainText().strip())))

    def values(self) -> tuple[str, str]:
        return self.message.toPlainText().strip(), self.email.text().strip()
