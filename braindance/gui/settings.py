import sys
import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QDialog, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QTabWidget, 
    QFormLayout, QComboBox, QInputDialog, QListWidget, QListWidgetItem, QWidget
)
import pathlib

home_dir = Path.home()
braindance_dir = home_dir / '.braindance'
pathlib.Path(braindance_dir).mkdir(parents=True, exist_ok=True)

preferences_file = braindance_dir / 'preferences.json'


default_metadata_fields = {
    'metadata_fields': [
        {
            'name': 'notes',
            'info': 'Notes about the recording',
            'type': 'Raw Text',
            'options': []
        },
        {
            'name': 'aggregation_date',
            'info': 'Age since aggregation/plating, use YYYY-MM-DD format',
            'type': 'Date',
            'options': []
        },
        {
            'name': 'adherence_date',
            'info': 'Date of adherence, use YYYY-MM-DD format',
            'type': 'Date',
            'options': []
        },
        {
            'name': 'Patterning',
            'info': 'Patterning type',
            'type': 'Categorical',
            'options': []
        },
        {
            'name': 'cell_line',
            'info': 'Cell line used',
            'type': 'Categorical',
            'options': []
        },
        {
            'name': 'parent',
            'info': 'Parents name :) (Who has handled the cells)',
            'type': 'Raw Text',
            'options': []
        }
    ]
}

class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Preferences')
        self.setMinimumSize(800, 600)
        self.initUI()
        self.loadPreferences()

    @classmethod
    def fromMainWindow(cls, mainWin):
        dialog = cls()
        dialog.metadataTab.loadMetadataFields(mainWin.metadata_fields)
        return dialog
    
    def initUI(self):
        layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        self.metadataTab = MetadataTab()
        self.tabs.addTab(self.metadataTab, "Metadata")
        
        layout.addWidget(self.tabs)
        
        button_layout = QHBoxLayout()
        
        self.saveButton = QPushButton("Save")
        self.saveButton.clicked.connect(self.savePreferences)
        button_layout.addWidget(self.saveButton)
        
        self.resetButton = QPushButton("Reset to Defaults")
        self.resetButton.clicked.connect(self.loadDefaults)
        button_layout.addWidget(self.resetButton)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def savePreferences(self):
        metadata_fields = self.metadataTab.getMetadataFields()
        preferences = {
            'metadata_fields': metadata_fields
        }
        with open(preferences_file, 'w') as f:
            json.dump(preferences, f, indent=4)
        QMessageBox.information(self, "Preferences", "Preferences saved successfully!")
        self.accept()

    def loadPreferences(self):
        try:
            with open(preferences_file, 'r') as f:
                preferences = json.load(f)
                metadata_fields = preferences.get('metadata_fields', [])
                self.metadataTab.loadMetadataFields(metadata_fields)
        except FileNotFoundError:
            QMessageBox.information(self, "First Time Setup", "No existing preferences found. Loading default settings.")
            self.loadDefaults()

    def loadDefaults(self):
        self.metadataTab.loadMetadataFields(default_metadata_fields['metadata_fields'])

class MetadataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        self.fieldsList = QListWidget()
        self.fieldsList.currentItemChanged.connect(self.showFieldDetails)
        layout.addWidget(self.fieldsList)

        # Add layout at the bottom for buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        self.addFieldButton = QPushButton("Add Field")
        self.addFieldButton.clicked.connect(self.addField)
        button_layout.addWidget(self.addFieldButton)

        self.moveUpButton = QPushButton("Move Up")
        self.moveUpButton.clicked.connect(self.moveFieldUp)
        button_layout.addWidget(self.moveUpButton)
        
        self.moveDownButton = QPushButton("Move Down")
        self.moveDownButton.clicked.connect(self.moveFieldDown)
        button_layout.addWidget(self.moveDownButton)
        
        self.trashButton = QPushButton("Trash")
        self.trashButton.clicked.connect(self.trashField)
        button_layout.addWidget(self.trashButton)
        
        self.fieldDetailsLabel = QLabel("Field Details:")
        layout.addWidget(self.fieldDetailsLabel)
        
        self.setLayout(layout)
    
    def addField(self):
        text, ok = QInputDialog.getText(self, 'Add Field', 'Enter field name:')
        if ok and text:
            item = QListWidgetItem(text)
            self.fieldsList.addItem(item)
            self.showFieldOptionsDialog(item)
    
    def showFieldOptionsDialog(self, item):
        dialog = FieldOptionsDialog()
        if dialog.exec_():
            field_type, options = dialog.getFieldOptions()
            item.setData(1, (field_type, options))
            self.updateFieldListItem(item)
    
    def updateFieldListItem(self, item):
        field_name = item.text()
        field_type, options = item.data(1) if item.data(1) else ("Raw Text", [])
        item.setText(f"{field_name} ({field_type})")
    
    def showFieldDetails(self, current, previous):
        if current is not None:
            field_type, options = current.data(1) if current.data(1) else ("Raw Text", [])
            details = f"Field Type: {field_type}\n"
            if field_type == "Categorical":
                details += "Options:\n" + "\n".join(options)
            self.fieldDetailsLabel.setText(details)
    
    def getMetadataFields(self):
        fields = []
        for index in range(self.fieldsList.count()):
            item = self.fieldsList.item(index)
            field_name = item.text().split(' (')[0]
            field_data = item.data(1)
            if field_data:
                field_type, options = field_data
                fields.append({
                    'name': field_name,
                    'type': field_type,
                    'options': options
                })
        return fields

    def loadMetadataFields(self, metadata_fields):
        self.fieldsList.clear()
        for field in metadata_fields:
            item = QListWidgetItem(f"{field['name']} ({field['type']})")
            item.setData(1, (field['type'], field['options']))
            self.fieldsList.addItem(item)
        if self.fieldsList.count() > 0:
            self.fieldsList.setCurrentRow(0)

    def moveFieldUp(self):
        current_row = self.fieldsList.currentRow()
        if current_row > 0:
            current_item = self.fieldsList.takeItem(current_row)
            self.fieldsList.insertItem(current_row - 1, current_item)
            self.fieldsList.setCurrentRow(current_row - 1)

    def moveFieldDown(self):
        current_row = self.fieldsList.currentRow()
        if current_row < self.fieldsList.count() - 1:
            current_item = self.fieldsList.takeItem(current_row)
            self.fieldsList.insertItem(current_row + 1, current_item)
            self.fieldsList.setCurrentRow(current_row + 1)

    def trashField(self):
        current_row = self.fieldsList.currentRow()
        if current_row >= 0:
            self.fieldsList.takeItem(current_row)

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QDialogButtonBox, QListWidget, QInputDialog, QListWidgetItem
)
class FieldOptionsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Field Options')
        self.setMinimumSize(400, 300)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        self.fieldTypeCombo = QComboBox()
        self.fieldTypeCombo.addItem('Raw Text')
        self.fieldTypeCombo.addItem('Categorical')
        self.fieldTypeCombo.addItem('Date')
        self.fieldTypeCombo.currentIndexChanged.connect(self.onFieldTypeChanged)
        layout.addWidget(QLabel("Field Type:"))
        layout.addWidget(self.fieldTypeCombo)
        
        self.optionsList = QListWidget()
        layout.addWidget(QLabel("Options (for Categorical type):"))
        layout.addWidget(self.optionsList)
        
        self.addOptionButton = QPushButton("Add Option")
        self.addOptionButton.clicked.connect(self.addOption)
        layout.addWidget(self.addOptionButton)
        
        self.confirmButton = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.confirmButton.accepted.connect(self.accept)
        self.confirmButton.rejected.connect(self.reject)
        layout.addWidget(self.confirmButton)
        
        self.setLayout(layout)
        self.onFieldTypeChanged(0)  # Default to Raw Text

    def onFieldTypeChanged(self, index):
        is_categorical = self.fieldTypeCombo.currentText() == 'Categorical'
        self.optionsList.setVisible(is_categorical)
        self.addOptionButton.setVisible(is_categorical)

    def addOption(self):
        text, ok = QInputDialog.getText(self, 'Add Option', 'Enter option:')
        if ok and text:
            self.optionsList.addItem(text)
    
    def getFieldOptions(self):
        field_type = self.fieldTypeCombo.currentText()
        options = [self.optionsList.item(i).text() for i in range(self.optionsList.count())] if field_type == 'Categorical' else []
        return field_type, options