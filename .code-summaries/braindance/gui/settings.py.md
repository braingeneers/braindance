# settings.py

**Path:** `braindance/gui/settings.py`
**Module:** `braindance.gui.settings`
**Feature Area:** `Configuration`
**Entry point:** no — library or imported component

## Overview
Provides Qt dialogs for editing the experiment launcher's metadata field definitions. Saves ordered field types and categorical options in per-user preferences.

## Connections
- **Used by:** `braindance.gui.experiment_launcher` — import consumer hint; not a proven runtime call.
- **Shared data:** PreferencesDialog owns MetadataTab and FieldOptionsDialog; experiment launcher consumes metadata_fields.

## Dependencies
- `PyQt5.QtWidgets.QAction` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QApplication` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QComboBox` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QDialog` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QDialogButtonBox` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QFileDialog` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QFormLayout` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QHBoxLayout` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QInputDialog` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QLabel` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QLineEdit` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QListWidget` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QListWidgetItem` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QMainWindow` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QMessageBox` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QPushButton` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QTabWidget` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QVBoxLayout` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QWidget` — external or unresolved local import; source import evidence.

## Classes
### PreferencesDialog(QDialog)
> unclear — see source
**Source:** `braindance/gui/settings.py:60`
**Kind:** class. **Instantiated by:** braindance/gui/experiment_launcher.py:1648 (named-call hint)
**Constructor:** `__init__(self, parent=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `metadataTab` | inferred at runtime | `MetadataTab()` |
| `resetButton` | inferred at runtime | `QPushButton('Reset to Defaults')` |
| `saveButton` | inferred at runtime | `QPushButton('Save')` |
| `tabs` | inferred at runtime | `QTabWidget()` |
**Methods:**
#### `fromMainWindow(cls, mainWin)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:69`
#### `initUI(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/settings.py:74`
#### `savePreferences(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/gui/settings.py:97`
#### `loadPreferences(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:107`
#### `loadDefaults(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:117`
### MetadataTab(QWidget)
> unclear — see source
**Source:** `braindance/gui/settings.py:120`
**Kind:** class. **Instantiated by:** braindance/gui/settings.py:78 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `addFieldButton` | inferred at runtime | `QPushButton('Add Field')` |
| `fieldDetailsLabel` | inferred at runtime | `QLabel('Field Details:')` |
| `fieldsList` | inferred at runtime | `QListWidget()` |
| `moveDownButton` | inferred at runtime | `QPushButton('Move Down')` |
| `moveUpButton` | inferred at runtime | `QPushButton('Move Up')` |
| `trashButton` | inferred at runtime | `QPushButton('Trash')` |
**Methods:**
#### `initUI(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/settings.py:125`
#### `addField(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:157`
#### `showFieldOptionsDialog(self, item)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:164`
#### `updateFieldListItem(self, item)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:171`
#### `showFieldDetails(self, current, previous)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:176`
#### `getMetadataFields(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:184`
#### `loadMetadataFields(self, metadata_fields)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:199`
#### `moveFieldUp(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:208`
#### `moveFieldDown(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:215`
#### `trashField(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:222`
### FieldOptionsDialog(QDialog)
> unclear — see source
**Source:** `braindance/gui/settings.py:231`
**Kind:** class. **Instantiated by:** braindance/gui/settings.py:165 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `addOptionButton` | inferred at runtime | `QPushButton('Add Option')` |
| `confirmButton` | inferred at runtime | `QDialogButtonBox(QDialogButtonBox.Ok \| QDialogButtonBox.Cancel)` |
| `fieldTypeCombo` | inferred at runtime | `QComboBox()` |
| `optionsList` | inferred at runtime | `QListWidget()` |
**Methods:**
#### `initUI(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/settings.py:238`
#### `onFieldTypeChanged(self, index)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:265`
#### `addOption(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:270`
#### `getFieldOptions(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/settings.py:275`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| ~/.braindance/preferences.json; field types Raw Text,Categorical,Date |

## Data Shapes
- JSON {'metadata_fields':[{'name':str,'type':str,'options':list}]}

## Notes
- Creates ~/.braindance on import; saving drops default field info descriptions.
