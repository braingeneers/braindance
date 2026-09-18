# experiment_launcher.py

**Path:** `braindance/gui/experiment_launcher.py`
**Module:** `braindance.gui.experiment_launcher`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Provides a legacy Qt desktop launcher for project/chip/experiment JSON records, hardware status and analysis commands. Displays electrode mappings and footprints, edits configuration, manages metadata preferences and offers experiment uploads.

## Connections
- **Uses:** `AnalysisDAO` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `plot_helper` from `braindance.analysis` — imports (static evidence).
- **Uses:** `get_maxwell_status` from `braindance.core.maxwell.maxwell_utils` — imports (static evidence).
- **Uses:** `SmartPlug` from `braindance.core.utils` — imports (static evidence).
- **Uses:** `CommandRunner` from `braindance.gui.command_runner` — imports (static evidence).
- **Uses:** `PreferencesDialog` from `braindance.gui.settings` — imports (static evidence).

## Dependencies
- `PyQt5.QtCore.QDate` — external or unresolved local import; source import evidence.
- `PyQt5.QtCore.Qt` — external or unresolved local import; source import evidence.
- `PyQt5.QtGui.QColor` — external or unresolved local import; source import evidence.
- `PyQt5.QtGui.QIcon` — external or unresolved local import; source import evidence.
- `PyQt5.QtGui.QImage` — external or unresolved local import; source import evidence.
- `PyQt5.QtGui.QPixmap` — external or unresolved local import; source import evidence.
- `PyQt5.QtGui.QTextCursor` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QAction` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QApplication` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QCalendarWidget` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QCheckBox` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QComboBox` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QDialog` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QDialogButtonBox` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QFileDialog` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QFormLayout` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QHBoxLayout` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QInputDialog` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QLabel` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QLineEdit` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QMainWindow` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QMessageBox` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QPushButton` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QRadioButton` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QSizePolicy` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QSplitter` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QTabWidget` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QTableWidget` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QTableWidgetItem` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QTextEdit` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QTreeWidget` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QTreeWidgetItem` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QVBoxLayout` — external or unresolved local import; source import evidence.
- `PyQt5.QtWidgets.QWidget` — external or unresolved local import; source import evidence.
- `braindance.analysis.data_loader.AnalysisDAO` — intra-repo import; source import evidence.
- `braindance.analysis.plot_helper` — intra-repo import; source import evidence.
- `braindance.core.maxwell.maxwell_utils.get_maxwell_status` — intra-repo import; source import evidence.
- `braindance.core.utils.SmartPlug` — intra-repo import; source import evidence.
- `braindance.gui.command_runner.CommandRunner` — intra-repo import; source import evidence.
- `braindance.gui.settings.PreferencesDialog` — intra-repo import; source import evidence.
- `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg` — external or unresolved local import; source import evidence.
- `matplotlib.backends.backend_qt5agg.NavigationToolbar2QT` — external or unresolved local import; source import evidence.
- `matplotlib.figure.Figure` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `maxlab` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### MainWindow(QMainWindow)
> Qt desktop launcher for experiment and analysis workflows.
**Source:** `braindance/gui/experiment_launcher.py:83`
**Kind:** class. **Instantiated by:** braindance/gui/experiment_launcher.py:1879 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `analysisRequirementText` | inferred at runtime | `QTextEdit()` |
| `analysisSelector` | inferred at runtime | `QComboBox(self.analysisTab)` |
| `analysisTab` | inferred at runtime | `QWidget()` |
| `analysis_obj` | inferred at runtime | `None` |
| `analysis_running` | inferred at runtime | `False` |
| `canvas` | inferred at runtime | `FigureCanvas(self.figure)` |
| `changeMainPathButton` | inferred at runtime | `QPushButton()` |
| `chipSelector` | inferred at runtime | `QComboBox(self)` |
| `cleanStimElectrodesButton` | inferred at runtime | `QPushButton('Clean', self.launchTab)` |
| `clearButton` | inferred at runtime | `QPushButton('Clear', self.launchTab)` |
| `clearButtonAnalysis` | inferred at runtime | `QPushButton('Clear', self.analysisTab)` |
| `cloudSyncButton` | inferred at runtime | `QPushButton(self)` |
| `commandRunner` | inferred at runtime | `CommandRunner(command)` |
| `configTab` | inferred at runtime | `QWidget()` |
| `dummy_maxlab` | inferred at runtime | `False` |
| `electrodeSelector` | inferred at runtime | `QComboBox(dialog)` |
| `emergencyStopAnalysisButton` | inferred at runtime | `QPushButton('Emergency Stop', self.analysisTab)` |
| `emergencyStopButton` | inferred at runtime | `QPushButton('Emergency Stop', self.launchTab)` |
| `expParamsUpdateButton` | inferred at runtime | `QPushButton('Update', self.expWidget)` |
| `expRawText` | inferred at runtime | `QTextEdit('', centralWidget)` |
| `expWidget` | inferred at runtime | `QTreeWidget(centralWidget)` |
| `experimentLaunchSelector` | inferred at runtime | `QComboBox(self.launchTab)` |
| `experimentSelector` | inferred at runtime | `QComboBox(self)` |
| `experiment_running` | inferred at runtime | `False` |
| `figure` | inferred at runtime | `Figure()` |
| `footprint_chs` | inferred at runtime | `None` |
| `footprint_waves` | inferred at runtime | `None` |
| `getFootprintButton` | inferred at runtime | `QPushButton('Get Footprint', self)` |
| `height` | inferred at runtime | `800` |
| `json_data` | inferred at runtime | `None` |
| `launchAnalysisButton` | inferred at runtime | `QPushButton('Launch Analysis', self.analysisTab)` |
| `launchExperimentButton` | inferred at runtime | `QPushButton('Neural Config Recording', self.launchTab)` |
| `launchTab` | inferred at runtime | `QWidget()` |
| `mainPath` | inferred at runtime | `None` |
| `mapping` | inferred at runtime | `None` |
| `meanAmpButton` | inferred at runtime | `QRadioButton('Mean Amp', self)` |
| `menubar` | inferred at runtime | `self.menuBar()` |
| `metadata_fields` | inferred at runtime | `None` |
| `notesText` | inferred at runtime | `QTextEdit(notesTab)` |
| `outputText` | inferred at runtime | `QTextEdit(self.launchTab)` |
| `outputTextAnalysis` | inferred at runtime | `QTextEdit(self.analysisTab)` |
| `pictureWidget` | inferred at runtime | `QLabel()` |
| `picturesList` | inferred at runtime | `QComboBox(self)` |
| `projectSelector` | inferred at runtime | `QComboBox(self)` |
| `refreshButton` | inferred at runtime | `QPushButton('Refresh', self)` |
| `requirementText` | inferred at runtime | `QTextEdit(requirementsTab)` |
| `scatSelectedCheckbox` | inferred at runtime | `QCheckBox('Selected Electrodes', self)` |
| `scatStimCheckbox` | inferred at runtime | `QCheckBox('Stim Electrodes', self)` |
| `scat_selected_electrodes` | inferred at runtime | `None` |
| `scat_stim_electrodes` | inferred at runtime | `None` |
| `selected_chip` | inferred at runtime | `None` |
| `selected_project` | inferred at runtime | `None` |
| `smartPlugLabel` | inferred at runtime | `None` |
| `smartPlugOffButton` | inferred at runtime | `QPushButton('Off', self)` |
| `smartPlugOnButton` | inferred at runtime | `QPushButton('On', self)` |
| `smartplug` | inferred at runtime | `None` |
| `spikeCountButton` | inferred at runtime | `QRadioButton('Spike Count', self)` |
| `statusCircle` | inferred at runtime | `QLabel()` |
| `stim_electrodes` | inferred at runtime | `QLineEdit(self.launchTab)` |
| `tabWidget` | inferred at runtime | `QTabWidget(centralWidget)` |
| `title` | inferred at runtime | `'BrainDance Experiment Launcher'` |
| `toolbar` | inferred at runtime | `NavigationToolbar(self.canvas, self)` |
| `updateStimElectrodesButton` | inferred at runtime | `QPushButton('Update', self.launchTab)` |
| `validateStimElectrodesButton` | inferred at runtime | `QPushButton('Validate', self.launchTab)` |
| `width` | inferred at runtime | `1000` |
**Methods:**
#### `initMenuBar(self)`
> Qt slot `initMenuBar` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:113`
#### `initUI(self)`
> Qt slot `initUI` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:125`
#### `smartPlugOn(self)`
> Qt slot `smartPlugOn` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:463`
#### `smartPlugOff(self)`
> Qt slot `smartPlugOff` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:470`
#### `get_hardware_status(self)`
> Get the hardware status from the maxlab library
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:478`
#### `setupLaunchTab(self)`
> Qt slot `setupLaunchTab` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:500`
#### `checkLaunchButtonDependencies(self)`
> Qt slot `checkLaunchButtonDependencies` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:636`
#### `onLaunchExperimentChanged(self, index)`
> Update the button, what it says, and what it launches. Also update the text box which shows the parameters and the requirements
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:668`
#### `launchExperiment(self)`
> Qt slot `launchExperiment` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:695`
#### `checkExperimentRequirements(self, requirements=[])`
> Checks if the experiment has the requirements Returns a list of missing requirements
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:740`
#### `validateStimElectrodes(self, remove_duplicates=False)`
> Use validation script bdquery to validate the stim electrodes
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:761`
#### `updateStimElectrodes(self)`
> Updates stim electrodes in the json data from the text box if it is different from the json data
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/gui/experiment_launcher.py:807`
#### `updateExpParams(self)`
> Save the text in the expRawText box to the json file
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/gui/experiment_launcher.py:842`
#### `handleExperimentError(self, error_message)`
> Qt slot `handleExperimentError` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:864`
#### `handleExperimentCompleted(self)`
> Qt slot `handleExperimentCompleted` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:872`
#### `updateRequirementText(self, text, color=None)`
> Update the text box with the requirements
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:893`
#### `updateOutputText(self, text, color=None)`
> Qt slot `updateOutputText` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:903`
#### `updateNotes(self)`
> Update the notes in the json file
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:909`
#### `emergencyStop(self)`
> Qt slot `emergencyStop` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:927`
#### `setupAnalysisTab(self)`
> Qt slot `setupAnalysisTab` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:946`
#### `onAnalysisChanged(self, index)`
> Update the button, what it says, and what it launches. Also update the text box which shows the parameters and the requirements
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1026`
#### `launchAnalysis(self)`
> Qt slot `launchAnalysis` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1051`
#### `updateAnalysisRequirementText(self, text, color=None)`
> Update the text box with the requirements
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1097`
#### `updateAnalysisOutputText(self, text, color=None)`
> Qt slot `updateAnalysisOutputText` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1106`
#### `handleAnalysisError(self, error_message)`
> Qt slot `handleAnalysisError` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1113`
#### `handleAnalysisCompleted(self)`
> Qt slot `handleAnalysisCompleted` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1121`
#### `getFootprint(self)`
> Try to load from .npy file in the folder, or create a dialogue to select .raw.h5 file to get footprint
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1133`
#### `getFootprintFromH5(self, h5_file, dialog)`
> Qt slot `getFootprintFromH5` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1209`
#### `setMapping(self, mapping)`
> Qt slot `setMapping` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1240`
#### `scatSelectedToggle(self, state)`
> Qt slot `scatSelectedToggle` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1248`
#### `scatStimToggle(self, state)`
> Qt slot `scatStimToggle` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1253`
#### `plotScatter(self, color_mode='mean_amp')`
> Qt slot `plotScatter` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1258`
#### `getProjectsList(self)`
> Qt slot `getProjectsList` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1389`
#### `onProjectSelect(self, index)`
> Qt slot `onProjectSelect` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1396`
#### `updateProjectsList(self)`
> Qt slot `updateProjectsList` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1403`
#### `addProject(self)`
> Open up a new window, ask for the project name, and create the project folder
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1412`
#### `onChipSelect(self, index)`
> Qt slot `onChipSelect` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1429`
#### `updateChipsList(self)`
> Qt slot `updateChipsList` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1437`
#### `addChip(self)`
> Open up a new window, ask for the chip name, and create the chip folder
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1443`
#### `onExperimentSelect(self, index, clear=True)`
> Load the json file and display it as a table
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1461`
#### `updateExperimentsList(self)`
> Qt slot `updateExperimentsList` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1481`
#### `loadExperiment(self, json_file)`
> Load the json file and display it as a tree
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1494`
#### `addExperiment(self)`
> Qt slot `addExperiment` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1529`
#### `loadPicture(self)`
> Load a .png file and displays it
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1573`
#### `updatePicturesList(self)`
> Qt slot `updatePicturesList` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1600`
#### `syncToCloud(self)`
> Qt slot `syncToCloud` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1614`
#### `syncToCloudFinished(self)`
> Qt slot `syncToCloudFinished` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1638`
#### `openSettings(self)`
> Qt slot `openSettings` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1646`
#### `loadPreferences(self)`
> Qt slot `loadPreferences` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1651`
#### `setMetadataFields(self, metadata_fields)`
> Qt slot `setMetadataFields` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/experiment_launcher.py:1671`
#### `buttonSetMainPathClicked(self)`
> Qt slot `buttonSetMainPathClicked` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1675`
#### `setMainPath(self, mainPath=None)`
> Qt slot `setMainPath` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/gui/experiment_launcher.py:1678`
#### `get_text_position(self, text, text_widget)`
> Qt slot `get_text_position` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1695`
#### `jsonItemClicked(self, item, column)`
> Qt slot `jsonItemClicked` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1707`
### AddExperimentDialog(QDialog)
> Dialog that collects experiment name, type, config file, smart-plug, and metadata fields.
**Source:** `braindance/gui/experiment_launcher.py:1728`
**Kind:** class. **Instantiated by:** braindance/gui/experiment_launcher.py:1530 (named-call hint)
**Constructor:** `__init__(self, parent=None, metadata_fields=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `buttons` | inferred at runtime | `QDialogButtonBox(QDialogButtonBox.Ok \| QDialogButtonBox.Cancel, self)` |
| `configButton` | inferred at runtime | `QPushButton('Select Config File', self)` |
| `configFile` | inferred at runtime | `None` |
| `formLayout` | inferred at runtime | `QFormLayout()` |
| `metadataInputs` | inferred at runtime | `{}` |
| `metadata_fields` | inferred at runtime | `metadata_fields or []` |
| `nameEdit` | inferred at runtime | `QLineEdit(self)` |
| `parent` | inferred at runtime | `parent` |
| `smartplugEdit` | inferred at runtime | `QLineEdit(self)` |
| `typeEdit` | inferred at runtime | `QLineEdit(self)` |
**Methods:**
#### `getInputs(self)`
> Qt slot `getInputs` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1791`
#### `getFieldValue(self, field)`
> Qt slot `getFieldValue` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1801`
#### `buttonSetConfigFileClicked(self)`
> Qt slot `buttonSetConfigFileClicked` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1812`
#### `addCategory(self, field_name, combo)`
> Qt slot `addCategory` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1817`
#### `updateMetadataFieldOptions(self, field_name, new_option)`
> Qt slot `updateMetadataFieldOptions` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/gui/experiment_launcher.py:1824`
#### `closeEvent(self, event)`
> Qt slot `closeEvent` updates the launcher state or UI for its named action.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1834`

## Functions
### `fill_item(item, value)`
> Recursively populates a QTreeWidget item from nested JSON values.
> **Called by:** braindance/gui/experiment_launcher.py:1849 (named-call hint); braindance/gui/experiment_launcher.py:1856 (named-call hint); braindance/gui/experiment_launcher.py:1859 (named-call hint); braindance/gui/experiment_launcher.py:1875 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1842`
### `fill_widget(widget, value)`
> Renders a JSON value into the appropriate Qt editor widget.
> **Called by:** braindance/gui/experiment_launcher.py:1506 (named-call hint); braindance/gui/experiment_launcher.py:920 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1873`
### `main()`
> Creates the QApplication, MainWindow, and Qt event loop.
> **Called by:** braindance/gui/experiment_launcher.py:1885 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/gui/experiment_launcher.py:1877`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| ~/.braindance/preferences.txt stores project root; preferences.json stores metadata fields |
| exp_dict and analysis_dict map menu entries to proj.* shell commands |
| Cloud destination s3://braingeneersdev/asrobbin/<project>/<chip>/<experiment>/ at s3-west.nrp-nautilus.io |

## Data Shapes
- Experiment JSON keys include name,config,stim_electrodes,motor_electrodes,sensory_electrodes,mapping_file_path,py_file_path,plug
- Mapping CSV with electrode,channel,x,y and optional mean_amp/spike_count

## Notes
- Imports set global Matplotlib colors and create ~/.braindance.
- Launch commands depend on external proj.* modules and unquoted shell paths; CommandRunner is POSIX-specific.
- Existing footprint.npy branch is unfinished; metadata_args are built but omitted from creation command.
