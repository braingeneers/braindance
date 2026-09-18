import sys
import os
import json
import numpy as np
import shutil

from braindance.analysis.data_loader import AnalysisDAO
from braindance.analysis import plot_helper
from braindance.core.maxwell.maxwell_utils import get_maxwell_status
from braindance.core.utils import SmartPlug

from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QSplitter
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QTabWidget, QPushButton
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QTreeWidget, QTreeWidgetItem
from PyQt5.QtWidgets import QRadioButton, QCheckBox
from PyQt5.QtWidgets import QDialog, QAction
from PyQt5.QtWidgets import  QTextEdit, QSizePolicy
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QImage, QTextCursor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


from matplotlib.figure import Figure

from braindance.gui.command_runner import CommandRunner

home_dir = Path.home()
braindance_dir = home_dir / '.braindance'  # Example: '.experiment_manager'
# Create the config directory if it doesn't exist
os.makedirs(braindance_dir, exist_ok=True)

preferences_file = braindance_dir / 'preferences.txt'
preferences_json = braindance_dir / 'preferences.json'

script_dir = Path(__file__).parent
resources_dir = script_dir / 'resources'


# Plotting aesthetics
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['figure.facecolor'] = '#2c3e50' 
# legend color black
plt.rcParams['legend.facecolor'] = 'black'
# plt.rcParams['axes.facecolor'] = '#2c3e50'


# name : (command, [requirements])
exp_dict = {
    'Recording': ('python -m proj.cartpole_v1.neural_config_analysis', ['config']),
    'Causal': ('python -m proj.cartpole_v1.causal_analysis', ['config', 'stim_electrodes']),
    'Cartpole': ('python -m proj.cartpole_v1.cartpole', ['config', 'stim_electrodes', 'motor_electrodes', 'sensory_electrodes']),
    'Cartpole Scheduled': ('python -m proj.cartpole_v1.cartpole_long', ['config', 'stim_electrodes', 'motor_electrodes', 'sensory_electrodes']),
    'Busy Bee': ('python -m proj.busy_bee.continuous', ['config', 'stim_electrodes']),
    'Rank pairs': ('python -m proj.cartpole_v2.ranked_pairs', ['config', 'stim_electrodes']),
    'Cartpole-Force Train': ('python -m proj.cartpole_v1.cartpole_force_train', ['config', 'stim_electrodes', 'motor_electrodes', 'sensory_electrodes']),
    'Ant': ('python -m proj.robot_control.ant_test', ['config','stim_electrodes','motor_electrodes']),
    'Test': ('python -m proj.cartpole_v1.test', []),
    'Cartpole full': ('python -m proj.cartpole_v2.full_cartpole', ['config']),
    'Reward Check': ('python -m proj.analysis.reward_peek', ['config', 'stim_electrodes', 'motor_electrodes', 'sensory_electrodes']),
    'OLD Reward Check': ('python -m proj.cartpole_v1.reward_peek', ['config', 'stim_electrodes', 'motor_electrodes', 'sensory_electrodes']),
    'Food Land': ('python -m proj.for_david.food_land_experiment', ['config', 'stim_electrodes', 'motor_electrodes', 'sensory_electrodes']),
    'Observe Motor': ('python -m proj.cartpole_v1.observe_motor', ['config', 'motor_electrodes']),
}

analysis_dict = {
    'Recording Metrics': ('python -m proj.cartpole_v1.recording_metrics', ['Recording']),
    'Causal Analysis': ('python -m proj.cartpole_v1.do_causal_analysis', ['Causal Reactivity']),
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "BrainDance Experiment Launcher"
        self.width = 1000
        self.height = 800

        self.mainPath = None
        self.selected_project = None
        self.selected_chip = None
        self.json_data = None
        self.mapping = None
        self.analysis_obj = None
        self.experiment_running = False
        self.analysis_running = False

        self.footprint_chs = None
        self.footprint_waves = None

        self.scat_selected_electrodes = None
        self.scat_stim_electrodes = None
        self.smartplug = None
        self.smartPlugLabel = None
        self.metadata_fields = None




        self.initUI()

    def initMenuBar(self):
        self.menubar = self.menuBar()
        settingsMenu = self.menubar.addMenu('Settings')

        preferencesAction = QAction('Preferences', self)
        preferencesAction.triggered.connect(self.openSettings)
        settingsMenu.addAction(preferencesAction)
        # Change color of background of menu to grey
        settingsMenu.setStyleSheet("background-color: grey; color: white;")
        # Change color of text to white
        

    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, self.width, self.height)
        self.setStyleSheet("background-color: #2c3e50; color: white;")

        # Set Application Icon
        self.setWindowIcon(QIcon(str(resources_dir / 'brain.png')))

        # Central Widget with Vertical Layout
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        layout = QVBoxLayout(centralWidget)
        layout.setAlignment(Qt.AlignTop)

        self.cloudSyncButton = QPushButton(self)
        pixmap = QPixmap(str(resources_dir / 'cloud-upload.png'))
        self.cloudSyncButton.setIcon(QIcon(pixmap))
        # Hover text
        self.cloudSyncButton.setToolTip("Sync current experiment to the cloud.")
        self.cloudSyncButton.clicked.connect(self.syncToCloud)
        layout.addWidget(self.cloudSyncButton, alignment=Qt.AlignRight)

        

        #

        # self.settingsButton = QPushButton(self)
        # pixmap = QPixmap(str(resources_dir / 'settings.png'))
        # self.settingsButton.setIcon(QIcon(pixmap))
        # # Hover text
        # self.settingsButton.setToolTip("Settings")
        # self.settingsButton.clicked.connect(self.openSettings)
        # layout.addWidget(self.settingsButton, alignment=Qt.AlignRight)


        self.initMenuBar()


        # Logo
        logo = QLabel(centralWidget)
        pixmap = QPixmap(str(resources_dir / 'brain.png'))
        logo.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
        # Make circle
        
        

        layout.addWidget(logo, alignment=Qt.AlignCenter)
        

        # Welcome Label
        # label = QLabel("Welcome to the BrainDance Experimenter!", centralWidget)
        # label.setAlignment(Qt.AlignCenter)
        # label.setStyleSheet("font-size: 18px; margin: 10px;")
        # layout.addWidget(label)

        # horizontal layout for the selections
        hlayout = QHBoxLayout()
        hlayout.setAlignment(Qt.AlignCenter)
        layout.addLayout(hlayout)

        # Add button to change main path, icon is a folder
        self.changeMainPathButton = QPushButton()
        pixmap = QPixmap(str(resources_dir / 'folder.png'))
        self.changeMainPathButton.setIcon(QIcon(pixmap))
        # attach function
        self.changeMainPathButton.clicked.connect(self.buttonSetMainPathClicked)
        hlayout.addWidget(self.changeMainPathButton)



        # ~~~~~~~ Status ~~~~~~~~~~

        self.statusBar().showMessage("Welcome. Please select or create a project to begin.")
        # Add a circle in the status bar as a status color indicator
        self.statusCircle = QLabel()
        self.statusCircle.setFixedSize(10, 10)
        # If dummy, make color grey
     # Add refresh button which calls get_hardware_status
        #smartplugon button
        self.smartPlugLabel = QLabel("None :(", self)
        self.smartPlugOnButton = QPushButton("On", self)
        # Set width to be small
        self.smartPlugOnButton.setFixedWidth(50)
        self.smartPlugOnButton.clicked.connect(self.smartPlugOn)
        self.smartPlugOffButton = QPushButton("Off", self)
        self.smartPlugOffButton.setFixedWidth(50)
        self.smartPlugOffButton.clicked.connect(self.smartPlugOff)

        self.refreshButton = QPushButton("Refresh", self)
        self.refreshButton.clicked.connect(self.get_hardware_status)
        self.get_hardware_status()

        self.statusBar().addPermanentWidget(self.smartPlugLabel)
        self.statusBar().addPermanentWidget(self.smartPlugOnButton)
        self.statusBar().addPermanentWidget(self.smartPlugOffButton)
        self.statusBar().addPermanentWidget(self.refreshButton)
        self.statusBar().addPermanentWidget(self.statusCircle)
        

        # ~~~~~~~ Project, Chip, Experiment Selection ~~~~~~~~~~

        # Label for Project Selection
        label = QLabel("Project:", centralWidget)
        label.setAlignment(Qt.AlignLeft)
        label.setStyleSheet("font-size: 14px; margin: 10px;")
        hlayout.addWidget(label)

        # Dropdown for Project Selection
        self.projectSelector = QComboBox(self)
        self.projectSelector.currentIndexChanged.connect(self.onProjectSelect)
        hlayout.addWidget(self.projectSelector)

        # Button to add a new project
        addProjectButton = QPushButton()
        pixmap = QPixmap(str(resources_dir / 'plus.png'))
        addProjectButton.setIcon(QIcon(pixmap))
        # attach function
        addProjectButton.clicked.connect(self.addProject)
        hlayout.addWidget(addProjectButton, alignment=Qt.AlignCenter)



        # Label for Chip Selection
        label = QLabel("Chip:", centralWidget)
        label.setAlignment(Qt.AlignLeft)
        label.setStyleSheet("font-size: 14px; margin: 10px;")
        hlayout.addWidget(label)

        # Dropdown for Chip Selection
        self.chipSelector = QComboBox(self)
        self.chipSelector.currentIndexChanged.connect(self.onChipSelect)
        hlayout.addWidget(self.chipSelector)

        # Button to add a new chip
        addChipButton = QPushButton()
        pixmap = QPixmap(str(resources_dir / 'plus.png'))
        addChipButton.setIcon(QIcon(pixmap))
        # attach function
        addChipButton.clicked.connect(self.addChip)
        hlayout.addWidget(addChipButton, alignment=Qt.AlignCenter)


        # Label for Experiment Selection
        label = QLabel("Experiment:", centralWidget)
        label.setAlignment(Qt.AlignLeft)
        label.setStyleSheet("font-size: 14px; margin: 10px;")
        hlayout.addWidget(label)

        # Dropdown for Experiment Selection
        self.experimentSelector = QComboBox(self)
        self.experimentSelector.currentIndexChanged.connect(self.onExperimentSelect)
        hlayout.addWidget(self.experimentSelector)

        # Button to add a new experiment
        addExperimentButton = QPushButton()
        pixmap = QPixmap(str(resources_dir / 'plus.png'))
        addExperimentButton.setIcon(QIcon(pixmap))
        # attach function
        addExperimentButton.clicked.connect(self.addExperiment)
        hlayout.addWidget(addExperimentButton, alignment=Qt.AlignCenter)


        # Make tabs for the experiment
        self.tabWidget = QTabWidget(centralWidget)
        layout.addWidget(self.tabWidget)

        

        exp_params_splitter = QSplitter(Qt.Vertical)
        exp_tab_widget = QWidget()
        # exp_layout.setAlignment(Qt.AlignTop)
        self.expWidget = QTreeWidget(centralWidget, )
        self.expWidget.itemClicked.connect(self.jsonItemClicked)
        self.expWidget.setHeaderLabels([""])
        # self.expWidget.setStyleSheet("background-color: #2c3e50; color: white;")
        # exp_layout.addWidget(self.expWidget)

        self.expRawText = QTextEdit('', centralWidget)
        self.expRawText.setStyleSheet("background-color: lightgrey; color: black;")
        expParamsUpdateButtonLayout = QHBoxLayout()
        self.expParamsUpdateButton = QPushButton("Update", self.expWidget)
        self.expParamsUpdateButton.setFixedWidth(100)
        self.expParamsUpdateButton.setStyleSheet("background-color: green; color: white;")
        self.expParamsUpdateButton.clicked.connect(self.updateExpParams)
        expParamsUpdateButtonLayout.addWidget(self.expParamsUpdateButton)
        # self.expRawText.setReadOnly(True)  # Make it read-only

        # add text window and update button to the layout

        exp_params_splitter.addWidget(self.expWidget)
        exp_params_splitter.addWidget(self.expRawText)
        # Init splitter sizes
        exp_params_splitter.setSizes([200, 100])
        exp_tab_widget.layout = QVBoxLayout()
        exp_tab_widget.layout.addWidget(exp_params_splitter)
        exp_tab_widget.layout.addWidget(self.expParamsUpdateButton)
        exp_tab_widget.setLayout(exp_tab_widget.layout)









        pictureLayout = QHBoxLayout()

        # Make pictures list
        self.picturesList = QComboBox(self)
        self.picturesList.currentIndexChanged.connect(self.loadPicture)
        self.picturesList.setStyleSheet("background-color: #2c3e50; color: white;")
        # Make 1/4 of the width
        self.picturesList.setFixedWidth(self.width // 4)
        self.picturesList.setContentsMargins(0, 0, 0, 0)
        pictureLayout.addWidget(self.picturesList, stretch=2)


        # Make picture layout and widget
        # Widget to display the picture
        self.pictureWidget = QLabel()
        # Add image to the widget
        self.pictureWidget.setPixmap(QPixmap())
        self.pictureWidget.setScaledContents(True)
        # self.pictureWidget.setFixedSize(400, 400)
        self.pictureWidget.setContentsMargins(0, 0, 0, 0)
        pictureLayout.addWidget(self.pictureWidget)
        
        # add to layout

        
        
        # Make resizeable
        # self.pictureWidget.setScaledContents(True)
        # self.pictureWidget.setContentsMargins(0, 0, 0, 0)

        
        
        
        #keep aspect ratio
        
        # self.pictureWidget.setFixedSize(400, 400)
        # self.pictureWidget.resize(698,402)
        # Make high quality
        


        

        # Make tabs
        pictureTab = QWidget(centralWidget)
        self.configTab = QWidget()

        self.launchTab = QWidget()
        self.analysisTab = QWidget()

        pictureTab.setLayout(pictureLayout)
        # Tabs
        self.tabWidget.addTab(self.launchTab, "Launch")
        self.tabWidget.addTab(exp_tab_widget, "Experiment Params")
        self.tabWidget.addTab(self.configTab, "Config")
        self.tabWidget.addTab(self.analysisTab, "Analysis")
        self.tabWidget.addTab(pictureTab, "Pictures")
        
        

        # Launch tab layout
        self.setupLaunchTab()

        self.setupAnalysisTab()


        
        configLayout = QVBoxLayout(self.configTab)
        configLayout.setAlignment(Qt.AlignTop)
        configLayout.setContentsMargins(0, 0, 0, 0)


        # Plotting in the config tab
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        # make canvas expandable
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Buttons to change plots
        buttonLayout = QHBoxLayout()

        self.meanAmpButton = QRadioButton("Mean Amp", self)
        self.meanAmpButton.setChecked(True)
        self.meanAmpButton.toggled.connect(lambda: self.plotScatter(color_mode='mean_amp'))
        buttonLayout.addWidget(self.meanAmpButton)

        self.spikeCountButton = QRadioButton("Spike Count", self)
        self.spikeCountButton.toggled.connect(lambda: self.plotScatter(color_mode='spike_count'))
        buttonLayout.addWidget(self.spikeCountButton)

        self.scatSelectedCheckbox = QCheckBox("Selected Electrodes", self)
        self.scatSelectedCheckbox.setChecked(True)
        self.scatSelectedCheckbox.toggled.connect(self.scatSelectedToggle)
        buttonLayout.addWidget(self.scatSelectedCheckbox)

        self.scatStimCheckbox = QCheckBox("Stim Electrodes", self)
        self.scatStimCheckbox.setChecked(True)
        self.scatStimCheckbox.toggled.connect(self.scatStimToggle)
        buttonLayout.addWidget(self.scatStimCheckbox)




        self.getFootprintButton = QPushButton("Get Footprint", self)
        self.getFootprintButton.setStyleSheet("background-color: green;")
        # set size
        self.getFootprintButton.setFixedWidth(100)
        self.getFootprintButton.clicked.connect(self.getFootprint)
        buttonLayout.addWidget(self.getFootprintButton)



        configLayout.addLayout(buttonLayout)
        configLayout.addWidget(self.toolbar)
        configLayout.addWidget(self.canvas)

        self.configTab.layout = configLayout
        self.configTab.layout.addWidget(self.toolbar)
        self.configTab.layout.addWidget(self.canvas)
        


        self.loadPreferences()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardware ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

    def smartPlugOn(self):
        if self.smartplug is None:
            self.statusBar().showMessage("No smart plug found.")
            return
        self.smartplug.turn_on()
        self.statusBar().showMessage("Smart Plug On")

    def smartPlugOff(self):
        if self.smartplug is None:
            self.statusBar().showMessage("No smart plug found.")
            return
        self.smartplug.turn_off()
        self.statusBar().showMessage("Smart Plug Off")

    
    def get_hardware_status(self):
        """Get the hardware status from the maxlab library
        """
        try:
            import maxlab
            self.dummy_maxlab = False
            # Get the hardware status
            status = get_maxwell_status()
            if status:
                self.statusCircle.setStyleSheet("background-color: green; border-radius: 5px;")
            else:
                self.statusCircle.setStyleSheet("background-color: yellow; border-radius: 5px;")
        except:
            self.dummy_maxlab = True
            self.statusCircle.setStyleSheet("background-color: grey; border-radius: 5px;")
            self.statusBar().showMessage("Maxlab library not found. Please install maxlab.")
            return


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Launch Tab ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def setupLaunchTab(self):
        launchLayout = QVBoxLayout(self.launchTab)
        buttonLayout = QHBoxLayout()

        # Experiment Launch Dropdown
        
        # for exp_name, exp_command in exp_dict.items():
        self.experimentLaunchSelector = QComboBox(self.launchTab)
        self.experimentLaunchSelector.addItems(exp_dict.keys())
        self.experimentLaunchSelector.currentIndexChanged.connect(self.onLaunchExperimentChanged)
        buttonLayout.addWidget(self.experimentLaunchSelector)

            


        self.launchExperimentButton = QPushButton("Neural Config Recording", self.launchTab)
        self.launchExperimentButton.setStyleSheet("background-color: green; color: white;")
        self.launchExperimentButton.setFixedWidth(200)
        self.launchExperimentButton.clicked.connect(self.launchExperiment)
        buttonLayout.addWidget(self.launchExperimentButton)

        

        self.emergencyStopButton = QPushButton("Emergency Stop", self.launchTab)
        # start disabled
        self.emergencyStopButton.setEnabled(False)
        self.emergencyStopButton.setStyleSheet("background-color: grey; color: white;")
        self.emergencyStopButton.clicked.connect(self.emergencyStop)
        self.emergencyStopButton.setFixedWidth(200)

        
        buttonLayout.addWidget(self.emergencyStopButton)

        # Parameter inputs
        parameterLayout = QHBoxLayout()
        stim_electrodes_label = QLabel("Stim Electrodes:", self.launchTab)
        # itemized
        self.stim_electrodes = QLineEdit(self.launchTab)
        self.stim_electrodes.setPlaceholderText("Enter electrodes separated by commas")
        # self.stim_electrodes.setFixedWidth(200)
        self.stim_electrodes.setFixedHeight(30)
        self.stim_electrodes.setStyleSheet("background-color: #2c3e50; color: white;")
        self.validateStimElectrodesButton = QPushButton("Validate", self.launchTab)
        self.validateStimElectrodesButton.setStyleSheet("background-color: green; color: white;")
        self.validateStimElectrodesButton.clicked.connect(self.validateStimElectrodes)
        self.validateStimElectrodesButton.setFixedWidth(100)

        self.cleanStimElectrodesButton = QPushButton("Clean", self.launchTab)
        self.cleanStimElectrodesButton.setStyleSheet("background-color: blue; color: white;")
        # Use validateStimElectrodes with remove_duplicates=True
        self.cleanStimElectrodesButton.clicked.connect(lambda: self.validateStimElectrodes(remove_duplicates=True))
        self.cleanStimElectrodesButton.setFixedWidth(100)

        # Try importing maxlab, if it fails, disable the validate button
        try:
            import maxlab
            self.dummy_maxlab = False
        except:
            self.dummy_maxlab = True
            self.validateStimElectrodesButton.setEnabled(False)
            self.validateStimElectrodesButton.setStyleSheet("background-color: grey; color: white;")
            self.cleanStimElectrodesButton.setEnabled(False)
            self.cleanStimElectrodesButton.setStyleSheet("background-color: grey; color: white;")

        self.updateStimElectrodesButton = QPushButton("Update", self.launchTab)
        self.updateStimElectrodesButton.setStyleSheet("background-color: green; color: white;")
        self.updateStimElectrodesButton.clicked.connect(self.updateStimElectrodes)
        self.updateStimElectrodesButton.setFixedWidth(100)

        parameterLayout.addWidget(stim_electrodes_label)
        parameterLayout.addWidget(self.stim_electrodes)
        parameterLayout.addWidget(self.validateStimElectrodesButton)
        parameterLayout.addWidget(self.cleanStimElectrodesButton)
        parameterLayout.addWidget(self.updateStimElectrodesButton)


        # Build the layout
        launchLayout.addLayout(buttonLayout)
        launchLayout.addLayout(parameterLayout)

        # change size
        

        # Splitter for text
        splitter = QSplitter(Qt.Vertical)
        # Create a tab widget for requirements and notes
        tabWidget = QTabWidget(self.launchTab)
        
        # Requirements tab
        requirementsTab = QWidget()
        requirementsLayout = QVBoxLayout(requirementsTab)
        requirementLabel = QLabel("Requirements:", requirementsTab)
        self.requirementText = QTextEdit(requirementsTab)
        self.requirementText.setReadOnly(True)
        self.requirementText.setStyleSheet("background-color: grey;")
        requirementsLayout.addWidget(requirementLabel)
        requirementsLayout.addWidget(self.requirementText)
        tabWidget.addTab(requirementsTab, "Requirements")
        
        # Notes tab
        notesTab = QWidget()
        notesLayout = QVBoxLayout(notesTab)
        notesLabel = QLabel("Notes:", notesTab)
        self.notesText = QTextEdit(notesTab)
        # On change, update the notes
        self.notesText.textChanged.connect(self.updateNotes)
        notesLayout.addWidget(notesLabel)
        notesLayout.addWidget(self.notesText)
        tabWidget.addTab(notesTab, "Notes")
        
        launchLayout.addWidget(tabWidget)
        
        blankWidget = QWidget()
        outputLayout = QVBoxLayout()
        # Text Window to display output
        self.outputText = QTextEdit(self.launchTab)
        self.outputText.setReadOnly(True)  # Make it read-only
        # Clear button
        self.clearButton = QPushButton("Clear", self.launchTab)
        self.clearButton.clicked.connect(lambda: self.outputText.clear())
        self.clearButton.setFixedWidth(100)
        self.clearButton.setStyleSheet("background-color: blue; color: white;")
        outputLayout.addWidget(self.outputText)
        outputLayout.addWidget(self.clearButton)

        blankWidget.setLayout(outputLayout)
        splitter.addWidget(tabWidget)
        splitter.addWidget(blankWidget)
        splitter.setSizes([50, 200])

        launchLayout.addWidget(splitter)

        # Check dependencies
        self.checkLaunchButtonDependencies()


    def checkLaunchButtonDependencies(self):
        if self.experiment_running:
            # Enable emergency stop button
            self.emergencyStopButton.setEnabled(True)
            self.emergencyStopButton.setStyleSheet("background-color: red; color: white;")
            
            # Disable other buttons
            self.launchExperimentButton.setEnabled(False)
            self.launchExperimentButton.setStyleSheet("background-color: grey; color: white;")
            return
        else:
            self.emergencyStopButton.setEnabled(False)
            self.emergencyStopButton.setStyleSheet("background-color: grey; color: white;")
        
        # Enable/disable buttons based on self.json_data
        if self.json_data is not None:  # Assuming self.json_data is a dictionary
            # Example condition - adjust according to your actual data and requirements
            # Update stim electrodes
            stim_electrodes = self.json_data.get('stim_electrodes', None)
            if stim_electrodes is not None:
                stim_electrodes_str = ','.join([str(e) for e in stim_electrodes])
                self.stim_electrodes.setText(stim_electrodes_str)
            else:
                self.stim_electrodes.setText('None')

            self.launchExperimentButton.setEnabled(True)
            self.launchExperimentButton.setStyleSheet("background-color: green; color: white;")

        else:
            self.launchExperimentButton.setEnabled(False)
            self.launchExperimentButton.setStyleSheet("background-color: grey; color: white;")

    def onLaunchExperimentChanged(self, index):
        """Update the button, what it says, and what it launches. Also update the text box which shows the 
        parameters and the requirements
        """
        # get the name of the experiment
        exp_name = self.experimentLaunchSelector.itemText(index)
        # get the command and requirements
        command, requirements = exp_dict[exp_name]
        # update the button
        self.launchExperimentButton.setText(exp_name)
        # Check if all requirements are met
        missing_requirements = self.checkExperimentRequirements(requirements)
        # update the text box
        # self.outputText.setText(f"Command: {command} -j [path]\nRequirements: {requirements}")
        # self.outputText.setText(f"Command: {command}\n")
        self.requirementText.clear()
        self.updateRequirementText(f"Command: {command}\n")
        # Append requirements with missing requirements in red
        for req in requirements:
            if req in missing_requirements:
                # light red
                self.updateRequirementText(f"Missing requirement: {req}", color=QColor.fromRgb(240, 128, 128))
            else:
                self.updateRequirementText(f"Requirement: {req}", color=QColor.fromRgb(128, 240, 128))



    def launchExperiment(self):
        # Launches the chosen experiment
        if self.experiment_running:
            self.statusBar().showMessage("Experiment already running.")
            return
        
        if self.json_data is None:
            self.statusBar().showMessage("No experiment selected.")
            return
        
        # get the name of the experiment
        exp_name = self.experimentLaunchSelector.currentText()
        # get the command and requirements
        command, requirements = exp_dict[exp_name]
        # check if all requirements are met
        missing_requirements = self.checkExperimentRequirements(requirements)
        if len(missing_requirements) > 0:
            self.updateOutputText(f"Missing requirements: {missing_requirements}")
            self.statusBar().showMessage(f"Missing requirements: {missing_requirements}")
            return
        
        experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, self.json_data['name'] + ".json")
        if exp_name == 'Reward Check':
            command += f" -d {self.mainPath + '/'} --chip {self.selected_chip} --proj {self.selected_project} --exp {self.json_data['name']}"
        elif exp_name == 'OLD Reward Check':
            command += f" -d {experimentPath.split('.')[0]}"
        else:
            command += f" -j {experimentPath}"
        self.commandRunner = CommandRunner(command)
        self.commandRunner.output.connect(self.updateOutputText)
        self.commandRunner.error.connect(self.handleExperimentError)
        self.commandRunner.finished.connect(self.handleExperimentCompleted)
        

        self.experiment_running = True
        self.checkLaunchButtonDependencies()
        # Lock the experiment selection
        self.experimentLaunchSelector.setEnabled(False)
        self.statusBar().showMessage("Experiment started.")
        self.outputText.append("Experiment started.")
        self.outputText.append(f"Command: {command}")

        self.commandRunner.start()
        

    def checkExperimentRequirements(self, requirements = []):
        """Checks if the experiment has the requirements
        Returns a list of missing requirements
        """
        if self.json_data is None:
            self.statusBar().showMessage("No experiment selected.")
            return requirements
        missing_requirements = []
        for req in requirements:
            req_val = self.json_data.get(req, None)
            if type(req_val) == list:
                if len(req_val) == 0:
                    missing_requirements.append(req)

            if type(req_val) == str:
                if req_val == '':
                    missing_requirements.append(req)
            if req_val is None:
                missing_requirements.append(req)
        return missing_requirements

    def validateStimElectrodes(self, remove_duplicates = False):
        """Use validation script bdquery to validate the stim electrodes
        """
        # Get the stim electrodes
        stim_electrodes = self.stim_electrodes.text()
        # Check that it is not empty, and is a list of integers separated by commas
        if stim_electrodes == '':
            self.statusBar().showMessage("Stim electrodes is empty.")
            return
        try:
            #remove spaces
            stim_electrodes = [int(e.strip()) for e in stim_electrodes.split(',')]
        except:
            self.updateOutputText("Error parsing stim electrodes, please use comma separated integers", color=QColor.fromRgb(240, 128, 128))
            self.statusBar().showMessage("Error parsing stim electrodes, please use comma separated integers")
            return
        # Check if text is different from the json data
        if self.json_data is not None:
            if self.json_data.get('stim_electrodes', None) != stim_electrodes:
                # Prompt the user to update the stim electrodes
                reply = QMessageBox.question(self, 'Message', "Stim electrodes are different from the json data. Update?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.updateStimElectrodes()
                elif reply == QMessageBox.No:
                    stim_electrodes = self.json_data.get('stim_electrodes', None)
                    self.stim_electrodes.setText(','.join([str(e) for e in stim_electrodes]))
                elif reply == QMessageBox.Cancel:
                    self.statusBar().showMessage("Cancelled Validate Stim Electrodes")
                    return
        # Run bdquery
        stim_electrodes_str = ' '.join([str(e) for e in stim_electrodes])
        json_path = experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, self.json_data['name'] + ".json")
        if remove_duplicates:
            command = f"bdquery -j {json_path} -r"
        else:
            command = f"bdquery -j {json_path}"
        self.commandRunner = CommandRunner(command)
        self.commandRunner.output.connect(self.updateOutputText)
        self.commandRunner.error.connect(self.handleExperimentError)
        self.commandRunner.finished.connect(self.handleExperimentCompleted)
        self.commandRunner.start()
        # After its done, if remove_duplicates, update the stim electrodes
        if remove_duplicates:
            # reload stim electrodes from the json file
            self.onExperimentSelect(self.experimentSelector.currentIndex())

    def updateStimElectrodes(self):
        """Updates stim electrodes in the json data from the text box if it is different from the json data
        """
        # Get the stim electrodes
        stim_electrodes = self.stim_electrodes.text()
        # Check that it is not empty, and is a list of integers separated by commas
        if stim_electrodes == '':
            self.statusBar().showMessage("Stim electrodes is empty.")
            return
        try:
            #remove spaces
            stim_electrodes = [int(e.strip()) for e in stim_electrodes.split(',')]
        except:
            self.updateOutputText("Error parsing stim electrodes, please use comma separated integers", color=QColor.fromRgb(240, 128, 128))
            self.statusBar().showMessage("Error parsing stim electrodes, please use comma separated integers")
            return
        # Check if text is different from the json data
        if self.json_data is not None:
            if self.json_data.get('stim_electrodes', None) != stim_electrodes:
                
                self.json_data['stim_electrodes'] = stim_electrodes
                # Update the json file
                experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, self.json_data['name'] + ".json")
                json.dump(self.json_data, open(experimentPath, 'w'), indent=4)
                
                # Select experiment to update the info throughout the gui
                self.onExperimentSelect(self.experimentSelector.currentIndex())

                self.statusBar().showMessage("Updated stim electrodes.")
                self.outputText.append("Updated stim electrodes.")
                
        else:
            self.statusBar().showMessage("No experiment selected.")
            return

    def updateExpParams(self):
        """Save the text in the expRawText box to the json file
        """
        if self.json_data is None:
            self.statusBar().showMessage("No experiment selected.")
            return
        # Get the text
        text = self.expRawText.toPlainText()
        # Parse the text
        try:
            json_data = json.loads(text)
        except:
            self.statusBar().showMessage("Error parsing json data")
            return
        # Update the json file
        experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, self.json_data['name'] + ".json")
        json.dump(json_data, open(experimentPath, 'w'), indent=4)
        # Select experiment to update the info throughout the gui
        self.onExperimentSelect(self.experimentSelector.currentIndex())
        self.statusBar().showMessage("Updated experiment parameters.")
        self.outputText.append("Updated experiment parameters.")

    def handleExperimentError(self, error_message):
        self.experiment_running = False
        self.checkLaunchButtonDependencies()
        self.experimentLaunchSelector.setEnabled(True)
        self.statusBar().showMessage("Experiment error.")
        # self.outputText.append(f"Error: {error_message}")
        self.updateOutputText(f"Error: {error_message}", color=QColor.fromRgb(240, 128, 128))

    def handleExperimentCompleted(self):
        self.experiment_running = False
        self.checkLaunchButtonDependencies()
        self.experimentLaunchSelector.setEnabled(True)
        self.statusBar().showMessage(f"{self.experimentLaunchSelector.currentText()} completed.")
        self.outputText.append(f"{self.experimentLaunchSelector.currentText()} completed.")
        # Select experiment to update the plots
        self.onExperimentSelect(self.experimentSelector.currentIndex(),clear=False)




        # if self.json_data:
        #     experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, self.json_data['name'] + ".json")
        #     command = f"python -m proj.cartpole_v1.neural_config_analysis -j {experimentPath}"
        #     self.commandRunner = CommandRunner(command)
        #     self.commandRunner.output.connect(self.updateOutputText)
        #     self.commandRunner.start()
        #     self.experiment_running = True
        #     self.checkLaunchButtonDependencies()

    def updateRequirementText(self, text, color=None):
        """Update the text box with the requirements
        """
        # clear
        if color is not None:
            self.requirementText.setTextColor(color)
        self.requirementText.append(text)
        self.requirementText.setTextColor(QColor("white"))
    
    # update, optional color
    def updateOutputText(self, text, color=None):
        if color is not None:
            self.outputText.setTextColor(color)
        self.outputText.append(text)
        self.outputText.setTextColor(QColor("white"))

    def updateNotes(self):
        """Update the notes in the json file
        """
        if self.json_data is None:
            self.statusBar().showMessage("No experiment selected.")
            return
        # Get the text
        text = self.notesText.toPlainText()
        # Update the json file
        # experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, self.json_data['name'] + ".json")
        self.json_data['notes'] = text
        fill_widget(self.expWidget, self.json_data)
        # close all items
        self.expWidget.collapseAll()
        self.expRawText.setPlainText(json.dumps(self.json_data, indent=4))
        # print("Updated notes with:", text)


    def emergencyStop(self):
        # This is just an example - adjust the logic based on how you want to stop the experiment
        if hasattr(self, 'commandRunner') and self.commandRunner.isRunning():
            self.commandRunner.stop()
            self.experiment_running = False
            self.checkLaunchButtonDependencies()
            self.outputText.append("Experiment stopped.")
            self.statusBar().showMessage("Experiment stopped.")

            self.experimentLaunchSelector.setEnabled(True)
            # Select experiment to update the plots
            self.onExperimentSelect(self.experimentSelector.currentIndex(),clear=False)
        else:
            self.outputText.append("No experiment running.")
            self.statusBar().showMessage("No experiment running.")
            self.experiment_running = False
            self.checkLaunchButtonDependencies()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Analysis Tab ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setupAnalysisTab(self):
        analysisLayout = QVBoxLayout(self.analysisTab)
        buttonLayout = QHBoxLayout()

        # Experiment Launch Dropdown
        
        # for exp_name, exp_command in exp_dict.items():
        self.analysisSelector = QComboBox(self.analysisTab)
        self.analysisSelector.addItems(analysis_dict.keys())
        self.analysisSelector.currentIndexChanged.connect(self.onAnalysisChanged)
        buttonLayout.addWidget(self.analysisSelector)

            


        self.launchAnalysisButton = QPushButton("Launch Analysis", self.analysisTab)
        self.launchAnalysisButton.setStyleSheet("background-color: green; color: white;")
        self.launchAnalysisButton.setFixedWidth(200)
        self.launchAnalysisButton.clicked.connect(self.launchAnalysis)
        buttonLayout.addWidget(self.launchAnalysisButton)

        

        self.emergencyStopAnalysisButton = QPushButton("Emergency Stop", self.analysisTab)
        # start disabled
        self.emergencyStopAnalysisButton.setEnabled(False)
        self.emergencyStopAnalysisButton.setStyleSheet("background-color: grey; color: white;")
        # self.emergencyStopAnalysisButton.clicked.connect(self.emergencyStopAnalysis)
        self.emergencyStopAnalysisButton.setFixedWidth(200)

        
        buttonLayout.addWidget(self.emergencyStopAnalysisButton)

        # Parameter inputs
        parameterLayout = QHBoxLayout()

        # self.outputTextAnalysis = QTextEdit('', self.analysisTab)
        # self.outputTextAnalysis.setReadOnly(True)  # Make it read-only

        splitter = QSplitter(Qt.Vertical)

        requirementLabel = QLabel("Requirements:", self.launchTab)
        # Add more buttons as needed, following the same pattern
        self.analysisRequirementText = QTextEdit()
        self.analysisRequirementText.setReadOnly(True)  # Make it read-only
        # Set background color
        self.analysisRequirementText.setStyleSheet("background-color: grey;")
        requirementLayout = QVBoxLayout()
        requirementLayout.addWidget(requirementLabel)
        requirementLayout.addWidget(self.analysisRequirementText)
        requirementWidget = QWidget()
        requirementWidget.setLayout(requirementLayout)


        blankWidget = QWidget()
        outputLayout = QVBoxLayout()
        # Text Window to display output
        self.outputTextAnalysis = QTextEdit(self.analysisTab)
        self.outputTextAnalysis.setReadOnly(True)  # Make it read-only
        # Clear button
        self.clearButtonAnalysis = QPushButton("Clear", self.analysisTab)
        self.clearButtonAnalysis.clicked.connect(lambda: self.outputTextAnalysis.clear())
        self.clearButtonAnalysis.setFixedWidth(100)
        self.clearButtonAnalysis.setStyleSheet("background-color: blue; color: white;")
        outputLayout.addWidget(self.outputTextAnalysis)
        outputLayout.addWidget(self.clearButtonAnalysis)

        blankWidget.setLayout(outputLayout)

        splitter.addWidget(requirementWidget)
        splitter.addWidget(blankWidget)
        splitter.setSizes([50, 200])

        analysisLayout.addLayout(buttonLayout)
        analysisLayout.addLayout(parameterLayout)
        analysisLayout.addWidget(splitter)

        self.onAnalysisChanged(0)


    def onAnalysisChanged(self, index):
        """Update the button, what it says, and what it launches. Also update the text box which shows the 
        parameters and the requirements
        """
        # get the name of the experiment
        analysis_name = self.analysisSelector.itemText(index)
        # get the command and requirements
        command, requirements = analysis_dict[analysis_name]
        # update the button
        self.launchAnalysisButton.setText(analysis_name)
        # Check if all requirements are met
        missing_requirements = [None]#self.checkAnalysisRequirements(requirements)
        # update the text box
        # self.outputText.setText(f"Command: {command} -j [path]\nRequirements: {requirements}")
        # self.outputText.setText(f"Command: {command}\n")
        self.analysisRequirementText.clear()
        self.updateAnalysisRequirementText(f"Command: {command}\n")
        # Append requirements with missing requirements in red
        for req in requirements:
            if req in missing_requirements:
                # light red
                self.updateAnalysisRequirementText(f"Missing requirement: {req}", color=QColor.fromRgb(240, 128, 128))
            else:
                self.updateAnalysisRequirementText(f"Requirement: {req}", color=QColor.fromRgb(128, 240, 128))

    def launchAnalysis(self):
        # Launches the chosen experiment
        if self.analysis_running:
            self.statusBar().showMessage("Analysis already running.")
            return
        
        if self.json_data is None:
            self.statusBar().showMessage("No experiment selected.")
            return
        
        # get the name of the experiment
        analysis_name = self.analysisSelector.currentText()
        # get the command and requirements
        command, requirements = analysis_dict[analysis_name]
        # check if all requirements are met
        # missing_requirements = self.checkAnalysisRequirements(requirements)
        # if len(missing_requirements) > 0:
        #     self.updateOutputText(f"Missing requirements: {missing_requirements}")
        #     self.statusBar().showMessage(f"Missing requirements: {missing_requirements}")
        #     return
        
        experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, self.json_data['name'])
        command += f" -j {experimentPath}.json -f {experimentPath}/{self.json_data['name']} -s "\
                f"{experimentPath}/plots/" 
        
        if analysis_name == 'Causal Analysis':
            command += f" -j {experimentPath}.json -f {experimentPath}/{self.json_data['name']}_causal -s "\
                f"{experimentPath}/plots/" 
        # TODO: Make it so it checks causal, and can take any of them
        # print("I WILL RUN", command)
        self.commandRunner = CommandRunner(command)
        self.commandRunner.output.connect(self.updateAnalysisOutputText)
        self.commandRunner.error.connect(self.handleAnalysisError)
        self.commandRunner.finished.connect(self.handleAnalysisCompleted)
        

        self.analysis_running = True
        # self.checkAnalysisButtonDependencies()
        # # Lock the experiment selection
        self.analysisSelector.setEnabled(False)
        self.statusBar().showMessage("Analysis started.")
        self.outputTextAnalysis.append("Analysis started.")
        self.outputTextAnalysis.append(f"Command: {command}")

        self.commandRunner.start()

    def updateAnalysisRequirementText(self, text, color=None):
        """Update the text box with the requirements
        """
        # clear
        if color is not None:
            self.analysisRequirementText.setTextColor(color)
        self.analysisRequirementText.append(text)
        self.analysisRequirementText.setTextColor(QColor("white"))

    def updateAnalysisOutputText(self, text, color=None):
        if color is not None:
            self.outputTextAnalysis.setTextColor(color)
        self.outputTextAnalysis.append(text)
        self.outputTextAnalysis.setTextColor(QColor("white"))


    def handleAnalysisError(self, error_message):
        self.analysis_running = False
        # self.checkAnalysisButtonDependencies()
        self.analysisSelector.setEnabled(True)
        self.statusBar().showMessage("Analysis error.")
        # self.outputText.append(f"Error: {error_message}")
        self.updateAnalysisOutputText(f"Error: {error_message}", color=QColor.fromRgb(240, 128, 128))

    def handleAnalysisCompleted(self):
        self.analysis_running = False
        # self.checkAnalysisButtonDependencies()
        self.analysisSelector.setEnabled(True)
        self.statusBar().showMessage(f"{self.analysisSelector.currentText()} completed.")
        self.outputTextAnalysis.append(f"{self.analysisSelector.currentText()} completed.")
        # Select experiment to update the plots
        # self.onExperimentSelect(self.experimentSelector.currentIndex(),clear=False)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Footprint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getFootprint(self):
        """Try to load from .npy file in the folder, or create a dialogue to select .raw.h5 file to get footprint
        """
        experiment = self.experimentSelector.currentText().split('.')[0]
        experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, experiment)
        npy_file = os.path.join(experimentPath, 'footprint.npy')
        if os.path.exists(npy_file):
            # Load the npy file
            footprints = np.load(npy_file)
            # Plot the footprint
            # self.plotFootprint(footprint)
            pass
        else:
            # Prompt the user to select a .raw.h5 file
            h5_file, _ = QFileDialog.getOpenFileName(self, "Select .raw.h5 file", experimentPath, "HDF5 Files (*.raw.h5)")
            if h5_file:
                # Get the footprint
                # make dialog to choos either stim electrodes, selected electrodes, or custom electrodes

                dialog = QDialog(self)
                dialog.setWindowTitle("Get Footprint")
                dialog.setFixedWidth(300)
                dialog.setFixedHeight(200)
                dialog.setStyleSheet("background-color: #2c3e50; color: white;")
                dialog.setModal(True)

                layout = QVBoxLayout(dialog)

                # Label for Project Selection
                label = QLabel("Select Electrodes:", dialog)
                label.setAlignment(Qt.AlignLeft)
                label.setStyleSheet("font-size: 14px; margin: 10px;")
                layout.addWidget(label)

                # Dropdown for Project Selection
                self.electrodeSelector = QComboBox(dialog)
                self.electrodeSelector.addItems(['Stim Electrodes', 'Selected Electrodes', 'Custom Electrodes'])

                # if custom electrodes, make a text box to enter the electrodes

                customElectrodesPrompt = QLabel("Enter electrodes separated by commas", dialog)
                customElectrodesPrompt.setAlignment(Qt.AlignLeft)
                customElectrodesPrompt.setStyleSheet("font-size: 14px; margin: 10px;")
                customElectrodesPrompt.hide()
                self.electrodeSelector.customElectrodes = QLineEdit(dialog)
                self.electrodeSelector.customElectrodes.setPlaceholderText("Enter electrodes separated by commas")
                self.electrodeSelector.customElectrodes.setFixedWidth(200)
                self.electrodeSelector.customElectrodes.setFixedHeight(30)
                self.electrodeSelector.customElectrodes.setStyleSheet("background-color: #2c3e50; color: white;")
                self.electrodeSelector.customElectrodes.hide()
                layout.addWidget(self.electrodeSelector.customElectrodes)

                customElectrodes = self.electrodeSelector.customElectrodes

                def electrodeSelectorChanged(index):
                    if index == 2:
                        customElectrodesPrompt.show()
                        customElectrodes.show()
                    else:
                        customElectrodesPrompt.hide()
                        customElectrodes.hide()

                self.electrodeSelector.currentIndexChanged.connect(electrodeSelectorChanged)

                layout.addWidget(self.electrodeSelector)

                # Button to add a new project
                getFootprintButton = QPushButton("Get Footprint", dialog)
                getFootprintButton.setStyleSheet("background-color: green;")
                # attach function
                getFootprintButton.clicked.connect(lambda: self.getFootprintFromH5(h5_file, dialog))
                layout.addWidget(getFootprintButton, alignment=Qt.AlignCenter)

                dialog.exec_()
                print("Getting footprint")
        
    def getFootprintFromH5(self, h5_file, dialog):
        print("Getting footprint from h5 file", h5_file)

        # do analysis
        # Get which electrodes to use
        if self.electrodeSelector.currentIndex() == 0:
            electrodes = self.json_data.get('stim_electrodes', None)
        elif self.electrodeSelector.currentIndex() == 1:
            electrodes = self.json_data.get('selected_electrodes', None)
        elif self.electrodeSelector.currentIndex() == 2:
            try:
                electrodes = self.electrodeSelector.customElectrodes.text().split(',')
                electrodes = [int(e) for e in electrodes]
            except:
                self.statusBar().showMessage("Error parsing custom electrodes, please use comma separated integers")
                return

        print("Electrodes", electrodes)
        # exit
        dialog.close()

        analysis_obj = AnalysisDAO(file_path=h5_file)
        footprint_chs, footprint_waves, mapping = plot_helper.get_footprints(analysis_obj,
                                                                selected_electrodes=electrodes)
        self.footprint_chs = footprint_chs
        self.footprint_waves = footprint_waves
        self.mapping = mapping
        print("Footprint chs", footprint_chs)
        self.plotScatter()

    
    def setMapping(self, mapping):
        self.mapping = mapping
        if self.analysis_obj is not None:
            self.analysis_obj.set_mapping(mapping)
        else:
            self.analysis_obj = AnalysisDAO(mapping = self.mapping)
    

    def scatSelectedToggle(self, state):
        if self.scat_selected_electrodes is not None:
            self.scat_selected_electrodes.set_visible(state)
            self.canvas.draw_idle()

    def scatStimToggle(self, state):
        if self.scat_stim_electrodes is not None:
            self.scat_stim_electrodes.set_visible(state)
            self.canvas.draw_idle()

    def plotScatter(self, color_mode='mean_amp'):
        if self.json_data is None:
            return

        mapping_path = self.json_data.get('mapping_file_path', None)
        if mapping_path is None or mapping_path == '':
            return
        
        py_params_path = self.json_data.get('py_file_path', None)

        selected_electrodes = self.json_data.get('selected_electrodes', None)
        stim_electrodes = self.json_data.get('stim_electrodes', None)
        
        if py_params_path is not None:
            try:
                py_params_path = os.path.join(self.mainPath, self.selected_project, self.selected_chip, py_params_path.split(self.selected_chip)[1].strip('/'))
                # import from the py file
                sys.path.append(os.path.dirname(py_params_path))
                module_name = os.path.basename(py_params_path).split('.')[0]
                params_module = __import__(module_name)
                selected_footprint_chans = params_module.selected_footprint_chans
            except:
                print('Error importing params file')
                selected_footprint_chans = None
            # print (dir(params_module))
            # print(selected_footprint_chans)


        # Construct full path to the file
        mapping_path = os.path.join(self.mainPath, self.selected_project, self.selected_chip, mapping_path.split(self.selected_chip)[1].strip('/'))
        
        # Load the mapping file
        mapping = pd.read_csv(mapping_path)
        self.setMapping(mapping)

        # Clear previous figure
        self.figure.clear()
        self.figure.subplots_adjust(left=0.05, right=.7)

        ax = self.figure.add_subplot(111)
        
        # Choose color based on the mode
        color = mapping[color_mode] if color_mode in mapping else 'blue'

        # Handle NaN values in color
        color = np.where(color.isna(), 0, color)

        ax.scatter(mapping['x'], mapping['y'], c=color, s=10, cmap='magma', marker='s', label='all')

        if selected_electrodes is not None:
            poses = self.analysis_obj.get_positions(electrodes=selected_electrodes)
            self.scat_selected_electrodes = ax.scatter(poses[:, 0], poses[:, 1], c='white', s=50, alpha=.8, marker='x', label='selected')
            # Text labels
            # On hover show the electrode number
            # Get mouse position
            
            # Arrow from mouse pos to electrode
            annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='white'))
            
            annot.set_visible(False)

            def update_annot(ind):
                pos = self.scat_selected_electrodes.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = "{}".format(" ".join([str(mapping['electrode'].iloc[n]) for n in ind["ind"]]))
                annot.set_text(text)
                annot.get_bbox_patch().set_facecolor('white')
                annot.get_bbox_patch().set_alpha(0.4)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = self.scat_selected_electrodes.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        self.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            self.canvas.draw_idle()

            self.canvas.mpl_connect("motion_notify_event", hover)


            

        if stim_electrodes is not None:
            poses = self.analysis_obj.get_positions(electrodes=stim_electrodes)
            self.scat_stim_electrodes = ax.scatter(poses[:, 0], poses[:, 1], c='green', s=50, alpha=.8, marker='x', label='stim')

        # Plot footprints if available
        if self.footprint_chs is not None and self.footprint_waves is not None:
            # Plot footprints
            x_scale = .05
            y_scale = .3
            for fp_ch, fp_wave in zip(self.footprint_chs, self.footprint_waves):
                for i, (ch, wave) in enumerate(zip(fp_ch, fp_wave)):
                    # Get the position of the channel
                    pos = self.analysis_obj.get_positions(channels=[ch])[0]
                    x, y = pos[0], pos[1]
                    # Plot the footprint
                    t = np.arange(len(wave)) * x_scale + x - len(wave) * x_scale / 2
                    wave = -wave
                    wave = wave - np.median(wave)
                    wave = wave * y_scale + y
                    ax.plot(t, wave, c='blue', alpha=.5)# label=f'footprint {ch}')

        ax.set_title(f'Scatterplot: Colored by {color_mode}')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_aspect('equal', 'box')
        #colorbar
        norm = plt.Normalize(vmin=color.min(), vmax=color.max())
        sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
        sm.set_array([])
        self.figure.colorbar(sm, ax=ax)

        ax.legend(loc='upper left')
        ax.set_facecolor('black')
        #invert y axis
        ax.invert_yaxis()



        self.canvas.draw()

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Projects ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def getProjectsList(self):
        # Function to list directories in the root experiment folder
        # Example: return ["20217", "P001354", ...]
        projects = [name for name in os.listdir(self.mainPath) if os.path.isdir(os.path.join(self.mainPath, name))]
        return projects


    def onProjectSelect(self, index):
        # List out chips for the experiment, which should be the subdirectories of the selected project
        self.selected_project = self.projectSelector.itemText(index)
        self.updateChipsList()
        self.statusBar().showMessage("Project set. Please select a chip.")

    
    def updateProjectsList(self):
        # Enumerate all directories in the main path
        if self.mainPath is None or self.mainPath == "":
            return
        projects = [name for name in os.listdir(self.mainPath) if os.path.isdir(os.path.join(self.mainPath, name))]
        
        self.projectSelector.clear()
        self.projectSelector.addItems(projects)

    def addProject(self):
       """Open up a new window, ask for the project name, and create the project folder
       """
       # Open up a new window, ask for the project name, and create the project folder
       text, okPressed = QInputDialog.getText(self, "New Project", "Project Name:", QLineEdit.Normal, "")
       if okPressed and text != '':
           # Create the project folder
           projectPath = os.path.join(self.mainPath, text)
           os.makedirs(projectPath, exist_ok=True)
           # Update the project list
           self.updateProjectsList()
           # Select the new project
           self.projectSelector.setCurrentIndex(self.projectSelector.findText(text))
           # Update the status bar
           self.statusBar().showMessage("Project created. Please select a chip.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Chips ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def onChipSelect(self, index):
        # List JSON files for the selected chip
        self.selected_chip = self.chipSelector.itemText(index)
        # check 1 or 2 levels deep
        self.updateExperimentsList()

        self.statusBar().showMessage("Chip set. Please select an experiment.")

    def updateChipsList(self):
        # Enumerate all directories in the main path
        chips = [name for name in os.listdir(os.path.join(self.mainPath, self.selected_project)) if os.path.isdir(os.path.join(self.mainPath, self.selected_project, name))]
        self.chipSelector.clear()
        self.chipSelector.addItems(chips)

    def addChip(self):
        """Open up a new window, ask for the chip name, and create the chip folder
        """
        # Open up a new window, ask for the chip name, and create the chip folder
        text, okPressed = QInputDialog.getText(self, "New Chip", "Chip Name:", QLineEdit.Normal, "")
        if okPressed and text != '':
            # Create the chip folder
            chipPath = os.path.join(self.mainPath, self.selected_project, text)
            os.makedirs(chipPath, exist_ok=True)
            # Update the chip list
            self.updateChipsList()
            # Select the new chip
            self.chipSelector.setCurrentIndex(self.chipSelector.findText(text))
            # Update the status bar
            self.statusBar().showMessage("Chip created. Please select an experiment.")


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Experiments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def onExperimentSelect(self, index, clear=True):
        """ Load the json file and display it as a table
        """
        # Load the json file
        selected_experiment = self.experimentSelector.itemText(index)
        json_file = os.path.join(self.mainPath, self.selected_project, self.selected_chip, selected_experiment)
        self.statusBar().showMessage("Experiment set. Loading experiment file: " + json_file)

        # Load the json file and display it as a table
        self.expWidget.setHeaderLabels([selected_experiment])
        self.loadExperiment(json_file)
        
        if self.json_data is not None and self.selected_chip != "":
            self.plotScatter()
        else:
            self.figure.clear()
            self.canvas.draw()
        self.checkLaunchButtonDependencies()
        self.onLaunchExperimentChanged(self.experimentLaunchSelector.currentIndex())

    def updateExperimentsList(self):
        # Enumerate all directories in the main path
        experiments = []
        self.json_data = None
        for root, dirs, files in os.walk(os.path.join(self.mainPath, self.selected_project, self.selected_chip)):
            for file in files:
                if file.endswith(".json"):
                    # Append the relative path from the chip directory
                    experiments.append(os.path.relpath(os.path.join(root, file), os.path.join(self.mainPath, self.selected_project, self.selected_chip)))
        self.experimentSelector.clear()
        self.experimentSelector.addItems(experiments)
        

    def loadExperiment(self, json_file):
        """ Load the json file and display it as a tree
        """
        try:
            self.json_data = json.load(open(json_file))
        except:
            # Clear the tree
            self.json_data = None
            self.expWidget.clear()
            self.statusBar().showMessage("Error loading experiment file: " + json_file)

            return
        fill_widget(self.expWidget, self.json_data)

        try:
            plug_name = self.json_data.get('plug', None)
            if plug_name == 'drew':
                plug_name = 'none'
            self.smartplug = SmartPlug(plug_name, verbose=True)
            self.smartPlugLabel.setText(f"{plug_name}")
            
        except Exception as e:
            print(e)
            self.smartplug = None

        # unexpand all items
        for item in self.expWidget.findItems("", Qt.MatchContains | Qt.MatchRecursive):
            item.setExpanded(False)

        self.expRawText.setPlainText(json.dumps(self.json_data, indent=4))

        # Update the pictures list
        self.updatePicturesList()

    
    def addExperiment(self):
        dialog = AddExperimentDialog(self, self.metadata_fields)
        if dialog.exec_() == QDialog.Accepted:
            inputs = dialog.getInputs()
            name = inputs['name']
            exp_type = inputs['type']
            smartplug = inputs['smartplug']
            config_file = inputs['configFile']
            metadata = inputs['metadata']

            # if config_file is not under the chip folder, copy it over
            if config_file is not None:
                # copy the file
                new_config_file = os.path.join(self.mainPath, self.selected_project, self.selected_chip, os.path.basename(config_file))
                # check if same file
                if new_config_file != config_file:
                    shutil.copy(config_file, os.path.join(self.mainPath, self.selected_project, self.selected_chip))
                config_file = new_config_file
            else:
                config_file = ''

            if name:
                # Create the experiment json file
                experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, name + ".json")
                
                # Construct the command with metadata
                metadata_args = ' '.join([f"-m {k}={v}" for k, v in metadata.items()])
                command = f"python -m proj.cartpole_v1.start_experiment -j {experimentPath} -s {smartplug} -e {exp_type} -c {config_file}"# {metadata_args}"
                
                # Update the status bar
                self.statusBar().showMessage("Experiment created.")

                # Launch start_experiment script
                os.system(command)  # You might want to run this in a separate thread or process
                # Update the experiment list
                self.updateExperimentsList()

                # Select the new experiment
                self.experimentSelector.setCurrentIndex(self.experimentSelector.findText(name + ".json"))

        


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pictures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def loadPicture(self):
        """Load a .png file and displays it"""
        selected_picture = self.picturesList.currentText()
        picture_file = os.path.join(self.mainPath, self.selected_project, self.selected_chip, selected_picture)
        self.statusBar().showMessage("Loading picture: " + picture_file)
        
        pixmap = QPixmap(picture_file)
        # Get current width and height of the whole window        
        # scaled_pixmap = pixmap.scaled(int(self.width // 1.5), int(self.height // 1.5), Qt.KeepAspectRatio)
        # self.pictureLabel.setPixmap(scaled_pixmap)
        # self.pictureLabel.adjustSize()
        # print(self.pictureWidget.size())
        if pixmap.isNull():
            # self.statusBar().showMessage("Error loading picture: " + picture_file)
            return
        self.pictureWidget.setPixmap(pixmap.scaled(self.pictureWidget.size()))#, Qt.KeepAspectRatio))


        # Scale the whole window to fit the picture
        



        #self.pictureWidget.setPixmap(pi
        # xmap.scaled(self.pictureWidget.size(), Qt.KeepAspectRatio))
    

    def updatePicturesList(self):
        # Enumerate all directories in the main path
        pictures = []
        for root, dirs, files in os.walk(os.path.join(self.mainPath, self.selected_project, self.selected_chip)):
            for file in files:
                if file.endswith(".png"):
                    # Append the relative path from the chip directory
                    pictures.append(os.path.relpath(os.path.join(root, file), os.path.join(self.mainPath, self.selected_project, self.selected_chip)))
        self.picturesList.clear()
        self.picturesList.addItems(pictures)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cloud sync ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def syncToCloud(self):
        # Syncs the current experiment to the cloud
        # Use command runner to run s3 sync command
        if self.json_data is None:
            self.statusBar().showMessage("No experiment selected.")
            return
        experimentPath = os.path.join(self.mainPath, self.selected_project, self.selected_chip, self.json_data['name'])
        command = f'aws s3 --endpoint="https://s3-west.nrp-nautilus.io" sync {experimentPath} s3://braingeneersdev/asrobbin/{self.selected_project}/{self.selected_chip}/{self.json_data["name"]}/'
        command += ' --exclude "*__pycache__*"'
        self.updateOutputText(f"Syncing to cloud: {command}", color=QColor.fromRgb(128, 240, 240))
        # Disable the sync button, and the launch button
        self.cloudSyncButton.setEnabled(False)
        self.launchExperimentButton.setEnabled(False)

        self.commandRunner = CommandRunner(command)
        self.commandRunner.output.connect(self.updateOutputText)
        self.commandRunner.error.connect(self.handleExperimentError)
        self.commandRunner.finished.connect(self.syncToCloudFinished)

        self.commandRunner.start()




    def syncToCloudFinished(self):
        # Re-enable the sync button
        self.cloudSyncButton.setEnabled(True)
        self.launchExperimentButton.setEnabled(True)
        self.statusBar().showMessage("Sync to cloud completed.")
        self.updateOutputText("Sync to cloud completed.", color=QColor.fromRgb(128, 240, 128))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Preferences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def openSettings(self):
        from braindance.gui.settings import PreferencesDialog
        dialog = PreferencesDialog(self)
        dialog.exec_()

    def loadPreferences(self):
        # Try to load the configuration file
        try:
            with open(preferences_file, 'r') as f:
                mainPath = f.read().strip()
                # Set the main path and update the projects list
                self.setMainPath(mainPath)
        except FileNotFoundError:
            # If the config file doesn't exist, prompt the user to select a directory
            self.setMainPath()

        try:
            with open(preferences_json, 'r') as f:
                preferences = json.load(f)
                metadata_fields = preferences.get('metadata_fields', [])
                self.setMetadataFields(metadata_fields)
        except FileNotFoundError:
            QMessageBox.information(self, "First Time Setup", "No existing preferences found. Creating new settings.")
            self.setMetadataFields([])

    def setMetadataFields(self, metadata_fields):
        self.metadata_fields = metadata_fields


    def buttonSetMainPathClicked(self):
        self.setMainPath()

    def setMainPath(self, mainPath=None):
        if mainPath is None:
            # Prompt the user to select a directory
            mainPath = QFileDialog.getExistingDirectory(self, "Select Main Project Directory")
            if not mainPath:
                QMessageBox.warning(self, "Warning", "No directory selected.")
                # sys.exit(1)  # Exit if no directory is selected

        # Save the selected path to the config file
        with open(preferences_file, 'w') as f:
            f.write(mainPath)

        # Update the main path and projects list
        self.mainPath = mainPath
        self.updateProjectsList()


    def get_text_position(self, text, text_widget):

        cursor = text_widget.textCursor()
        cursor.movePosition(QTextCursor.Start)
        line_number = 0
        while not cursor.atEnd():
            if text in cursor.block().text():
                return line_number
            cursor.movePosition(QTextCursor.Down)
            line_number += 1
        return None

    def jsonItemClicked(self, item, column):
        text = item.text(column).strip()
        line_number = self.get_text_position(text, self.expRawText)
        if line_number is not None:
            cursor = self.expRawText.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.MoveAnchor, line_number)
            self.expRawText.setTextCursor(cursor)
            # self.expRawText.ensureCursorVisible()
            # Set cursor to be top of vox
            self.expRawText.setFocus()


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtWidgets import QFormLayout, QDialogButtonBox, QComboBox, QHBoxLayout
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QCalendarWidget
# import QDateEdit

# from 

class AddExperimentDialog(QDialog):
    def __init__(self, parent=None, metadata_fields=None):
        super().__init__(parent)
        self.setWindowTitle("New Experiment")
        self.parent = parent
        self.metadata_fields = metadata_fields or []
        self.configFile = None

        layout = QVBoxLayout(self)

        # Form layout for dynamic fields
        self.formLayout = QFormLayout()
        layout.addLayout(self.formLayout)

        # Static fields
        self.nameEdit = QLineEdit(self)
        self.formLayout.addRow(QLabel("Experiment Name:"), self.nameEdit)

        self.typeEdit = QLineEdit(self)
        self.formLayout.addRow(QLabel("Experiment Type:"), self.typeEdit)

        self.smartplugEdit = QLineEdit(self)
        self.formLayout.addRow(QLabel("Smartplug Name:"), self.smartplugEdit)

        self.configButton = QPushButton("Select Config File", self)
        self.configButton.clicked.connect(self.buttonSetConfigFileClicked)
        self.formLayout.addRow(QLabel("Config File:"), self.configButton)

        # Dynamic metadata fields
        self.metadataInputs = {}
        for field in self.metadata_fields:
            field_name = field['name']
            field_type = field['type']
            if field_type == 'Raw Text':
                edit = QLineEdit(self)
                self.metadataInputs[field_name] = edit
                self.formLayout.addRow(QLabel(field_name), edit)
            elif field_type == 'Categorical':
                combo_layout = QHBoxLayout()
                combo = QComboBox(self)
                combo.addItems(field.get('options', []))
                self.metadataInputs[field_name] = combo
                add_button = QPushButton(QIcon('resources/plus.png'), '', self)
                add_button.clicked.connect(lambda _, fn=field_name, c=combo: self.addCategory(fn, c))
                combo_layout.addWidget(combo)
                combo_layout.addWidget(add_button)
                self.formLayout.addRow(QLabel(field_name), combo_layout)
            elif field_type == 'Date':
                calendar = QCalendarWidget(self)
                calendar.setGridVisible(True)
                calendar.setSelectionMode(QCalendarWidget.SingleSelection)
                calendar.setSelectedDate(QDate.currentDate())
                # Text black
                calendar.setStyleSheet("background-color : lightgreen; color : black;")
                self.metadataInputs[field_name] = calendar
                self.formLayout.addRow(QLabel(field_name), calendar)

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def getInputs(self):
        inputs = {
            'name': self.nameEdit.text(),
            'type': self.typeEdit.text(),
            'smartplug': self.smartplugEdit.text(),
            'configFile': self.configFile,
            'metadata': {field['name']: self.getFieldValue(field) for field in self.metadata_fields}
        }
        return inputs

    def getFieldValue(self, field):
        field_name = field['name']
        field_type = field['type']
        widget = self.metadataInputs[field_name]
        if field_type == 'Raw Text':
            return widget.text()
        elif field_type == 'Categorical':
            return widget.currentText()
        elif field_type == 'Date':
            return widget.selectedDate().toString('yyyy-MM-dd')

    def buttonSetConfigFileClicked(self):
        self.configFile, _ = QFileDialog.getOpenFileName(self, "Select Config File", self.parent.mainPath, "Config Files (*.cfg)")
        if self.configFile:
            self.configButton.setText(os.path.basename(self.configFile))

    def addCategory(self, field_name, combo):
        text, ok = QInputDialog.getText(self, 'Add Category', f'Enter new category for {field_name}:')
        if ok and text:
            combo.addItem(text)
            combo.setCurrentText(text)
            self.updateMetadataFieldOptions(field_name, text)

    def updateMetadataFieldOptions(self, field_name, new_option):
        # Update the metadata preferences with the new option
        for field in self.metadata_fields:
            if field['name'] == field_name:
                field['options'].append(new_option)
                break
        preferences = {'metadata_fields': self.metadata_fields}
        with open(preferences_json, 'w') as f:
            json.dump(preferences, f, indent=4)

    def closeEvent(self, event):
        # Check if the thread is running
        if hasattr(self, 'commandRunner') and self.commandRunner.isRunning():
            self.commandRunner.terminate()  # Terminate the thread
            self.commandRunner.wait()       # Wait for the thread to finish
        event.accept()  # Accept the close event

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ misc ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fill_item(item, value):
  item.setExpanded(True)
  if type(value) is dict:
    for key, val in sorted(value.items()):
      child = QTreeWidgetItem()
      child.setText(0, key)
      item.addChild(child)
      fill_item(child, val)
  elif type(value) is list:
    for val in value:
      child = QTreeWidgetItem()
      item.addChild(child)
      if type(val) is dict:      
        child.setText(0, 'dict')
        fill_item(child, val)
      elif type(val) is list:
        child.setText(0, 'list')
        fill_item(child, val)
      else:
        child.setText(0, str(val))              
      child.setExpanded(True)
  elif type(value) is int or type(value) is bool or type(value) is float:
    child = QTreeWidgetItem()
    child.setText(0, str(value))
    item.addChild(child)

  else:
    child = QTreeWidgetItem()
    child.setText(0, value)
    item.addChild(child)

def fill_widget(widget, value):
  widget.clear()
  fill_item(widget.invisibleRootItem(), value)

def main():
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        main()
    finally:
        print("Closing application...")