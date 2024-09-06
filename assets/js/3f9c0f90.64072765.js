"use strict";(self.webpackChunkbraindance_docs=self.webpackChunkbraindance_docs||[]).push([[573],{2160:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>a,contentTitle:()=>r,default:()=>h,frontMatter:()=>c,metadata:()=>o,toc:()=>l});var s=i(4848),t=i(8453);const c={},r="module phases.py",o={id:"docs/phases.py",title:"phases.py",description:"Experiment phases",source:"@site/docs/docs/phases.py.mdx",sourceDirName:"docs",slug:"/docs/phases.py",permalink:"/BrainLoop/docs/docs/phases.py",draft:!1,unlisted:!1,editUrl:"https://github.com/braingeneers/brainloop/docs/docs/phases.py.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"params.py",permalink:"/BrainLoop/docs/docs/params.py"},next:{title:"plot.py",permalink:"/BrainLoop/docs/docs/plot.py"}},a={},l=[{value:"<kbd>class</kbd> <code>FrequencyStimPhase</code>",id:"class-frequencystimphase",level:2},{value:"<kbd>function</kbd> <code>__init__</code>",id:"function-__init__",level:3},{value:"<kbd>function</kbd> <code>info</code>",id:"function-info",level:3},{value:"<kbd>function</kbd> <code>predicted_time</code>",id:"function-predicted_time",level:3},{value:"<kbd>function</kbd> <code>run</code>",id:"function-run",level:3},{value:"<kbd>function</kbd> <code>time_elapsed</code>",id:"function-time_elapsed",level:3},{value:"<kbd>class</kbd> <code>NeuralSweepPhase</code>",id:"class-neuralsweepphase",level:2},{value:"<kbd>function</kbd> <code>__init__</code>",id:"function-__init__-1",level:3},{value:"<kbd>function</kbd> <code>generate_stim_commands</code>",id:"function-generate_stim_commands",level:3},{value:"<kbd>function</kbd> <code>info</code>",id:"function-info-1",level:3},{value:"<kbd>function</kbd> <code>predicted_time</code>",id:"function-predicted_time-1",level:3},{value:"<kbd>function</kbd> <code>run</code>",id:"function-run-1",level:3},{value:"<kbd>function</kbd> <code>time_elapsed</code>",id:"function-time_elapsed-1",level:3},{value:"<kbd>class</kbd> <code>Phase</code>",id:"class-phase",level:2},{value:"<kbd>function</kbd> <code>__init__</code>",id:"function-__init__-2",level:3},{value:"<kbd>function</kbd> <code>info</code>",id:"function-info-2",level:3},{value:"<kbd>function</kbd> <code>predicted_time</code>",id:"function-predicted_time-2",level:3},{value:"<kbd>function</kbd> <code>run</code>",id:"function-run-2",level:3},{value:"<kbd>function</kbd> <code>time_elapsed</code>",id:"function-time_elapsed-2",level:3},{value:"<kbd>class</kbd> <code>PhaseManager</code>",id:"class-phasemanager",level:2},{value:"<kbd>function</kbd> <code>__init__</code>",id:"function-__init__-3",level:3},{value:"<kbd>function</kbd> <code>add_phase</code>",id:"function-add_phase",level:3},{value:"<kbd>function</kbd> <code>add_phase_group</code>",id:"function-add_phase_group",level:3},{value:"<kbd>function</kbd> <code>log_phase</code>",id:"function-log_phase",level:3},{value:"<kbd>function</kbd> <code>log_summary</code>",id:"function-log_summary",level:3},{value:"<kbd>function</kbd> <code>run</code>",id:"function-run-3",level:3},{value:"<kbd>function</kbd> <code>summary</code>",id:"function-summary",level:3},{value:"<kbd>class</kbd> <code>RecordPhase</code>",id:"class-recordphase",level:2},{value:"<kbd>function</kbd> <code>__init__</code>",id:"function-__init__-4",level:3},{value:"<kbd>function</kbd> <code>info</code>",id:"function-info-3",level:3},{value:"<kbd>function</kbd> <code>predicted_time</code>",id:"function-predicted_time-3",level:3},{value:"<kbd>function</kbd> <code>run</code>",id:"function-run-4",level:3},{value:"<kbd>function</kbd> <code>time_elapsed</code>",id:"function-time_elapsed-3",level:3}];function d(e){const n={a:"a",br:"br",code:"code",em:"em",h1:"h1",h2:"h2",h3:"h3",header:"header",hr:"hr",li:"li",p:"p",pre:"pre",ul:"ul",...(0,t.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L0",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsx)(n.header,{children:(0,s.jsxs)(n.h1,{id:"module-phasespy",children:[(0,s.jsx)("kbd",{children:"module"})," ",(0,s.jsx)(n.code,{children:"phases.py"})]})}),"\n",(0,s.jsx)(n.p,{children:"Experiment phases\n----------------- Phases act as parts of experiments which have a specific purpose, such as"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:"spontaneous recording"}),"\n",(0,s.jsx)(n.li,{children:"amplitude sweep\n--- over a neuron to find the stimulus amplitude that elicits a spike"}),"\n",(0,s.jsxs)(n.li,{children:["frequency stimulation",(0,s.jsx)(n.br,{}),"\n","---to stimulate one or more neurons at a certain frequency"]}),"\n",(0,s.jsx)(n.li,{children:"custom\n---where users can build exact logic for stimulation"}),"\n"]}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsxs)(n.h2,{id:"class-frequencystimphase",children:[(0,s.jsx)("kbd",{children:"class"})," ",(0,s.jsx)(n.code,{children:"FrequencyStimPhase"})]}),"\n",(0,s.jsx)(n.p,{children:"Phase for stimulating a command at a certain frequency"}),"\n",(0,s.jsx)(n.p,{children:"Parameters\n---------- env : base_env.BaseEnv  The environment to run the phase in stim_command : stim command or list of stim commands  If a single stim command, it will be repeated at the given frequency  If a list of stim commands, each stim command will be run sequentially stim_freq : float, optional  The frequency of the stimulation, by default 1 duration : int, optional  The duration of the stimulation in seconds, by default 10 tag : str, list of str, optional  The tag to use for the stimulation, by default 'frequency_stim'  If a list of strings, each tag will be used for the corresponding stim command\n--- Must be the same length as stim_command verbose : bool, optional  Whether to print out information about the stimulation, by default False connect_units : list, optional  The units to connect to the stimulation electrodes, by default None, leaves as is  If a list is input, the stimulation units of the corresponding indexes in the environmen  will be connected."}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L438",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-__init__",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"__init__"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"__init__(\n    env: BaseEnv,\n    stim_command: list,\n    stim_freq: float = 1,\n    duration: int = 10,\n    tag='frequency_stim',\n    verbose=False,\n    connect_units=None\n)\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L506",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-info",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"info"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"info()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L37",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-predicted_time",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"predicted_time"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"predicted_time()\n"})}),"\n",(0,s.jsx)(n.p,{children:"Returns the predicted time for the phase to run in seconds"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L464",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-run",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"run"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"run()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L34",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-time_elapsed",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"time_elapsed"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"time_elapsed()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsxs)(n.h2,{id:"class-neuralsweepphase",children:[(0,s.jsx)("kbd",{children:"class"})," ",(0,s.jsx)(n.code,{children:"NeuralSweepPhase"})]}),"\n",(0,s.jsx)(n.p,{children:"Sweep the amplitude of a stimulation to find the minimum amplitude that elicits a spike"}),"\n",(0,s.jsx)(n.p,{children:"Parameters\n---------- env : base_env.BaseEnv  The environment to run the phase in neuron_list : list  The neurons to stimulate amp_bounds : tuple  The bounds of the amplitude sweep  The step size defaults to 10% of the range, but   if a third element is provided, it will be used as the number of steps   (start, end, n_step) stim_freq : float, optional  The frequency of the stimulation, by default 1 replicates : int  The number of times to repeat the stimulation, by default 30 phase_length : int, optional  The length of one of the phases in the stimulation pulse, by default 100 type : str, optional  The type of amplitude sweep to perform, by default 'ran', determined by the   order of the following characters"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:"'r': Iterate through replicates"}),"\n",(0,s.jsx)(n.li,{children:"'a': Iterate through the amplitudes"}),"\n",(0,s.jsx)(n.li,{children:"'s': Iterate through the neurons  If you put 'random', it will randomly iterate through everything  Options:"}),"\n",(0,s.jsx)(n.li,{children:"'ran': First iterate through the replicates, then the amplitudes, then the neurons\n--- ex: (r1, a1, n1), (r2, a1, n1),...,(r1, a2, n1), (r2, a2, n1),..."}),"\n",(0,s.jsx)(n.li,{children:"'rna': First iterate through the replicates, then the neurons, then the amplitudes\n--- ex: (r1, a1, n1), (r2, a1, n1),...,(r1, a1, n2), (r2, a1, n2),..."}),"\n",(0,s.jsx)(n.li,{children:"'arn': First iterate through the amplitudes, then the replicates, then the neurons\n--- ex: (r1, a1, n1), (r1, a2, n1),...,(r2, a1, n1), (r2, a2, n1),..."}),"\n",(0,s.jsx)(n.li,{children:"etc."}),"\n"]}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L261",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-__init__-1",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"__init__"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"__init__(\n    env: BaseEnv,\n    neuron_list: list,\n    amp_bounds=[150, 150, 1],\n    stim_freq: float = 1,\n    replicates=30,\n    phase_length: int = 100,\n    order='ran',\n    single_connect=False,\n    verbose=False,\n    tag='neural_sweep'\n)\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L301",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-generate_stim_commands",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"generate_stim_commands"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"generate_stim_commands()\n"})}),"\n",(0,s.jsx)(n.p,{children:"Generates the stimulation commands for the amplitude sweep"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L397",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-info-1",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"info"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"info()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L37",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-predicted_time-1",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"predicted_time"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"predicted_time()\n"})}),"\n",(0,s.jsx)(n.p,{children:"Returns the predicted time for the phase to run in seconds"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L350",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-run-1",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"run"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"run()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L34",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-time_elapsed-1",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"time_elapsed"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"time_elapsed()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsxs)(n.h2,{id:"class-phase",children:[(0,s.jsx)("kbd",{children:"class"})," ",(0,s.jsx)(n.code,{children:"Phase"})]}),"\n",(0,s.jsx)(n.p,{children:"Base class for all phases"}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L27",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-__init__-2",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"__init__"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"__init__(env: BaseEnv)\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L41",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-info-2",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"info"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"info()\n"})}),"\n",(0,s.jsx)(n.p,{children:"Returns a dictionary of information about the phase"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L37",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-predicted_time-2",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"predicted_time"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"predicted_time()\n"})}),"\n",(0,s.jsx)(n.p,{children:"Returns the predicted time for the phase to run in seconds"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L31",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-run-2",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"run"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"run()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L34",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-time_elapsed-2",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"time_elapsed"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"time_elapsed()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsxs)(n.h2,{id:"class-phasemanager",children:[(0,s.jsx)("kbd",{children:"class"})," ",(0,s.jsx)(n.code,{children:"PhaseManager"})]}),"\n",(0,s.jsx)(n.p,{children:"Manages phases of an experiment"}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L50",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-__init__-3",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"__init__"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"__init__(env: BaseEnv, verbose=False)\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L61",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-add_phase",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"add_phase"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"add_phase(phase: Phase)\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L64",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-add_phase_group",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"add_phase_group"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"add_phase_group(phase_group: list)\n"})}),"\n",(0,s.jsx)(n.p,{children:"Adds a group of phases to the manager,  each group will belong to the same save file"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L79",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-log_phase",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"log_phase"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"log_phase(phase)\n"})}),"\n",(0,s.jsx)(n.p,{children:"Appends the phase and filename to the log file"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L71",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-log_summary",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"log_summary"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"log_summary()\n"})}),"\n",(0,s.jsx)(n.p,{children:"Logs the summary of the experiment to a text file"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L92",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-run-3",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"run"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"run()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L151",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-summary",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"summary"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"summary()\n"})}),"\n",(0,s.jsx)(n.p,{children:"Returns a summary of the experiment as a string"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsxs)(n.h2,{id:"class-recordphase",children:[(0,s.jsx)("kbd",{children:"class"})," ",(0,s.jsx)(n.code,{children:"RecordPhase"})]}),"\n",(0,s.jsx)(n.p,{children:"Phase for recording spontaneous activity"}),"\n",(0,s.jsx)(n.p,{children:"Parameters\n---------- env : base_env.BaseEnv  The environment to run the phase in duration : int  The duration of the recording in seconds, by default 10"}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L201",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-__init__-4",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"__init__"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"__init__(env: BaseEnv = None, duration: int = 10)\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L217",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-info-3",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"info"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"info()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L37",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-predicted_time-3",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"predicted_time"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"predicted_time()\n"})}),"\n",(0,s.jsx)(n.p,{children:"Returns the predicted time for the phase to run in seconds"}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L206",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-run-4",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"run"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"run(env=None)\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/phases.py#L34",children:(0,s.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,s.jsxs)(n.h3,{id:"function-time_elapsed-3",children:[(0,s.jsx)("kbd",{children:"function"})," ",(0,s.jsx)(n.code,{children:"time_elapsed"})]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"time_elapsed()\n"})}),"\n",(0,s.jsx)(n.hr,{}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsxs)(n.em,{children:["This file was automatically generated via ",(0,s.jsx)(n.a,{href:"https://github.com/ml-tooling/lazydocs",children:"lazydocs"}),"."]})})]})}function h(e={}){const{wrapper:n}={...(0,t.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},8453:(e,n,i)=>{i.d(n,{R:()=>r,x:()=>o});var s=i(6540);const t={},c=s.createContext(t);function r(e){const n=s.useContext(c);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:r(e.components),s.createElement(c.Provider,{value:n},e.children)}}}]);