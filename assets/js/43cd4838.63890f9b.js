"use strict";(self.webpackChunkbraindance_docs=self.webpackChunkbraindance_docs||[]).push([[184],{5146:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>c,contentTitle:()=>o,default:()=>p,frontMatter:()=>i,metadata:()=>t,toc:()=>l});var a=r(4848),s=r(8453);const i={sidebar_position:2},o="Core Concepts",t={id:"core-concepts",title:"Core Concepts",description:"Understanding the core concepts of BrainDance will help you design and execute more complex experiments.",source:"@site/docs/core-concepts.md",sourceDirName:".",slug:"/core-concepts",permalink:"/brainloop/docs/core-concepts",draft:!1,unlisted:!1,editUrl:"https://github.com/braingeneers/brainloop/docs/core-concepts.md",tags:[],version:"current",sidebarPosition:2,frontMatter:{sidebar_position:2},sidebar:"tutorialSidebar",previous:{title:"Quick Start Guide",permalink:"/brainloop/docs/quick-start"},next:{title:"RT-Sort",permalink:"/brainloop/docs/category/rt-sort"}},c={},l=[{value:"Maxwell Environment",id:"maxwell-environment",level:2},{value:"Phases",id:"phases",level:2},{value:"Phase Manager",id:"phase-manager",level:2}];function d(e){const n={code:"code",h1:"h1",h2:"h2",header:"header",li:"li",p:"p",pre:"pre",ul:"ul",...(0,s.R)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(n.header,{children:(0,a.jsx)(n.h1,{id:"core-concepts",children:"Core Concepts"})}),"\n",(0,a.jsx)(n.p,{children:"Understanding the core concepts of BrainDance will help you design and execute more complex experiments."}),"\n",(0,a.jsx)(n.h2,{id:"maxwell-environment",children:"Maxwell Environment"}),"\n",(0,a.jsxs)(n.p,{children:["The ",(0,a.jsx)(n.code,{children:"MaxwellEnv"})," class is the central component that interfaces with the micro electrode array hardware. It handles:"]}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsx)(n.li,{children:"Initialization of the hardware"}),"\n",(0,a.jsx)(n.li,{children:"Data acquisition"}),"\n",(0,a.jsx)(n.li,{children:"Stimulation control"}),"\n",(0,a.jsx)(n.li,{children:"Data saving"}),"\n"]}),"\n",(0,a.jsx)(n.p,{children:"Example usage:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from brainloop.core.maxwell_env import MaxwellEnv\nfrom brainloop.core.params import maxwell_params\n\nparams = maxwell_params\nparams['save_dir'] = './experiment_data'\nparams['name'] = 'my_experiment'\nenv = MaxwellEnv(**params)\n"})}),"\n",(0,a.jsx)(n.h2,{id:"phases",children:"Phases"}),"\n",(0,a.jsx)(n.p,{children:"Phases are the building blocks of experiments in BrainDance. Each phase represents a specific experimental action or protocol. Common phase types include:"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"RecordPhase"}),": Simple recording of neural activity"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"NeuralSweepPhase"}),": Systematic stimulation of specified electrodes"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"FrequencyStimPhase"}),": Stimulation at a specified frequency"]}),"\n"]}),"\n",(0,a.jsx)(n.p,{children:"Example of creating a phase:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from brainloop.core.phases import RecordPhase\n\nrecord_phase = RecordPhase(env, duration=60*5)  # 5-minute recording phase\n"})}),"\n",(0,a.jsx)(n.h2,{id:"phase-manager",children:"Phase Manager"}),"\n",(0,a.jsxs)(n.p,{children:["The ",(0,a.jsx)(n.code,{children:"PhaseManager"})," class orchestrates the execution of multiple phases in an experiment. It allows you to:"]}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsx)(n.li,{children:"Add phases to the experiment"}),"\n",(0,a.jsx)(n.li,{children:"Specify the order of phase execution"}),"\n",(0,a.jsx)(n.li,{children:"Run the entire experiment"}),"\n"]}),"\n",(0,a.jsx)(n.p,{children:"Example usage:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from brainloop.core.phases import PhaseManager\n\nphase_manager = PhaseManager(env, verbose=True)\nphase_manager.add_phase_group([record_phase, stim_phase, record_phase])\nphase_manager.run()\n"})}),"\n",(0,a.jsx)(n.p,{children:"By combining these core concepts, you can create complex, multi-stage experiments with precise control over stimulation and recording parameters."})]})}function p(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,a.jsx)(n,{...e,children:(0,a.jsx)(d,{...e})}):d(e)}},8453:(e,n,r)=>{r.d(n,{R:()=>o,x:()=>t});var a=r(6540);const s={},i=a.createContext(s);function o(e){const n=a.useContext(i);return a.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function t(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:o(e.components),a.createElement(i.Provider,{value:n},e.children)}}}]);