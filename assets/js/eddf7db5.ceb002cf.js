"use strict";(self.webpackChunkbraindance_docs=self.webpackChunkbraindance_docs||[]).push([[860],{9173:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>d,contentTitle:()=>c,default:()=>h,frontMatter:()=>o,metadata:()=>r,toc:()=>l});var t=s(4848),i=s(8453);const o={},c="module utils.py",r={id:"docs/utils.py",title:"utils.py",description:"---",source:"@site/docs/docs/utils.py.mdx",sourceDirName:"docs",slug:"/docs/utils.py",permalink:"/BrainLoop/docs/docs/utils.py",draft:!1,unlisted:!1,editUrl:"https://github.com/braingeneers/brainloop/docs/docs/utils.py.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"trainer.py",permalink:"/BrainLoop/docs/docs/trainer.py"}},d={},l=[{value:"<kbd>function</kbd> <code>confusion_matrix</code>",id:"function-confusion_matrix",level:2},{value:"<kbd>function</kbd> <code>confusion_stats</code>",id:"function-confusion_stats",level:2},{value:"<kbd>function</kbd> <code>random_seed</code>",id:"function-random_seed",level:2},{value:"<kbd>function</kbd> <code>get_time</code>",id:"function-get_time",level:2},{value:"<kbd>function</kbd> <code>copy_file</code>",id:"function-copy_file",level:2},{value:"<kbd>function</kbd> <code>round</code>",id:"function-round",level:2},{value:"<kbd>function</kbd> <code>torch_to_np</code>",id:"function-torch_to_np",level:2}];function a(e){const n={a:"a",code:"code",em:"em",h1:"h1",h2:"h2",header:"header",hr:"hr",p:"p",pre:"pre",...(0,i.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/spikedetector/utils.py#L0",children:(0,t.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,t.jsx)(n.header,{children:(0,t.jsxs)(n.h1,{id:"module-utilspy",children:[(0,t.jsx)("kbd",{children:"module"})," ",(0,t.jsx)(n.code,{children:"utils.py"})]})}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/spikedetector/utils.py#L9",children:(0,t.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,t.jsxs)(n.h2,{id:"function-confusion_matrix",children:[(0,t.jsx)("kbd",{children:"function"})," ",(0,t.jsx)(n.code,{children:"confusion_matrix"})]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"confusion_matrix(preds, labels)\n"})}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/spikedetector/utils.py#L20",children:(0,t.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,t.jsxs)(n.h2,{id:"function-confusion_stats",children:[(0,t.jsx)("kbd",{children:"function"})," ",(0,t.jsx)(n.code,{children:"confusion_stats"})]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"confusion_stats(confusion_matrix)\n"})}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/spikedetector/utils.py#L29",children:(0,t.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,t.jsxs)(n.h2,{id:"function-random_seed",children:[(0,t.jsx)("kbd",{children:"function"})," ",(0,t.jsx)(n.code,{children:"random_seed"})]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"random_seed(seed, silent=False)\n"})}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/spikedetector/utils.py#L37",children:(0,t.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,t.jsxs)(n.h2,{id:"function-get_time",children:[(0,t.jsx)("kbd",{children:"function"})," ",(0,t.jsx)(n.code,{children:"get_time"})]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"get_time()\n"})}),"\n",(0,t.jsxs)(n.p,{children:["Gets the time in the string format of yymmdd_HHMMSS_ffffff (",(0,t.jsx)(n.a,{href:"https://www.programiz.com/python-programming/datetime/strftime",children:"https://www.programiz.com/python-programming/datetime/strftime"}),") Ex: If run at 9/26/22 at 3:53pm and 30 seconds and 1 microsecond, yymmdd_HHMMSS_ffffff = 220926_155330_000001"]}),"\n",(0,t.jsx)(n.p,{children:":return: str  Formatted time"}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/spikedetector/utils.py#L49",children:(0,t.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,t.jsxs)(n.h2,{id:"function-copy_file",children:[(0,t.jsx)("kbd",{children:"function"})," ",(0,t.jsx)(n.code,{children:"copy_file"})]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"copy_file(src_path, dest_folder)\n"})}),"\n",(0,t.jsx)(n.p,{children:"Copies file at src_path and stores in dest_folder"}),"\n",(0,t.jsxs)(n.p,{children:[":param"," src_path: Path or str  Path of source file ",":param"," dest_folder: Path or str  Path to folder in which source file is saved  The copied file has the same name as the source file  Ex) test.py saved on 9/26/22 at 1:30pm will be saved as 220926_1330_test.py"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/spikedetector/utils.py#L65",children:(0,t.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,t.jsxs)(n.h2,{id:"function-round",children:[(0,t.jsx)("kbd",{children:"function"})," ",(0,t.jsx)(n.code,{children:"round"})]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"round(n)\n"})}),"\n",(0,t.jsx)(n.p,{children:"Rounds a float (n) to the nearest integer Uses standard math rounding, i.e. 0.5 rounds up to 1"}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)("a",{href:"https://github.com/braingeneers/brainloop/blob/main/brainloop/core/spikedetector/utils.py#L79",children:(0,t.jsx)("img",{align:"right",style:{float:"right"},src:"https://img.shields.io/badge/-source-cccccc?style=flat-square"})}),"\n",(0,t.jsxs)(n.h2,{id:"function-torch_to_np",children:[(0,t.jsx)("kbd",{children:"function"})," ",(0,t.jsx)(n.code,{children:"torch_to_np"})]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"torch_to_np(tensor)\n"})}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.p,{children:(0,t.jsxs)(n.em,{children:["This file was automatically generated via ",(0,t.jsx)(n.a,{href:"https://github.com/ml-tooling/lazydocs",children:"lazydocs"}),"."]})})]})}function h(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(a,{...e})}):a(e)}},8453:(e,n,s)=>{s.d(n,{R:()=>c,x:()=>r});var t=s(6540);const i={},o=t.createContext(i);function c(e){const n=t.useContext(o);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:c(e.components),t.createElement(o.Provider,{value:n},e.children)}}}]);