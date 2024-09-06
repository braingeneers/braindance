import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/BrainLoop/__docusaurus/debug',
    component: ComponentCreator('/BrainLoop/__docusaurus/debug', 'a2a'),
    exact: true
  },
  {
    path: '/BrainLoop/__docusaurus/debug/config',
    component: ComponentCreator('/BrainLoop/__docusaurus/debug/config', '7df'),
    exact: true
  },
  {
    path: '/BrainLoop/__docusaurus/debug/content',
    component: ComponentCreator('/BrainLoop/__docusaurus/debug/content', '9dc'),
    exact: true
  },
  {
    path: '/BrainLoop/__docusaurus/debug/globalData',
    component: ComponentCreator('/BrainLoop/__docusaurus/debug/globalData', '3d1'),
    exact: true
  },
  {
    path: '/BrainLoop/__docusaurus/debug/metadata',
    component: ComponentCreator('/BrainLoop/__docusaurus/debug/metadata', 'f14'),
    exact: true
  },
  {
    path: '/BrainLoop/__docusaurus/debug/registry',
    component: ComponentCreator('/BrainLoop/__docusaurus/debug/registry', 'b9d'),
    exact: true
  },
  {
    path: '/BrainLoop/__docusaurus/debug/routes',
    component: ComponentCreator('/BrainLoop/__docusaurus/debug/routes', 'b73'),
    exact: true
  },
  {
    path: '/BrainLoop/docs',
    component: ComponentCreator('/BrainLoop/docs', '4b9'),
    routes: [
      {
        path: '/BrainLoop/docs',
        component: ComponentCreator('/BrainLoop/docs', 'b47'),
        routes: [
          {
            path: '/BrainLoop/docs',
            component: ComponentCreator('/BrainLoop/docs', 'dc2'),
            routes: [
              {
                path: '/BrainLoop/docs',
                component: ComponentCreator('/BrainLoop/docs', 'f36'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/category/documentation',
                component: ComponentCreator('/BrainLoop/docs/category/documentation', '3be'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/category/rt-sort',
                component: ComponentCreator('/BrainLoop/docs/category/rt-sort', '0ea'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/category/usage',
                component: ComponentCreator('/BrainLoop/docs/category/usage', '147'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/core-concepts',
                component: ComponentCreator('/BrainLoop/docs/core-concepts', '092'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/artifact_removal.py',
                component: ComponentCreator('/BrainLoop/docs/docs/artifact_removal.py', '0e2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/base_env.py',
                component: ComponentCreator('/BrainLoop/docs/docs/base_env.py', '677'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/data_loader.py',
                component: ComponentCreator('/BrainLoop/docs/docs/data_loader.py', '052'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/dummy_maxlab.py',
                component: ComponentCreator('/BrainLoop/docs/docs/dummy_maxlab.py', '77f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/dummy_zmq_np.py',
                component: ComponentCreator('/BrainLoop/docs/docs/dummy_zmq_np.py', 'd2a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/maxwell_env.py',
                component: ComponentCreator('/BrainLoop/docs/docs/maxwell_env.py', '610'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/maxwell_utils.py',
                component: ComponentCreator('/BrainLoop/docs/docs/maxwell_utils.py', '495'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/params.py',
                component: ComponentCreator('/BrainLoop/docs/docs/params.py', '997'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/phases.py',
                component: ComponentCreator('/BrainLoop/docs/docs/phases.py', '137'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/plot.py',
                component: ComponentCreator('/BrainLoop/docs/docs/plot.py', '6bc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/query_electrodes.py',
                component: ComponentCreator('/BrainLoop/docs/docs/query_electrodes.py', 'c05'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/select_electrodes.py',
                component: ComponentCreator('/BrainLoop/docs/docs/select_electrodes.py', 'ba0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/spikedetector.py',
                component: ComponentCreator('/BrainLoop/docs/docs/spikedetector.py', '7a0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/train.py',
                component: ComponentCreator('/BrainLoop/docs/docs/train.py', 'ae0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/trainer.py',
                component: ComponentCreator('/BrainLoop/docs/docs/trainer.py', '218'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/docs/utils.py',
                component: ComponentCreator('/BrainLoop/docs/docs/utils.py', 'cf9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/quick-start',
                component: ComponentCreator('/BrainLoop/docs/quick-start', '0ff'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/RT-sort/api-reference',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/api-reference', '256'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/RT-sort/installation',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/installation', '74a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/RT-sort/introduction',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/introduction', '818'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/RT-sort/usage/load-detection-model',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/usage/load-detection-model', 'eb1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/RT-sort/usage/real-time-application',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/usage/real-time-application', '961'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/RT-sort/usage/sequence-detection',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/usage/sequence-detection', 'e75'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/BrainLoop/docs/RT-sort/usage/training-models',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/usage/training-models', 'c76'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/BrainLoop/',
    component: ComponentCreator('/BrainLoop/', 'bd8'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
