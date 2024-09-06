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
    component: ComponentCreator('/BrainLoop/docs', '2c1'),
    routes: [
      {
        path: '/BrainLoop/docs',
        component: ComponentCreator('/BrainLoop/docs', 'd0e'),
        routes: [
          {
            path: '/BrainLoop/docs',
            component: ComponentCreator('/BrainLoop/docs', 'a5e'),
            routes: [
              {
                path: '/BrainLoop/docs',
                component: ComponentCreator('/BrainLoop/docs', 'f8d'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/core-concepts',
                component: ComponentCreator('/BrainLoop/docs/core-concepts', '30c'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/artifact_removal.py',
                component: ComponentCreator('/BrainLoop/docs/docs/artifact_removal.py', '4bb'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/base_env.py',
                component: ComponentCreator('/BrainLoop/docs/docs/base_env.py', '507'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/data_loader.py',
                component: ComponentCreator('/BrainLoop/docs/docs/data_loader.py', 'ebd'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/dummy_maxlab.py',
                component: ComponentCreator('/BrainLoop/docs/docs/dummy_maxlab.py', 'ab9'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/dummy_zmq_np.py',
                component: ComponentCreator('/BrainLoop/docs/docs/dummy_zmq_np.py', 'a26'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/maxwell_env.py',
                component: ComponentCreator('/BrainLoop/docs/docs/maxwell_env.py', 'b89'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/maxwell_utils.py',
                component: ComponentCreator('/BrainLoop/docs/docs/maxwell_utils.py', '8fe'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/params.py',
                component: ComponentCreator('/BrainLoop/docs/docs/params.py', '3a7'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/phases.py',
                component: ComponentCreator('/BrainLoop/docs/docs/phases.py', '1b1'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/plot.py',
                component: ComponentCreator('/BrainLoop/docs/docs/plot.py', '745'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/query_electrodes.py',
                component: ComponentCreator('/BrainLoop/docs/docs/query_electrodes.py', 'ca7'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/select_electrodes.py',
                component: ComponentCreator('/BrainLoop/docs/docs/select_electrodes.py', '969'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/spikedetector.py',
                component: ComponentCreator('/BrainLoop/docs/docs/spikedetector.py', '4d3'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/train.py',
                component: ComponentCreator('/BrainLoop/docs/docs/train.py', '186'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/trainer.py',
                component: ComponentCreator('/BrainLoop/docs/docs/trainer.py', 'ab5'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/docs/utils.py',
                component: ComponentCreator('/BrainLoop/docs/docs/utils.py', '423'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/quick-start',
                component: ComponentCreator('/BrainLoop/docs/quick-start', 'd99'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/RT-sort/api-reference',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/api-reference', '829'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/RT-sort/installation',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/installation', '39f'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/RT-sort/introduction',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/introduction', 'a8b'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/RT-sort/usage/load-detection-model',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/usage/load-detection-model', '2ea'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/RT-sort/usage/real-time-application',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/usage/real-time-application', 'e15'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/RT-sort/usage/sequence-detection',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/usage/sequence-detection', '391'),
                exact: true
              },
              {
                path: '/BrainLoop/docs/RT-sort/usage/training-models',
                component: ComponentCreator('/BrainLoop/docs/RT-sort/usage/training-models', '41c'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
