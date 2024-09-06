import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/braindance/docs',
    component: ComponentCreator('/braindance/docs', '553'),
    routes: [
      {
        path: '/braindance/docs',
        component: ComponentCreator('/braindance/docs', '483'),
        routes: [
          {
            path: '/braindance/docs',
            component: ComponentCreator('/braindance/docs', '4c6'),
            routes: [
              {
                path: '/braindance/docs',
                component: ComponentCreator('/braindance/docs', '101'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/category/documentation',
                component: ComponentCreator('/braindance/docs/category/documentation', '5cd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/category/rt-sort',
                component: ComponentCreator('/braindance/docs/category/rt-sort', '88e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/category/usage',
                component: ComponentCreator('/braindance/docs/category/usage', 'e09'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/core-concepts',
                component: ComponentCreator('/braindance/docs/core-concepts', '8b3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/artifact_removal.py',
                component: ComponentCreator('/braindance/docs/docs/artifact_removal.py', 'ff1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/base_env.py',
                component: ComponentCreator('/braindance/docs/docs/base_env.py', '5a4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/data_loader.py',
                component: ComponentCreator('/braindance/docs/docs/data_loader.py', '252'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/dummy_maxlab.py',
                component: ComponentCreator('/braindance/docs/docs/dummy_maxlab.py', '756'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/dummy_zmq_np.py',
                component: ComponentCreator('/braindance/docs/docs/dummy_zmq_np.py', '905'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/maxwell_env.py',
                component: ComponentCreator('/braindance/docs/docs/maxwell_env.py', '54b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/maxwell_utils.py',
                component: ComponentCreator('/braindance/docs/docs/maxwell_utils.py', 'bf2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/params.py',
                component: ComponentCreator('/braindance/docs/docs/params.py', '1e5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/phases.py',
                component: ComponentCreator('/braindance/docs/docs/phases.py', '350'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/plot.py',
                component: ComponentCreator('/braindance/docs/docs/plot.py', '826'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/query_electrodes.py',
                component: ComponentCreator('/braindance/docs/docs/query_electrodes.py', '7b9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/select_electrodes.py',
                component: ComponentCreator('/braindance/docs/docs/select_electrodes.py', '912'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/spikedetector.py',
                component: ComponentCreator('/braindance/docs/docs/spikedetector.py', '717'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/train.py',
                component: ComponentCreator('/braindance/docs/docs/train.py', 'da2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/trainer.py',
                component: ComponentCreator('/braindance/docs/docs/trainer.py', '258'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/docs/utils.py',
                component: ComponentCreator('/braindance/docs/docs/utils.py', '93c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/quick-start',
                component: ComponentCreator('/braindance/docs/quick-start', '510'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/RT-sort/api-reference',
                component: ComponentCreator('/braindance/docs/RT-sort/api-reference', '3c1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/RT-sort/installation',
                component: ComponentCreator('/braindance/docs/RT-sort/installation', 'ae4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/RT-sort/introduction',
                component: ComponentCreator('/braindance/docs/RT-sort/introduction', 'cc2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/RT-sort/usage/load-detection-model',
                component: ComponentCreator('/braindance/docs/RT-sort/usage/load-detection-model', '07a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/RT-sort/usage/real-time-application',
                component: ComponentCreator('/braindance/docs/RT-sort/usage/real-time-application', '333'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/RT-sort/usage/sequence-detection',
                component: ComponentCreator('/braindance/docs/RT-sort/usage/sequence-detection', 'a3f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/braindance/docs/RT-sort/usage/training-models',
                component: ComponentCreator('/braindance/docs/RT-sort/usage/training-models', '3ce'),
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
    path: '/braindance/',
    component: ComponentCreator('/braindance/', '649'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
