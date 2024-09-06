import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  image: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Modular Experiment Design',
    image: require('@site/static/img/modular_design.png').default,
    description: (
      <>
        Design complex neural stimulation experiments with ease. BrainDance's
        phase-based system allows you to combine recording, stimulation, and
        analysis modules for sophisticated protocols.
      </>
    ),
  },
  {
    title: 'Real-time Control',
    image: require('@site/static/img/realtime_control.png').default,
    description: (
      <>
        Achieve precise control over stimulation parameters and timing. Interact
        with micro electrode arrays in real-time, adjusting your experiment on
        the fly based on neural responses.
      </>
    ),
  },
  {
    title: 'Integrated Analysis Tools',
    image: require('@site/static/img/data_analysis.png').default,
    description: (
      <>
        From raw data processing to advanced statistical methods, BrainDance
        provides built-in tools for visualizing and analyzing your experimental
        results, streamlining your research workflow.
      </>
    ),
  },
];

function Feature({title, image, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img className={styles.featureImg} src={image} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}