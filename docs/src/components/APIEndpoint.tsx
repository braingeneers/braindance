import React from 'react';

interface Parameter {
  name: string;
  description: string;
}

interface APIEndpointProps {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  endpoint: string;
  description: string;
  parameters: Parameter[];
  returns: string;
}

const APIEndpoint: React.FC<APIEndpointProps> = ({ method, endpoint, description, parameters, returns }) => (
  <div className="api-endpoint">
    <h3>
      <span className={`method ${method.toLowerCase()}`}>{method}</span>
      <code>{endpoint}</code>
    </h3>
    <p>{description}</p>
    <h4>Parameters:</h4>
    <ul>
      {parameters.map((param, index) => (
        <li key={index}>
          <code>{param.name}</code>: {param.description}
        </li>
      ))}
    </ul>
    <h4>Returns:</h4>
    <p>{returns}</p>
  </div>
);

export default APIEndpoint;