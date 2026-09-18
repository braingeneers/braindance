// Run with: node --test tests/test_workshop_simulator_ui.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function editor() {
  const elements = new Map();
  function element() {
    return {value: '0', checked: false, style: {}, children: [], attributes: {},
      replaceChildren(...children) { this.children = children; },
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener() {}, append() {}, insertBefore() {},
      setPointerCapture() {}, getBoundingClientRect() { return {left: 0, top: 0, width: 400, height: 400}; },
      parentElement: {append() {}},
    };
  }
  const $ = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  $('channels').value = 4;
  $('dataSource').value = 'simulation';
  $('simEdge').value = '.35';
  $('simMode').value = 'move';
  let live = false;
  const errors = [];
  const context = vm.createContext({$, document: {createElement: element},
    Option: function(text, value) { return {text, value}; },
    views: {simulation: element()}, monitorGrid: element(),
    settings: {channels: 4, num_neurons: 4}, matrixValues: [], matrixDirty: false,
    renderMatrix() {}, applyMatrix() {}, running: () => live, error: message => errors.push(message),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../braindance/examples/streaming_workshop/simulator.js'), 'utf8'), context);
  return {$, errors, context, run: code => vm.runInContext(code, context), start: () => { live = true; }};
}

test('selecting a neuron synchronizes connection source and sets multiple directed edges', () => {
  const e = editor();
  e.$('simSelected').value = '2';
  e.$('simSelected').onchange();
  assert.equal(e.$('simFrom').value, 2);
  e.$('simTargets').children[0].onclick();
  e.$('simTargets').children[3].onclick();
  e.$('simConnectMany').onclick();
  assert.equal(e.context.matrixValues[0][2], .35);
  assert.equal(e.context.matrixValues[3][2], .35);
  assert.equal(e.context.matrixValues[2][0], 0);
  e.$('simDisconnectMany').onclick();
  assert.equal(e.context.matrixValues[0][2], 0);
  assert.equal(e.context.matrixValues[3][2], 0);
});

test('canvas connect mode retains source across targets and Shift removes a connection during a run', () => {
  const e = editor();
  e.run('neuronPositions=[[0,0],[17.5,0],[35,0],[52.5,0]];selectSimulatorNeuron(0)');
  e.$('simMode').value = 'connect';
  e.start();
  const point = index => e.run(`simProjection(400,400,simulatorElectrodes()).point(neuronPositions[${index}])`);
  const click = (index, shiftKey = false) => {
    const [clientX, clientY] = point(index);
    e.$('simCulture').onpointerdown({clientX, clientY, shiftKey});
  };
  click(1); click(2);
  assert.equal(e.run('simulatorSelection'), 0);
  assert.equal(e.context.matrixValues[1][0], .35);
  assert.equal(e.context.matrixValues[2][0], .35);
  click(1, true);
  assert.equal(e.context.matrixValues[1][0], 0);
  assert.equal(e.context.matrixValues[2][0], .35);
  assert.equal(e.run('simulatorDrag'), false);
});

test('bulk target selection excludes self and invalid weights never partially change the matrix', () => {
  const e = editor();
  e.$('simTargetsAll').onclick();
  assert.equal(e.run('simulatorTargets.size'), 3);
  e.$('simEdge').value = '3';
  e.$('simConnectMany').onclick();
  assert.ok(e.context.matrixValues.flat().every(value => value === 0));
  assert.match(e.errors.pop(), /between -2 and 2/);
  e.$('simEdge').value = '-.5';
  e.$('simConnectMany').onclick();
  assert.equal(e.context.matrixValues[0][0], 0);
  assert.equal(e.context.matrixValues[3][0], -.5);
  e.$('simFrom').value = '3'; e.$('simFrom').onchange();
  assert.equal(e.run('simulatorTargets.size'), 0);
  assert.equal(e.$('simSelected').value, 3);
});

test('external sources cannot change simulator connections', () => {
  const e = editor();
  e.$('dataSource').value = 'replay';
  e.$('simTargetsAll').onclick(); e.$('simConnectMany').onclick();
  assert.ok(e.context.matrixValues.flat().every(value => value === 0));
  assert.match(e.errors.pop(), /simulated data only/);
});
