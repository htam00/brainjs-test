const brain = require('brain.js')
const tf    = require('@tensorflow/tfjs')

const network = new brain.NeuralNetwork()

// Train a simple model:
const model = tf.sequential()
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}))
model.add(tf.layers.dense({units: 1, activation: 'linear'}))
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})

const xs = tf.randomNormal([100, 10])
const ys = tf.randomNormal([100, 1])

model.fit(xs, ys, {
 epochs: 100,
 callbacks: {
  onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
 }
})

/*network.train([
 {input: [0,0,0], output: [0]},
 {input: [0,0,1], output: [0]},
 {input: [0,1,1], output: [0]},
 {input: [1,0,1], output: [1]},
 {input: [1,1,1], output: [1]}
])
*/

network.train([
 {input: [1,2], output: [1]}, // Team 2 wins
 {input: [1,3], output: [1]}, // Team 3 wins
 {input: [2,3], output: [0]}, // Team 2 wins
 {input: [2,4], output: [1]}, // Team 4 wins
 {input: [1,2], output: [0]}, // Team 1 wins
 {input: [1,3], output: [0]}, // Team 1 wins
 {input: [3,4], output: [0]}, // Team 3 wins
])


// output Data 
const output = network.run('I fixed the power supply')

console.log(`Prob: ${output}`)
