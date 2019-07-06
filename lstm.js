// Dependencies
const brain = require('brain.js')
const data = require('./data.json') // getting file json

// initialize brain network 
const network = new brain.recurrent.LSTM()

// Mapping data callback item 
// input data text
// output data category
const trainingData = data.map(item => ({
  input: item.text, 
  output: item.category
}))

// Training Data
network.train(trainingData, {
 iterations: 2000
})

const output = network.run('my unit test faild')

console.log(`Category: ${output}`)
