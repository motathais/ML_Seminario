const fs = require('fs');

// Função para carregar dados
function loadData(filePath) {
  const fileContent = fs.readFileSync(filePath, 'utf8');
  const lines = fileContent.trim().split('\n');
  const headers = lines[0].split(',');
  const data = lines.slice(1).map(line => line.split(',').map(Number));
  
  const features = data.map(row => row.slice(0, -1));
  const labels = data.map(row => row[row.length - 1]);
  
  return { features, labels };
}

// Função para calcular o Gini impurity
function giniImpurity(groups, classes) {
  const nInstances = groups.reduce((acc, group) => acc + group.length, 0);
  let gini = 0.0;

  for (const group of groups) {
    const size = group.length;
    if (size === 0) continue;

    let score = 0.0;
    for (const classVal of classes) {
      const proportion = group.filter(row => row[row.length - 1] === classVal).length / size;
      score += proportion * proportion;
    }
    gini += (1.0 - score) * (size / nInstances);
  }
  return gini;
}

// Dividir o dataset com base em um atributo
function testSplit(index, value, dataset) {
  const left = [];
  const right = [];
  for (const row of dataset) {
    if (row[index] < value) left.push(row);
    else right.push(row);
  }
  return [left, right];
}

// Selecionar a melhor divisão
/*function getSplit(dataset) {
  const classValues = [...new Set(dataset.map(row => row[row.length - 1]))];
  let bestIndex, bestValue, bestScore = 1, bestGroups;
  
  for (let index = 0; index < dataset[0].length - 1; index++) {
    for (const row of dataset) {
      const groups = testSplit(index, row[index], dataset);
      const gini = giniImpurity(groups, classValues);
      if (gini < bestScore) {
        bestIndex = index;
        bestValue = row[index];
        bestScore = gini;
        bestGroups = groups;
      }
    }
  }
  return { index: bestIndex, value: bestValue, groups: bestGroups };
}*/
function getSplit(dataset) {
  const classValues = [...new Set(dataset.map(row => row[row.length - 1]))];
  let bestIndex, bestValue, bestScore = 1, bestGroups;

  // Selecionar 90% das features de forma aleatória
  const numFeatures = dataset[0].length - 1; // Exclui a coluna de label
  const featureSubsetSize = Math.floor(numFeatures * 0.9);
  const featureIndices = [];

  // Gera um array com índices de features aleatórios
  while (featureIndices.length < featureSubsetSize) {
    const randomIndex = Math.floor(Math.random() * numFeatures);
    if (!featureIndices.includes(randomIndex)) {
      featureIndices.push(randomIndex);
    }
  }

  // Tentar divisão com apenas as features no subset selecionado
  for (let index of featureIndices) {
    for (const row of dataset) {
      const groups = testSplit(index, row[index], dataset);
      const gini = giniImpurity(groups, classValues);
      if (gini < bestScore) {
        bestIndex = index;
        bestValue = row[index];
        bestScore = gini;
        bestGroups = groups;
      }
    }
  }
  return { index: bestIndex, value: bestValue, groups: bestGroups };
}

// Criar um nó folha
function toTerminal(group) {
  const outcomes = group.map(row => row[row.length - 1]);
  return outcomes.sort((a,b) =>
    outcomes.filter(v => v === a).length - outcomes.filter(v => v === b).length
  ).pop();
}

// Dividir o nó ou criar folha
function split(node, maxDepth, minSize, depth) {
  const [left, right] = node.groups;
  delete node.groups;

  if (!left.length || !right.length) {
    node.left = node.right = toTerminal(left.concat(right));
    return;
  }

  if (depth >= maxDepth) {
    node.left = toTerminal(left);
    node.right = toTerminal(right);
    return;
  }

  if (left.length <= minSize) node.left = toTerminal(left);
  else {
    node.left = getSplit(left);
    split(node.left, maxDepth, minSize, depth + 1);
  }

  if (right.length <= minSize) node.right = toTerminal(right);
  else {
    node.right = getSplit(right);
    split(node.right, maxDepth, minSize, depth + 1);
  }
}

// Construir árvore de decisão
function buildTree(train, maxDepth, minSize) {
  const root = getSplit(train);
  split(root, maxDepth, minSize, 1);
  return root;
}

// Fazer uma previsão com uma árvore
function predict(node, row) {
  if (row[node.index] < node.value) {
    if (typeof node.left === 'object') return predict(node.left, row);
    else return node.left;
  } else {
    if (typeof node.right === 'object') return predict(node.right, row);
    else return node.right;
  }
}

// Criar uma floresta de árvores de decisão
function randomForest(train, maxDepth, minSize, sampleSize, nTrees) {
  const forest = [];
  for (let i = 0; i < nTrees; i++) {
    const sample = [];
    while (sample.length < sampleSize) {
      sample.push(train[Math.floor(Math.random() * train.length)]);
    }
    const tree = buildTree(sample, maxDepth, minSize);
    forest.push(tree);
  }
  return forest;
}

// Fazer uma previsão com a floresta
function baggingPredict(forest, row) {
  const predictions = forest.map(tree => predict(tree, row));
  return predictions.sort((a, b) =>
    predictions.filter(v => v === a).length - predictions.filter(v => v === b).length
  ).pop();
}

// Avaliar a precisão do modelo
function accuracyMetric(actual, predicted) {
  let correct = 0;
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] === predicted[i]) correct++;
  }
  return correct / actual.length * 100.0;
}

// Carregar e treinar o modelo
const trainData = loadData('./treino.csv');
const testData = loadData('./teste.csv');

const nTrees = 10;
const maxDepth = 5;
const minSize = 10;
const sampleSize = Math.floor(trainData.features.length * 0.8);

const forest = randomForest(trainData.features.map((row, i) => [...row, trainData.labels[i]]), maxDepth, minSize, sampleSize, nTrees);

// Previsões e avaliação
const predictions = testData.features.map(row => baggingPredict(forest, row));
const accuracy = accuracyMetric(testData.labels, predictions);
console.log(`Acurácia do modelo: ${accuracy.toFixed(2)}%`);
