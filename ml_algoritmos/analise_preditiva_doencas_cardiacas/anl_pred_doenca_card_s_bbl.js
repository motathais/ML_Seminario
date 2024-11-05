// Algoritmo que não utiliza biblioteca para análise preditiva de doença cardíaca com Random Forest

const fs = require('fs'); // Importa o módulo fs para manipulação de arquivos.

// Função para carregar dados
function loadData(filePath) {
  const fileContent = fs.readFileSync(filePath, 'utf8'); // Lê o conteúdo do arquivo CSV.
  const lines = fileContent.trim().split('\n'); // Divide o conteúdo do arquivo em linhas.
  const headers = lines[0].split(','); // Extrai os cabeçalhos das colunas (primeira linha).
  const data = lines.slice(1).map(line => line.split(',').map(Number)); // Converte as linhas em arrays de números.

  const features = data.map(row => row.slice(0, -1)); // Extrai os atributos das amostras, ignorando a última coluna (etiqueta).
  const labels = data.map(row => row[row.length - 1]); // Extrai as etiquetas (última coluna).
  
  return { features, labels }; // Retorna as características e etiquetas.
}

// Função para calcular o Gini impurity
function giniImpurity(groups, classes) {
  const nInstances = groups.reduce((acc, group) => acc + group.length, 0); // Calcula o número total de amostras.
  let gini = 0.0;

  for (const group of groups) {
    const size = group.length;
    if (size === 0) continue; // Evita divisões por zero.

    let score = 0.0;
    for (const classVal of classes) {
      const proportion = group.filter(row => row[row.length - 1] === classVal).length / size; // Proporção de cada classe no grupo.
      score += proportion * proportion; // Soma dos quadrados das proporções.
    }
    gini += (1.0 - score) * (size / nInstances); // Calcula o Gini para o grupo e ajusta pela proporção de instâncias.
  }
  return gini; // Retorna o valor de Gini, indicando a "impureza" da divisão.
}

// Dividir o dataset com base em um atributo
function testSplit(index, value, dataset) {
  const left = []; // Subconjunto de amostras que atendem a condição.
  const right = []; // Subconjunto de amostras que não atendem a condição.
  for (const row of dataset) {
    if (row[index] < value) left.push(row);
    else right.push(row);
  }
  return [left, right]; // Retorna os grupos divididos.
}

// Selecionar a melhor divisão
function getSplit(dataset) {
  const classValues = [...new Set(dataset.map(row => row[row.length - 1]))]; // Identifica classes únicas nas etiquetas.
  let bestIndex, bestValue, bestScore = 1, bestGroups;

  // Selecionar 80% das features de forma aleatória
  const numFeatures = dataset[0].length - 1; // Total de características, excluindo a etiqueta.
  const featureSubsetSize = Math.floor(numFeatures * 0.8); // Define 80% das características.
  const featureIndices = [];

  while (featureIndices.length < featureSubsetSize) { // Seleciona índices aleatórios para as features.
    const randomIndex = Math.floor(Math.random() * numFeatures);
    if (!featureIndices.includes(randomIndex)) {
      featureIndices.push(randomIndex);
    }
  }

  // Testa a divisão apenas nas características selecionadas aleatoriamente
  for (let index of featureIndices) {
    for (const row of dataset) {
      const groups = testSplit(index, row[index], dataset); // Divide o dataset.
      const gini = giniImpurity(groups, classValues); // Calcula a impureza de Gini da divisão.
      if (gini < bestScore) { // Atualiza se a divisão melhorar a impureza.
        bestIndex = index;
        bestValue = row[index];
        bestScore = gini;
        bestGroups = groups;
      }
    }
  }
  return { index: bestIndex, value: bestValue, groups: bestGroups }; // Retorna o melhor ponto de divisão.
}

// Criar um nó folha
function toTerminal(group) {
  const outcomes = group.map(row => row[row.length - 1]); // Obtém as classes no grupo.
  return outcomes.sort((a, b) =>
    outcomes.filter(v => v === a).length - outcomes.filter(v => v === b).length
  ).pop(); // Retorna a classe mais comum.
}

// Dividir o nó ou criar folha
function split(node, maxDepth, minSize, depth) {
  const [left, right] = node.groups;
  delete node.groups; // Remove os grupos do nó atual.

  if (!left.length || !right.length) { // Caso não haja divisão, cria um nó folha.
    node.left = node.right = toTerminal(left.concat(right));
    return;
  }

  if (depth >= maxDepth) { // Caso a profundidade máxima seja atingida, cria nós folhas.
    node.left = toTerminal(left);
    node.right = toTerminal(right);
    return;
  }

  if (left.length <= minSize) node.left = toTerminal(left); // Garante o tamanho mínimo para dividir.
  else {
    node.left = getSplit(left); // Realiza a divisão do lado esquerdo.
    split(node.left, maxDepth, minSize, depth + 1); // Continua dividindo recursivamente.
  }

  if (right.length <= minSize) node.right = toTerminal(right); // Garante o tamanho mínimo para dividir.
  else {
    node.right = getSplit(right); // Realiza a divisão do lado direito.
    split(node.right, maxDepth, minSize, depth + 1); // Continua dividindo recursivamente.
  }
}

// Construir árvore de decisão
function buildTree(train, maxDepth, minSize) {
  const root = getSplit(train); // Determina o ponto de divisão inicial.
  split(root, maxDepth, minSize, 1); // Realiza a divisão recursiva.
  return root; // Retorna a árvore construída.
}

// Fazer uma previsão com uma árvore
function predict(node, row) {
  if (row[node.index] < node.value) { // Verifica se vai para a esquerda ou direita.
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
    while (sample.length < sampleSize) { // Seleciona uma amostra com reposição.
      sample.push(train[Math.floor(Math.random() * train.length)]);
    }
    const tree = buildTree(sample, maxDepth, minSize); // Constrói uma árvore com a amostra.
    forest.push(tree); // Adiciona a árvore à floresta.
  }
  return forest; // Retorna a floresta de árvores.
}

// Fazer uma previsão com a floresta
function baggingPredict(forest, row) {
  const predictions = forest.map(tree => predict(tree, row)); // Coleta previsões de todas as árvores.
  return predictions.sort((a, b) =>
    predictions.filter(v => v === a).length - predictions.filter(v => v === b).length
  ).pop(); // Retorna a previsão mais comum (voto majoritário).
}

// Avaliar a precisão do modelo
function accuracyMetric(actual, predicted) {
  let correct = 0;
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] === predicted[i]) correct++; // Conta as previsões corretas.
  }
  return (correct / actual.length) * 100.0; // Calcula a acurácia.
}

// Carregar e treinar o modelo
const trainData = loadData('../bases/treino.csv'); // Carrega os dados de treino.
const testData = loadData('../bases/teste.csv'); // Carrega os dados de teste.

const nTrees = 10; // Número de árvores na floresta.
const maxDepth = 5; // Profundidade máxima da árvore.
const minSize = 10; // Número mínimo de amostras para dividir um nó.
const sampleSize = Math.floor(trainData.features.length * 0.8); // Tamanho da amostra para cada árvore.

const forest = randomForest(trainData.features.map((row, i) => [...row, trainData.labels[i]]), maxDepth, minSize, sampleSize, nTrees); // Cria a floresta de decisão.

// Previsões e avaliação
const predictions = testData.features.map(row => baggingPredict(forest, row)); // Gera previsões para o conjunto de teste.
const accuracy = accuracyMetric(testData.labels, predictions); // Calcula a acurácia.
console.log(`Acurácia do modelo: ${accuracy.toFixed(2)}%`); // Exibe a acurácia do modelo.

