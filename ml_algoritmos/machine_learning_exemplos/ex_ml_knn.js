/*algoritmo knn*/

// Função para calcular a distância euclidiana entre dois pontos
function euclideanDistance(point1, point2) {
    const distance = Math.sqrt(
        point1.reduce((acc, val, i) => acc + (val - point2[i]) ** 2, 0)
    );
    return distance;
}

// Função para encontrar os K vizinhos mais próximos
function getNeighbors(trainingSet, testInstance, k) {
    const distances = trainingSet.map(instance => {
        const distance = euclideanDistance(instance.slice(0, -1), testInstance);
        return { instance, distance };
    });

    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, k).map(neighbor => neighbor.instance);
}

// Função para fazer a predição com base nos vizinhos
function predict(trainingSet, testInstance, k) {
    const neighbors = getNeighbors(trainingSet, testInstance, k);
    const classVotes = {};

    neighbors.forEach(neighbor => {
        const response = neighbor[neighbor.length - 1]; // Última coluna é a classe
        classVotes[response] = (classVotes[response] || 0) + 1;
    });

    // Retorna a classe mais votada
    return Object.keys(classVotes).reduce((a, b) => classVotes[a] > classVotes[b] ? a : b);
}

// Função para rodar o algoritmo KNN em um conjunto de teste
function kNearestNeighbors(trainingSet, testSet, k) {
    return testSet.map(testInstance => predict(trainingSet, testInstance, k));
}

// Exemplo de uso
const dataset = [
    [2.7810836, 2.550537003, 'A'],
    [1.465489372, 2.362125076, 'A'],
    [3.396561688, 4.400293529, 'B'],
    [1.38807019, 1.850220317, 'A'],
    [3.06407232, 3.005305973, 'B'],
    [7.627531214, 2.759262235, 'B'],
    [5.332441248, 2.088626775, 'B'],
    [6.922596716, 1.77106367, 'B'],
    [8.675418651, -0.242068655, 'B'],
    [7.673756466, 3.508563011, 'B']
];

const testSet = [
    [2.5, 2.5],
    [6.5, 3.0]
];

// Número de vizinhos (K)
const k = 3;

// Executa o KNN e faz as previsões
const predictions = kNearestNeighbors(dataset, testSet, k);
console.log(predictions);

/*
Explicação do Algoritmo
Distância Euclidiana:

A função euclideanDistance calcula a distância entre dois pontos no espaço multi-dimensional. No caso da classificação KNN, os pontos são representados pelos atributos de cada exemplo.
Seleção dos K Vizinhos:

A função getNeighbors encontra os K exemplos de treinamento mais próximos (com menor distância) da instância de teste. Ela ordena as distâncias e seleciona os K primeiros.
Predição:

A função predict calcula qual classe tem o maior número de votos entre os K vizinhos mais próximos e retorna essa classe.
Execução do Algoritmo:

A função kNearestNeighbors usa o conjunto de treinamento para prever a classe de cada instância no conjunto de teste, usando o valor de K fornecido.
*/
