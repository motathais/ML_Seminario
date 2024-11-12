/*algoritmo regressão losistica*/

// Função para calcular a função sigmoide
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

// Função para treinar o modelo de regressão logística
function logisticTrain(trainingSet, labels, learningRate, epochs) {
    let weights = new Array(trainingSet[0].length).fill(0);
    let bias = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
        for (let i = 0; i < trainingSet.length; i++) {
            const input = trainingSet[i];
            const label = labels[i];

            // Calcula o valor da função linear
            const linearOutput = dotProduct(weights, input) + bias;
            const prediction = sigmoid(linearOutput);

            // Atualiza os pesos e o bias com base no erro
            const error = label - prediction;

            for (let j = 0; j < weights.length; j++) {
                weights[j] += learningRate * error * prediction * (1 - prediction) * input[j];
            }
            bias += learningRate * error * prediction * (1 - prediction);
        }
    }

    return { weights, bias };
}

// Função para fazer a predição
function logisticPredict(input, model) {
    const { weights, bias } = model;
    const linearOutput = dotProduct(weights, input) + bias;
    const probability = sigmoid(linearOutput);
    return probability >= 0.5 ? 1 : 0; // Classe 1 se a probabilidade >= 0.5, senão classe 0
}

// Função para calcular o produto escalar
function dotProduct(vector1, vector2) {
    return vector1.reduce((acc, val, i) => acc + val * vector2[i], 0);
}

// Exemplo de uso
const trainingSet = [
    [2.5, 2.1],
    [1.5, 2.3],
    [3.5, 3.3],
    [1.1, 1.1],
    [3.3, 3.0],
    [2.7, 2.8],
];

const labels = [1, 0, 1, 0, 1, 0]; // Classes: 1 e 0

// Treina o modelo de regressão logística
const learningRate = 0.1;
const epochs = 1000;
const model = logisticTrain(trainingSet, labels, learningRate, epochs);

// Conjunto de teste
const testSet = [
    [2.6, 2.0],
    [1.2, 1.5],
    [3.1, 3.1]
];

testSet.forEach(input => {
    const prediction = logisticPredict(input, model);
    console.log(`Predição para ${input}: Classe ${prediction}`);
});

/*
Explicação do Algoritmo
Função Sigmoide:
A função sigmoid converte a saída linear em uma probabilidade entre 0 e 1. Isso permite que o modelo faça predições probabilísticas.
Treinamento:
A função logisticTrain ajusta os pesos e o viés (bias) usando o gradiente descendente. Em cada iteração, o erro entre a previsão do modelo e o rótulo real é usado para ajustar os pesos.
Predição:
A função logisticPredict calcula a probabilidade para uma nova instância. Se a probabilidade for maior ou igual a 0,5, a classe predita será 1; caso contrário, será 0.
*/