/*algoritmo SVM*/

// Função para calcular o produto escalar entre dois vetores
function dotProduct(vector1, vector2) {
    return vector1.reduce((acc, val, i) => acc + val * vector2[i], 0);
}

// Função SVM que treina o modelo e ajusta os pesos para encontrar o hiperplano
function svmTrain(trainingSet, labels, learningRate, epochs) {
    let weights = new Array(trainingSet[0].length).fill(0);
    let bias = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
        for (let i = 0; i < trainingSet.length; i++) {
            const input = trainingSet[i];
            const label = labels[i];

            const linearOutput = dotProduct(weights, input) + bias;
            const prediction = label * linearOutput;

            // Se a predição estiver incorreta, atualize os pesos
            if (prediction <= 0) {
                for (let j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * label * input[j];
                }
                bias += learningRate * label;
            }
        }
    }

    return { weights, bias };
}

// Função para fazer a predição usando o modelo treinado
function svmPredict(input, model) {
    const { weights, bias } = model;
    const linearOutput = dotProduct(weights, input) + bias;
    return linearOutput >= 0 ? 1 : -1;
}

// Exemplo de uso
const trainingSet = [
    [2, 3],
    [1, 1],
    [2, 0],
    [0, 1],
    [3, 3],
    [3, 0]
];

const labels = [1, -1, 1, -1, 1, -1]; // Classes: 1 e -1

// Treina o modelo SVM
const learningRate = 0.1;
const epochs = 100;
const model = svmTrain(trainingSet, labels, learningRate, epochs);

// Testa o modelo com novos dados
const testSet = [
    [2, 2],
    [0, 0],
    [3, 1]
];

testSet.forEach(input => {
    const prediction = svmPredict(input, model);
    console.log(`Predição para ${input}: Classe ${prediction}`);
});

/*Explicação do Algoritmo
Produto Escalar:

A função dotProduct calcula o produto escalar entre dois vetores. Isso é importante para calcular a distância de um ponto em relação ao hiperplano.
Treinamento do Modelo (SVM):

A função svmTrain ajusta os pesos do hiperplano usando gradiente descendente para maximizar a margem entre as classes. O modelo é ajustado por um número fixo de epochs (iterações), e os pesos são atualizados apenas se a predição estiver incorreta.
Predição:

A função svmPredict usa os pesos treinados para fazer predições. Se o valor do produto escalar mais o viés (bias) for positivo, a classe será 1; caso contrário, será -1. */