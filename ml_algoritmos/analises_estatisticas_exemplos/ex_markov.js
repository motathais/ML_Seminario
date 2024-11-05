//Método Markov:

function buildMarkovChain(data) {
    const markovChain = {};
 
    // Dividindo o texto em palavras
    const words = data.split(' ');
 
    // Construindo a cadeia de Markov
    for (let i = 0; i < words.length - 1; i++) {
        const word = words[i];
        const nextWord = words[i + 1];
 
        if (!markovChain[word]) {
            markovChain[word] = [];
        }
 
        markovChain[word].push(nextWord);
    }
 
    return markovChain;
}
 
function generateText(markovChain, startWord, numWords = 20) {
    let currentWord = startWord;
    const result = [currentWord];
 
    for (let i = 0; i < numWords - 1; i++) {
        const nextWords = markovChain[currentWord];
 
        if (!nextWords) {
            break; // Se não houver palavras seguintes, termina a geração
        }
 
        // Escolhe uma palavra aleatória entre as opções
        currentWord = nextWords[Math.floor(Math.random() * nextWords.length)];
        result.push(currentWord);
    }
 
    return result.join(' ');
}
 
// Exemplo de uso
const textData = "O gato pulou sobre o muro. O cachorro latiu para o gato. O gato saiu correndo.";
const markovChain = buildMarkovChain(textData);
const generatedText = generateText(markovChain, "O", 15);
console.log(generatedText);