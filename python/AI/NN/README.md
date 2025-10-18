# Neural Network

Dette prosjektet er et lite eksperiment med 'Neural Network' med inspirasjon fra forelesning og egen research og notater. 

Dette prosjektet startet som et lite eksperiment med nevrale nettverk. 

Hovedmålet var å lage en enkel **3-2-1 feedforward neural network** for å klassifisere e-poster som spam eller ikke-spam basert på tre nøkkelord: `"free"`, `"win"`, og `"offer"`.
<br>(Kjent eksempel, også fra forelesning)

Senere ble denne modellen utvidet til en mer **generell arkitektur** som kan håndtere flere antall input-noder, flere hidden layers, og ulike output-noder. <br>
Modellen støtter også eget valg av `Activation` og `Loss`, med tilsvarende deriverte funksjoner. <br>
Dette gjør at samme kode kan brueks på andre prosjekter og datasett, ikke bare det spesielle tilfelle med spam.

Tester på blir i ulike filer, som f.eks. `spamtest.py` som tester NN på spam-ideen.

## Arkitektur
- **Input layer:** 3 noder (en for hver feature)
- **Hidden layer:** 2 noder, bruker ReLU aktivering
- **Output layer:** 1 node, bruker sigmoid for sannsynlighet (0-1)
- Generell implementasjon støtter flere antall layers og noder

## Funksjonalitet
- Neural Networket trenes med **Binary Cross-Entropy loss** (men modellen støtter også CCE)
- Data normaliseres med min-max scaling basert på treningssettet
- En enkel "generate sample" lager syntetisk data basert på enkel regel: minst 2 nøkkelord bestemmer spam
- Støtte for å lagre modellen som .pt fil og laste opp modell for bruk senere.

# Hvordan bruke
* Egne filer for egne tester kan kjøres som vanlig, e.g. `spamtest.py`

1. Importer Neural Network klassen:
```python
from neural_network import NeuralNet
from utility import relu_act, sigmoid_act, bce_loss
```
2. Lag datasett, gjerne med innebygd generator:
```python
X, y = generate_sample(#)
```
3. Lag og tren modellen
```python
nn = NeuralNet(input_size=3, hidden_size=[2], output_size=1,
               hidden_activation=relu_act,
               output_activation=sigmoid_act,
               loss_function=bce_loss,
               lr=0.1)
nn.train(X, y, epochs=500)
```
4. Evaluer på data som ikke er brukt i trening (evaluering ikke implementert, men print-funksjon)
```python
probs, classes = get_predictions(nn, X_test)
print_prediction(X_test, y_test, probs, classes)
```
5. Lagre modellen:
```python
n.save("spam_detector.pt")
```
6. Laast inn modellen:
```python
nn = NeuralNet.load("spam_detector.pt")
```