# Signal Plates Identifier

## Introdução
Este projeto é um modelo de inteligência artificial que utiliza de uma Rede Neural Convolucional Baseada em Regiões (RCNN). O objetivo deste modelo é identificar placas de trânsito do CONTRAN de fotografias através de visão computacional. 

Abaixo buscamos explicar como foi realizado o desenvolvimento deste modelo e também as dificuldades encontradas.

## Construção e Análise do Dados
### 1. Coleta de Dados
Dada a falta de imagens públicas de placas de trânsito em um banco de dados centralizado, decidimos obter os dados necessários através da técnica de _web scraping_. 
Para isso criamos um [repositório separado](https://github.com/caribeedu/Google-Image-Scraper) onde implementamos um _web scraper_ em Python focado em obter imagens do Google Imagens, por ele conseguimos passar termos de busca específicos e as imagens de resultado são baixadas na máquina local automaticamente.

Usando o seguinte padrão para os termos de busca; `placa <nome-da-placa> rua`, conseguimos obter mais de **7000** imagens como resultado. Com isso, fazendo uma seleção manual, atingimos **mais de 500** fotografias reais de diferentes placas de trânsito do CONTRAN. Este _dataset_ já filtrado pode ser acessado [aqui](https://drive.google.com/file/d/1d1ni3fdETzC6S7itu7IwAhSsYAypCoBC/view?usp=drive_link).

### 2. Anotação dos Dados
Uma vez com um _dataset_ sólido, seguimos para a anotação das imagens. Para esta tarefa, utilizamos o [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html), uma aplicação web simples que permite a anotação necessária. 

Decidimos neste momento utilizar o tipo de máscaras de poligonal, tentando buscar posteriormente uma maior precisão no nosso modelo. Ao todo, foram aproximadamente **566** imagens revisadas, onde destas, apenas 19 não eram válidas e foram descartadas.

Caso tenha interesse, você pode explorar este _dataset_ localmente, basta [baixá-lo](https://drive.google.com/file/d/1d1ni3fdETzC6S7itu7IwAhSsYAypCoBC/view?usp=drive_link), abrir o arquivo `via.html` com qualquer navegador, clicar em "Project > Load Project" e em seguida selecionar o arquivo `signal-plates-identifier-vgg-project.json`. A parir disso, é possível visualizar as anotações existentes e também adicionar novas imagens/anotações.

### 3. Análise dos Dados
Agora com o _dataset_ devidamente anotado, fizemos uma breve análise exploratória para entender melhor a relação das imagens obtidas. Basicamente a distribuição final de placas foi a seguinte:

| Código da Placa | Quantidade de Ocorrências |
|------------|-------------|
| r-1        | 71          |
| r-2        | 13          |
| r-3        | 35          |
| r-4a       | 31          |
| r-4b       | 23          |
| r-5a       | 13          |
| r-5b       | 1           |
| r-6a       | 117         |
| r-6b       | 125         |
| r-6c       | 48          |
| r-7        | 8           |
| r-8a       | -           |
| r-8b       | 1           |
| r-9        | 29          |
| r-10       | 1           |
| r-11       | 3           |
| r-12       | 5           |
| r-13       | -           |
| r-14       | 1           |
| r-15       | 12          |
| r-16       | -           |
| r-17       | 1           |
| r-18       | -           |
| r-19       | 51          |
| r-20       | 5           |
| r-21       | -           |
| r-22       | -           |
| r-23       | -           |
| r-24a      | 11          |
| r-24b      | 9           |
| r-25a      | 3           |
| r-25b      | 8           |
| r-25c      | 9           |
| r-25d      | 3           |
| r-26       | 9           |
| r-27       | 5           |
| r-28       | 5           |
| r-29       | 5           |
| r-30       | -           |
| r-31       | -           |
| r-32       | 15          |
| r-33       | 9           |
| r-34       | 13          |
| r-35a      | 1           |
| r-35b      | 1           |
| r-36a      | 3           |
| r-36b      | 3           |
| r-37       | 3           |
| r-38       | -           |
| r-39       | 3           |
| r-40       | 1           |

Apesar da distribuição má balanceada, decidimos seguir com o _dataset_ da maneira apresentada acima para que pudéssemos validar quais resultados conseguiríamos alcançar.

### 4. Transformação dos Dados
Dado que neste ponto já tínhamos o _dataset_ pronto, era necessário distribuí-lo em dois _datasets_, de treino e de validação, antes de seguirmos com o desenvolvimento do modelo. 
Para esta tarefa utilizamos o _notebook_ [`transform_data`](https://github.com/caribeedu/signal-plates-identifier/blob/main/transform_data.ipynb), com ele pegamos uma imagem de cada placa para o _dataset_ de validação e o restante foi para o _dataset_ de treinamento.

O resultado final do _dataset_ transformado pode ser acessado [aqui](https://drive.google.com/file/d/1xjfm5-oBZgm-RAe2CaXg5tEV6CYYP3e_/view?usp=sharing). É este _dataset_ que iremos utilizar durante o treinamento.

Observação: É importante destacar que para as placas que contínhamos apenas uma ocorrência, destinamos as imagens das mesmas para o _dataset_ de treinamento, uma vez que não faria sentido usá-las no _dataset_ de validação.

## Desenvolvimento e Avaliação do Modelo

### 1. Decisão de Tecnologias

### 2. Desenvolvimento do Modelo

### 3. Avaliação do Modelo

### 4. Comparativo com o YOLOv8

## Conclusões
