# Análise Exploratória do dataset Ames.

# Como utilizar
* É possível instalar a aplicação de duas formas:
  - Clonagem do repositório utilizando o seguinte comando no terminal: `git clone https://github.com/JoaoLucasMBC/ames.git`.
  - Ou baixar o arquivo zip desse repositório em `Code > Download Zip`. E descompactá-lo onde preferir.

## Arrumando seu ambiente

### Anaconda

It is preferrable to use Anaconda (https://anaconda.org/) for this project.

Install Anaconda and then create the environment for this project with the command:

``` bash
conda env create -f environment.yml
```

This will install an environment called "ames" with the latest Python for you. 

Just activate the environment.

``` bash
conda activate ames
```

### Virtual Enviroment | Python Puro 
* Para máquina virtual execute: 
``` bash 
python venv env
``` 
* Depois ative a mesma. 

* Em seguida execute o comando No diretório principal do projeto clonado: 
``` bash 
pip install -r requirements.txt
```  

# API REST Preditora de Preço
Foi desenvolvida uma API que permite com que o usuário envie um JSON contendo as features de uma casa (seguindo as features originais do dataset AMES) e receba uma estimativa do valor da casa (em doláres). 

### Como Usar:

* É necessário estar com o projeto baixado ou clonado

* Execute o comando abaixo no diretório ```/api```:

``` bash 
python app.py
```

* A rota ```'/ames/predict'``` é uma rota POST, que recebe um JSON, e realiza a predição

* Se a rota não receber um JSON ela retorna um erro

* Se a rota receber uma feature que não se encontra no AMES original ela retorna um erro

* Se a rota não receber alguma feature do AMES original mas nenhum dos erros acima for cometido, ela completa os mesmos (Categóricas -> Mediana , Numéricas -> Média)

### Video Tutorial
