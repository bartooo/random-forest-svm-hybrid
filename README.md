<h1 align="center">Hybrid of Random Forest with SVM</h1>
<h3 align="center"><a href="https://github.com/bartooo">Bartosz Cywiński</a> & <a href="https://github.com/lukasz-staniszewski">Łukasz Staniszewski</a> (Warsaw Univerity of Technology)</h3>
<div align="center">
<img src="https://user-images.githubusercontent.com/59453698/168869615-95ac05a3-49b5-4e45-9181-5c9fa80bb679.png" alt="banner">
</div>

<div align="center">
  ENG: This repository contains college Machine Learning course project - the implementation of hybrid Random Forest model with Support Vector Machines (SVM).

  PL: Repozytorium zawiera projekt wykonany w ramach przedmiotu Uczenie Maszynowe - jest to implementacja hybrydy Lasu Losowego z Maszyną Wektorów Nośnych (SVM).
</div>

## About
ENG: Together with the project documentation has been created describing models from the algorithmic and mathematical side, as well as its implementation. It's right <a href="https://github.com/bartooo/random-forest-svm-hybrid/blob/main/doc/doc.pdf">here</a>.

PL: Razem z projektem powstała dokumentacja opisująca projekt, poszczególne modele od strony algorytmicznej i matematycznej, a także implementację. Znajduje się <a href="https://github.com/bartooo/random-forest-svm-hybrid/blob/main/doc/doc.pdf">tutaj</a>.

## Used technologies / Użyte technologie:
1. Ubuntu 20.04. LTS.
2. Python 3.8.10 with modules: NumPy, Black, PyTest, Jupyter.
## Instalation / Instalacja:
1. Download this repository and change cd. / Pobierz repozytorium i zmień w konsoli aktualny folder.
2. Create your own Python virtual environment. / Stwórz swoje własne środowisko wirtualne Python.

``` 
$ python3 -m venv venv
```

3. Activate venv. / Aktywuj środowisko.

``` 
> REPO_PATH\venv\scripts\activate (WINDOWS)
$ source REPO_PATH\venv\bin\activate (LINUX) 
```

4. Install necessary packages. / Zainstaluj odpowiednie biblioteki.

``` 
$ python3 -m pip install -r requirements.txt 
```

5. Run tests. / Uruchom Testy.

``` 
$ python3 −m pytest
```
6. Run hybrid model. / Uruchom model hybrydowy.
```
$ python3 main.py −−dataset breast_cancer −−n_folds 5
    −−num_classifiers 4 −−tree_max_depth 4 
    −−tree_min_entropy_diff 0.001
    −−tree_min_node_size 34 −−svm_lambda 0.05
```
7. Run model comparison experiments. / Uruchom eksperymenty dotyczące porównywania modeli.
```
$ python3 main_experiments.py −WHAT models
```
8. Run models parameters experiments. / Uruchom eksperymenty dotyczące parametrów modeli.
```
$ python3 main_experiments.py −WHAT parameters
```

