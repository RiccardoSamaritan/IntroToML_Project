# Proximal Policy Optimization (PPO) - Implementazione Custom

Un'implementazione completa dell'algoritmo PPO (Proximal Policy Optimization) in PyTorch con confronto sperimentale contro Stable-Baselines3 e baseline random, realizzata per l'esame del corso di "Introduzione al Machine Learning" dell'A.A. 2024/25 dell'Università di Trieste.

## 📋 Indice

- [Panoramica](#panoramica)
- [Architettura](#architettura)
- [Installazione](#installazione)
- [Struttura del Progetto](#struttura-del-progetto)
- [Utilizzo](#utilizzo)
- [Implementazione Tecnica](#implementazione-tecnica)
- [Risultati Sperimentali](#risultati-sperimentali)
- [Confronti e Benchmark](#confronti-e-benchmark)
- [Contributi](#contributi)
- [Riferimenti](#riferimenti)

## 🎯 Panoramica

Questo progetto implementa l'algoritmo **Proximal Policy Optimization (PPO)** partendo dal paper di Schulman et al. (2017).

### Caratteristiche Principali

- ✅ **Implementazione fedele al [paper originale](https://arxiv.org/abs/1707.06347)** con tutti i componenti chiave
- ✅ **Architettura Actor-Critic** con reti neurali condivise
- ✅ **Clipped Surrogate Objective** per stabilità degli aggiornamenti
- ✅ **Generalized Advantage Estimation (GAE)** per ridurre la varianza
- ✅ **Minibatch training** con multiple epoche per efficienza
- ✅ **Confronto sperimentale** con Stable-Baselines3 e baseline random
- ✅ **Sistema di analisi e visualizzazione** delle performance

## 🏗️ Architettura

Il progetto è strutturato in moduli modulari e riutilizzabili:

```
├── ppo.py              # Implementazione PPO custom
├── sb3_model.py        # Wrapper Stable-Baselines3
├── random_model.py     # Baseline random agent
├── comparison.py       # Sistema di confronto e analisi
├── requirements.txt   
└── README.md          
```

## 🚀 Installazione

```bash
# Clona il repository
git clone <repository-url>
cd ppo-implementation

# Installa le dependencies
pip install -r requirements.txt
```

## 📁 Struttura del Progetto

```
ppo-implementation/
│
├── ppo.py                 
│   ├── ActorCritic        # Rete neurale Actor-Critic
│   ├── PPOAgent          # Agente PPO con algoritmo completo
│   └── train_ppo()       # Funzione di training
│
├── sb3_model.py          # Integrazione Stable-Baselines3
│   ├── SB3TrainingCallback
│   └── train_sb3_ppo()
│
├── random_model.py       # Baseline random
│   ├── RandomAgent
│   └── train_random_agent()
│
├── comparison.py         # Sistema di confronto
│   └── compare()        # Analisi comparativa modelli
│
└── requirements.txt      # Dipendenze Python
```
