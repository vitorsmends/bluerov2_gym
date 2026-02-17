#!/bin/bash

# Define o caminho para o executavel do python no seu venv
PYTHON=".venv/bin/python3"
EXP_DIR="experiments"

echo "=========================================================="
echo "Iniciando Bateria de Experimentos - BlueROV2"
echo "=========================================================="

# 1. Executar PID Simples
echo "[1/5] Executando PID Simples..."
$PYTHON $EXP_DIR/pid_experiment.py

# 2. Executar Cascaded PID
echo "[2/5] Executando Cascaded PID..."
$PYTHON $EXP_DIR/cascaded_pid_experiment.py

# 3. Executar SMC (Sliding Mode Control)
echo "[3/5] Executando Sliding Mode Control..."
$PYTHON $EXP_DIR/smc_experiment.py

# 4. Executar MPC (Model Predictive Control)
echo "[4/5] Executando Model Predictive Control..."
$PYTHON $EXP_DIR/mpc_experiment.py

# 5. Executar PPO (Agente treinado)
echo "[5/5] Executando PPO Agent..."
$PYTHON $EXP_DIR/ppo_experiment.py

echo "=========================================================="
echo "Todos os dados coletados. Gerando comparativo final..."
echo "=========================================================="

# Gerar o Plot
$PYTHON $EXP_DIR/plot_comparison.py

echo "Processo concluido. Verifique o arquivo de imagem gerado."