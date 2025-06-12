import numpy as np
import pandas as pd
import skfuzzy as fuzz #pip install scikit-fuzzy
from skfuzzy import control as ctrl
from datetime import datetime
import time
start_time = time.time()

data = []
dc = {}
dc['result1'] = 0
dc['result2'] = 0
dc['result3'] = 0


def fuzzy(VE1, VE2, VE3): 
    start_time = time.time()
    CFTV = ctrl.Antecedent(np.arange(0, 100, .1), 'disponibilidade_CFTV')
    SCAF = ctrl.Antecedent(np.arange(0, 100, .1), 'disponibilidade_SCAF')
    SA = ctrl.Antecedent(np.arange(0, 100, .1), 'disponibilidade_SA')
    # Funcoes de pertinencia das variaveis de entrada
    # disponibilidade_CFTV
    CFTV['baixa'] = fuzz.trapmf(CFTV.universe, [0, 0, 25,50]) #FP trapezoidal
    CFTV['media'] = fuzz.trimf(CFTV.universe,  [25,50,75]) #FP tringular
    CFTV['alta'] = fuzz.trapmf(CFTV.universe,  [50,75,100,100]) #FP trapezoidal
    #disponibilidade_SCAF
    SCAF['baixa'] = fuzz.trapmf(SCAF.universe, [0, 0, 25,50]) #FP trapezoidal
    SCAF['media'] = fuzz.trimf(SCAF.universe,  [25,50,75]) #FP tringular
    SCAF['alta'] = fuzz.trapmf(SCAF.universe,  [50,75,100,100]) #FP trapezoidal
    # disponibilidade_SA
    SA['baixa'] = fuzz.trapmf(SA.universe, [0, 0, 25,50]) #FP trapezoidal
    SA['media'] = fuzz.trimf(SA.universe, [25,50,75]) #FP tringular
    SA['alta'] = fuzz.trapmf(SA.universe, [50,75,100,100]) #FP trapezoidal
    # Domínio da variável de saída
    IVSF = ctrl.Consequent(np.arange(0, 100, .1), 'IVSF')
    # Funcoes de pertinencia das variaveis de saida
    IVSF['baixa'] = fuzz.trapmf(IVSF.universe,[0, 0, 12.5, 25])
    IVSF['media-baixa'] = fuzz.trimf(IVSF.universe, [12.5, 25, 50]) 
    IVSF['media'] = fuzz.trimf(IVSF.universe, [25, 50, 75]) 
    IVSF['media-alta'] = fuzz.trimf(IVSF.universe, [50, 75, 87.5 ]) 
    IVSF['alta'] = fuzz.trapmf(IVSF.universe, [75, 87.5, 100, 100]) 
    # Regras Fuzzy
    R1 =  ctrl.Rule(CFTV['baixa'] & SCAF['baixa'] & SA['baixa'], IVSF['alta'], label = 'R1')
    R2 =  ctrl.Rule(CFTV['baixa'] & SCAF['baixa'] & SA['media'], IVSF['alta'], label = 'R2')
    R3 =  ctrl.Rule(CFTV['baixa'] & SCAF['baixa'] & SA['alta'],  IVSF['alta'], label = 'R3')
    R4 =  ctrl.Rule(CFTV['baixa'] & SCAF['media'] & SA['baixa'], IVSF['alta'], label = 'R4')
    R5 =  ctrl.Rule(CFTV['baixa'] & SCAF['media'] & SA['media'], IVSF['media-alta'], label = 'R5')
    R6 =  ctrl.Rule(CFTV['baixa'] & SCAF['media'] & SA['alta'],  IVSF['media-alta'], label = 'R6')
    R7 =  ctrl.Rule(CFTV['baixa'] & SCAF['alta']  & SA['baixa'], IVSF['alta'], label = 'R7')
    R8 =  ctrl.Rule(CFTV['baixa'] & SCAF['alta']  & SA['media'], IVSF['media-alta'], label = 'R8')
    R9 =  ctrl.Rule(CFTV['baixa'] & SCAF['alta']  & SA['alta'],  IVSF['media'], label = 'R9')
    R10 = ctrl.Rule(CFTV['media'] & SCAF['baixa'] & SA['baixa'], IVSF['alta'], label = 'R10')
    R11 = ctrl.Rule(CFTV['media'] & SCAF['baixa'] & SA['media'], IVSF['media-alta'], label = 'R11')
    R12 = ctrl.Rule(CFTV['media'] & SCAF['baixa'] & SA['alta'],  IVSF['media'], label = 'R12')
    R13 = ctrl.Rule(CFTV['media'] & SCAF['alta']  & SA['baixa'], IVSF['media-alta'], label = 'R13')
    R14 = ctrl.Rule(CFTV['media'] & SCAF['alta']  & SA['media'], IVSF['media-baixa'], label = 'R14')
    R15 = ctrl.Rule(CFTV['media'] & SCAF['alta']  & SA['alta'],  IVSF['media-baixa'], label = 'R15')
    R16 = ctrl.Rule(CFTV['media'] & SCAF['media'] & SA['baixa'], IVSF['media-alta'], label = 'R16')
    R17 = ctrl.Rule(CFTV['media'] & SCAF['media'] & SA['media'], IVSF['media'], label = 'R17')
    R18 = ctrl.Rule(CFTV['media'] & SCAF['media'] & SA['alta'],  IVSF['media-baixa'], label = 'R18')
    R19 = ctrl.Rule(CFTV['alta']  & SCAF['baixa'] & SA['baixa'], IVSF['alta'], label = 'R19')
    R20 = ctrl.Rule(CFTV['alta']  & SCAF['media'] & SA['baixa'], IVSF['media'], label = 'R20')
    R21 = ctrl.Rule(CFTV['alta']  & SCAF['baixa'] & SA['alta'],  IVSF['media-baixa'], label = 'R21')
    R22 = ctrl.Rule(CFTV['alta']  & SCAF['media'] & SA['baixa'], IVSF['media-alta'], label = 'R22')
    R23 = ctrl.Rule(CFTV['alta']  & SCAF['media'] & SA['media'], IVSF['media-baixa'], label = 'R23')
    R24 = ctrl.Rule(CFTV['alta']  & SCAF['media'] & SA['alta'],  IVSF['media-baixa'], label = 'R24')
    R25 = ctrl.Rule(CFTV['alta']  & SCAF['alta']  & SA['baixa'], IVSF['media-baixa'], label = 'R25')
    R26 = ctrl.Rule(CFTV['alta']  & SCAF['alta']  & SA['media'], IVSF['baixa'], label = 'R26')
    R27 = ctrl.Rule(CFTV['alta']  & SCAF['alta']  & SA['alta'],  IVSF['baixa'], label = 'R27')
    # Dicionário de testes
    dic = {}
    dic['disponibilidade_CFTV'] = [VE1]
    dic['disponibilidade_SCAF'] = [VE2]
    dic['disponibilidade_SA'] =   [VE3]
    # Controlador Fuzzy
    controlador_fuzzy = ctrl.ControlSystem(rules = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27])
    engine = ctrl.ControlSystemSimulation(controlador_fuzzy)
    list_IVFS_fuzzy = []
    list_IVFS_medio = []
    list_disp_medio = []

    def predicao_fuzzy (a,b,c):
        engine.input['disponibilidade_CFTV'] = a 
        engine.input['disponibilidade_SCAF'] = b 
        engine.input['disponibilidade_SA'] = c 
        engine.compute()  
        disp_medio = np.mean([a,b,c])
        IVSF_medio = 100 - disp_medio
        list_IVFS_fuzzy.append(round(engine.output['IVSF'],2))
        list_IVFS_medio.append(round(IVSF_medio,2)) 
        list_disp_medio.append(round(disp_medio,2))
    for i in range(len(dic['disponibilidade_CFTV'])):
        predicao_fuzzy(dic['disponibilidade_CFTV'][i],
                       dic['disponibilidade_SCAF'][i],
                       dic['disponibilidade_SA'][i])
    list_IVFS_fuzzy = np.array(list_IVFS_fuzzy)
    list_IVFS_medio = np.array(list_IVFS_medio)
    list_disp_medio = np.array(list_disp_medio)
    data = {'Cenario':list(range(1, len(dic['disponibilidade_CFTV'])+1)),
            'Disponibilidade CFTV (%)': dic['disponibilidade_CFTV'],
            'Disponibilidade SCAF (%)': dic['disponibilidade_SCAF'],
            'Disponibilidade SA (%)': dic['disponibilidade_SA'],
            'Disponibilidade Media (%)': list_disp_medio,
            'IVSF Estimado(%)': list_IVFS_fuzzy,
            'IVSF Medio(%)': list_IVFS_medio,       
           }
    IVSF = data['IVSF Estimado(%)'][0] 
    time_fuzzy = time.time() - start_time
    
def run():

    return

if __name__ == '__main__':
    run()