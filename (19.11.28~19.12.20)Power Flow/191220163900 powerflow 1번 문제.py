import pandas as pd
import numpy as np
import math

Bus = pd.read_excel(".\\data\\IEEE24BUS(수정).xlsx","Bus INFO")
Branch = pd.read_excel(".\\data\\IEEE24BUS(수정).xlsx","Branch")
Transformer = pd.read_excel(".\\data\\IEEE24BUS(수정).xlsx","Transformer")
Gen = pd.read_excel(".\\data\\IEEE24BUS(수정).xlsx","GEN")
Load = pd.read_excel(".\\data\\IEEE24BUS(수정).xlsx","LOAD")

########################## Y BUS MATRIX ############################
Branch_number = len(Branch)
Bus_number = len(Bus)
Trans_number =len(Transformer)

incidence_matrix = np.zeros((Branch_number,Bus_number))
Y_matrix = np.zeros((Branch_number,Branch_number),dtype=np.complex)
B_matrix = np.zeros((Bus_number,Bus_number),dtype=np.complex)
T_matrix = np.zeros((Bus_number,Bus_number),dtype=np.complex)

#incidence matrix 만들기

for i in range(Branch_number):
    incidence_matrix[i][Branch[Branch.columns[0]][Branch.index[i]]-1] = 1
    incidence_matrix[i][Branch[Branch.columns[1]][Branch.index[i]]-1] = -1

#Y 대각행렬 만들기
for i in range(Branch_number):
    Y_matrix[i,i] += 1/complex(Branch[Branch.columns[2]][Branch.index[i]], Branch[Branch.columns[3]][Branch.index[i]])

# B 값 더해주기
for i in range(len(Branch)):
    B_matrix[Branch[Branch.columns[0]][Branch.index[i]]-1, Branch[Branch.columns[0]][Branch.index[i]]-1]  += complex(0,Branch[Branch.columns[4]][Branch.index[i]]/2)
    B_matrix[Branch[Branch.columns[1]][Branch.index[i]]-1, Branch[Branch.columns[1]][Branch.index[i]]-1]  += complex(0,Branch[Branch.columns[4]][Branch.index[i]]/2)


# #변압기 고려
for i in range(Trans_number):
    T_matrix[Transformer[Transformer.columns[0]][Transformer.index[i]]-1,Transformer[Transformer.columns[1]][Transformer.index[i]]-1] = -1/complex(Transformer[Transformer.columns[2]][Transformer.index[i]],Transformer[Transformer.columns[3]][Transformer.index[i]])
    T_matrix[Transformer[Transformer.columns[1]][Transformer.index[i]]-1,Transformer[Transformer.columns[0]][Transformer.index[i]] - 1] = -1 / complex(Transformer[Transformer.columns[2]][Transformer.index[i]],Transformer[Transformer.columns[3]][Transformer.index[i]])

for i in range(Bus_number):
    T_matrix[i,i] = -1* np.sum(T_matrix[i])

# A^T * Y * A 로 Y-Buse matrix 만들기
Y_matrix = np.matmul(np.transpose(incidence_matrix), Y_matrix)
Y_matrix = np.matmul(Y_matrix, incidence_matrix)
Y_matrix = Y_matrix + B_matrix + T_matrix

######################## Power Flow Equation ########################
# PV Bus 정보 가진 set 구하기
PV_Bus_set = np.zeros(len(Gen))
for i in range(len(Gen)):
    PV_Bus_set[i] = Gen[Gen.columns[0]][i]
PV_Bus_set = PV_Bus_set.tolist()

# PQ Bus 정보 가진 set 구하기
PQ_Bus_set = []
for i in range(len(Bus)):
    if Bus[Bus.columns[2]][i] == 1:
        PQ_Bus_set.append(Bus[Bus.columns[0]][i])


# 모든 Bus의 theta, v 값을 가진 matrix
x_total_matrix = np.zeros((Bus_number*2))
for i in range(len(x_total_matrix)):
    if i < Bus_number: # theta에 대한 식
        x_total_matrix[i] = 0

    else : # V에 대한 식
        if Bus[Bus.columns[2]][i-Bus_number] != 1:      #PV, Slack Bus 에 대해 abs(V) 값 넣어줌
            node_name = Bus[Bus.columns[0]][i-Bus_number]
            node_index = PV_Bus_set.index(node_name)
            x_total_matrix[i] = Gen[Gen.columns[2]][node_index]

        else:       # PQ Bus에서 abs(V) 값 넣어줌
            x_total_matrix[i] = 1


# 풀이를 위한 x_matrix 만들기
x_iteration_matrix = np.zeros(Bus_number+len(PQ_Bus_set)-1)
for i in range(len(x_iteration_matrix)):
    if i < Bus_number-1:        # Theta에 대한 식 넣기:
        x_iteration_matrix[i] = 0

    else:       # abs(V) 값 넣기
        x_iteration_matrix[i] = 1


################################################    P, Q 만들기    #######################################################
F_matrix = np.zeros((Bus_number-1+len(PQ_Bus_set)))
for i in range(len(F_matrix)):
    if i < Bus_number-1:    # P에 대한 내용
        F_matrix[i] = 0
        for sigma in range(Bus_number):
            F_matrix[i] += abs(x_total_matrix[i+Bus_number+1]) * abs(x_total_matrix[sigma+Bus_number]) \
                           *(Y_matrix[i+1][sigma].real*math.cos(x_total_matrix[i+1]-x_total_matrix[sigma])\
                             +Y_matrix[i+1][sigma].imag*math.sin(x_total_matrix[i+1]-x_total_matrix[sigma]))

    else:   # Q에 대한 내용
        F_matrix[i] = 0
        node_index = i-(Bus_number-1)  # (Bus_number-1) 은 PV 만큼 무시해준 값. -1은 1번 버스를 무시해주는 값(전압 영역에도 존재하므로 -1 한 번 더 실시)
        node_name = PQ_Bus_set[node_index]
        for sigma in range(Bus_number):
            F_matrix[i] += abs(x_total_matrix[node_name-1+Bus_number]) * abs(x_total_matrix[sigma+Bus_number])\
                           *(Y_matrix[node_name-1][sigma].real * math.sin(x_total_matrix[node_name-1]-x_total_matrix[sigma])\
                            -Y_matrix[node_name-1][sigma].imag * math.cos(x_total_matrix[node_name-1]-x_total_matrix[sigma]))

F_total_matrix = np.zeros(2*Bus_number)
for i in range(len(F_total_matrix)):
    if i < Bus_number-1:        # 모든 Bus의 P에 대한 내용
        F_total_matrix[i] = 0
        for sigma in range(Bus_number):
            F_total_matrix[i] += abs(x_total_matrix[i+Bus_number]) * abs(x_total_matrix[sigma+Bus_number]) \
                                 * (Y_matrix[i,sigma].real * math.cos(x_total_matrix[i]-x_total_matrix[sigma]) \
                                + Y_matrix[i,sigma].imag * math.sin(x_total_matrix[i]-x_total_matrix[sigma]))

    else:       # 모든 Bus의 Q에 대한 내용
        F_total_matrix[i] = 0
        for sigma in range(Bus_number):
            F_total_matrix[i] += abs(x_total_matrix[i]) * abs(x_total_matrix[sigma+Bus_number]) \
                                 * (Y_matrix[i-Bus_number,sigma].real * math.sin(x_total_matrix[i-Bus_number]-x_total_matrix[sigma]) \
                                -Y_matrix[i-Bus_number,sigma].imag * math.cos(x_total_matrix[i-Bus_number]-x_total_matrix[sigma]))

#################################################  Given 값 행렬 만들기  ############################################################################

Given_matrix = np.zeros(Bus_number-1 + len(PQ_Bus_set))

# P_Gen 더해주기

for i in range(len(Gen)):
    node_name = Gen[Gen.columns[0]][i]

    if node_name != 1:
        P_Gen = Gen[Gen.columns[3]][i]
        P_Base = Gen[Gen.columns[9]][i]
        P_Gen = P_Gen/P_Base
        Given_matrix[node_name-2] = P_Gen

# P_load  빼주기

for i in range(len(Load)-1):
    node_name = Load[Load.columns[0]][i]

    if node_name != 1:
        P_Load = Load[Load.columns[13]][i]
        P_Base = Gen[Gen.columns[9]][1]
        P_Load = P_Load/P_Base
        Given_matrix[int(node_name-2)] -= P_Load

# Q_load 빼주기
    if node_name !=1:
        Q_Load = Load[Load.columns[14]][i]
        Q_Base = Gen[Gen.columns[9]][1]
        Q_Load = Q_Load/Q_Base

        if Load[Load.columns[3]][i]==1:
            node_index = PQ_Bus_set.index(node_name)
            Given_matrix[int(node_index +(Bus_number-1))] -= Q_Load

Given_total_matrix = np.zeros(Bus_number*2)        # F_total_matrix 를 위한 Given값 구하기
for i in range(len(Gen)): # P_Gen 값 더하기
    node_name = Gen[Gen.columns[0]][i]

    P_Gen = Gen[Gen.columns[3]][i]
    P_base = Gen[Gen.columns[9]][i]
    P_Gen = P_Gen/P_base

    Given_total_matrix[node_name-1] = P_Gen

for i in range(len(Load)-1): #P_Load 및 Q_Load 값 구하기
    node_name = Load[Load.columns[0]][i]

    P_Load = Load[Load.columns[13]][i]
    Q_Load = Load[Load.columns[14]][i]
    P_base = Gen[Gen.columns[9]][1]

    P_Load = P_Load/P_base
    Q_Load = Q_Load/P_base

    Given_total_matrix[int(node_name)-1] -= P_Load

    if Load[Load.columns[3]][i]==1:
        Given_total_matrix[int(node_name)-1 +Bus_number] -= Q_Load




cnt=0
while True:
    ################################################    jacobian 만들기    ####################################################

    jacobian = np.zeros((len(x_iteration_matrix), len(x_iteration_matrix)))

    for row in range(len(x_iteration_matrix)):
        for column in range(len(x_iteration_matrix)):

            if (row != column) and (row < Bus_number - 1) and (column < Bus_number - 1):  # 대각 X, 2 사분면
                jacobian[row, column] = abs(x_total_matrix[row + 1 + Bus_number]) * abs(x_total_matrix[column + 1 + Bus_number]) \
                                        * (Y_matrix[row + 1, column + 1].real * math.sin(x_total_matrix[row + 1] - x_total_matrix[column + 1]) \
                                        - Y_matrix[row + 1, column + 1].imag * math.cos(x_total_matrix[row + 1] - x_total_matrix[column + 1]))

            elif (row == column) and (row < Bus_number - 1) and (column < Bus_number - 1):  # 대각 O, 2 사분면
                jacobian[row, column] = -1 * F_total_matrix[row + 1 + Bus_number] \
                                        - Y_matrix[row + 1, row + 1].imag * (abs(x_total_matrix[row + 1 + Bus_number]) ** 2)


            elif (row >= Bus_number - 1) and (column >= Bus_number - 1):  # 대각 X, 4 사분면
                node_index_row = row - (Bus_number - 1)
                node_name_row = PQ_Bus_set[node_index_row]
                node_index_col = column - (Bus_number - 1)
                node_name_col = PQ_Bus_set[node_index_col]

                if node_name_col != node_name_row:
                    jacobian[row, column] = abs(x_total_matrix[node_name_col - 1 + Bus_number]) \
                                            * (Y_matrix[node_name_row - 1, node_name_col - 1].real * math.sin(x_total_matrix[node_name_row - 1] - x_total_matrix[node_name_col - 1]) \
                                            - Y_matrix[node_name_row - 1, node_name_col - 1].imag * math.cos(x_total_matrix[node_name_row - 1] - x_total_matrix[node_name_col - 1]))


                else:  # 대각 O, 4 사분면
                    node_index = row - (Bus_number - 1)
                    node_name = PQ_Bus_set[node_index]
                    jacobian[row, column] = F_total_matrix[node_name - 1 + Bus_number] / abs(x_total_matrix[node_name - 1 + Bus_number]) \
                                            - Y_matrix[node_name - 1, node_name - 1].imag * abs(x_total_matrix[node_name - 1 + Bus_number])



            elif (row < Bus_number - 1) and (column >= Bus_number - 1):  # 1 사분면
                node_index = column - (Bus_number - 1)
                node_number = PQ_Bus_set[node_index]

                if row != (node_number - 2):
                    jacobian[row, column] = abs(x_total_matrix[row + 1 + Bus_number]) \
                                            * (Y_matrix[row + 1, node_number - 1].real * math.cos(x_total_matrix[row + 1] - x_total_matrix[node_number - 1])\
                                               + Y_matrix[row + 1, node_number - 1].imag * math.sin(x_total_matrix[row + 1] - x_total_matrix[node_number - 1]))

                else:  # 1사분면 p==q 일 경우
                    jacobian[row, column] = F_total_matrix[row + 1] / abs(x_total_matrix[row + 1 + Bus_number]) \
                                            + Y_matrix[row + 1, row + 1].real * abs(x_total_matrix[row + 1 + Bus_number])


            elif (row >= Bus_number - 1) and (column < Bus_number - 1):  # 3 사분면
                node_index = row - (Bus_number - 1)
                node_number = PQ_Bus_set[node_index]

                if (node_number - 2) != column:
                    jacobian[row, column] = -1 * abs(x_total_matrix[node_number - 1 + Bus_number]) * abs(x_total_matrix[column + 1 + Bus_number]) \
                                            * (Y_matrix[node_number - 1, column + 1].real * math.cos(x_total_matrix[node_number - 1] - x_total_matrix[column + 1]) \
                                               + Y_matrix[node_number - 1, column + 1].imag * math.sin(x_total_matrix[node_number - 1] - x_total_matrix[column + 1]))

                else:  # 3사분면 P==Q 일 경우
                    jacobian[row, column] = F_total_matrix[column + 1] \
                                            - Y_matrix[column + 1, column + 1].real * (abs(x_total_matrix[column + 1 + Bus_number]) ** 2)


            else:
                pass


    ########  inverse of jacobian  ##########################################
    jacobian_inv = np.linalg.inv(jacobian)

################################################  F given - F matrix  ###############

    F_iteration = Given_matrix - F_matrix
    F_compare = F_iteration.copy()
    F_total_matrix = Given_total_matrix - F_total_matrix

    delta_x = np.matmul(jacobian_inv, F_iteration)
    x_iteration_matrix = x_iteration_matrix + delta_x

    #################################################  x_iteration_matrix를 x_total_matrix로 최신화  ##########################################

    for i in range(len(x_iteration_matrix)):
        if i < Bus_number-1:
            x_total_matrix[i+1] = x_iteration_matrix[i]

        else:
            node_name = PQ_Bus_set[i-(Bus_number-1)]
            x_total_matrix[node_name-1 + Bus_number] = x_iteration_matrix[i]


    ################################################ F_iteration 최신화  #########################################################
    F_matrix = np.zeros((Bus_number - 1 + len(PQ_Bus_set)))

    for i in range(len(F_matrix)):
        if i < Bus_number - 1:  # P에 대한 내용
            F_matrix[i] = 0
            for sigma in range(Bus_number):
                F_matrix[i] += abs(x_total_matrix[i + Bus_number + 1]) * abs(x_total_matrix[sigma + Bus_number]) \
                               * (Y_matrix[i + 1][sigma].real * math.cos(x_total_matrix[i + 1] - x_total_matrix[sigma]) \
                                  + Y_matrix[i + 1][sigma].imag * math.sin(x_total_matrix[i + 1] - x_total_matrix[sigma]))

        else:  # Q에 대한 내용
            F_matrix[i] = 0
            node_index = i - (Bus_number - 1)  # (Bus_number-1) 은 PV 만큼 무시해준 값. -1은 1번 버스를 무시해주는 값(전압 영역에도 존재하므로 -1 한 번 더 실시)
            node_name = PQ_Bus_set[node_index]
            for sigma in range(Bus_number):
                F_matrix[i] += abs(x_total_matrix[node_name - 1 + Bus_number]) * abs(x_total_matrix[sigma + Bus_number]) \
                               * (Y_matrix[node_name - 1][sigma].real * math.sin(x_total_matrix[node_name - 1] - x_total_matrix[sigma]) \
                                  - Y_matrix[node_name - 1][sigma].imag * math.cos(x_total_matrix[node_name - 1] - x_total_matrix[sigma]))

    F_total_matrix = np.zeros(2 * Bus_number)
    for i in range(len(F_total_matrix)):
        if i < Bus_number - 1:  # 모든 Bus의 P에 대한 내용
            F_total_matrix[i] = 0
            for sigma in range(Bus_number):
                F_total_matrix[i] += abs(x_total_matrix[i + Bus_number]) * abs(x_total_matrix[sigma + Bus_number]) \
                                     * (Y_matrix[i, sigma].real * math.cos(x_total_matrix[i] - x_total_matrix[sigma]) \
                                        + Y_matrix[i, sigma].imag * math.sin(x_total_matrix[i] - x_total_matrix[sigma]))

        else:  # 모든 Bus의 Q에 대한 내용
            F_total_matrix[i] = 0
            for sigma in range(Bus_number):
                F_total_matrix[i] += abs(x_total_matrix[i]) * abs(x_total_matrix[sigma + Bus_number]) \
                                     * (Y_matrix[i - Bus_number, sigma].real * math.sin(
                    x_total_matrix[i - Bus_number] - x_total_matrix[sigma]) \
                                        - Y_matrix[i - Bus_number, sigma].imag * math.cos(
                            x_total_matrix[i - Bus_number] - x_total_matrix[sigma]))

    F_iteration = Given_matrix - F_matrix
    F_total_matrix = Given_total_matrix - F_total_matrix
    ############################################### 정지 조건 확인 #################################################################



    if abs(np.sum(F_iteration-F_compare)) < 0.0000001:
        break

    cnt +=1

##################################################    선로 용량 초과 확인   ####################################################################

Rate_A = np.zeros(len(Branch))      # 선로 용량 입력
for i in range(len(Branch)):
    Rate_A[i] = Branch[Branch.columns[5]][i]/Gen[Gen.columns[9]][2]

Power_on_line = np.zeros(len(Branch))
for i in range(len(Branch)):
    Power_on_line[i] = ((abs(x_total_matrix[Branch[Branch.columns[0]][i]-1+Bus_number]))**2 \
                        + (abs(x_total_matrix[Branch[Branch.columns[1]][i]-1+Bus_number]))**2 \
                        -2*abs(x_total_matrix[Branch[Branch.columns[0]][i]-1+Bus_number])*abs(x_total_matrix[Branch[Branch.columns[1]][i]-1+Bus_number])\
                        *math.cos(x_total_matrix[Branch[Branch.columns[0]][i]-1]-x_total_matrix[Branch[Branch.columns[1]][i]-1]))\
                       *Y_matrix[Branch[Branch.columns[0]][i]-1][Branch[Branch.columns[1]][i]-1]



cnt2=0
for i in range(len(Branch)):
    if i!=0 and Power_on_line[i] >= Rate_A[i]:
        print("{node1}, {node2} node 간 송전 용량 초과".format(node1= Branch[Branch.columns[0]][i], node2 = Branch[Branch.columns[1]][i]))
        cnt+=1

if cnt2 ==0:
    print("송전용량 초과 해당사항 없음")