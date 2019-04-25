import numpy as np
import pandas as pd

def dw(C1, C2): #ward distance
    return (len(C1) * len(C2)) * np.sum(np.square(np.subtract(np.mean(C1, axis=0), np.mean(C2, axis=0)))) / (len(C1) + len(C2))

def gen_matrix(df_list):
    mat = list()
    temp = list()

    for C1 in df_list:
        for C2 in df_list:
            temp.append(dw(C1, C2))

        mat.append(temp[:])
        temp.clear()

    return mat

def update(df_list, mat, minX, minY):
    if isinstance(df_list[minX][:][0], list):
        temp = df_list[minX][:]
    else:
        temp = [df_list[minX][:]]

    if isinstance(df_list[minY][:][0], list):
        for l in df_list[minY][:]:
            temp.append(l)
    else:
        temp.append(df_list[minY][:])

    df_list.append(temp[:])
    del df_list[max(minY, minX)]
    del df_list[min(minY, minX)]


    del mat[max(minY, minX)]
    del mat[min(minY, minX)]
    for row in mat:
        del row[max(minY, minX)]
        del row[min(minY, minX)]

    temp.clear()
    for C1 in df_list:
        temp.append(dw(C1, df_list[-1]))
    mat.append(temp[:])
    #bad code start
    count = 0
    for row in mat:
        if count < len(mat) - 1:
            row.append(temp[count])
            count += 1
    #bad code end

    return (df_list, mat)

def aglomerativeClustering(df):
    df_list = df.values.tolist()
    mat = gen_matrix(df_list)

    while(len(df_list) > 1):
        minX = 1
        minY = 0
        mn = mat[minX][minY]

        for x in range(len(mat)):
            for y in range(len(mat[0])):

                if y >= x: continue

                if mat[x][y] < mn:
                    minX = x
                    minY = y
                    mn = mat[x][y]

        (df_list, mat) = update(df_list, mat, minX, minY)


df = pd.DataFrame([[5,2,1],[3,4,3],[2,8,8],[1,1,1],[4,5,5],[2,3,2],[5,5,5],[2,3,4],[2,2,2]])

#print(df.values.tolist())
#print(gen_matrix(df))
aglomerativeClustering(df)